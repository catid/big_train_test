# Stability brief mapped to the canonical H-Net run

This note records how the supplied “stabilize and pretrain an H-Net with
GDN2, LoL, and full attention” brief applies to the repository as it exists
today.  It is a research and release-gate note; it does not change the
production model or trainer.

## Audit basis

- Repository: `/home/catid/big_train_test`
- Audited commit: `ab501a5` (`Document progressive middle-depth growth research`)
- The working tree is dirty.  The uncommitted changes include the S=8192
  configuration/trainer switch and related documentation/tool edits.  Results
  below describe the current files, not only the committed tree.
- The canonical trainer is `train_hnet_600m.py`; artifacts and checkpoints
  belong outside this code repository.

## Current architecture and update flow

The production configuration is
`configs/hnet_1stage_600m_loop2.json` and is validated by
`hnet/models/config_hnet.py`.  The full runtime reports 603,935,676 trainable
parameters for the raw-byte model with `d_model=[1024,1536]`, vocabulary 256, and static sequence length
8192 (the uncommitted S8192 switch; earlier measurements are S4096).

The hierarchy in `hnet/models/hnet.py` and `hnet/models/mixer_seq.py` is:

```text
fixed byte lift [B,S] -> [B,S,1024]
  -> outer encoder: S2M3 (2 x 128-token SWA, then 3 x Mamba-3)
  -> learned causal chunker, left-aligned and capped at 512 latent rows
  -> middle: (S3T1)^5 (3 x SWA, then 1 x global FA), shared inner loop x2
  -> learned dechunker
  -> outer decoder: S2M3
  -> FP32 byte LM head [B,S,1024] -> [B,S,256]
```

The two middle executions share parameters and the layer KV slot is rewritten
for each refinement.  MSAR is enabled only in the innermost stack, at stream
boundaries every four layers (`hnet/modules/isotropic.py` and
`hnet/modules/msar_triton.py`).  FBT is a two-pass chunk-level objective at
the outer boundary (`hnet/models/hnet.py`); it does not alter chunk masks or
outer encoder/decoder state.  Inference caches keep bounded Mamba/MSAR tails,
the recurrent H-Net state, and VHA KV state.

SWA uses a 128-token window, shared 48-dimensional LeRoPE, and FlashAttention
windows.  Global FA is explicitly NoPE, uses VHA with two physical KV heads
and rank 4, and enables the patched CroissAttn auxiliary path.  The exact
layer selection and position contract are in `hnet/modules/isotropic.py`,
`hnet/modules/mha.py`, and `hnet/modules/rotary.py`.

The FFN replacement is `hnet/modules/latent_moe.py`: eight top-1 routed
experts, fixed capacity, and one always-on shared expert.  The latent path is
384 wide in the 1536-wide stack, routed experts have 1664 intermediate units,
and the shared expert has 512.  Router logits/probabilities, auxiliary terms,
and accumulation are FP32 around TE FP8 grouped GEMMs.  The trainer records
router balance and expert Gini, but capacity overflow is not yet exposed as a
metric.

The byte code is a persistent, row-normalized `[256,1024]` Gaussian codebook
(`FixedByteFeatures` in `hnet/models/mixer_seq.py`), with no trainable input
embedding.  The output head is trainable and FP32.  Document-aware sampling in
`train_hnet_600m.py` supplies an active mask and a loss mask that prevents a
target from crossing a document boundary; this mask also gates recurrent,
chunk, FBT, MSAR, and KV-cache paths.

### Precision and loss contracts

BF16 is the activation default.  FP8 TE covers attention projections, global
FA DPA, latent-MoE grouped/shared/router projections, and FBT projections.
The fixed byte lift, Mamba-3 fused state-space core, router reductions,
residual controls, norms, and LM head remain higher precision by contract.
`ce()` in `train_hnet_600m.py` casts logits and reductions to FP32.  The
reported `loss`/`standard_ce` is plain byte cross-entropy; the optimized
`objective` adds the two-pass FBT term, chunk ratio loss, and MoE auxiliary
loss.  The current z-loss is a router z-loss folded into the MoE auxiliary
term; a PaLM output-logit z-loss and LM-head μ-loss/centering intervention are
not currently production options.

### Optimizer partition

`optimizer_experiments.py` implements AnchorMuon: SODA anchor pull, row-only
PMuonEq, Gram Newton-Schulz, NorMuon, decoupled weight decay, and an
AdamATan2 path for vectors/skinny controls.  `build_param_groups()` currently
partitions by effective tensor shape (matrix-like versus fallback) and
deduplicates shared parameters by object identity.  This is operationally
safe, but it is less explicit than the brief's desired semantic table:

| Category in the brief | Current status |
|---|---|
| Wide trunk/attention/MoE matrices | Shape-based AnchorMuon matrix group |
| Gates, biases, norms, router controls, chunk controls | Shape-based AdamATan2 fallback, with forced fallback markers where present |
| Byte lift | Fixed buffer, no optimizer entry |
| LM head | Matrix-like classification is shape-driven; no dedicated head LR/SGD-M ablation in the canonical trainer |
| Weight decay | Implemented in both optimizer paths, with `_no_weight_decay` exclusions |

The exact group summary is serialized in checkpoints, but gradient/update
norms, update-to-parameter ratios, cosine, skipped-step counts, and stale
state age for rarely used experts are not all logged yet.

### Distributed and checkpoint flow

The trainer uses one DDP process per GPU, BF16/FP8 CUDA graphs for the fixed
shape, direct reducer storage when `gradient_as_bucket_view` is available,
and a dedicated capture stream.  Runtime provenance includes loaded binary
hashes, CUDA/TE settings, architecture signature, optimizer state, RNG state
per rank, data seed/manifest hashes, schedule, world size, batch, and sequence
length.  Resume now rejects a checkpoint whose static `max_seq_len` differs
from the current graph contract.

## Discrepancies from the supplied brief

1. The brief names GDN2 as a required mixer.  The current canonical repository
   uses `Mamba3Wrapper` in `hnet/modules/block.py` and does not instantiate
   GDN2.  GDN2 guidance therefore cannot be claimed as implemented; the
   equivalent Mamba-3 state/reference diagnostics remain to be specified.
2. The brief names LoL's causal, window-level dynamic layer-library router.
   The current production middle layout is fixed `S3T1` ordering with shared
   depth loops.  It is not the LoL paper's learned runtime layer-order router.
   Continue labeling this as fixed S3T1 until a separately reviewed LoL router
   is implemented and passes causal/reference parity.
3. The brief suggests a 40M staged research track.  Project instructions
   require the complete approximately 600M model; small deterministic probes
   may test mathematics and kernels, but no smaller production run should be
   presented as evidence for the canonical run.
4. The S8192 edits retain a 512-latent cap and set the ratio target to 16.
   This leaves no average-capacity headroom (`8192/16 = 512`) and has not yet
   been measured on a full graph step.

## Advice already reflected in code

- Raw-byte CE/BPB remains visible, with separate ratio, FBT, and MoE terms.
- Document manifest sampling and boundary masks preserve causal document
  structure; separator bytes are not used as a fake state reset.
- Static chunk capacity, left-aligned routing, and causal clipping keep graph
  shapes fixed and prevent a future boundary from evicting an earlier one.
- Router, auxiliary, CE, and softmax-sensitive reductions are FP32.
- FBT has a fixed two-pass objective and fresh caches; recurrent and KV state
  are not silently reused between training passes.
- FP8/CroissAttn runtime checks fail closed on unreviewed binaries, and
  checkpoints contain runtime/dependency provenance.
- DDP graph capture retains reducer bucket views where possible and avoids
  capture-unsafe DDP logging.
- `expert_gini`, expert load, route ratios, LR phase, throughput, MFU, power,
  and evaluation records are already part of run telemetry.

## Release gates before a serious S8192 run

The following are evidence gaps, not claims that the implementation is wrong:

1. Run fixed-input eager versus compiled, FP8, and CUDA-graph parity for
   logits, every loss component, gradients, optimizer updates, and optimizer
   state. Include both FBT passes and both shared inner loops.
2. Test future-byte perturbations, document resets, chunk-cap behavior, and
   chunkwise versus tokenwise recurrent reference behavior at the full static
   shape. Add explicit chunk-cap overflow and post-cap-byte metrics.
3. Add MoE dropped/overflow-token fraction per layer and compare it with expert
   utilization; current capacity overflow is silently sent to the shared path.
4. Add Mamba-3 state RMS/max/update and decay/gate diagnostics. GDN2-specific
   diagnostics are applicable only if the architecture is intentionally moved
   back to GDN2.
5. Validate the fixed S3T1 routing contract and document that LoL router parity
   is not yet applicable. If adopting dynamic LoL routing later, route by
   causal window, begin top-1, keep router statistics FP32, and checkpoint every
   depth transition.
6. Add LM-head mean/norm and raw-logit RMS/max diagnostics, plus FP32-versus-
   production logit RMS/KL checks. Evaluate untied μ-centering, a dedicated
   head optimizer group, and PaLM z-loss as isolated ablations.
7. Emit per-group gradient/update norms, update ratios, cosine, zero-gradient
   frequency, and optimizer-state age. Verify every parameter belongs to one
   intended optimizer group, including shared loop weights and expert banks.
8. Validate checkpoint/resume and deterministic rank-sharded data at world
   sizes 1/2/4/8/12 where hardware permits. Confirm global reduction semantics
   for chunk counts, route loads, means, losses, and diagnostics.
9. Measure complete S8192 graph steps, memory, power, and throughput before
   selecting `CANONICAL_MAX_BATCH=8`. The previous S4096 B18 result cannot be
   extrapolated. Update historical S4096 numbers in `TRAINING.md` only after a
   measured S8192 result exists.

## S8192 performance notes from the asynchronous review

The Claude review (OAuth subscription session, source-only review; no run was
executed) identified these high-value checks:

- Measure chunk-cap overflow. At the current ratio/cap combination dense rows
  can hit the cap, after which later bytes are decoded from the last latent.
- Add cap overflow, post-cap-byte, and MoE dropped-token fractions to the
  existing single diagnostic all-reduce.
- `DeChunkLayer` receives sort indices but currently redoes an `argsort`; reuse
  the chunk sort/p and `dt` across the two FBT passes after proving parity.
- The host-side finite flag check (`finite_flag.item()`) can serialize every
  step; move the skip decision into the captured device update and inspect on
  report steps only.
- Profile the full-width FP32 residual projection at S8192. Preserve FP32
  accumulation while testing BF16 input if it is a material bottleneck.
- Update S8192 runbook numbers and reject old S4096 checkpoints explicitly;
  the current static-shape validation is the right safety behavior.

These are optimization or instrumentation candidates, not changes to apply
before eager correctness and parity evidence.

## Pinned references and provenance

The following links are a durable copy of the reading list from the brief;
the local SHAs below are the revisions used for source inspection:

- [H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) and [official code](https://github.com/goombalab/hnet)
- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), [official code](https://github.com/NVlabs/GatedDeltaNet-2), and [cuDNN linear-attention operators](https://docs.nvidia.com/cudnn/operations/LinearAttention)
- [Library of Layers](https://arxiv.org/abs/2611.03526) and [official code](https://github.com/arahmani1/library-of-layers)
- [The Role of Quantization in Loss Spikes](https://arxiv.org/abs/2610.38277) and [reproduction code](https://github.com/tingtang2/quantization-loss-spikes)
- [Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031)
- [PaLM z-loss](https://arxiv.org/abs/2204.02311)
- [adam-atan2](https://github.com/imoneoi/adam-atan2) and [adam-atan2-pytorch](https://github.com/lucidrains/adam-atan2-pytorch) (comparison references only; the checked-in implementation is authoritative)

The reviewed local revisions currently recorded by the repository are:

- GDN2 checkout: `5ec53d56da98841f6613a0a1c1a59ca1aad9b44c`
- `library_of_layers`: `03b85f42d30d628a973ec759eeb3491b56f1d9e4`
- Transformer Engine (`~/fp8/TransformerEngine`):
  `9646c3ef22455b6948cbe870f1aaf20fc8238d69`
- BioNeMo Mixtral TE routing reference: `2686b2418889ce346c5e50ea8614bb3b27415b97`

The usable `/tmp/hnet-venv` runtime satisfies the reviewed binary gate for the
full run: PyTorch `2.10.0+cu130`, cuBLASLt version code `130802`, cuDNN
`9.23.2`, FlashAttention `2.8.3.post2`, Mamba-SSM `2.3.2.post1`, and the
Transformer Engine checkout recorded above.  A complete eight-rank S8192
FP8/DDP/CUDA-graph run is now finite at B=1, B=8, and B=16.  B=32 completes
eager calibration but fails closed during graph capture because the MSAR path
exhausts the 267.7 GiB device.  The measurements and exact commands are in
`docs/research/s8192_calibration_2026-11-06.md`; older S4096 numbers remain
historical and are not extrapolated.

## Execution order

1. Finish inventory and eager reference/parity tests for the current full
   architecture and checkpoint contract.
2. Run full 600M S8192 eager, BF16/FP8, and graph parity, including the
   diagnostics above; do not stage a smaller production architecture.
3. Calibrate complete DDP batches and profile the actual dispatched kernels,
   power, memory, and scaling before a long run.
4. Run the controlled stability ablations (LM head, optimizer grouping,
   Mamba-3 safeguards, routing diagnostics) with fixed data/schedule, then
   perform the extended-horizon run only on configurations with evidence.

