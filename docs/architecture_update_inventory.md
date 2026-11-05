# Canonical H-Net architecture and update-flow inventory

This inventory is an audit of the checkout at commit `48fb2f4` (the working
tree also contains uncommitted runtime edits). It describes what the code does,
not what the research brief intends. Items marked **unverified** or **TODO**
must be resolved before a long pretraining run.

## Configuration and external revisions

The only accepted configuration is
`configs/hnet_1stage_600m_gdn2_lol_8192.json`:

| Setting | Current value | Evidence |
|---|---:|---|
| byte vocabulary | 256 | config `vocab_size`; `FixedByteFeatures` |
| outer width | 3072 | config `d_model[0]`, GDN2 width |
| latent width | 1280 | config `d_model[1]`, LoL/FA width |
| byte sequence | 8192 | `max_seq_len`, `canonical_seq_len` |
| latent capacity | 2048 rows | `latent_max_seq_len` and `ChunkLayer` |
| local attention window | 128 | `lol_window_size` |
| GDN2 shape | 48 heads × 64 | config and `GDN2Adapter` |
| LoL/FA shape | 20 query heads × 4 KV heads × 64 | config |
| trunk repeats | 2 | `inner_loops` |
| FBT passes | 2 by trainer default | `--fbt-passes` |
| routed experts | 8, top-4 | config and `LatentMoE` |
| external GDN2 checkout | `/home/catid/gdn2`, `5ec53d5` in config | config metadata; current checkout was inspected at this revision |
| external LoL metadata | `/home/catid/library_of_layers`, `03b85f4` in config | config metadata |

The user supplied GDN2 reproducibility handoff names commit `11d9f76`, cuDNN
1.32, and the CUTLASS DSL path. The current config still records `5ec53d5`;
this is a release-blocking provenance mismatch, not an optimization detail.
Record the exact dependency commits and `pip`/CUDA/driver versions in every
run manifest after resolving it. Transformer Engine is not declared in
`pyproject.toml`; the optional import in `LatentMoE` can only prove FP8 after a
working CUDA/PyTorch extension is installed.

## Tensor and sequence flow

All activation tensors use BF16 unless a row below explicitly says FP32. The
LM head and residual projection are FP32. The boundary-router projection follows
the model activation dtype and is normalized/reduced in FP32; the LatentMoE
router projection itself is allocated in FP32. Sensitive probabilities, losses,
and reductions are explicitly promoted to FP32.

| Stage | Shape and operation | Sequence/mask contract |
|---|---|---|
| Input | `input_ids: [B, 8192]`, integer byte IDs | Trainer supplies an all-true mask. `ByteStream` slices a contiguous byte file; no document IDs or reset mask are carried. |
| Fixed byte lift | immutable `[256, 3072]` code buffer gathers to `[B, 8192, 3072]` | Buffer is persistent and non-trainable. It is not a learned embedding, but the code still calls `F.embedding`; this is only a gather implementation. |
| Encoder | `HybridStack(stage=0)`: `GDN2(3072) -> LoL(1280) -> GDN2(3072) -> LoL(1280)`, with explicit linear bridges and final projection to 3072 | Outer stack drops the all-true mask in static mode to select causal/window kernels. LoL attention uses learned RoPE and shifted causal windows. |
| Boundary router | fused `[3072 -> 6144]` Q/K projection; normalized adjacent Q/K cosine; `boundary_prob: [B,8192,2]` FP32 and hard `boundary_mask: [B,8192]` | First position is forced to a boundary. With packed inputs, document starts are forced to boundaries. The trainer has no document-boundary metadata. |
| Chunker | selected boundary rows packed to `[B, 2048, 3072]`; `next_mask: [B,2048]` | The static cap retains earliest boundaries when more than 2048 are selected. This changes the learned resolution and must be logged as a phase/constraint. |
| Latent input bridge | `Linear(3072 -> 1280)` | Padding rows remain in the fixed buffer and are represented by `next_mask`. |
| Middle trunk | two executions of `GDN2(3072) -> LoL(1280) -> GDN2(3072) -> LoL(1280) -> FA(1280)` | The middle `GDN2` core is shared between repeats; the middle `FA` core is shared; LoL cores and bridges are distinct. LoL uses learned RoPE shifted windows; FA uses causal NoPE GQA. Current code executes this fixed order; it does not perform LoL library routing. |
| Latent output bridge | `Linear(1280 -> 3072)` | Returns fixed latent buffer and mask to dechunker. |
| Dechunker | causal EMA/scan expands latent rows to `[B,8192,3072]` | `p` is clamped to `[1e-4, 1-1e-4]`; scan uses `mamba_chunk_scan_combined` with FP32 `A` and BF16 scan tensors. This is a dechunking primitive, not a Mamba mixer, but the dependency remains in `pyproject.toml`. |
| Residual and decoder | FP32 `residual_proj(3072 -> 3072)` is added with selected boundary probability; decoder repeats the outer `GDN2 -> LoL -> GDN2 -> LoL` stack | FBT repeats the latent trunk/dechunk/decoder path. FBT feedback is shifted within each document in packed mode and zeroed at document starts. |
| Output | FP32 `lm_head: [3072,256]`, logits `[B,8192,256]` | Primary byte CE is computed on `logits.float()`. Input and output are untied. |

The H-Net hierarchy is causal in the intended direction: encoder and router
see the current prefix, chunk rows retain source order, the middle stack is
causal, and the dechunk scan gathers only the most recent selected chunk. This
has not been demonstrated by a future-byte perturbation test in this checkout.

## What is actually routed

`hnet/modules/hybrid.py` names local attention blocks “LoL”, but
`HybridStack.__init__` constructs a fixed `ModuleList` for every loop and
`_HybridSequence.forward` iterates every step. There is no current implementation
of the LoL paper’s:

- global block library with a selectable library size;
- causal window router input based on prior-window summaries;
- top-1 argmax plus load-balancing bias;
- probability-weighted residual interpolation;
- router-noise schedule, loss-free balancing, route entropy, or route-change logs;
- depth-growth schedule.

The current local shifted-window attention is a useful fixed-shape attention
primitive and should be kept as a reference path. It must not be reported as
LoL algorithm parity until the router and fixed-route control exist.

## Optimizer and update flow

The trainer defaults to `AnchorMuon` (`--optimizer gramns`), with Muon/GramNS
for parameters passing an effective-shape matrix test and an AdamATan2-style
fallback for the rest. The current defaults are:

| Item | Current value |
|---|---|
| matrix LR | `1.2e-3` |
| fallback LR | `6e-4` |
| momentum | `0.95` |
| fallback betas | `(0.9, 0.95)` |
| matrix weight decay | `0.05` |
| fallback weight decay | `0.05` |
| SODA anchor pull | `0.003` |
| PMU-1Q beta / row gamma | `0.90` / `0.45` |
| NorMuon beta2 | `0.93` |
| gradient clipping | global norm 1.0 |
| schedule | 2,000-step linear warmup, cosine to 10% of base LR |

`build_param_groups` deduplicates shared tensors by object identity. It excludes
the LM head and convolution-like names from Muon, but otherwise routes by shape;
there is no explicit router/gate/chunker/FA parameter-category table. Thus a
two-dimensional router matrix can enter Muon, contrary to the brief’s starting
hypothesis that router and gate parameters should use AdamAtan2. The trainer
does not log `group_summary`, per-module update norms, update/parameter ratios,
or optimizer-state age. `AdamW` and `ANVIL-II` remain command-line alternatives,
so the canonical claim should be limited to the GramNS/AdamAtan2 mapping until
those alternatives are deliberately audited.

The update path is eager forward, FP32 loss, backward, global clip, finite check,
and optimizer step. CUDA graph mode captures forward/backward only; the optimizer
step remains eager. This is safer for the custom optimizer but does not satisfy a
requirement to capture the complete optimizer update. No optimized/eager parity
test exists.

## Losses and precision

The objective currently contains:

1. final byte cross-entropy in FP32;
2. FBT auxiliary CE for earlier passes;
3. H-Net boundary/ratio regularizer;
4. LatentMoE auxiliary load regularizer;
5. LatentMoE router z-loss.

The trainer logs only total loss, primary CE, ratio, gradient norm, and
non-finite step numbers. It does not log each loss component independently.
There is no LM-head z-loss, output-embedding μ-centering, μ-loss, low/high
precision logit RMS error, or softmax KL diagnostic. `clip_grad_norm_` is called
with its default non-error behavior; a non-finite step is recorded and skipped,
then training continues. The brief requires preserving the original failure,
saving the last healthy checkpoint, and stopping for diagnosis.

FP8 is entered through Transformer Engine’s delayed-scaling context in
`LatentMoE` when the extension is present. Router logits, routing probabilities,
auxiliary losses, CE, and the output head stay FP32. Amax history, saturation,
underflow, actual dispatch backend, and FP8 reference error are not logged.

## Checkpoint and distributed state

The checkpoint currently contains `step`, serialized config, model state, and
optimizer state. It does **not** contain RNG states (Python/CPU/CUDA), data
position, world size, schedule phase, router-noise state, chunker state,
recurrent/dechunk state, or a loader manifest. There is no `--resume` path, so
checkpoint/resume parity is not established.

DDP uses one process per GPU, a 256 MB bucket, gradient-as-bucket-view, and
`static_graph=True` only in CUDA-graph mode. Eager routing sets
`find_unused_parameters=True`; graph mode assumes every fixed-capacity expert
path participates. Loss logging reduces the primary scalar but leaves ratio and
MoE diagnostics rank-local. No 1/2/4/8/12-GPU trajectory or rank-statistics
parity test is present.

## Release-blocking TODOs

- Add deterministic causality tests for encoder, chunker, dechunker, GDN2, LoL
  windows, FA, and FBT document starts.
- Add fixed-input eager/BF16/FP8/compiled/CUDA-graph parity for logits, every
  loss term, activations, gradients, parameter updates, and optimizer state.
- Implement or explicitly defer the actual LoL router; keep fixed shifted
  attention labeled as the reference primitive until then.
- Add a document-aware byte packer and boundary reset mask; raw concatenated
  bytes cannot prove no state leakage across documents.
- Add FP32 reference GDN2 recurrence and chunkwise/tokenwise parity using the
  exact external commit.
- Instrument H-Net chunk counts/lengths, compression, boundary entropy, GDN2
  state/gate distributions, LoL routes, FA logits/entropy, LM-head drift, and
  optimizer update geometry.
- Make parameter groups explicit by semantic category and assert every trainable
  parameter belongs to exactly one intended group.
- Extend checkpoint/resume to include all stochastic, schedule, loader, and
  recurrent state; validate uninterrupted versus resumed trajectories.
- Run deterministic DDP parity at each available world size before enabling
  compilation or graph capture.
- Resolve the GDN2 commit mismatch and pin/install a verified Transformer Engine
  extension; record actual kernel dispatch and FP8 scale telemetry.

