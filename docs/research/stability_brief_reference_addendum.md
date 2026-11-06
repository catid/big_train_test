# Stability brief reference addendum

This addendum is the current, source-linked reading note for the stability
brief supplied with the project.  It was checked against repository `HEAD`
`f4b4f82` (`Fuse LatentMoE route weighting into TE unpermute`) on 2026-11-06.
It preserves the useful recommendations without treating proposed experiments
or paper terminology as features that are already present in the canonical
trainer.

The earlier [`stability_brief_mapping.md`](stability_brief_mapping.md) remains
the detailed implementation inventory.  Its opening audit metadata and the
older `CANONICAL_MAX_BATCH=8` release-gate wording describe an earlier state;
the current S8192 calibration record supersedes those measurements:
[`s8192_calibration_2026-11-06.md`](s8192_calibration_2026-11-06.md) records a
complete graph-validated per-GPU batch of 16 and the B32 capture OOM.

## Source record

These are the primary sources named in the brief.  URLs are kept here so a
future audit can repeat the reading; local revisions are recorded only where
the project already inspected a checkout.

| Topic | Primary source | Local revision or provenance |
| --- | --- | --- |
| H-Net dynamic chunking | [paper](https://arxiv.org/abs/2507.07955), [official code](https://github.com/goombalab/hnet) | inspect the repository's H-Net implementation before changing chunk semantics |
| GDN2 | [paper](https://arxiv.org/abs/2605.22791), [NVlabs code](https://github.com/NVlabs/GatedDeltaNet-2), [cuDNN linear attention docs](https://docs.nvidia.com/cudnn/operations/LinearAttention) | the inspected GDN2 checkout is `5ec53d56da98841f6613a0a1c1a59ca1aad9b44c`; GDN2 is not instantiated by the current trainer |
| LoL | [paper](https://arxiv.org/abs/2611.03526), [official code](https://github.com/arahmani1/library-of-layers) | inspected `library_of_layers` revision `03b85f42d30d628a973ec759eeb3491b56f1d9e4`; the current middle stack is fixed S3T1, not a learned LoL router |
| Quantization loss spikes | [paper](https://arxiv.org/abs/2610.38277), [reproduction code](https://github.com/tingtang2/quantization-loss-spikes) | use as a diagnostic motivation, not as evidence that the current LM head is unstable |
| Output embedding centering | [paper](https://arxiv.org/abs/2601.02031) | proposed ablation; no production centering is enabled |
| PaLM z-loss | [paper](https://arxiv.org/abs/2204.02311) | proposed output-loss ablation; current z-loss is router-specific |
| Optimizers | [adam-atan2](https://github.com/imoneoi/adam-atan2), [PyTorch reference](https://github.com/lucidrains/adam-atan2-pytorch) | checked-in `optimizer_experiments.py` is authoritative; external code is comparison material only |
| LatentMoE/dispatch | [paper](https://arxiv.org/abs/2601.18089), [reference](https://github.com/kyegomez/Latent-MoE), [Megatron MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) | local hand-edited transcription is `docs/latentmoe_2601.18089.md`; TE/BioNeMo source was inspected at `9646c3ef22455b6948cbe870f1aaf20fc8238d69` and `2686b2418889ce346c5e50ea8614bb3b27415b97` |

The measured runtime used for the latest S8192 evidence is outside the
repository at `/tmp/hnet-venv`: PyTorch `2.10.0+cu130`, cuDNN `9.23.2`,
FlashAttention `2.8.3.post2`, Mamba-SSM `2.3.2.post1`, and the Transformer
Engine revision above.  These versions are provenance, not an invitation to
silently upgrade dependencies.

## Advice to carry into implementation and review

### Establish a reference before optimizing

1. Keep one complete raw-byte architecture and a fixed-shape eager reference.
   The production model is approximately 604M parameters with learned causal
   chunking, Mamba-3 outer stacks, fixed S3T1 middle ordering, MSAR, FBT,
   LatentMoE, FP8 TE, and AnchorMuon/AdamATan2.  GDN2 and dynamic LoL are
   research references in the brief, not hidden fallbacks.
2. Compare eager, compiled, FP8, and CUDA-graph paths on the same full shape.
   Compare logits, each loss component, gradients, optimizer updates, and
   optimizer state.  A finite loss alone is not parity evidence.
3. Keep optimization changes separate from correctness changes.  Record the
   dispatched backend and dependency revision with every timing result.

### Preserve causal hierarchy and state contracts

- Test future-byte perturbations at the chunker, dechunker, FBT, MSAR, KV,
  recurrent, and byte-loss interfaces.  Document boundaries must reset the
  appropriate state; a separator byte is not a reset.
- Log chunk count, length statistics, compression ratio, boundary rate,
  hard-cap hits, and bytes decoded after the latent cap.  The current 512-row
  cap is a fixed graph contract, so cap overflow must be visible even when
  the graph remains finite.
- If a future experiment changes hierarchy ratio, depth, routing semantics, or
  gate parameterization, save checkpoints immediately before and after the
  named transition and compare optimizer-state policies.

### Keep sensitive math in FP32

- Keep CE, router logits/probabilities, load statistics, softmax-sensitive
  reductions, norms, and optimizer state in FP32 unless a reference comparison
  demonstrates an equivalent lower-precision implementation.
- For recurrent mixers, compare chunkwise and tokenwise recurrence on short
  fixed inputs and log state RMS/max/update norms plus gate or decay ranges.
  The current canonical mixer is Mamba-3; GDN2-specific erase/write
  diagnostics apply only if the architecture is deliberately changed.
- For attention, verify causal masking, Q/K scale, positional encoding, logit
  range, entropy, output-gate scale, and backend dispatch against an FP32
  reference.  The current contract is LeRoPE in 128-token SWA and NoPE in
  global FA.

### Treat the output head and optimizer partition as separate systems

- Log output-head mean/norm, raw-logit mean/std/max, and the low/high-precision
  logit RMS and softmax KL on a diagnostic batch.  Test μ-centering, a
  dedicated head optimizer, and PaLM z-loss as isolated, measured ablations;
  do not fold them into the canonical objective without evidence.
- Validate that every parameter belongs to exactly one intended group.  The
  current shape-based AnchorMuon matrix group plus AdamATan2 fallback is
  operational, but the brief's desired semantic groups (router/gates/norms,
  chunk controls, LM head, shared/reused blocks) still need per-group
  gradient/update, zero-gradient, and stale-state telemetry.

### Distributed and graph evidence

- Check world-size 1/2/4/8/12 where hardware permits using deterministic
  rank-sharded data and the same global byte batch.  Verify all-reduce dtype,
  normalization, route counts, chunk statistics, and checkpoint/resume RNG and
  data position.
- The current graph setup constructs DDP and performs its warmups on the
  capture stream, then captures `model.module` directly after reducer hooks
  have been installed.  The audit still observes PyTorch `AccumulateGrad`
  stream-mismatch warnings at this boundary.  Treat this as an open release
  gate: either prove the reducer-hook stream contract with warnings enabled,
  or change the capture arrangement and rerun full eager/graph/DDP parity.
  Do not suppress the warning or call the graph path fully validated while it
  remains unexplained.
- Do not overlap collectives on a second stream while TE delayed-scaling uses
  the same communicator.  Any overlap experiment must use a separate process
  group and pass a complete parity and deadlock test before consideration for
  production.
- Graph capture must fail closed on a shape or backend mismatch.  The current
  S8192 B32 capture OOM is a capacity finding, not a reason to fall back to an
  eager partial path.

## Current evidence gaps

The following are the highest-information next checks.  They are intentionally
listed as work to do, not claims that the current code already satisfies them:

- full-shape eager/FP8/graph parity including both FBT passes and both shared
  middle-loop executions;
- chunk-cap overflow and post-cap-byte metrics, plus MoE overflow/dropped-token
  metrics (overflow currently uses the shared path);
- Mamba-3 state/gate diagnostics and fixed-route controls for the S3T1 stack;
- LM-head precision/centering/z-loss ablations with raw-logit diagnostics;
- per-optimizer-group update geometry and stale expert-state age;
- deterministic checkpoint/resume and reduction parity at additional world
  sizes.

These checks should precede a long pretraining run.  They do not justify
introducing a smaller production model, a tokenizer path, or an unvalidated
GDN2/LoL fallback.

