# Applying the supplied stability brief

The supplied Codex brief is retained here as an implementation checklist for
the canonical H-Net run. Existing hand-cleaned source copies are:

- [`docs/lol_2611.03526.md`](lol_2611.03526.md), the LoL paper transcription;
- [`docs/latentmoe_2601.18089.md`](latentmoe_2601.18089.md), the LatentMoE
  paper and reference-implementation comparison;
- [`docs/activation_bias_2610.14500.md`](activation_bias_2610.14500.md), the
  activation-centering paper and its mapping to SwiGLU;
- [`HNET_REPRODUCTION.md`](../HNET_REPRODUCTION.md), the current runtime
  contract and exact-shape command.

The source claims below are the primary references named in the brief. The
repository stores cleaned notes rather than binary papers or training artifacts.

| Component | Primary source to pin and inspect | Application to this checkout |
|---|---|---|
| H-Net chunking | arXiv:2507.07955; `goombalab/hnet` | Keep causal learned boundaries and byte CE. Add soft → straight-through → hard tests before changing the current immediate argmax. |
| GDN2 | arXiv:2605.22791; `/home/catid/gdn2`; cuDNN Linear Attention docs | Treat the external recurrence as the reference. Compare fused chunk mode against FP32 tokenwise recurrence; instrument decay, erase, write, state, and reset behavior. |
| LoL | arXiv:2611.03526; `/home/catid/library_of_layers` | Current shifted windows are only the attention primitive. Implement the causal window-level route, selected-probability residual interpolation, loss-free balancing, and fixed-route control before calling it LoL. |
| LM-head spikes | arXiv:2610.38277; `tingtang2/quantization-loss-spikes` | Add FP32 reference logits, low/high precision RMS/KL, mean-logit ratio, and failure snapshots. Do not hide non-finite values. |
| Output centering | arXiv:2601.02031 | Make untied output μ-centering and no-centering explicit ablations; global vocabulary reduction is required if vocab sharding is introduced. |
| PaLM z-loss | arXiv:2204.02311 | Add a named LM-head z-loss flag and report its contribution separately from the existing MoE router z-loss. |
| Activation bias | arXiv:2610.14500; local cleaned note | The existing LatentMoE SwiGLU path can test per-token FP32 mean subtraction after the gated product, but this is an intervention and needs full-model late-horizon evidence. |
| LatentMoE | arXiv:2601.18089; `kyegomez/Latent-MoE`; Megatron `MoELayer` | Keep full-width router/shared expert and latent routed projections. Validate grouped dispatch, stream ordering, padding/capacity, and FP8 scales against an eager reference. |
| Optimizers | exact `scripts/optimizer_experiments.py`; AdamATan2 reference only as comparison | Preserve Muon/GramNS + AdamATan2. Replace shape-only assignment with an auditable category map and log update geometry before tuning rates. |

## Ordered work plan

### A. Establish an eager reference

1. Build deterministic tiny fixed-shape inputs that still exercise the complete
   hierarchy, two FBT passes, all eight experts, and FA. The shape may be a
   testing shape, but the model graph and loss terms must be complete.
2. Check future-byte perturbation, document-start reset, first-boundary, chunk
   packing/dechunking, and FA causal-mask invariants.
3. Compare BF16 production tensors against FP32 for router probabilities, GDN2
   summaries, attention outputs, dechunk outputs, logits, CE, gradients, and one
   optimizer update.

### B. Instrument before optimizing

Log at every smoke step and periodically thereafter:

- every objective component and global/non-finite counts;
- chunk count, length quantiles, hard-limit fraction, ratio, boundary entropy,
  and decoder reconstruction error;
- GDN2 per-layer/head state RMS/max/Frobenius, retained/write energy, and gate
  distributions;
- LoL window route entropy, utilization, route changes, effective depth,
  repeated selections, noise, and optimizer-state age;
- FA Q/K RMS, logits, entropy, saturation, residual/output-gate ratio, and
  backend dispatch;
- LM-head/output embedding mean and norm, logit mean/std/max, low/high precision
  error and KL, and any z/μ-loss;
- per-category gradient, parameter, update, update/parameter ratio, cosine, and
  skipped-gradient statistics.

Do not aggregate these into one global gradient norm. The brief treats H-Net,
GDN2, LoL, FA, LM head, optimizer, and distributed precision as separate
dynamical systems.

### C. Checkpoint and DDP parity

Add a manifest containing config hash, source/dependency commits, CUDA/driver/
PyTorch/TE versions, data-file hash, rank/world size, global token position,
RNG states, optimizer states, router/chunker noise state, and schedule phase.
Compare uninterrupted and resumed runs from the same checkpoint. Compare
single-rank and DDP runs using the same global batch and deterministic rank
shards; reduce route, chunk, μ-centering, and loss statistics globally.

### D. Only then enable compiled/FP8/graph paths

The current graph wrapper captures forward/backward and applies the custom
optimizer eagerly. Keep this conservative path until reference parity passes.
If complete optimizer capture is later required, prove that every optimizer
state update and non-finite guard is graph-safe; do not silently capture an
unverified custom Muon implementation. Record graph breaks and actual FP8/FA/
GDN2 dispatch rather than inferring them from configuration.

## Interventions worth testing, in brief order

These are named experiments, not production defaults:

1. FP32 loss/reference-logit diagnostics with no architecture change.
2. Explicit Muon/AdamATan2 category mapping and state-age logging.
3. Untied output-head μ-centering; compare with head-only SGD-M and no-centering.
4. LM-head z-loss around `1e-4`, swept against byte BPB and logit statistics.
5. GDN2 gate/state dtype and reset safeguards against the official recurrence.
6. LoL router correctness, fixed-route control, and then router noise/load-free
   balancing. Add depth growth only after a fixed-depth route is stable.
7. Activation-bias centering in the LatentMoE SwiGLU product, with late-horizon
   validation and activation-statistics evidence.
8. FP8 grouped experts, compiled blocks, and CUDA graphs, each with fixed-input
   parity and a separate throughput/memory measurement.

Do not draw conclusions from a short loss curve. The brief requires at least
three seeds for controlled 40M ablations and 4–8× horizon stress tests before
claiming stability. The repository’s production model remains the full named
architecture; short tests are correctness probes, not smaller training targets.

## Failure triage map

| First diagnostic that moves | Most likely subsystem | First checks |
|---|---|---|
| output mean/logit mean and low/high-precision KL | LM head / quantization | FP32 CE/reference logits, μ-centering, head optimizer and z-loss |
| recurrent state/write energy or gate saturation | GDN2 | recurrence dtype, official parameterization, document/chunk resets, fused/reference parity |
| route entropy/utilization or repeated depth jumps | LoL | causal window router, balancing bias/noise, stale expert state, fixed-route control |
| QK/logit max or softmax entropy collapse | FA | QK norm, causal mask, NoPE contract, FP32 reductions, output gate/residual scale |
| only multi-GPU or only resume diverges | DDP/checkpoint | reduction normalization, rank sharding, RNG/data position, optimizer/router state |
| eager/fused mismatch with finite values | kernel/precision | isolate DDP, BF16, compile, FP8, and each fused dispatch path one at a time |

The required recovery is to save the last healthy checkpoint, preserve the
first failing diagnostics, and stop the run for classification. Skipping a bad
batch or applying unconditional `nan_to_num` would hide the mechanism.

