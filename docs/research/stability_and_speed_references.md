# Stability and speed reference audit

This note records the primary references supplied for the H-Net S=8192
pretraining system and maps their actionable guidance to the current checkout.
It is an evidence register, not a second model configuration. The production
target remains the complete 636,756,976-parameter raw-byte model at `B=1,
S=8192`, with its fixed latent cap and two FBT passes.

## Sources and provenance

| Area | Primary reference | Local evidence |
|---|---|---|
| H-Net dynamic chunking | [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955), [goombalab/hnet](https://github.com/goombalab/hnet) | `hnet/models/hnet.py`, `hnet/modules/dc.py`; byte CE remains the output objective and the routing/chunking path is active from the first update. |
| GDN2 | [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2), [cuDNN Linear Attention](https://docs.nvidia.com/cudnn/operations/LinearAttention) | `/home/catid/gdn2` resolves to `5ec53d56da98841f6613a0a1c1a59ca1aad9b44c`; `hnet/modules/gdn2_adapter.py` loads `lit_gpt.gdn2:GatedDeltaNet2` through the pinned root. The requested `11d9f76` is not present locally. |
| Library of Layers | [LoL](https://arxiv.org/abs/2611.03526), [library_of_layers](https://github.com/arahmani1/library-of-layers) | `/home/catid/library_of_layers` resolves to `03b85f42d30d628a973ec759eeb3491b56f1d9e4`; `docs/lol_2611.03526.md` is the cleaned transcription. The current block is a fixed shifted-window attention primitive, not the paper's causal window-level library router. |
| Quantization loss spikes | [The Role of Quantization in Loss Spikes](https://arxiv.org/abs/2610.38277), [reproduction code](https://github.com/tingtang2/quantization-loss-spikes) | `hnet/models/byte_lm.py` computes CE from FP32 logits. Low/high-precision logit RMS, softmax KL, and first-failure snapshots are still instrumentation work. |
| Output-head centering | [Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031) | The output head is untied and FP32. μ-centering is documented as an ablation, not silently enabled. Global vocabulary reduction is required if vocabulary sharding is added. |
| PaLM z-loss | [PaLM](https://arxiv.org/abs/2204.02311) | The model has a LatentMoE router z-loss. A separate LM-head z-loss coefficient and metric are not yet implemented. They must not be conflated. |
| LatentMoE | [LatentMoE](https://arxiv.org/abs/2601.18089), [reference](https://github.com/kyegomez/Latent-MoE), [Megatron MoE layer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) | `hnet/modules/latent_moe.py` uses eight routed experts, a shared expert, FP8 Transformer Engine grouped linears, and TE permutation. `docs/latentmoe_2601.18089.md` records the implementation comparison. |
| Optimizers | exact `scripts/optimizer_experiments.py`; [AdamAtan2 reference](https://github.com/imoneoi/adam-atan2) | `AnchorMuon` applies GramNS/PMuonEq/NorMuon to selected matrices and AdamATan2-style updates to fallback tensors. Trainer defaults are GramNS/AdamATan2 with decoupled decay. |

The exact repository revisions and runtime caveats are also recorded in
[`docs/stability_brief_sources.md`](../stability_brief_sources.md). PDFs,
checkpoints, generated datasets, and benchmark artifacts stay outside the code
repository.

The required architecture/update-flow inventory is
[`docs/architecture_update_inventory.md`](../architecture_update_inventory.md).
It records the tensor shapes, sequence changes, mask contracts, precision
boundaries, optimizer routing, losses, and current checkpoint/DDP behavior.
The brief's requested 40M production track conflicts with the standing project
instruction to train the full approximately 600M model; the inventory therefore
audits the full model and treats any smaller shapes only as correctness probes.

## Findings that affect speed and correctness

1. **Preserve the hierarchy while optimizing dispatch.** The fixed byte lift,
   boundary router, 2048-row latent cap, dechunk scan, and outer decoder are
   causal sequence transformations. Optimizing the inner attention or expert
   path must preserve their masks and fixed shapes.
2. **The dominant measured speed target is expert dispatch.** The current TE
   index-map permutation plus fused weighted unpermute avoids the old Python
   index accumulation. An isolated `N=2048` latent expert backward measured
   about 8.3 ms after the change versus 17.6 ms before it. The full model's
   real-TE graph replay was about 179 ms per update in a three-replay sample;
   these figures need repetition after a dependency or driver change.
3. **Graph capture requires a dedicated stream.** The trainer warms and captures
   forward/backward on a fixed stream and leaves clipping and the custom
   optimizer update eager. A default-stream capture can raise
   `cudaErrorStreamCaptureImplicit`. TE FP8 metadata must be warmed before
   capture, and the first replay is not a reliable output-parity reference.
4. **FP32 reduction contracts are deliberate.** Router logits/probabilities,
   CE, z-loss, expert merge accumulation, norms, and optimizer statistics are
   kept in FP32. Expert GEMMs may run BF16/FP8, but a fused merge must accept
   FP32 inputs if it is replacing the reference `index_add_` reduction.
5. **The LoL name currently describes a primitive, not full paper parity.**
   `hnet/modules/hybrid.py` has learned RoPE, GQA, shifted causal windows, and
   static compilation, but no causal window-level layer-library router,
   route-noise schedule, loss-free balancing, or depth-growth schedule. This
   is a documented architecture discrepancy, not a reason to add an alternate
   fallback path.

## Required evidence checklist

The brief calls for evidence at the same scope as the intended run. The current
status is:

| Required evidence | Current status | Authoritative location |
|---|---|---|
| Full tensor/sequence/update-flow inventory | documented; unresolved items are marked TODO | `docs/architecture_update_inventory.md` |
| Eager reference before compile/graphs/FP8 | full eager `B1/S8192` forward/backward completed; formal tensor/gradient parity suite remains | `docs/full_shape_runtime.md` |
| BF16 versus FP32 and optimized-path parity | partial: finite full-shape BF16 and isolated dispatch parity; end-to-end logits/optimizer-state parity remains | `docs/stability_brief_application.md` |
| Causality and document-boundary resets | mask contracts are documented; future-byte and document-aware packing tests are still missing | `hnet/modules/dc.py`, `hnet/models/hnet.py` |
| GDN2 recurrence/state diagnostics | external FROST execution is measured; FP32 tokenwise/chunkwise parity and gate/state telemetry remain | `hnet/modules/gdn2_adapter.py` |
| LoL route/depth diagnostics | unavailable for paper semantics because current path has fixed shifted windows; document this before calling it LoL parity | `hnet/modules/hybrid.py` |
| FA mask/logit/entropy diagnostics | causal GQA path exists; explicit logit/entropy telemetry and reference comparison remain | `hnet/modules/hybrid.py` |
| LM-head drift and low/high-precision diagnostics | FP32 CE/head exists; μ-centering, z-loss, logit RMS, and softmax KL are open ablations | `hnet/models/byte_lm.py` |
| Optimizer category/update diagnostics | GramNS/AdamATan2 mapping exists; per-category update geometry and stale-route state age are not logged | `scripts/optimizer_experiments.py` |
| Checkpoint/resume parity | not established; current checkpoint lacks RNG, loader position, schedule phase, and dependency manifest | `scripts/train_hnet_600m.py` |
| DDP 1/2/4/8/12-GPU parity | launcher and fixed-shape DDP path exist; deterministic world-size trajectory evidence is missing | `scripts/train_hnet_600m.py` |

For a serious run, the missing rows are release gates rather than optional
instrumentation. In particular, a finite loss or a fast graph replay does not
prove that rank reductions, recurrent states, expert routes, or optimizer state
are reproducible.

## Evidence-backed next work

- Keep separate eager, compiled, FP8, and graph measurements at `B=1/S=8192`;
  record median warm forward/backward time, peak memory, tokens/s, and actual
  TE/GDN2 backend dispatch.
- Add fixed-input parity checks for logits, each loss component, expert routes,
  gradients, and one optimizer transition before changing more kernels.
- Instrument chunk count/compression, GDN2 state and gates, FA logits and
  entropy, output-head drift, and per-group optimizer updates. The current
  trainer logs only a subset of these values.
- Add deterministic checkpoint/resume and DDP world-size parity evidence before
  interpreting throughput or loss curves. The current checkpoint does not yet
  include RNG/data-position/schedule manifests.
- Add a document-aware byte packer with a boundary reset mask. The current
  contiguous byte stream has no document IDs, so it cannot demonstrate that
  GDN2, the chunker, dechunker, FBT shift, and FA never carry state across
  documents.
- Preserve separate metrics for byte CE/BPB, boundary/ratio/reconstruction
  losses, MoE auxiliary/z losses, and FBT auxiliary CE. Do not let a single
  total loss or gradient norm stand in for those subsystem signals.
- Resolve the GDN2 revision mismatch and build a matching Transformer Engine
  extension before claiming production FP8 parity. The Python TE package alone
  is not sufficient evidence.

## Explicit ambiguities

- The supplied GDN2 handoff names commit `11d9f76`, but that object is absent
  from the local checkout; current measurements therefore use `5ec53d5`.
- The LoL paper's routing semantics and the project's heterogeneous GDN2/LoL/FA
  block contract are not identical. The current fixed shifted-window path must
  remain labeled as such until a true router is implemented and tested.
- The reference papers do not establish that their stabilization interventions
  transfer unchanged to this hierarchical byte model. μ-centering, LM-head
  z-loss, activation centering, gate safeguards, and routing schedules require
  full-shape ablations with their own diagnostics.
