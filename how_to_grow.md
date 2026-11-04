# Research notes: growing the middle stack during pretraining

This is a research plan only. It does not change the model or trainer.

## What should grow

The canonical model is `S2M3 -> (S3T1)x5 -> S2M3`. The middle stack therefore has 20 blocks: five groups of three 128-token SWA blocks followed by one global NoPE FA block. It also runs the configured two shared-weight inner executions. The outer encoder and decoder implement the byte/chunk interfaces and should remain at full depth throughout. MSAR is only in the middle stack and its stream boundaries are every four blocks.

The safest definition of active depth is a contiguous prefix of the 20 middle blocks. Keep all 20 modules, parameters, optimizer groups, and checkpoint keys instantiated from step zero, but execute only the active prefix. Growing by four blocks (`4 -> 8 -> 12 -> 16 -> 20`) preserves the `S3T1` unit and the MSAR boundary cadence. Arbitrary layer selection would change the SWA/FA ratio and MSAR source semantics and should not be the first design.

The final checkpoint must run all 20 middle blocks. A depth value is training state, not a different model configuration.

## What the literature says

- **On the Transformer Growth for Progressive BERT Training** ([Gu et al., 2020/2021](https://arxiv.org/abs/2010.12562)) separates a growth scheduler (when) from a growth operator (how). Its controlled comparisons favor compound growth when more than one model dimension is allowed to change. For this project, the relevant lesson is to define the stage schedule independently from the layer activation mechanism and keep the final model fixed.
- **Progressively Stacking 2.0** ([Yang et al., 2020](https://arxiv.org/abs/2011.13635)) starts shallow, adds layers near the output, freezes previously trained lower layers during intermediate stages, and reserves a final joint-retraining stage. It reports that roughly 10–20% of the total training steps in the joint stage is useful. The freezing rule is attractive for compute savings but should not be copied literally here: the H-Net middle stack has MSAR and shared inner loops, and freezing would change optimizer/decay semantics.
- **LayerDrop** ([Fan et al., 2019](https://arxiv.org/abs/1909.11556)) trains with structured layer dropout so several depths remain usable. It is useful if depth elasticity at inference is a goal, but stochastic per-step dropping adds variance and does not give the deterministic early-training compute reduction requested here.
- **Apollo: Preparing Lessons for Progressive Training on Language Models** ([Pan et al., 2024](https://arxiv.org/abs/2401.09192)) samples shallower effective depths more often and uses interpolation/weight sharing to prepare higher-layer functionality before expansion. It reports that interpolation is more stable than direct stacking. This is a possible later refinement, but it complicates a heterogeneous SWA/FA stack and is not necessary for a first deterministic schedule.
- **Sparse Growing Transformer** ([Chen et al., 2026](https://arxiv.org/abs/2603.23998)) warms up first, then grows at fixed intervals. Its reported deep-to-shallow maturation signal and entropy-guided looping are relevant to adaptive looping, but our requested mechanism is simpler: add complete middle groups in order, with no data-dependent routing of depth. Its schedule parameters (`t_start`, interval, maximum depth, and a fixed phase) are a useful template.
- **Depth-Adaptive Transformer** ([Elbayad et al., 2019](https://arxiv.org/abs/1910.10073)) addresses input-dependent inference depth rather than training-time growth. It supports keeping training-time depth decisions separate from any eventual decode-time policy.

The papers agree on three practical points: growth should be staged, the order and operator must be explicit, and the final full-depth phase matters. Reported speedups are not directly transferable because they use BERT-like stacks, different objectives, and often freeze or copy parameters.

## Recommended schedule to implement later

Use optimizer update count, never wall-clock time, as the schedule coordinate. Every rank computes the same integer active depth from the global step; no rank may make a local decision. A simple stage schedule is:

| Stage | Active middle blocks | Purpose |
|---|---:|---|
| 0 | 4 | Train the first complete `S3T1` group and establish MSAR streams. |
| 1 | 8 | Add one complete group. |
| 2 | 12 | Add one complete group. |
| 3 | 16 | Add one complete group. |
| 4 | 20 | Full canonical architecture and final convergence. |

The schedule should expose `initial_depth`, `start_step`, `end_step`, `growth_interval`, and `final_depth` (20). Quantize every transition to four blocks and clamp at 20. A linear alternative is

```
depth(step) = 4 * ceil(clamp(step - start, 0, end - start)
                    / (end - start) * 4 + 1)
```

with the exact integer transition table saved in the run provenance. A fixed table is easier to audit and resume than a floating-point calculation. Keep the final full-depth phase for at least 10–20% of the run as a starting point, then compare loss per consumed byte and total wall time against a full-depth control.

Growth must happen before the forward pass for an optimizer step. Both FBT passes in that step use the same active depth. Evaluation and logs should report the active depth. A checkpoint taken at a transition must record the completed step, active depth, schedule version, and all schedule fields; on resume, restore the depth before any DDP warmup or graph capture.

## Optimizer and initialization choices

Construct Muon/GramNS and AdamATan2 groups once over the full parameter set. Do not add groups or rebuild the optimizer as layers activate. With prefix execution, inactive parameters receive no gradient; their optimizer state is then created on first activation by the current lazy optimizer implementation, so their bias-correction age starts at activation. That behavior must be recorded explicitly and compared with a run that materializes state for every parameter at step zero.

Weight decay needs an explicit decision. If decay is applied only when a parameter has a gradient, inactive layers do not decay during the shallow stages. If the intended regularization is global-step decay, the optimizer must apply decay to inactive parameters without fabricating a data gradient. This is a numerical-policy choice and should be settled before an apples-to-apples run.

All full-depth parameters can be initialized at step zero, but newly activated residual branches may cause a loss jump. Candidate growth operators, in order of simplicity, are:

1. zero or small residual output/gate for a newly active group, then release it over a short fixed ramp;
2. copy a nearby block only when the block type matches (SWA copies SWA, FA copies FA), followed by a small perturbation;
3. use Apollo-style interpolation only after the deterministic prefix schedule is characterized.

Do not copy an SWA block into an FA block or alter the fixed byte-lifting/output-head contract.

## DDP and CUDA-graph consequences

Changing the executed prefix changes which parameters receive gradients. The current production DDP setup uses `static_graph=True` and assumes a stable used-parameter set; it cannot silently replay a changing-depth schedule. There are two correct implementation strategies:

1. **Stage-specific graph recapture (preferred for throughput):** synchronize all ranks at a transition, finish and destroy the old graph, set the new prefix, warm up the exact new graph, and capture one graph per active depth. Keep optimizer parameter identities unchanged. The DDP reducer must be initialized for the stage's used-parameter set, and every rank must recapture in the same order.
2. **Eager growth phase (simpler correctness):** use `static_graph=False` with `find_unused_parameters=True` while depth is changing, and disable CUDA-graph replay until depth reaches 20. Recreate the stable full-depth graph only after the final transition. This pays graph-capture/setup overhead only during growth and avoids replaying a stale graph.

Never put a Python depth branch inside a captured graph and never replay a graph captured at a previous depth. The graph transition should be a visible event in logs and checkpoints. Recapture cost must be amortized by making stage intervals long compared with one capture.

The manual post-capture DDP reduction path also needs a stage audit: skipped parameters must either be represented by an explicit zero gradient in the reduction layout or be excluded consistently on every rank. A missing or rank-dependent gradient is a correctness failure, not a performance optimization.

## H-Net-specific invariants

- Only the deepest `Isotropic` main network is depth-scaled. Encoder, decoder, chunker, dechunker, and all outer S2M3 stacks remain full depth.
- Keep `height`, MSAR parameter arrays, latent capacity, and the full module topology sized for 20 blocks. Slice execution, not parameter construction.
- Use the same active prefix for both recurrent inner loops, FBT passes, eager training, graph replay, and cached prefill. Decode should force the final 20-block path unless a separately specified shallow inference mode is added later.
- Keep the KV-cache layout and layer identities stable. A stage transition may require a fresh prefill/cache allocation, but it must not reorder layers or reuse a stale cache from another depth.
- Preserve FP8/TE, CroissAttn, MoE routing, MSAR, and loss fixes at every active depth. A shallow stage is still the complete H-Net computation around a shorter middle prefix.

## Evidence required before implementation is accepted

1. A full canonical configuration run that logs active depth, elapsed time, tokens/s, CE, objective, router/MoE metrics, and transition points.
2. A matched full-depth run with the same data order, optimizer, batch, FP8 recipe, and total consumed bytes.
3. Finite-loss/gradient checks immediately before and after every transition, including both FBT passes and every DDP rank.
4. Checkpoint/resume at an intermediate depth and immediately after a transition, with identical next-batch data and matching optimizer state IDs.
5. Per-stage graph-capture evidence showing that no stale graph is replayed and that all ranks use the same active depth.
6. A timing breakdown that includes capture overhead, steady-state throughput, DDP communication, and the final full-depth phase. Compare total wall time and loss at equal consumed bytes; early tokens/s alone is not sufficient.

The open decisions are the exact stage boundaries, whether to use lazy or materialized optimizer state, the residual initialization/ramp for new groups, and whether graph recapture or eager growth is the better throughput/correctness tradeoff on the target GPUs.

## Sources

- https://arxiv.org/abs/2010.12562
- https://arxiv.org/abs/2011.13635
- https://arxiv.org/abs/1909.11556
- https://arxiv.org/abs/2401.09192
- https://arxiv.org/abs/2603.23998
- https://arxiv.org/abs/1910.10073
