# Current stability brief audit

This is a source audit of the canonical H-Net trainer at commit
`f4b4f82` (`Fuse LatentMoE route weighting into TE unpermute`). It applies the
stability brief supplied with this project to the current checked-in code. It
does not change the production architecture or enable an alternate path.

## What is implemented

The single production configuration is
`configs/hnet_1stage_600m_loop2.json`. Its model construction is rejected
unless it is the complete byte model: fixed 256-row byte codebook, widths
`[1024, 1536]`, static sequence length 8192, learned causal chunking with a
512-latent cap, two FBT passes, two shared middle executions, MSAR, eight
top-1 routed experts plus an always-on shared expert, and the required
attention/SSM settings. The corresponding validation is in
`hnet/models/config_hnet.py:95-167`.

The current hierarchy is implemented as:

```text
fixed byte code -> outer S2M3 -> learned causal chunker -> (S3T1)x5,
shared middle loop x2 -> dechunker/residual -> outer S2M3 -> FP32 byte head
```

`hnet/models/hnet.py:292-596` keeps the encoder/router/chunker/dechunker and
FBT passes explicit. `hnet/modules/dc.py:87-151` computes the boundary score
from the current and immediately preceding state, forces document starts to
be boundaries, and applies the active mask. The static chunk path retains the
earliest boundaries when its cap is exceeded, preserving causality.

The attention contract is explicit in
`hnet/modules/isotropic.py:199-217` and `hnet/modules/mha.py:103-210`:
128-token SWA uses shared LeRoPE, global FA uses NoPE, VHA has two physical KV
heads and rank four, and the global FA branch enables the pinned CroissAttn TE
path. FA and KV-cache calls use causal FlashAttention/TE masks.

The production mixer is Mamba-3, not GDN2. `hnet/modules/block.py:30-238`
wraps the pinned Mamba-3 SISO kernel, converts its projections to the reviewed
TE path, and keeps the sensitive state preparation in the native fused core.
This is a deliberate repository choice and must not be described as GDN2
evidence.

`hnet/modules/latent_moe.py` uses the TE fused score/router and fixed-count
permutation, capacity-trimmed top-1 maps, grouped expert GEMMs, and the shared
expert. Router probabilities, z-loss, auxiliary loss, capacity decisions, and
route counts are handled in FP32. The recent route-weighting optimization is
at commit `f4b4f82`; it passes the original FP32 probabilities to TE
unpermute for FP32 merge accumulation.

The trainer computes byte CE from FP32 logits in
`train_hnet_600m.py:1500-1588`, keeps the FBT/router/MoE terms separate, and
records evals every 40 steps by default. `make_optimizer` at
`train_hnet_600m.py:1479-1497` uses the repository AnchorMuon/GramNS matrix
path and AdamATan2 fallback with decoupled decay. The grouping is currently
shape/marker driven in `optimizer_experiments.py:605-749`; shared parameters
are deduplicated by identity.

The FP8 runtime has a fail-closed dependency gate. `hnet/te.py:418-507`
requires the reviewed Transformer Engine checkout, CroissAttn environment
flags, runtime binary hashes, and the pinned cuDNN release. CUDA graph capture
preserves DDP reducer bucket views and fixed input/output shapes in
`train_hnet_600m.py:2301-2517`; checkpoint provenance includes model config,
optimizer state, per-rank RNG, data/manifest hashes, runtime binaries, and
world size (`train_hnet_600m.py:1192-1389, 3360-3390`).

## Discrepancies and evidence gaps

These are findings for the release gate, not claims that the current code is
incorrect.

1. **GDN2 and LoL are not the active architecture.** The code has Mamba-3 and
   fixed `S3T1` ordering. There is no GDN2 recurrence diagnostic and no
   causal window-level dynamic LoL layer-library router. The brief's GDN2 gate
   and LoL route/depth transition tests therefore cannot be marked complete.

2. **Chunk-cap behavior is not fully observable.** The chunker clips later
   boundaries to the static capacity, and overflow rows follow the residual or
   shared paths, but the training telemetry does not expose cap-overflow count,
   post-cap byte count, or latent-length distribution. Add these metrics before
   interpreting ratio or router stability.

3. **LM-head stability diagnostics are incomplete.** The head is FP32 and CE
   is reduced in FP32, but the run log has no output-weight mean/norm, raw-logit
   mean/std/max, low/high-precision logit error, or softmax KL. There is no
   isolated output-head μ-centering, SGD-M, μ-loss, or PaLM output z-loss
   intervention. The current z-loss is a router regularizer in
   `LatentMoE`, not the PaLM LM-head guard.

4. **Optimizer evidence is coarser than the brief requests.** The explicit
   Muon/AdamATan2 implementation and weight-decay exclusions are present, but
   per-category gradient/update norms, update/parameter ratios, gradient/update
   cosine, zero-gradient frequency, and optimizer-state age for rarely routed
   experts are not emitted. Shape-based grouping should be checked against
   semantic categories (router, gates, chunk controls, head) before a long
   ablation.

5. **Reference-parity coverage is still a release gate.** The repository has
   loader tests and external full-run evidence, but no checked-in deterministic
   fixed-input suite proving eager versus compiled/FP8/graph equality for
   logits, every loss component, gradients, optimizer updates, both FBT passes,
   and both shared middle loops. The required causality perturbation,
   document-reset, chunkwise/tokenwise state, FA reference, and checkpoint
   resume tests should be run before a long run.

6. **DDP and TE collective ordering needs explicit evidence.** The graph replay
   path launches explicit gradient all-reduces in
   `train_hnet_600m.py:2631-2667`, while `hnet/te.py:418-429` gives TE FP8
   delayed-scaling reductions `torch.distributed.group.WORLD`. The code
   currently starts gradient collectives asynchronously and then performs TE
   reductions before waiting. Even if the operations are mathematically
   independent, they share a communicator; NCCL ordering must be validated on
   the exact S8192 graph run or moved to a separate communicator. Do not infer
   safety from a single-rank result.

7. **FP8 availability is environment-specific.** Source/runtime provenance is
   pinned to GDN2 `5ec53d56`, `library_of_layers` `03b85f4`, Transformer Engine
   `9646c3ef`, and the reviewed binary manifest. On this audit host, importing
   the package failed closed because the resolved cuBLASLt reports version
   `13.8.0` while the runtime gate requires at least `13.8.1`. A run is not
   FP8 evidence until the reviewed CUDA patch is installed and the exact
   binary hashes pass `assert_fp8_runtime()`.

8. **Checkpoint state scope is limited to training state.** The trainer saves
   model/optimizer/RNG/data and runtime provenance. It does not serialize a
   live recurrent/KV/chunker inference cache; that is acceptable for ordinary
   graph training, but cached prefill/decode parity must be tested separately
   if inference state is part of an experiment.

## Primary references and pinned local source

Keep these links with experiment records and inspect the local source before
using a fused dependency:

- H-Net paper and implementation: <https://arxiv.org/abs/2507.07955>,
  <https://github.com/goombalab/hnet>
- GDN2 paper, implementation, and cuDNN operators:
  <https://arxiv.org/abs/2605.22791>,
  <https://github.com/NVlabs/GatedDeltaNet-2>,
  <https://docs.nvidia.com/cudnn/operations/LinearAttention>
- LoL paper and implementation: <https://arxiv.org/abs/2611.03526>,
  <https://github.com/arahmani1/library-of-layers>
- Quantization loss spikes: <https://arxiv.org/abs/2610.38277>,
  <https://github.com/tingtang2/quantization-loss-spikes>
- Output embedding centering: <https://arxiv.org/abs/2601.02031>
- PaLM z-loss: <https://arxiv.org/abs/2204.02311>
- AdamAtan2 comparison sources: <https://github.com/imoneoi/adam-atan2>,
  <https://github.com/lucidrains/adam-atan2-pytorch>

The checked-out revisions inspected for this audit are:

| Dependency | Revision |
|---|---|
| `/home/catid/gdn2` | `5ec53d56da98841f6613a0a1c1a59ca1aad9b44c` |
| `/home/catid/library_of_layers` | `03b85f42d30d628a973ec759eeb3491b56f1d9e4` |
| `/home/catid/fp8/TransformerEngine` | `9646c3ef22455b6948cbe870f1aaf20fc8238d69` |

The first high-information follow-up is a full S8192 eager/FP8/graph parity
and DDP collective-ordering run. Only after that evidence should the missing
LM-head, chunk-cap, recurrent-state, and optimizer-group diagnostics be used
to choose stability interventions.

## Review provenance

`claude auth status` was verified before launching an asynchronous source-only
review: `authMethod=claude.ai`, `apiProvider=firstParty`, and
`subscriptionType=max`. The review was launched with ordinary
`claude --model opus` and no API key; its output is auxiliary and must not
override the pinned source or full-run evidence.
