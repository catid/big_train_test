# Pinned runtime dependencies

The H-Net trainer uses the vendored `hnet/` implementation and these reviewed
CUDA packages. Install from a lockfile or local wheel index; do not resolve
floating Git revisions during a training run.

- PyTorch 2.10.x with the installed CUDA 13.x build
- Triton 3.6.0
- FlashAttention 2.8.3.post2 (source-built for this CUDA/Python runtime)
- Mamba-SSM 2.3.2.post1 (source-built for this CUDA/Python runtime)
- causal-conv1d 1.7.0
- einops, optree, regex, omegaconf
- Transformer Engine from `/home/catid/fp8/TransformerEngine` at its checked-out commit
- BioNeMo Mixtral TE routing/GroupedMLP design reviewed at
  `NVIDIA-BioNeMo/bionemo-recipes` commit
  `2686b2418889ce346c5e50ea8614bb3b27415b97`; no BioNeMo runtime dependency is
  imported by the H-Net trainer. The H-Net MoE uses the same stacked-expert,
  fixed-capacity contract in its vendored PyTorch implementation.

Before a long run, record `pip freeze`, the CUDA driver/runtime versions, and
the exact git SHAs for every source-built package in the run artifact log.
