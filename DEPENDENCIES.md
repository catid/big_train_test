# Pinned runtime dependencies

The H-Net trainer uses the vendored `hnet/` implementation and these reviewed
CUDA packages. Install from a lockfile or local wheel index; do not resolve
floating Git revisions during a training run.

- PyTorch 2.10.x with the installed CUDA 13.x build
- Triton 3.8.x
- FlashAttention 2.8.3
- Mamba-SSM 2.3.2
- causal-conv1d 1.7.0
- einops, optree, regex, omegaconf
- Transformer Engine from `/home/catid/fp8/TransformerEngine` at its checked-out commit

Before a long run, record `pip freeze`, the CUDA driver/runtime versions, and
the exact git SHAs for every source-built package in the run artifact log.
