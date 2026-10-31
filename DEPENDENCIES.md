# Pinned runtime dependencies

The H-Net trainer uses the vendored `hnet/` implementation and these reviewed
CUDA packages. Install from a lockfile or local wheel index; do not resolve
floating Git revisions during a training run.

- PyTorch 2.10.x with the installed CUDA 13.x build
- Triton 3.6.0
- FlashAttention 2.8.3.post2 (source-built for this CUDA/Python runtime)
- Mamba-SSM 2.3.2.post1 (source-built for this CUDA/Python runtime; canonical
  outer stacks use its fused Mamba-3 SISO kernels and CuteDSL decode kernel)
- apache-tvm-ffi 0.1.11 (the minimum release that supports the
  `map_dataclass_to_tuple` wrapper used by the Mamba-3 CuteDSL decode path;
  0.1.9 is ABI-incompatible with nvidia-cutlass-dsl 4.8.1)
- causal-conv1d 1.7.0
- einops, optree, regex, omegaconf
- Transformer Engine from `/home/catid/fp8/TransformerEngine` at commit
  `9646c3ef22455b6948cbe870f1aaf20fc8238d69`, rebuilt against this environment
- cuDNN `9.23.2.1` (the packed fused FP8 DPA backend requires cuDNN >= 9.23 on B300)
- NVIDIA `nvidia-cublas==13.8.2.12` (wheel SHA-256
  `21721774d8e7044debec3034ecc1a8cd22e173eb6dae5d7f5f8a014852382894`), which
  supplies libcublasLt 13.8.2; 13.8.0.4 is rejected because it corrupts the
  pinned TE grouped-GEMM MoE path
- BioNeMo Mixtral TE routing/GroupedMLP design reviewed at
  `NVIDIA-BioNeMo/bionemo-recipes` commit
  `2686b2418889ce346c5e50ea8614bb3b27415b97`; no BioNeMo runtime dependency is
  imported by the H-Net trainer. The H-Net MoE uses the same stacked-expert,
  fixed-capacity contract in its vendored PyTorch implementation.

Before a long run, record `pip freeze`, the CUDA driver/runtime versions, and
the exact git SHAs for every source-built package in the run artifact log.

The committed `runtime_manifest.json` is the binary trust anchor for the
canonical environment. It records SHA-256 hashes for the loaded Transformer
Engine core/torch/NCCL extension, FlashAttention, selective-scan, causal-conv,
and CUDA cuBLAS libraries. `assert_fp8_runtime()` verifies those exact files
before model construction; generated TE build directories are permitted only
when the loaded binaries still match this manifest. The installed
FlashAttention, Mamba-SSM, and causal-conv1d distributions do not expose
authoritative source commit metadata. The trainer therefore records each
distribution's normalized installed-file manifest (per-file SHA-256, size, and
relative path) plus a tree SHA-256 in checkpoint provenance. Python bytecode
caches are excluded because imports may create or rewrite them; loaded CUDA
extension hashes remain recorded separately in `runtime_binaries`. A missing
manifest entry or a distribution without an immutable file list fails startup
rather than allowing an unreviewed package tree to run.
