# Direct H-Net run

`train_hnet_600m.py` is the canonical run. It reads a raw UTF-8 byte file,
uses the frozen 256-row byte code (zero trainable embedding parameters), and
trains `m4 -> T19 -> m4` with learned routing, MSAR in the innermost stack,
two shared-weight inner executions, chunk-level FBT feedback, FP32 CE/router
losses, Muon/GramNS plus AdamATan2 fallback, weight decay, and DDP. CUDA graph
capture is implemented for fixed-shape attention-only configurations. The
canonical `m4 -> T19 -> m4` path currently logs and skips capture because the
installed Mamba2/FlashAttention backward kernels use legacy CUDA streams; an
attempted capture fails or deadlocks, so running it would be unsafe.

The loop cache is indexed by recurrent depth for exact autoregressive context;
weights are shared while KV state is kept separate per loop. FBT training uses
fresh caches and fixed chunk boundaries, as required by its multipass
objective. The outer encoder, router, chunker, dechunker, and decoder do not
use MSAR or FBT.

Example launch (artifacts stay outside this repository):

```sh
torchrun --nproc_per_node=8 train_hnet_600m.py \
  --data /path/to/raw_utf8_bytes \
  --config configs/hnet_1stage_600m_loop2.json \
  --seq 4096 --batch 1 --optimizer gramns --cuda-graphs
```

The trainer logs per-GPU power. Adjust `--batch` only after observing the
steady-state draw; the target is approximately 1000 W per B300.

The selected FFN path uses a Mixtral/DeepSeek-style sparse layer from the
BioNeMo `mixtral_native_te` design: eight stacked routed experts with top-1
dispatch (1:8 active routed sparsity), plus one always-on shared SwiGLU expert.
Each routed expert has intermediate width 512, so the eight routed experts
have the same aggregate parameter count as the dense width 4096 FFN; the
shared expert adds a narrow 256-wide common path. This keeps the direct model
near 600M parameters (the shared path adds about 22M weights across T19).
Dispatch uses fixed per-expert
capacity and grouped tensor contractions so the token shape remains stable
for CUDA graphs. The router auxiliary loss is accumulated in FP32.

The CroissAttn correction in `~/fp8` uses the separate unrounded auxiliary
`O' = P @ V_f` for the softmax backward correction, rather than the rounded
FP8 output. The innermost H-Net attention config enables the vectorized
corrected path (`attn_cfg.croissattn_aux=true`) for regular padded training;
packed sequences, finite attention windows, and generation retain the fused
FlashAttention path. The TE FP8 trainer keeps the equivalent correction
enabled through `NVTE_FP8_DPA_BWD=1`/`NVTE_DPA_CROISSATTN_AUX=1`.
