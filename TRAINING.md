# Direct H-Net run

`train_hnet_600m.py` is the canonical run. It reads a raw UTF-8 byte file,
uses the frozen 256-row byte code (zero trainable embedding parameters), and
trains `M4 -> (S3T1)×5 -> M4` with learned routing. Each group starts with
three sliding-window attention (SWA) layers and ends with one full-attention
(FA) layer; the first layer is SWA. MSAR remains in the innermost stack,
two shared-weight inner executions, chunk-level FBT feedback, FP32 CE/router
losses, AnchorMuon/GramNS plus its AdamATan2 fallback, weight decay, and DDP.
The production SWA width is 128 tokens;
three SWA layers still provide a stacked receptive span above the 4096-byte
training context before each global FA layer.

The public Cartesia posts support the outer Mamba-2 `M4` stacks and the causal
adjacent-state router used here. They describe 1024-token SWA in encoder and
decoder Transformer blocks, while the exact unreleased 2026 internal 600M
configuration is unspecified. The 128-token SWA width and 3:1 SWA/FA schedule
here are this project's explicit 600M design choices, rather than claims about
the unreleased Cartesia model.

The canonical DDP + FBT2 + AnchorMuon update is eager. CUDA graph capture is
rejected explicitly because DDP gradient-bucket logging performs an operation
that CUDA forbids during capture; this avoids silently running a different
partial optimization path.

The loop cache is indexed by recurrent depth for exact autoregressive context;
weights are shared while KV state is kept separate per loop. FBT training uses
fresh caches and fixed chunk boundaries, as required by its multipass
objective. The outer encoder, router, chunker, dechunker, and decoder do not
use MSAR or FBT. MSAR currently has no recurrent stream-state implementation
for token-by-token decoding; `Isotropic.step` rejects that combination instead
of silently serving a model with different residual dynamics. Training and
full-sequence evaluation remain supported.

Example launch (artifacts stay outside this repository):

```sh
torchrun --nproc_per_node=8 train_hnet_600m.py \
  --data /path/to/raw_utf8_bytes \
  --seq 4096 --batch 1
```

The trainer logs per-GPU power. Adjust `--batch` only after observing the
steady-state draw; the target is approximately 1000 W per B300.

The selected FFN path uses a Mixtral/DeepSeek-style sparse layer from the
BioNeMo `mixtral_native_te` design: eight stacked routed experts with top-1
dispatch (1:8 active routed sparsity), plus one always-on shared SwiGLU expert.
Each routed expert has intermediate width 512, so the eight routed experts
have the same aggregate parameter count as the dense width 4096 FFN; the
shared expert adds a narrow 256-wide common path. This keeps the direct model
near 600M parameters (the shared path adds about 22M weights across the FA/SWA blocks).
Dispatch uses fixed per-expert
capacity and grouped tensor contractions so the token shape remains stable
for CUDA graphs. The router auxiliary loss is accumulated in FP32.

AnchorMuon applies GramNS/PMuonEq/NorMuon to sufficiently wide matrix banks
and AdamATan2 to vectors and skinny control projections. BF16 model weights use
persistent FP32 master copies for every update, SODA anchor, moment, and decay;
checkpoint reload restores those tensors without narrowing them.

The CroissAttn correction in `~/fp8` uses the separate unrounded auxiliary
`O' = P @ V_f` for the softmax backward correction, rather than the rounded
FP8 output. The innermost H-Net attention config enables the vectorized
corrected path (`attn_cfg.croissattn_aux=true`) for regular and packed training;
finite attention windows and generation retain the fused FlashAttention path.
The TE FP8 trainer keeps the equivalent correction
enabled through `NVTE_FP8_DPA_BWD=1`/`NVTE_DPA_CROISSATTN_AUX=1`.
