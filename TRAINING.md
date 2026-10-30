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

The canonical DDP + FBT2 + AnchorMuon update uses CUDA graphs by default for
fixed shapes; pass `--no-cuda-graphs` only for a diagnostic eager run.
Router-selected chunks are left-aligned with trailing padding; causal
attention cannot see that future padding, and active masks clear padded MLP and
residual rows. The graph path fails during capture if a TE kernel or DDP setup
still performs a host shape read. DDP capture uses a dedicated stream and
averages a flat gradient buffer after replay because DDP's runtime logger is not
capture-safe.

The looped attention executions share the layer KV slot and rewrite it on each
refinement; weights are shared across both executions. FBT training uses fresh
caches and fixed chunk boundaries, as required by its multipass objective. The
outer encoder, router, chunker, dechunker, and decoder do not use MSAR or FBT.
Cached prefill stores bounded MSAR stream tails for the base stream and each
inner layer, and decode advances those histories after both shared-weight loops.
The VHA cache is allocated per FA layer with two physical KV heads and rank-4
postmix compression.

FBT follows arXiv:2608.08888 Fig. 10: jitter is applied before document-local
shift-right, the shifted state is fused through a gated GLU with normalized
token input, and the fused input receives a second RMSNorm before the shared
inner stack. The current pass is not added back to the fused state.

Example launch (artifacts stay outside this repository):

```sh
torchrun --nproc_per_node=8 train_hnet_600m.py \
  --data /path/to/raw_utf8_bytes \
  --seq 4096
```

The trainer logs per-GPU power. Auto-batch calibration is enabled by default
(`--no-auto-batch` disables it) and runs complete FP8 calibration
steps at bounded candidate batches, averages readings across DDP ranks, and
selects a measured draw against the approximately 1000 W per-B300 target. If
every candidate remains below target, it selects the largest fully validated
candidate and records that gap rather than pretending the target was reached;
if every candidate is above target, it selects the smallest. Candidates are
powers of two up to `--auto-batch-max` (default 32), bounded by a memory
reserve for optimizer and DDP state. Calibration includes DDP synchronization,
gradient clipping, and the real AnchorMuon update, then restores model and
optimizer state before measuring the next candidate.

The run's `loss` field is the plain byte cross-entropy. The combined FBT,
router, and MoE objective is logged separately as `objective`; `--target-ce`
(default 4.0) records the first step that reaches the requested CE target.

The selected FFN path uses a Mixtral/DeepSeek-style sparse layer from the
BioNeMo `mixtral_native_te` design: eight stacked routed experts with top-1
dispatch (1:8 active routed sparsity), plus one always-on shared SwiGLU expert.
Each routed expert has intermediate width 512, so the eight routed experts
have the same aggregate parameter count as the dense width 4096 FFN; the
shared expert adds a narrow 256-wide common path. This keeps the direct model
near 600M parameters (the shared path adds about 22M weights across the FA/SWA blocks).
Dispatch uses fixed per-expert capacity and grouped tensor contractions. The
router auxiliary loss is accumulated in FP32. The chunker keeps a fixed padded
bound during graph replay; selected boundaries are left-aligned and trailing
padding is masked from residual and MLP outputs.

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

Transformer Engine FP8 covers every attention projection, the global FA DPA
(including the patched CroissAttn backward), sparse MoE GroupedLinear and
shared/router projection, and the FBT feedback projections. SWA keeps its
fused FlashAttention product path while its QKV/output projections remain TE
FP8.
The fixed byte lift, fused Mamba-2 state-space kernel, causal router's Q/K
boundary scores, routing boundary residual projection, RMSNorms, and FP32
language-model head remain explicitly higher precision for their fixed-code or
numerical-stability contracts; these are deliberate precision boundaries, not
alternate training implementations.
