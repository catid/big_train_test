# Direct H-Net run

`train_hnet_600m.py` is the canonical run. It reads a versioned document
manifest over an immutable raw UTF-8 byte payload, uses the frozen 256-row byte
code (zero trainable embedding parameters), and
trains `S2M3 -> (S3T1)×5 -> S2M3` with learned routing. Each outer group is S2M3: two 128-token sliding-window attention (SWA)
layers followed by three Mamba-3 layers. The middle stack is (S3T1)×5:
three SWA layers followed by one global full-attention (FA) layer, repeated five times. MSAR remains in the innermost stack,
two shared-weight inner executions, chunk-level FBT feedback, FP32 CE/router
losses, AnchorMuon/GramNS plus its AdamATan2 fallback, weight decay, and DDP.
The production SWA width is 128 tokens; the outer stacks do not use FA or
MSAR because they preserve the byte/chunk boundary interfaces.
SWA uses 48 rotary dimensions with one shared LeRoPE log-frequency scale per
frequency band across all SWA layers and heads; global FA is NoPE. LeRoPE is
initialized at ordinary RoPE (`alpha=0`), keeps its FP32 frequency parameter
out of weight decay, and clips its gradient separately before the model-wide
clip. The stage configuration keeps the first (Mamba-only) entry at zero and
applies the 48-dimension LeRoPE only to the `S` layers, so FA never allocates
or rotates a positional cache.

The public Cartesia posts support the outer state-space stacks and the causal
adjacent-state router used here. They describe 1024-token SWA in encoder and
decoder Transformer blocks, while the exact unreleased 2026 internal 600M
configuration is unspecified. The 128-token SWA width and 3:1 SWA/FA schedule
here are this project's explicit 600M design choices, rather than claims about
the unreleased Cartesia model.

The canonical DDP + FBT2 + AnchorMuon update always uses CUDA graphs for the
fixed full-system training shape.

## S8192 validation

The canonical shape is now `S=8192`. No S8192 throughput or batch-capacity
number is recorded until a complete eight-rank run has executed the actual
FP8 forward/backward, DDP reduction, AnchorMuon/AdamATan2 update, and CUDA-graph
replay. The calibration command is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 train_hnet_600m.py \
  --manifest /path/to/s8192.manifest.json --seq 8192 \
  --auto-batch-max 1 --steps 3 --warmup 1 --eval-every 0 \
  --ckpt-every 0 --out /tmp/hnet_s8192_run.json \
  --checkpoint /tmp/hnet_s8192_last.pt
```

Raise `--auto-batch-max` only after the complete run is finite. Record the
selected batch, per-GPU power, median graph step time, tokens/s/GPU, peak
memory, and reducer storage mode in the external run report. Artifacts and
payloads remain outside this repository.
Router-selected chunks are left-aligned with trailing padding; causal
attention cannot see that future padding, and active masks clear padded MLP and
residual rows. The graph path fails during capture if a TE kernel or DDP setup
still performs a host shape read. DDP capture uses a dedicated stream and
disables reducer synchronization inside the graph because DDP's runtime logger
is not capture-safe. The eager warmup preserves `gradient_as_bucket_view`
storage through capture, so replay reduces six coalesced reducer storages
directly; if a runtime does not expose that layout, it falls back to one flat
reduction per parameter dtype. The exact B=14/S=4096 full step measured 1.78695
seconds (32,090 tokens/s/GPU) on eight B300 GPUs, versus 1.85275 seconds before
preserving bucket views; all four measured updates were finite. This is a
historical S4096 measurement and is not an S8192 capacity or throughput claim.

A clean full-model B=18/S=4096 validation with optimizer state and CUDA-graph
capture measured 611.8 ms per update (120,508 tokens/s/GPU) on eight B300 GPUs;
all 15 steady updates were finite. The current MSAR path removes a redundant
backward activation cast and uses the validated two-warp MSAR weight-gradient
reduction. A matched pre-change HEAD run measured 616.0 ms/update.

The committed direct-MSAR-destination optimization (writing the fused mix at
the block's BF16 destination and removing a singleton grouped-weight repeat)
was revalidated on the same complete graph shape: 608.05 ms/update median
(121,253 tokens/s/GPU), with all 15 steady updates finite and direct DDP
storage preserved. Production-shaped MSAR forward/backward parity was exact.

The graph contract retains at most 512 learned latent rows per outer sequence.
If routing emits more boundaries, the first 512 causal boundaries are kept; a
future byte cannot retroactively evict an earlier latent. The same effective
boundary set is passed to the inner stack and dechunker.
The canonical graph keeps MSAR and compatible mixer activations for the
backward pass. A complete B=18/S=4096 FP8 graph step with the first AnchorMuon
state allocation is validated on B300; it measures 120,508 tokens/s/GPU. B=20
and B=24 exhaust the CUDA graph pool near 260 GiB during the first complete
update, so the validated cap is 18. Power calibration still
selects the largest safe batch at or below the approximately 1000 W per-GPU
target. The fused Mamba-3 SISO kernel keeps its native custom-autograd path.
MoE capacity uses the same latent bound in eager, prefill, and graph execution.
Streaming decode
enforces the bound monotonically: once the prefill or decode count reaches 512,
later boundary predictions are suppressed because a stream cannot
retroactively replace an earlier boundary with a higher-scoring one.

The looped attention executions share the layer KV slot and rewrite it on each
refinement; weights are shared across both executions. FBT training uses fresh
caches and fixed chunk boundaries, as required by its multipass objective. The
outer encoder, router, chunker, dechunker, and decoder do not use MSAR or FBT.
Cached prefill stores bounded MSAR stream tails for the base stream and each
inner layer, and decode advances those histories after both shared-weight loops.
The VHA cache is allocated per FA layer with two physical KV heads and rank-4
postmix compression.

FBT follows arXiv:2608.08888 Fig. 9: jitter is applied before the
document-local shift-right, the shifted state is the GLU value pathway, the
RMS-normalized current input is the gate pathway, and the same RMSNorm is
applied to the fused input before the shared inner stack. The current pass is
not added back to the fused state. The two-pass objective sums the plain
first-pass next-byte CE and the feedback-pass CE; the logged `loss` and
`standard_ce` fields report the plain first-pass byte CE, while `feedback_ce`
reports the final feedback-conditioned pass.

Example launch (artifacts stay outside this repository):

```sh
torchrun --nproc_per_node=8 train_hnet_600m.py \
  --manifest /path/to/fineweb_text.manifest.json \
  --seq 8192
```

The trainer emits UTC, step-numbered pretraining telemetry in the same shape as
the production dashboards: `loss`, `lr`, `lr_phase`, `grad_norm`, aggregate
`tokens/s`, cluster MFU, `moe_balance_loss`, and `expert_gini`. It runs a
complete-model held-out byte evaluation every 40 optimizer steps by default
(`--eval-every 40`); set `--eval-manifest` to a separate manifest to keep the
evaluation documents disjoint from training. Optional JSONL multiple-choice
suites can be supplied with `--mmlu-data`, `--arc-challenge-data`, and
`--hellaswag-data`; missing suites are reported as `null` rather than guessed.
Use `--eval-batches` and `--eval-max-items` to bound evaluation work, and
`--peak-tflops` to set the per-GPU FP8 peak used for the aggregate-DP MFU
estimate. Evaluation records, including duration and item counts, are written
to the run JSON alongside the step metrics.

Each manifest entry identifies one document byte range and its source,
license, language, quality metadata, SHA-256, and token-equivalent count. The
sampler chooses one document-local window per fixed-shape batch row and pads
only at the right edge; `active_mask` prevents padded rows from entering the
model and `loss_mask` removes the final input byte when its next target would
cross the document boundary. A separator byte is not used as a substitute for
these masks because it would not reset recurrent H-Net, FBT, MSAR, or KV-cache
state. FineWeb production runs therefore require the manifest path.

The planned text pretraining corpus targets 100B GPT-2-token equivalents, not
100B raw byte positions. The raw UTF-8 budget will normally be about 300–500B
bytes and is recorded separately from the tokenizer-equivalent count. The
recommended open-data mixture is 45% recent FineWeb dumps, 20% strict
FineWeb-Edu (score at least 3), 10% score-2-only educational documents, 10%
curated Dolma non-web text, 10% quota-controlled FineWeb2 multilingual text,
and 5% other clearly licensed educational/reference sources. FineWeb-Edu
documents must be assigned to one tier before mixing because they are subsets
of FineWeb. A manifest records document ID, source and dump, URL, license,
language, quality fields, SHA-256, UTF-8 byte count, and token-equivalent
count; exact and near-duplicate removal happens across the complete mixture
before the train/validation split.

The trainer logs per-GPU power. It always runs complete FP8 auto-batch
calibration steps at bounded candidate batches, takes the worst-rank reading across DDP, and
selects a measured draw against the approximately 1000 W per-B300 target. If
every candidate remains below target, it selects the largest fully validated
candidate and records that gap rather than pretending the target was reached;
if every candidate is above target, it selects the smallest. Candidates are
powers of two plus the explicit upper bound up to `--auto-batch-max` (default
14). Calibration catches and synchronizes candidate OOMs across DDP ranks,
includes DDP synchronization,
gradient clipping, and the real AnchorMuon update, then restores model and
optimizer and module-local FP8 scaling tensors before measuring the next
candidate; TE's global registration state remains intact. Power is aggregated
by the maximum rank draw, so the selected batch is safe for every GPU.

When CUDA-graph capture preserves six direct DDP reducer storages, gradient
clipping uses the native foreach norm over those storages and scales them in
place. A full B14 parity run matched the ordinary parameter-wise clip exactly;
the complete eight-rank timing reduced the clipping phase from about 15.1 ms
to 1.8 ms and measured 1.767 s per steady step (32,448 tokens/s/GPU). Graph
replay remains the dominant cost, so this is a small step-level gain.

The run's `loss` field is the plain byte cross-entropy. The combined FBT,
router, and MoE objective is logged separately as `objective`; `--target-ce`
(default 4.0) records the first step that reaches the requested CE target.

The selected FFN path uses LatentMoE (arXiv:2601.18089) with the BioNeMo
`mixtral_native_te` grouped dispatch design: eight stacked routed experts with
top-1 dispatch (1:8 active routed sparsity), plus one always-on full-width
shared expert. For the 1536-wide middle stack, LatentMoE uses a fixed 384-wide
latent path (`d/4`), 1664 neurons per routed expert, and 512 neurons in the
shared expert. The down/up projections and widened latent experts preserve the
former dense width-4096 SwiGLU parameter budget while increasing total expert
capacity; the full runtime reports 603,935,676 trainable parameters. Router logits,
probabilities, auxiliary loss, and routed accumulation remain FP32 around FP8
GEMMs. Eager execution schedules the shared expert on a CUDA side stream with
explicit event ordering; the production CUDA-graph path keeps the shared and
routed work on its capture stream because the side-stream variant was neutral
on the complete B=14 benchmark.
Dispatch uses fixed per-expert capacity and grouped tensor contractions. The
router auxiliary loss is accumulated in FP32. The chunker keeps a fixed padded
bound during graph replay; selected boundaries are left-aligned and trailing
padding is masked from residual and MLP outputs.

AnchorMuon applies GramNS/PMuonEq/NorMuon to sufficiently wide matrix banks
and AdamATan2 to vectors and skinny control projections. BF16 model weights use
persistent FP32 master copies for every update, SODA anchor, moment, and decay;
checkpoint reload restores those tensors without narrowing them.
The matrix residual stage batches the SODA pull, decoupled decay, and update
with foreach kernels within each same-shape optimizer chunk. A deterministic
probe over all canonical parameter shapes (603,935,676 elements) produced
bit-identical BF16 parameters and FP32 optimizer state for two updates; the
full eight-rank B14 graph remained finite with no measured end-to-end
regression.

The CroissAttn correction in `~/fp8` uses the separate unrounded auxiliary
`O' = P @ V_f` for the softmax backward correction, rather than the rounded
FP8 output. The innermost H-Net attention config enables the vectorized
corrected path (`attn_cfg.croissattn_aux=[false,true]`) for the stage-specific
SWA/FA training paths;
finite attention windows and generation retain the fused FlashAttention path.
The TE FP8 trainer keeps the equivalent correction
enabled through `NVTE_FP8_DPA_BWD=1`/`NVTE_DPA_CROISSATTN_AUX=1`.

Transformer Engine FP8 covers every attention projection, the global FA DPA
(including the patched CroissAttn backward), sparse MoE GroupedLinear and
shared/router projection, and the FBT feedback projections. SWA keeps its
fused FlashAttention product path while its QKV/output projections remain TE
FP8.
The fixed byte lift, fused Mamba-3 state-space kernel, causal router's Q/K
boundary scores, routing boundary residual projection, RMSNorms, and FP32
language-model head remain explicitly higher precision for their fixed-code or
numerical-stability contracts; these are deliberate precision boundaries, not
alternate training implementations.
