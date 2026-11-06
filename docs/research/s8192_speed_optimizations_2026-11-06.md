# S=8192 speed optimization evidence

This note records the matched full-system timing used to evaluate speed changes
after commit `bfc92a7` (`Reuse chunk sort across dechunk passes`).  The runs
used the complete 603,935,676-parameter raw-byte model, BF16/FP8 Transformer
Engine, FBT two-pass objective, LatentMoE, MSAR, CUDA graphs, AnchorMuon, and
eight-rank DDP.  Fixtures and checkpoints remain under `/tmp`.

## Accepted change: reuse the chunk partition

`ChunkLayer` already computes a stable `sort_indices` tensor for padded
multi-row batches.  `DeChunkLayer` now accepts and reuses that tensor instead
of recomputing the same `argsort` on each feedback pass.  Callers without the
paired indices retain the existing inference/direct-module behavior, and a
shape check fails loudly on an invalid tensor.

Focused B16/S8192/D=1024 dechunk forward/backward timing was 3.51 ms with the
reuse versus 3.61 ms when recomputing the sort (about 2.8%).  The focused
parity check was bit-identical for output, hidden-state gradients, and
boundary-logit gradients; the malformed-shape check raised `ValueError`.

The matched full-shape commands were:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /tmp/hnet-venv/bin/torchrun --standalone --nproc_per_node=8 \
  train_hnet_600m.py --manifest /tmp/hnet_s8192_fixture_128/train.manifest.json \
  --seq 8192 --auto-batch-max 16 --steps 5 --warmup 1 --eval-every 0 \
  --log-every 1 --ckpt-every 0 \
  --out /tmp/hnet_s8192_sort_baseline.json \
  --checkpoint /tmp/hnet_s8192_sort_baseline.pt
```

The production run (sort reuse, no uncommitted candidate) recorded step times
`1581.2, 932.0, 702.4, 702.3, 702.5 ms`; the three steady steps averaged
`702.4 ms`, or about `1.493M` aggregate tokens/s (`186.6k/GPU`).  All five
steps were finite.

## Rejected for production: MSAR pointer weight-gradient tile

Changing the pointer-reduction weight-gradient tile from `BLOCK_D=256,
num_warps=2` to `BLOCK_D=1024, num_warps=1` reduced a focused backward
microbenchmark from about 48.3 to 30.4 microseconds.  It preserves exact
outputs and bank gradients, but changes FP32 accumulation grouping in the
weight gradient (canonical focused probe: maximum absolute difference
`3.05e-5`, RMS `5.75e-6`).  This is not the required exact gradient/update
parity for an automatic production change.

A matched full S=8192 B16 run with the candidate remained finite and recorded
`1553.2, 913.8, 699.8, 699.4, 699.6 ms`; the steady average was `699.6 ms`,
only about 0.4% faster than the production run.  The candidate was reverted
after the measurement, so `hnet/modules/msar_triton.py` remains at the
validated `256/2` setting.  It remains an explicit numerical/speed ablation
for a future study rather than an implicit change to optimizer updates.

The candidate run used the same full model and shape, had no nonfinite values,
and used the same graph/DDP path.  The small end-to-end gain does not justify
the changed accumulation order without a documented optimizer-state and
long-horizon comparison.

## Measurement limits

The timing is a throughput comparison, not a quality comparison: the fixture
is deterministic and the five-step run is only a short infrastructure probe.
The graph path still emits PyTorch `AccumulateGrad` stream-mismatch warnings;
those warnings remain a correctness/release gate and were not suppressed for
these measurements.  Future kernel changes must repeat full eager/graph
parity for logits, losses, gradients, updates, and optimizer state before
being promoted.

## Additional MSAR probes that were not promoted

The current `_msar_expand_batched` path already materializes contiguous
`[N,B,L,D]` stream banks.  A focused RMSNorm/score probe measured about
`0.385 ms` for the contiguous bank versus `0.647 ms` for a strided bank, so
removing that copy would move work into a slower hidden-copy path.  Pointer
forward/gradient-bank launch changes were bit-identical but timing was
inconsistent across source counts and shapes.  A flattened eager `matmul`
score spelling was about 10% faster in isolation, while its compiled fixed
graph specialization was slightly slower (`1.35` versus `1.38 ms`); it was
therefore not promoted without a full graph gain.

The byte-stream and manifest regression tests also pass in the reviewed
runtime.  The runtime image does not include `pytest`, so the repository's
standalone test scripts were used rather than claiming a pytest suite that was
not installed.
## Current full-system rerun

A fresh eight-rank run of the checked-in `dbed2ef` path was used as the current
optimization baseline. It exercised the complete 603,935,676-parameter model
at `B=16,S=8192` with FP8, FBT, LatentMoE, MSAR, CUDA graphs, DDP, and the
AnchorMuon/AdamATan2 update. Step times were `1576.1, 924.3, 705.0, 704.7,
705.4 ms`; the steady mean was `704.9 ms` (about `1.49M` aggregate bytes/s or
`186.6k` bytes/s/GPU). Batch calibration selected `B=16` with roughly `348 W`
peak per GPU, and all five updates were finite.

The dominant single-rank graph costs remain Mamba-3 DQKV backward (`91.45 ms`),
Mamba-3 forward (`40.32 ms`), generic elementwise/copy work (about `67.7 ms`),
FP8 quantization (about `32.0 ms`), MSAR pointer gradients (`15.02 ms`), and
Mamba rotary/bias backward (`14.65 ms`).

The only new Python-level candidate, caching FP32 MSAR query/pre/bias casts for
the recurrent loop, was rejected. Its forward and most gradients matched, but
`msar_queries.grad` changed (maximum absolute difference `9.77e-4`, RMS
`1.56e-4`) because the full-bank cast changed FP32 accumulation order. The
worktree was restored to `dbed2ef`; no production speed patch was promoted.
The DDP/data path is already covered by direct DMA, one-batch prefetch, and
direct reducer-storage reductions. Remaining high-value work is upstream
Mamba-3 kernel tuning or aggregation of Transformer Engine quantization
launches, each requiring full fixed-shape parity and an eight-rank rerun.

## Rejected: Mamba-3 chunk-size variants

A pinned `/tmp/hnet-venv` microbenchmark at `B=16,S=8192,D=1024` compared the
native Mamba-3 SISO chunk sizes. The canonical `chunk_size=32` was about
`25.24 ms` median for one forward/backward layer; `16` was `26.40 ms` (+4.6%)
and `64` was `30.47 ms` (+20.7%). They also changed the recurrence numerics:
relative to the canonical output, chunk 16 had max absolute difference `4.0`
and RMS `0.0265`, while chunk 64 had max `1374` and RMS `16.24`. Both variants
were rejected and the canonical chunk size remains 32.
