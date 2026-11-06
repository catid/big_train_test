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
