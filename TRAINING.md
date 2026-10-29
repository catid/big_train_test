# Direct H-Net run

`train_hnet_600m.py` is the canonical run. It reads a raw UTF-8 byte file,
uses the frozen 256-row byte code (zero trainable embedding parameters), and
trains `m4 -> T19 -> m4` with learned routing, MSAR in the innermost stack,
two shared-weight inner executions, chunk-level FBT feedback, FP32 CE/router
losses, Muon/GramNS plus AdamATan2 fallback, weight decay, DDP, and static
CUDA-graph capture.

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
