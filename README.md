# H-Net

This checkout contains the canonical approximately 600M-parameter raw-byte H-Net
trainer. It reads fixed UTF-8 byte IDs, applies a non-trainable byte lift, learns
hierarchical chunk boundaries from the first update, and predicts the next byte
with a trainable FP32 output head. There is no BPE tokenizer or trainable input
embedding path.

## Production model

The single supported configuration is
[`configs/hnet_1stage_600m_gdn2_lol_8192.json`](configs/hnet_1stage_600m_gdn2_lol_8192.json):

- fixed sequence length 8192 bytes and a 2048-row latent cap;
- encoder and decoder: `GDN2 -> LoL -> GDN2 -> LoL`;
- middle trunk, repeated twice with tied GDN2 and FA weights:
  `GDN2 -> LoL -> GDN2 -> LoL -> FA`;
- GDN2 width 3072 (48 heads of 64), and LoL/FA width 1280
  (20 query heads, 4 KV heads of 64);
- LoL uses learned RoPE and shifted causal windows of 128 tokens;
- FA uses causal NoPE GQA;
- LoL/FA feed-forward blocks are eight-expert LatentMoE with grouped FP8
  Transformer Engine projections;
- FBT uses two trunk passes; the outer fixed-shape path is prepared for DDP,
  pinned byte staging, compilation, and CUDA graph forward/backward capture.

The model requires the inspected external GDN2/cuDNN/FLa checkout and a working
Transformer Engine PyTorch FP8 extension. Construction fails clearly when these
pinned runtime dependencies are unavailable; no alternate architecture is
silently substituted.

## Training

Install the package in an environment containing the pinned CUDA extensions:

```sh
uv pip install -e .
```

Prepare a raw UTF-8 byte file (for example from FineWeb-derived text) and run
the complete fixed-shape model:

```sh
CUDA_VISIBLE_DEVICES=1 \
GDN2_ROOT=/home/catid/gdn2 \
PYTHONPATH=/home/catid/gdn2:/home/catid/gdn2-gb300/flash-linear-attention:/home/catid/cudnn132:/home/catid/cudnn132/nvidia_cutlass_dsl/dsl_packages \
uv run python scripts/train_hnet_600m.py \
  --data /path/to/fineweb.bytes \
  --config configs/hnet_1stage_600m_gdn2_lol_8192.json \
  --seq 8192 --batch 1
```

The trainer reports CE, bits per byte, routing loss, gradient norm, tokens per
second, power, and held-out evaluation every 40 optimizer steps by default.
Artifacts and checkpoints belong outside this source repository.

## Layout

```text
configs/                         canonical model configuration
hnet/models/                     H-Net hierarchy and byte language-model wrapper
hnet/modules/dc.py               learned chunking and dechunking
hnet/modules/gdn2_adapter.py     pinned external GDN2 loader
hnet/modules/hybrid.py           GDN2, LoL, FA, and fixed-shape block stack
hnet/modules/latent_moe.py       eight-expert FP8 LatentMoE
hnet/utils/losses.py             router regularizer
scripts/train_hnet_600m.py       full-system trainer
scripts/optimizer_experiments.py GramNS/AnchorMuon and AdamATan2
```

## Citation

If you use the H-Net dynamic-chunking mechanism, cite:

```bibtex
@article{hnet,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```
