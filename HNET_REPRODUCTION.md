# Canonical H-Net reproduction notes

The repository's reproducible target is the complete fixed-shape raw-byte model
in [`configs/hnet_1stage_600m_gdn2_lol_8192.json`](configs/hnet_1stage_600m_gdn2_lol_8192.json).
It is not a collection of interchangeable small models: all measurements must
use the same 8192-byte sequence shape, 2048-row latent cap, routing settings,
FP8 MoE path, optimizer, and data packing.

## Runtime

The production path needs the pinned GDN2 checkout, cuDNN/FROST and FLA runtime,
and a functioning Transformer Engine PyTorch extension. The empty TE metadata
package or a mismatched Python ABI is not sufficient. Verify these dependencies
before launching a long run; the model intentionally has no Mamba, attention, or
optimizer fallback.

## Architecture

Raw byte IDs are lifted by a fixed codebook at width 3072. The encoder and decoder
are `GDN2 -> LoL -> GDN2 -> LoL`. The latent trunk repeats
`GDN2 -> LoL -> GDN2 -> LoL -> FA` twice. GDN2 and FA cores are shared between
repetitions; LoL cores and width bridges remain independent. LoL is learned-RoPE
GQA in shifted 128-token causal windows. FA is causal NoPE GQA. Each LoL/FA FFN
is eight-expert LatentMoE with grouped Transformer Engine FP8 GEMMs. FBT supplies
two trunk passes with prefix mixing and jitter.

## Fixed-step benchmark

After dependency checks and compilation, benchmark the full model with forward and
backward on the exact production shape. Record median step time, tokens per second
per GPU, peak memory, gradient norm, power, router loss, and held-out CE. Do not
substitute a reduced depth, dynamic sequence length, alternate mixer, or ordinary
AdamW run when reporting canonical results.

```sh
CUDA_VISIBLE_DEVICES=1 \
GDN2_ROOT=/home/catid/gdn2 \
PYTHONPATH=/home/catid/gdn2:/home/catid/gdn2-gb300/flash-linear-attention:/home/catid/cudnn132:/home/catid/cudnn132/nvidia_cutlass_dsl/dsl_packages \
uv run python scripts/train_hnet_600m.py \
  --data /path/to/fineweb.bytes \
  --config configs/hnet_1stage_600m_gdn2_lol_8192.json \
  --seq 8192 --batch 1 --eval-every 40
```

Training artifacts and checkpoints are stored outside the code repository.
