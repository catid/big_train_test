# Stability brief source register

This register keeps the references and implementation advice that informed the
stability audit. It is intentionally text-only; PDFs, source archives, model
checkpoints, and training outputs stay outside the repository.

## Primary papers and official implementations

| Topic | Paper / official source | Local copy or audit note |
|---|---|---|
| H-Net dynamic chunking | [arXiv:2507.07955](https://arxiv.org/abs/2507.07955), [goombalab/hnet](https://github.com/goombalab/hnet) | Current hierarchy is in `hnet/models/hnet.py` and `hnet/modules/dc.py`. |
| Gated DeltaNet-2 | [arXiv:2605.22791](https://arxiv.org/abs/2605.22791), [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2), [cuDNN Linear Attention](https://docs.nvidia.com/cudnn/operations/LinearAttention) | Adapter is `hnet/modules/gdn2_adapter.py`. The supplied reproducible handoff requires `/home/catid/gdn2` commit `11d9f76`; the current config metadata says `5ec53d5` and must be reconciled. |
| Library of Layers | [arXiv:2611.03526](https://arxiv.org/abs/2611.03526), [arahmani1/library-of-layers](https://github.com/arahmani1/library-of-layers) | Hand-cleaned transcription: [`lol_2611.03526.md`](lol_2611.03526.md). The current stack has shifted-window attention but no causal window-level LoL router. |
| Quantization loss spikes | [arXiv:2610.38277](https://arxiv.org/abs/2610.38277), [tingtang2/quantization-loss-spikes](https://github.com/tingtang2/quantization-loss-spikes) | Add FP32 reference logits and low/high-precision KL/RMS diagnostics before FP8 tuning. |
| Output embedding centering | [arXiv:2601.02031](https://arxiv.org/abs/2601.02031) | Compare untied-head μ-centering, no centering, and head-only optimizer controls. |
| PaLM z-loss | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) | Keep LM-head z-loss separate from LatentMoE router z-loss. |
| LatentMoE | [arXiv:2601.18089](https://arxiv.org/abs/2601.18089), [kyegomez/Latent-MoE](https://github.com/kyegomez/Latent-MoE), [Megatron MoELayer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) | Hand-cleaned transcription and implementation comparison: [`latentmoe_2601.18089.md`](latentmoe_2601.18089.md). |
| Activation bias | [arXiv:2610.14500](https://arxiv.org/abs/2610.14500) | Hand-cleaned transcription and H-Net mapping: [`activation_bias_2610.14500.md`](activation_bias_2610.14500.md). |

## Repository state used for the audit

- H-Net checkout audit base: commit `48fb2f4` (`Remove legacy architectures
  and keep canonical H-Net`).
- GDN2 checkout observed: `/home/catid/gdn2`, commit
  `5ec53d56da98841f6613a0a1c1a59ca1aad9b44c`; user’s handoff names
  `11d9f76`, so these are not interchangeable provenance records.
- Library of Layers checkout observed: `/home/catid/library_of_layers`, commit
  `03b85f42d30d628a973ec759eeb3491b56f1d9e4`.
- LoL transcription PDF SHA-256 recorded in [`lol_2611.03526.md`](lol_2611.03526.md):
  `20bedd3c47371d1c79466fec964c579c0388ce5675ae7cb4aa562143238ede62`.
- GDN2 runtime environment supplied by the user:

  ```sh
  export CUDA_VISIBLE_DEVICES=1
  export GDN2_ROOT=/home/catid/gdn2
  export PYTHONPATH=/home/catid/gdn2:/home/catid/gdn2-gb300/flash-linear-attention:/home/catid/cudnn132:/home/catid/cudnn132/nvidia_cutlass_dsl/dsl_packages
  ```

  Required benchmark: `uv run python benchmarks/gdn2_lol_fa_repeat.py` with
  `GDN2_GRAPH=1`, fixed `B1/S4096`, BF16, FROST, fused projections, and CUDA
  graph capture. This benchmark is an external dependency check, not evidence
  that the full H-Net trainer has passed parity.

## Advice retained from the supplied brief

1. Preserve direct byte NLL/BPB even when boundary, reconstruction, router, or
   auxiliary losses are present.
2. Establish eager PyTorch reference paths before compile, CUDA graphs, Triton,
   FlashAttention, cuDNN custom ops, or FP8.
3. Keep softmax-sensitive reductions, recurrent summaries, router logits,
   optimizer statistics, and diagnostics in FP32 unless parity demonstrates a
   safe lower-precision implementation.
4. Compare optimized paths against eager for logits, each loss component,
   activations, gradients, parameter updates, and optimizer state transitions.
5. Treat H-Net boundaries, GDN2 state, LoL routes, FA logits, the output head,
   optimizer state, and DDP reductions as separate dynamical systems. Instrument
   each one instead of relying on a single global gradient norm.
6. A non-finite event is a diagnostic stop: preserve the first failing batch and
   last healthy checkpoint; do not hide it with `nan_to_num`, silent clipping,
   batch deletion, or unconditional rollback.
7. Validate deterministic checkpoint/resume and 1/2/4/8/12-GPU trajectories
   using the same global token batch before serious pretraining.
8. Run at least three seeds and an extended (4–8×) horizon stress test before
   calling an intervention stable.

