# Full-shape runtime evidence

These measurements are evidence for the canonical graph, not a reduced model
or a substitute training target. They were collected on the available B300
with `B=1`, `S=8192`, BF16 activations, two FBT passes, eight-expert
LatentMoE, and the external GDN2 adapter.

## Direct full-model eager check

The model instantiated with **636,756,976 trainable parameters**, five GDN2
adapters, and nine LatentMoE blocks. A fixed all-true mask was used because
the production stack rejects packed 2-D inputs and requires the explicit
`[B,S,D]` graph shape.

| Measurement | Result |
|---|---:|
| forward | 46.76 s |
| backward | 34.93 s |
| peak allocated memory | 18.85 GiB |
| final logits | `[1, 8192, 256]` |
| feedback logits | 2 passes |
| loss | 12.7926 (finite) |
| global clipped gradient norm | 66.12 |

The run completed without NaNs or Infs. The long first forward includes kernel
initialization and is not a throughput benchmark. It does establish that the
full hierarchy, FBT path, GDN2 constructor, FP8 expert modules, FA, and CE
loss can execute together at the requested shape.

## Kernel and optimization caveat

The Python Transformer Engine package came from `/home/catid/fp8/TransformerEngine`.
The installed `transformer_engine_torch` shared object also reports an ABI
symbol error when imported directly in this environment, so the exact FP8
GroupedLinear dispatch and its eager/graph parity still require a clean,
matching TE build. Do not claim this run as proof of a production FP8 kernel
until that dispatch is recorded.

The separate full-shape fake-TE profile measured 121–131 ms steady forward plus
backward, with CUDA time dominated by matrix multiplies (18.9 ms), copies
(6.5 ms), RMSNorm backward (5.36 ms), and indexed accumulation (4.22 ms).
Those figures are useful for locating overhead but are not comparable to a
real FP8 grouped-GEMM result.

With the actual TE extension and the current fixed-shape CUDA-graph path,
warm replay of forward/backward plus gradient clipping and GramNS/AdamATan2
update measured approximately 179 ms per step in a three-replay sample. The
graph capture setup took about 32 s. A real-TE eager profile measured 230–260
ms for warm forward plus backward before the fused TE unpermute change. The
current dispatch uses TE's index-map unpermute for the weighted expert merge;
an isolated `N=2048`, latent-width-640 FP8 expert backward measured 8.3 ms with
that path versus 17.6 ms with the prior `index_add_` merge. These are warm
measurements and should be repeated after any dependency or driver change.

## Reproduction

Use the canonical command in [`HNET_REPRODUCTION.md`](../HNET_REPRODUCTION.md)
with the pinned GDN2 environment. Record the actual TE extension path,
cuDNN/driver versions, dispatch backend, peak memory, finite diagnostics, and
median warm steady-state step time in the run manifest.
