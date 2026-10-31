# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

**Venmugil Elango, Nidhi Bhatia, Roger Waleffe, Rasoul Shafipour, Tomer Asida,
Abhinav Khattar, Nave Assaf, Maximilian Golub, Joey Guman, Tiyasa Mitra,
Ritchie Zhao, Ritika Borkar, Ran Zilberstein, Mostofa Patwary, Mohammad Shoeybi,
Bita Rouhani (NVIDIA)**  
**arXiv:2601.18089v1 — 26 January 2026**

This is a hand-edited Markdown transcription of the paper downloaded from
<https://arxiv.org/pdf/2601.18089>. Line-wrap artifacts from PDF extraction were
removed and equations, tables, and symbols were rewritten for readability.

## Abstract

Mixture-of-Experts (MoE) models are widely used to increase model capacity while
keeping the compute per token approximately fixed. This paper revisits their design
from a hardware/software co-design perspective. It characterizes memory-bandwidth and
communication bottlenecks in both high-throughput and latency-critical serving, then
introduces **LatentMoE**, which routes and computes experts in a shared, lower-
dimensional latent space. At an equal FLOP or parameter budget, the saved bandwidth
and communication are reinvested in more experts and more active experts. Experiments
up to 95B parameters and 1T training tokens report better accuracy per FLOP and per
parameter than standard MoEs.

## 1. Motivation and design principles

Let `d` be the model hidden width, `m` the expert intermediate width, `N` the
number of routed experts, and `K` the number selected for each token. For a token
batch containing `t_total` tokens, a uniformly balanced expert receives

```
t_exp = t_total K / N.
```

The paper's running example is Qwen3-235B-A22B (`N = 128`, `K = 8`, `d = 4096`,
`m = 1536`) deployed across `EP = 64` GB200 GPUs. A single GPU owns two experts.

### Memory bandwidth

At low serving concurrency, expert GEMMs are memory-bandwidth bound. With FP4
weights, the arithmetic intensity for the two local experts is

```
I = (2 t_exp d m) / (d m + t_exp (d + m)).
```

For the assumed GB200 peak (`F = 10 PFLOP/s`, `BW_HBM = 8 TB/s`), compute-bound
execution requires `I >= F/BW_HBM = 1250 FLOP/byte`, which implies approximately
`t_exp >= 1418`. Typical interactive batches have fewer tokens per expert, so loading
expert weights dominates.

### Communication

In throughput-oriented serving, all-to-all dispatch and combine become important.
For the running example, the per-GPU communication is proportional to

```
M_comm = 2.5 (N / EP) t_exp d = 5 t_exp d,
```

where `2.5` accounts for FP4 dispatch and BF16 aggregation. The corresponding local
expert compute is proportional to `(N/EP) t_exp d m`. Their ratio is independent of
`d`, so reducing the hidden width alone does not remove the communication bottleneck
unless the bytes dispatched are reduced as well.

### Expressivity

The effective nonlinear budget of a token is approximately

```
U_eff ∝ K m.
```

Reducing `K` or `m` therefore risks accuracy. Conversely, increasing `N` expands the
number of possible expert combinations. The number of `K`-expert subsets is
`C(N, K)`, which grows rapidly with both `N` and `K`.

The paper's design principles are:

1. Small-batch inference is generally bandwidth bound; optimize accuracy per
   parameter.
2. At larger batches, expert communication can dominate; optimize dispatched bytes.
3. Preserve `K` and `m` to preserve the nonlinear budget.
4. There is an intrinsic rank `r_eff`; the latent width must satisfy `l >= r_eff`.
5. Increasing the number and diversity of experts improves quality.

## 2. LatentMoE architecture

LatentMoE compresses only the routed expert path. For `x ∈ R^d`, a shared down
projection produces

```
z = W_down x,                 W_down ∈ R^(l × d).
```

The router still reads the original hidden state and computes

```
p' = softmax(W'_r x),          W'_r ∈ R^(N' × d).
```

Each routed expert remains a complete gated MLP, but its input and output width is
`l` rather than `d`:

```
E_i(z; l) = W_FC2^(i) [ act(W_FC1^(i) z) ⊙ gate^(i)(z) ],
W_FC1^(i), W_gate^(i) ∈ R^(m × l),
W_FC2^(i) ∈ R^(l × m).
```

The selected routed outputs are weighted and summed in latent space, then projected
back to the residual width:

```
y_routed = W_up Σ_{i ∈ TopK} p'_i E_i(z; l),
W_up ∈ R^(d × l).
```

The shared experts are not compressed. For `S` always-on experts `E_j(·; d)`, the
layer output is

```
ℓ-MoE(x) = W_up Σ_{i ∈ TopK} p'_i E_i(W_down x; l)
           + Σ_{j=1..S} E_j(x; d).
```

The router and shared experts therefore operate at `d`; token dispatch, routed expert
weights, and routed expert GEMMs operate at `l`.

### Scaling variants

Set `α = d/l`. The routed expert count is expanded to `N' = α N`.

* **`l`-MoE_eff:** select the original `K` experts. This reduces communication and
  routed weight traffic by approximately `α` while matching baseline accuracy.
* **`l`-MoE_acc (recommended):** select `K' = α K` experts. The extra active experts
  restore the original communication and bandwidth budget, while increasing expert
  diversity and accuracy.

The paper's experiments use `α = 4`, after finding that ratios up to four preserve
quality. This construction requires `l >= r_eff`; compression beyond that point can
cause a quality collapse. Scaling `N` is important: compressing `d` without expanding
`N` causes a substantial validation-loss increase.

### Cost comparison

| Architecture | Communication per GPU | Routed weight memory per expert | Accuracy | Inference efficiency |
|---|---:|---:|---|---|
| Standard MoE | `(N/EP) t_exp d` | `d m` | baseline | baseline |
| `l`-MoE_eff | `(N/EP) t_exp l` | `l m` | approximately unchanged | higher |
| `l`-MoE_acc | `(N/EP) t_exp d` | `d m` equivalent after `K` scaling | higher | approximately unchanged |

The table is normalized by hardware constants. In `l`-MoE_acc the routed expert
weights are smaller, but `K` is larger by `α`, so total communication and weight
traffic are approximately iso-cost with standard MoE.

## 3. Experiments

The authors first use a 16B-total/2B-active Transformer for ablations, then a
95B-total/8B-active Transformer, and finally a 73B-total/8B-active hybrid
Mamba–Attention model. Models use a sequence length of 8192, a global batch of 768
(about six million tokens), and 8.4B warmup tokens for the Transformer experiments.
The 8B-active Transformer uses a cosine schedule (`1.2e-3` to `3e-6`). The hybrid
model uses WSD (`8e-4` to `8e-6` during the last 15%). The 8B-active runs use a
load-balancing coefficient of `1e-4` and DeepSeek's auxiliary-loss-free balancing.

### Baseline configurations

| Configuration | Layers | `d` | Routed experts `N` | Active `K` | Shared `S` | `m` | Activation | Attention |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 16B total / 2B active | 27 | 2048 | 64 | 6 | 2 | 1408 | SwiGLU | 16 heads, GQA 16 |
| 95B total / 8B active | 32 | 4096 | 128 | 6 | 2 | 2688 | Squared-ReLU | 32 heads, GQA 8 |
| Hybrid 73B total / 8B active | 52 (24 Mamba/MoE, 4 attention) | 4096 | 128 | 6 | 2 | 2688 | Squared-ReLU | 32 heads, GQA 8 |

The hybrid Mamba blocks use Mamba-2 with 128 heads, head dimension 64, state
size 128, and 8 groups.

### Compression and expert-count ablations

Across the 16B model, `α <= 4` preserves validation loss; the authors use `α = 4`
for larger models. When `d` is compressed by four, multiplying the expert count by
four removes the degradation seen when the original expert count is retained.

At equal `l = 512` and `α = 4`, `l`-MoE_eff tracks the standard baseline while
`l`-MoE_acc has lower validation loss. The authors recommend `l`-MoE_acc for the
accuracy/efficiency Pareto frontier.

### Scaling results

For the 95B model (`l = 1024`, `α = 4`), after 300B tokens:

| Model | Active parameters | Total parameters | MMLU-Pro | MMLU | Code | Math | Commonsense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8.47B | 94.4B | 29.26 | 58.95 | 40.33 | 64.39 | 74.32 |
| `l`-MoE_acc | 8.44B | 94.8B | 34.91 | 62.23 | 41.50 | 64.88 | 75.18 |
| `l`-MoE_eff | 5.62B | 94.8B | 34.75 | 61.06 | 40.68 | 63.61 | 73.72 |

For the 73B hybrid after 1T tokens:

| Model | Active parameters | Total parameters | MMLU-Pro | MMLU | Code | Math | Commonsense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8.09B | 72.6B | 48.30 | 70.10 | 51.95 | 78.32 | 81.73 |
| `l`-MoE_acc | 8.02B | 72.8B | 52.87 | 72.11 | 55.14 | 80.19 | 82.10 |
| `l`-MoE_eff | 5.91B | 72.8B | 51.29 | 71.34 | 53.13 | 77.01 | 80.78 |

### Inference measurements

On two H100 GPUs with FP8 per-tensor quantization, the hybrid model's measured
throughput (tokens/s/GPU) for concurrency 1, 4, 16, 64, and 128 was respectively:

| Concurrency | LatentMoE | Standard MoE |
|---:|---:|---:|
| 1 | 181.6 | 206.6 |
| 4 | 528.5 | 509.8 |
| 16 | 1130.8 | 1204.6 |
| 64 | 1569.6 | 1549.3 |
| 128 | 1625.8 | 1725.9 |

The small throughput difference is attributed to smaller routed GEMMs. The authors
suggest separate CUDA streams for shared and routed experts, and specialized CUTLASS
small-matrix kernels, to recover utilization.

### Projected trillion-parameter serving

The paper defines an effective-parameter multiplier (EPM). If a treated model's score
is `S_treat`, invert a baseline scaling law `f` to obtain `N_eff = f⁻¹(S_treat)`;
then

```
λ = N_eff / N_treat.
```

For Kimi-K2-1T-LatentMoE, the fitted estimate is `λ ≈ 1.35`. A standard iso-accuracy
model therefore needs about 1.35T parameters (roughly 350B additional parameters),
constructed by increasing depth from 61 to 80 layers. The projected standard model is
1.24×–3.46× slower over the serving Pareto frontier. Latent projections add at most
about 9% versus native Kimi-K2-1T.

## 4. Reference implementation comparison

The `kyegomez/Latent-MoE` repository is a dependency-light PyTorch reference. Its
`LatentMoE` class matches the paper's tensor ordering: full-width router, shared
`W_down`/`W_up`, grouped latent dispatch, and full-width shared experts. It exposes
both a loop dispatcher and a single-device grouped dispatcher. The grouped path sorts
`(token, expert)` slots, runs contiguous slices through each expert, then uses an
`index_add` to restore token order. It is useful as an equation-level reference but
has no expert-parallel all-to-all, no fused router, no Transformer Engine FP8 path,
and no CUDA-graph constraints. Its optional multi-GPU mode copies latent activations
to each device and loops over experts, which is unsuitable for the production
throughput target.

Megatron-LM's `MoELayer` is the systems reference. It separates the forward pass into
`route`, `preprocess`, `dispatch`, `routed_experts_compute`, `combine`, and
`postprocess`. This decomposition allows CUDA-graph capture, expert-parallel token
dispatchers, TE grouped GEMMs, delayed expert weight-gradient scheduling, and
recompute. Its latent path inserts TE `fc1_latent_proj` before dispatch and
`fc2_latent_proj` after combine. Shared experts can be launched on an independent
CUDA stream after routing (or before dispatch where the backend supports it) and
joined during postprocess. In the latent training path, Megatron explicitly rejects
shared-expert overlap for unsupported dispatcher/graph combinations; silently
executing an overlap without stream events would be a race.

For this H-Net, the production implementation should retain the reference's
full-width router and shared expert, but use Megatron's staged execution and TE
fused/distributed kernels. Router logits, softmax probabilities, top-k weights,
auxiliary balancing terms, and the latent accumulation should be computed in FP32.
The selected probabilities should be renormalized after top-k selection. Expert
outputs may be cast back to the activation dtype only after the weighted latent
reduction. FP8 expert kernels must preserve the unrounded auxiliary values needed by
backward math, following the CroissAttn correction used elsewhere in this project.

## 5. Related work and limitations

Quantization, pruning, and expert merging are orthogonal and can be composed with
LatentMoE. MoLAE also uses low-rank latent expert weights, but is a post-training
compression method that compresses only part of the expert path and therefore does
not obtain the same dispatch and weight-bandwidth savings. Manifold-Constrained
Hyper-Connections changes residual connectivity and is complementary to LatentMoE.

The paper does not specify an exact capacity-factor/drop-token policy, router jitter
policy, or an implementation of expert parallel all-to-all. These are systems choices
that must be supplied by the training stack. Its shared-expert overlap suggestion is
an optimization, not part of the mathematical definition.

## 6. Conclusion

LatentMoE reduces the routed path from `d` to `l`, then reinvests the savings in more
experts and (for `l`-MoE_acc) more active experts. Under matched inference cost it
improves accuracy, while the efficiency variant lowers routed bandwidth and compute.
The central implementation invariant is that routing and shared experts remain in the
full residual width, while dispatch and all routed expert operations remain in latent
width until the final up-projection.

## References

* Paper: <https://arxiv.org/abs/2601.18089>
* Reference implementation reviewed for this project: <https://github.com/kyegomez/Latent-MoE>
* Megatron-LM MoE layer reviewed for dispatch, shared-expert overlap, and TE integration:
  <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py>
* DeepSeek-MoE: <https://arxiv.org/abs/2401.06066>
* DeepSeek-V2: <https://arxiv.org/abs/2405.04434>
