# Rethinking Activation Functions: Reducing Bias Can Improve Learning

**Paper:** Zirui Peng, Heyi Zheng, Ziming Liu (arXiv:2610.14500v1, 2026-10-10)  
**Source:** <https://arxiv.org/abs/2610.14500>  
**Local source archive:** downloaded from `https://export.arxiv.org/e-print/2610.14500`; the PDF and TeX archive are kept outside the repository.

## Abstract

Common activations such as ReLU, GELU, Swish, and Sigmoid have predominantly positive outputs. Their nonzero post-activation mean is called **activation bias**. Controlled teacher–student experiments show that this bias can speed early loss reduction while causing overshoot, nonsmooth trajectories, and a sharper loss landscape. Subtracting the bias slows early learning but gives lower final loss. ViT, DiT, and GPT-2 scale-up experiments show that reducing bias is associated with better late-stage performance.

## Main definitions

For a pre-activation `z` and activation `φ`,

`b_φ = E[φ(z)]`.

A fixed bias-reduced intervention is

`φ_shifted(z; b) = φ(z) − b`,

where `b` is estimated from the activation mean at initialization. For scale-up models the paper uses **adaptive shifting**, computed independently for every token/example and layer:

`φ̃(z) = φ(z) − mean(features, dim=feature)`.

The mean is taken over the post-activation feature dimension, so the shifted activation has zero feature mean for each token.

## Evidence

- In controlled pre-normalized residual MLPs, the unshifted activation learns faster at the start, then overshoots and is overtaken by the shifted model.
- The shifted trajectory has lower Hessian eigenvalues and smoother parameter updates in the controlled setting.
- The activation-bias component can carry task information early in image reconstruction; the model relies on it less later.
- Scale-up ViT, DiT, and GPT-2 experiments use adaptive shifting. GPT-2 pretraining reports about 0.01 lower loss and 1% lower perplexity for the shifted condition.
- The paper explicitly cautions that zero post-activation mean is a diagnostic intervention, not a universal prescription; activation scale, variance, selectivity, and task/model details still matter.

## Experimental design and findings

### Controlled teacher–student model

The controlled model is a pre-normalized residual MLP,

`x_(l+1) = x_l + W2 φ(W1 LN(x_l))`.

The teacher uses the ordinary activation. Two students start from the same
parameters and see the same batches; one uses the ordinary activation and the
other subtracts a shift. This isolates activation bias from data, optimizer,
and initialization differences. The ordinary student has a short-lived loss
advantage, but its residual mean overshoots and its loss later stalls. The
shifted student follows a smoother trajectory and finishes lower.

The authors compare trajectories in the plane spanned by the initial gradients
and in its orthogonal complement. The ordinary and shifted students take
different update directions, rather than merely moving at different speeds.
Hessian and Gauss–Newton measurements in the controlled setting associate the
ordinary activation with larger curvature. These results are mechanistic
observations for the controlled setup, not a claim that a particular scalar
shift is always optimal.

### Information in the bias component

For a post-activation vector `h`, the paper decomposes

`h = (mean(h) * 1) + (h − mean(h) * 1)`.

In a VGG-style reconstruction experiment, decoding only the first component
recovers recognizable image structure early in training for the ordinary model.
That reconstruction later diverges from the full output. The shifted model has
little recoverable content in this component. The result suggests that the
positive mean can act as a temporary information channel, which helps explain
the early loss advantage and later overshoot.

### Scale-up experiments

The fixed scalar used in the toy model is unsuitable when the activation mean
changes by layer and over training. The scale-up intervention therefore uses
the current mean for each sample and layer. The paper applies this to ViT-Base
classification, DiT-B/2 image generation, and GPT-2 language pretraining,
holding all other training choices fixed. The centered condition can start
behind on training loss, but ends with better test loss, accuracy, FID, or
perplexity. GPT-2 reports about `0.01` lower loss and `1%` lower perplexity.
Additional experiments report gains on ViT-L, OLMo-2-1B, and HyenaDNA.

### Theory and ablations

At initialization, the paper approximates an up-projection pre-activation as
`N(0, 1/3)` under LayerNorm and uniform weights. The resulting expected shifts
are approximately `0.230` (ReLU), `0.115` (GELU), `0.077` (Swish), `0` (Tanh),
and `0.500` (Sigmoid). Sweeps over the fixed shift show that these values are
near the final-loss optimum in the controlled model. Larger width and depth
make the centering benefit more consistent across seeds. The effect persists
when the toy model uses post-norm blocks, trainable linear biases, or a
soft-target cross-entropy objective.

The paper also reports that centering changes the post-activation magnitude
distribution less over training and keeps it closer to the residual-stream
distribution. This is offered as a possible explanation for smoother learning,
not as a proved causal law. The discussion explicitly lists open interactions
with initialization, architecture, task, activation variance, and selectivity.

## Application to this H-Net

The production model uses SwiGLU inside every routed and shared LatentMoE expert. We apply the paper's scale-up intervention after the SwiGLU product and before the expert down projection:

```python
y, gate = fc1(x).chunk(2, dim=-1)
h = silu(gate) * y
h = h - h.float().mean(dim=-1, keepdim=True).to(h.dtype)
y = fc2(h)
```

The mean is computed in FP32 for BF16/FP8 numerical stability and then cast back. It is per token, over the expert hidden features; it is not a trainable embedding, a fixed scalar guessed from initialization, or a fallback activation. The same production path is used by routed and shared experts. No centering is applied to the byte lifting, router logits, residual stream, attention, or chunk/dechunk boundaries.

This intervention changes the expert output statistics while preserving the matched LatentMoE parameter budget and FP8 TE linears. It may delay early loss improvement; validation should compare late-stage loss, activation means, and utilization against the exact same full configuration.

## Scope and limitations

The paper does not establish that centering is optimal for every architecture or training regime. H-Net's hierarchical chunking, recurrent Mamba path, and FP8 expert dispatch differ from the paper's GPT-2 setup. Therefore this change is an evidence-backed full-model intervention, and its effect must be measured in the intended 600M byte H-Net run rather than inferred from a reduced model.

## Reproducibility notes

The repository records this cleaned transcription and implementation mapping. The original PDF/source archive is not committed as a training artifact. Equations and numerical claims above were checked against the downloaded TeX source and abstract.
