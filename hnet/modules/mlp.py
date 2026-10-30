# Base code imported from
# https://github.com/state-spaces/mamba
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hnet.te import TEGroupedLinear, make_linear, te

class SparseMoESwiGLU(nn.Module):
    """Static-shape top-1 sparse SwiGLU with a DeepSeek-style shared expert.

    The router dispatches each token to one of ``num_experts`` routed FFNs;
    the shared FFN is evaluated for every token and added through a learned
    sigmoid gate.  Dispatch uses a fixed capacity tensor and grouped
    ``einsum`` operations, so it remains CUDA-graph friendly and avoids the
    Python-side per-expert loops used by reference MoE implementations.  The
    fixed capacity is also the same contract used by the BioNeMo Mixtral TE
    recipe (per-expert padding before grouped GEMM).

    This module intentionally supports top-1 routing only.  ``1:8`` means one
    of eight routed experts is active for each token; enabling top-2 would
    silently change the requested sparsity and is rejected at construction.
    """

    def __init__(
        self,
        d_model,
        d_intermediate,
        num_experts=8,
        top_k=1,
        expert_intermediate=0,
        shared_intermediate=0,
        capacity_factor=1.25,
        capacity_tokens=0,
        aux_loss_weight=0.01,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if top_k != 1:
            raise ValueError("SparseMoESwiGLU requires moe_top_k=1 for 1:8 sparsity")
        if num_experts < 2:
            raise ValueError("SparseMoESwiGLU requires at least two routed experts")
        if capacity_factor < 1.0:
            raise ValueError("moe_capacity_factor must be >= 1.0")
        # Split the dense FFN width across experts.  Configs can override the
        # rounded width, while retaining a DeepSeek-like narrow shared path.
        expert_intermediate = int(expert_intermediate or max(1, d_intermediate // num_experts))
        shared_intermediate = int(shared_intermediate or expert_intermediate)
        expert_intermediate = (expert_intermediate + multiple_of - 1) // multiple_of * multiple_of
        shared_intermediate = (shared_intermediate + multiple_of - 1) // multiple_of * multiple_of
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_intermediate = expert_intermediate
        self.shared_intermediate = shared_intermediate
        self.capacity_factor = float(capacity_factor)
        self.capacity_tokens = int(capacity_tokens)
        if self.capacity_tokens < 0:
            raise ValueError("capacity_tokens must be non-negative")
        self.aux_loss_weight = float(aux_loss_weight)

        # [E, 2I, D] and [E, D, I] are consumed as one grouped GEMM via
        # einsum.  The layout mirrors TE GroupedLinear's stacked experts.
        self.experts_fc1 = TEGroupedLinear(
            self.num_experts, d_model, 2 * expert_intermediate, bias=bias,
            params_dtype=dtype, device=device, use_grouped_tensor=True,
        )
        self.experts_fc2 = TEGroupedLinear(
            self.num_experts, expert_intermediate, d_model, bias=bias,
            params_dtype=dtype, device=device, use_grouped_tensor=True,
        )
        self.shared_fc1 = make_linear(d_model, 2 * shared_intermediate, bias=bias, **factory_kwargs)
        self.shared_fc2 = make_linear(shared_intermediate, d_model, bias=bias, **factory_kwargs)
        self.router = make_linear(d_model, self.num_experts, bias=False, **factory_kwargs)
        self.shared_gate = make_linear(d_model, 1, bias=True, **factory_kwargs)
        self._aux_loss = None
        self._aux_loss_count = 0
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.experts_fc1.named_parameters():
            if name.startswith("weight"):
                nn.init.normal_(weight, mean=0.0, std=0.02)
        for name, weight in self.experts_fc2.named_parameters():
            if name.startswith("weight"):
                nn.init.normal_(weight, mean=0.0, std=0.02 / math.sqrt(2.0))
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_fc2.weight, mean=0.0, std=0.02 / math.sqrt(2.0))
        if self.shared_fc1.bias is not None:
            nn.init.zeros_(self.shared_fc1.bias)
            nn.init.zeros_(self.shared_fc2.bias)
        nn.init.zeros_(self.shared_gate.weight)
        nn.init.zeros_(self.shared_gate.bias)
        # Preserve the intended zero-start shared-expert gate when the outer
        # H-Net initializer walks every Linear module.
        self.shared_gate.weight._no_reinit = True
        self.shared_gate.bias._no_reinit = True

    @property
    def aux_loss(self):
        """Mean router load-balancing loss for this model forward.

        The deepest stack may execute the same MoE layer multiple times from
        recurrent inner loops and FBT passes.  Keep every call in the
        objective, while averaging over calls so enabling a loop does not
        silently multiply the auxiliary-loss coefficient.
        """
        if self._aux_loss is None:
            return None
        return self._aux_loss / max(1, self._aux_loss_count)

    def reset_aux_loss(self):
        """Clear per-forward routing statistics before a new model call."""
        self._aux_loss = None
        self._aux_loss_count = 0

    def _shared(self, x):
        y, gate = self.shared_fc1(x).chunk(2, dim=-1)
        y = F.silu(gate) * y
        y = self.shared_fc2(y)
        return y * torch.sigmoid(self.shared_gate(x).float()).to(y.dtype)

    def forward(self, x, active_mask=None):
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        if active_mask is None:
            active = torch.ones(flat.shape[0], device=flat.device, dtype=torch.bool)
        else:
            if active_mask.shape != x.shape[:-1]:
                raise ValueError(
                    f"active_mask shape {tuple(active_mask.shape)} does not match "
                    f"FFN input {tuple(x.shape[:-1])}"
                )
            active = active_mask.reshape(-1).to(dtype=torch.bool)
        logits = self.router(flat)
        probs = torch.softmax(logits.float(), dim=-1)
        _, expert_idx = probs.max(dim=-1)

        # Auxiliary loss follows the Mixtral/DeepSeek load-balancing form:
        # mean router probability times dispatched token fraction.  Keep this
        # in FP32 so BF16/FP8 router logits cannot poison the training loss.
        active_f = active.to(dtype=torch.float32)
        active_count = active_f.sum().clamp_min(1.0)
        active_one_hot = F.one_hot(expert_idx, self.num_experts).to(torch.float32)
        active_one_hot = active_one_hot * active_f[:, None]
        dispatch_fraction = active_one_hot.sum(dim=0) / active_count
        mean_probs = (probs * active_f[:, None]).sum(dim=0) / active_count
        current_aux_loss = self.aux_loss_weight * self.num_experts * (
            dispatch_fraction * mean_probs
        ).sum()
        if self._aux_loss is None:
            self._aux_loss = current_aux_loss
        else:
            self._aux_loss = self._aux_loss + current_aux_loss
        self._aux_loss_count += 1

        # Fused TE permutation and grouped dispatch. Capacity is bounded by
        # the configured fixed-capacity contract; overflow tokens are left on
        # the shared path. We call TE's underlying fixed-count op directly
        # rather than ``moe_permute_and_pad_with_probs``: that convenience
        # wrapper calls ``.sum().item()`` to infer its output size, which would
        # synchronize the host and make CUDA graph replay unsafe. Here the
        # output row count is the static Python value E * capacity.
        n_tokens = flat.shape[0]
        # Keep the capacity derived from the static graph bound.  The active
        # mask still removes padding from the cumsum below, so inactive rows
        # cannot consume slots or displace real tokens.  Computing a smaller
        # device-dependent capacity from ``active_count`` would change tensor
        # shapes and make CUDA graph capture unsafe; the bounded allocation is
        # the same fixed-capacity contract used by the TE grouped router.
        capacity_tokens = max(n_tokens, self.capacity_tokens)
        capacity = max(
            1, int(math.ceil(self.capacity_factor * capacity_tokens / self.num_experts))
        )
        # Keep the same fixed capacity in training and evaluation.  Using a
        # larger inference-only capacity changes which tokens overflow, the
        # amount of expert compute, and the decoded function being evaluated.
        # A single capacity contract also keeps the grouped GEMM shapes and
        # checkpointed model semantics identical across train/eval.
        # Count only active rows. Inactive static-chunk padding receives a
        # zero route and is omitted from the fused map.
        positions = torch.cumsum(active_one_hot.to(torch.int32), dim=0) - 1
        token_pos = positions.gather(1, expert_idx[:, None]).squeeze(1)
        valid = (token_pos < capacity) & active
        routing_map = F.one_hot(expert_idx, self.num_experts).to(torch.int32)
        routing_map = routing_map * valid[:, None].to(torch.int32)
        routing_probs = probs * routing_map.to(probs.dtype)
        # ``permute_mask_map_fwd`` first packs routed rows by their actual
        # expert counts.  GroupedLinear, however, receives fixed ``capacity``
        # rows per expert.  Shift each expert's packed segment into its own
        # fixed-capacity block; a zero offset would send the tail of a short
        # expert through the next expert's weights whenever the router loads
        # are uneven.  Counts are derived from the already capacity-trimmed
        # routing map, so this remains a fixed-shape, CUDA-graph-safe tensor
        # computation (the values may vary on each replay).
        counts = routing_map.sum(dim=0, dtype=torch.int64)
        prefix_counts = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=flat.device), torch.cumsum(counts, dim=0)[:-1]]
        )
        expert_starts = torch.arange(
            self.num_experts, dtype=torch.int64, device=flat.device
        ) * capacity
        pad_offsets = expert_starts - prefix_counts
        dispatched, row_id_map, permuted_probs = torch.ops.te_moe.permute_mask_map_fwd(
            flat,
            routing_map,
            self.num_experts * capacity,
            routing_probs,
            pad_offsets,
        )
        split_key = (flat.device.type, flat.device.index, capacity)
        m_splits = getattr(self, "_m_splits_cache", {}).get(split_key)
        if m_splits is None:
            if not hasattr(self, "_m_splits_cache"):
                self._m_splits_cache = {}
            m_splits = torch.full(
                (self.num_experts,), capacity, dtype=torch.int64, device=flat.device
            )
            # Delayed-scaling TE lacks a native grouped quantizer and normally
            # converts this tensor with ``tolist()``.  Preserve a Python tuple
            # alongside the fixed-capacity device tensor so graph capture can
            # use TE's split-quantize grouped GEMM without a host sync.
            m_splits._hnet_static_splits = (capacity,) * self.num_experts
            self._m_splits_cache[split_key] = m_splits
        elif getattr(m_splits, "_hnet_static_splits", None) is None:
            m_splits._hnet_static_splits = (capacity,) * self.num_experts
        expert_hidden = self.experts_fc1(dispatched, m_splits)
        value, gate = expert_hidden.chunk(2, dim=-1)
        expert_hidden = F.silu(gate) * value
        expert_out = self.experts_fc2(expert_hidden, m_splits)
        # TE returns a flattened top-1 probability vector. Apply it before
        # unpermute so the custom op receives no narrow [N,1] probability
        # tensor (which older TE builds cannot differentiate correctly).
        expert_out = expert_out * permuted_probs.reshape(-1, 1).to(expert_out.dtype)
        routed = te.moe_unpermute(
            expert_out,
            row_id_map,
            restore_shape=flat.shape,
            pad_offsets=pad_offsets,
        )
        out = routed + self._shared(flat)
        return out.view(shape)
