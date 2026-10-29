# Base code imported from
# https://github.com/state-spaces/mamba
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.ops.activations import swiglu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_intermediate=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        d_intermediate = (
            d_intermediate if d_intermediate is not None else int(8 * d_model / 3)
        )
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x, active_mask=None):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y)
        y = self.fc2(y)
        return y


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
        self.aux_loss_weight = float(aux_loss_weight)

        # [E, 2I, D] and [E, D, I] are consumed as one grouped GEMM via
        # einsum.  The layout mirrors TE GroupedLinear's stacked experts.
        self.experts_fc1 = nn.Parameter(
            torch.empty(self.num_experts, 2 * expert_intermediate, d_model, **factory_kwargs)
        )
        self.experts_fc2 = nn.Parameter(
            torch.empty(self.num_experts, d_model, expert_intermediate, **factory_kwargs)
        )
        self.shared_fc1 = nn.Linear(d_model, 2 * shared_intermediate, bias=bias, **factory_kwargs)
        self.shared_fc2 = nn.Linear(shared_intermediate, d_model, bias=bias, **factory_kwargs)
        self.router = nn.Linear(d_model, self.num_experts, bias=False, **factory_kwargs)
        self.shared_gate = nn.Linear(d_model, 1, bias=True, **factory_kwargs)
        self._aux_loss = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.experts_fc1, mean=0.0, std=0.02)
        nn.init.normal_(self.experts_fc2, mean=0.0, std=0.02 / math.sqrt(2.0))
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_fc2.weight, mean=0.0, std=0.02 / math.sqrt(2.0))
        if self.shared_fc1.bias is not None:
            nn.init.zeros_(self.shared_fc1.bias)
            nn.init.zeros_(self.shared_fc2.bias)
        nn.init.zeros_(self.shared_gate.weight)
        nn.init.zeros_(self.shared_gate.bias)

    @property
    def aux_loss(self):
        """Router load-balancing loss from the most recent forward."""
        return self._aux_loss

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
        routing_prob, expert_idx = probs.max(dim=-1)

        # Auxiliary loss follows the Mixtral/DeepSeek load-balancing form:
        # mean router probability times dispatched token fraction.  Keep this
        # in FP32 so BF16/FP8 router logits cannot poison the training loss.
        active_f = active.to(dtype=torch.float32)
        active_count = active_f.sum().clamp_min(1.0)
        dispatch_fraction = (
            F.one_hot(expert_idx, self.num_experts).float() * active_f[:, None]
        ).sum(dim=0) / active_count
        mean_probs = (probs * active_f[:, None]).sum(dim=0) / active_count
        self._aux_loss = self.aux_loss_weight * self.num_experts * (
            dispatch_fraction * mean_probs
        ).sum()

        # Fixed per-expert capacity and grouped dispatch.  Capacity is derived
        # from the static token count of the graph; overflow tokens are left on
        # the shared path, preserving a finite output with no host sync.
        n_tokens = flat.shape[0]
        capacity = max(1, int(math.ceil(self.capacity_factor * n_tokens / self.num_experts)))
        # Generation batches are tiny and may legitimately route several
        # sequences to one expert. Avoid dropping tokens at inference; the
        # training graph keeps the fixed capacity above for bounded memory.
        if not self.training:
            # Generation batches are tiny; use the static token bound instead
            # of a device-to-host bincount/max read. This keeps decode capture
            # free of synchronization and guarantees no token is dropped.
            capacity = max(1, n_tokens)
        positions = torch.cumsum(
            F.one_hot(expert_idx, self.num_experts).to(torch.int32), dim=0
        ) - 1
        token_pos = positions.gather(1, expert_idx[:, None]).squeeze(1)
        valid = (token_pos < capacity) & active
        slot = expert_idx * capacity + token_pos.clamp_max(capacity - 1)
        # Include one trash row so every token follows the same-shape
        # index_copy_ path. Boolean compaction (`flat[valid]`) performs a
        # data-dependent nonzero/host synchronization and breaks CUDA graph
        # capture. Overflow rows are written to trash and discarded below.
        trash = self.num_experts * capacity
        slot_safe = torch.where(valid, slot, torch.full_like(slot, trash))
        dispatched = flat.new_zeros((trash + 1, self.d_model))
        dispatched.index_copy_(0, slot_safe, flat)
        dispatched = dispatched[:trash].view(self.num_experts, capacity, self.d_model)

        expert_hidden = torch.einsum("ecd,eod->eco", dispatched, self.experts_fc1)
        value, gate = expert_hidden.chunk(2, dim=-1)
        expert_hidden = F.silu(gate) * value
        expert_out = torch.einsum("eci,edi->ecd", expert_hidden, self.experts_fc2)
        gathered = expert_out.reshape(-1, self.d_model).index_select(0, slot.clamp_max(trash - 1))
        gathered = gathered * routing_prob[:, None].to(gathered.dtype)
        routed = torch.where(valid[:, None], gathered, torch.zeros_like(gathered))
        out = routed + self._shared(flat)
        return out.view(shape)
