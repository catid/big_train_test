from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from hnet.modules.utils import get_seq_idx
from hnet.te import make_linear
from torch.utils._pytree import register_pytree_node


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


def _flatten_routing_output(value):
    return (value.boundary_prob, value.boundary_mask, value.selected_probs), None


def _unflatten_routing_output(values, _context):
    return RoutingModuleOutput(*values)


register_pytree_node(
    RoutingModuleOutput,
    _flatten_routing_output,
    _unflatten_routing_output,
)


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)
    # Number of effective latent boundaries retained by the outer chunker.
    # This is separate from ``has_seen_tokens`` because the static training
    # path applies a fixed latent capacity and decode must enforce the same
    # bound online.
    boundary_count: torch.Tensor | None = None


@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = make_linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = make_linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
            boundary_count=torch.zeros(batch_size, device=device, dtype=torch.int32),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert (
                ~inference_params.has_seen_tokens
            ).all(), "Cannot have seen tokens when inference_params is not provided"

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        # Router probabilities feed hard selection and the EMA dechunker. Do
        # the normalization in FP32 so BF16 roundoff cannot create NaNs or
        # probabilities outside [0, 1] in long runs.
        # Cartesia's causal router scores the current state against the
        # immediately preceding state: p_t = 1/2(1-cos(W_q x_t,
        # W_k x_{t-1})). Keep the first position forced to a boundary.
        q = F.normalize(self.q_proj_layer(hidden_states[:, 1:]).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states[:, :-1]).float(), dim=-1)
        cos_sim = torch.einsum("b l d, b l d -> b l", q, k)
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(
                            hidden_states.shape[0], device=hidden_states.device
                        ),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = hidden_states.squeeze(1)
        q = F.normalize(self.q_proj_layer(hidden_states).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(inference_params.last_hidden_state).float(), dim=-1)
        cos_sim = torch.einsum("b d, b d -> b", q, k)
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
        )


class ChunkLayer(nn.Module):

    def __init__(self, static_max_seqlen: int | None = None):
        super().__init__()
        self.static_max_seqlen = static_max_seqlen

    def forward(
        self,
        hidden_states,
        boundary_mask,
        cu_seqlens=None,
        mask=None,
        boundary_prob=None,
    ):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if boundary_prob is None:
            boundary_prob = boundary_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            boundary_prob = torch.cat((1 - boundary_prob, boundary_prob), dim=-1)

        if cu_seqlens is not None:
            # Packed callers use exact per-document cumulative lengths.  Apply
            # the same latent capacity as the padded graph path before
            # gathering rows; otherwise a packed prefill could silently feed
            # more than the trained 512 latent rows into the inner stack.
            effective_boundary_mask = boundary_mask
            if self.static_max_seqlen is not None:
                capacity = int(self.static_max_seqlen)
                # Retain the first boundaries in each document. Selecting by
                # score over the complete prompt would let a future byte evict
                # an earlier latent, which violates causality and disagrees
                # with the online decoder's boundary counter. Build the
                # per-document rank entirely on-device so packed callers keep
                # the same tensor-only contract as graph training.
                flat_mask = boundary_mask.to(dtype=torch.int32)
                global_rank = torch.cumsum(flat_mask, dim=0)
                starts = cu_seqlens[:-1].to(
                    device=boundary_mask.device, dtype=torch.long
                )
                offsets = global_rank.index_select(0, starts) - flat_mask.index_select(
                    0, starts
                )
                doc_idx = get_seq_idx(
                    cu_seqlens,
                    device=boundary_mask.device,
                    total_length=boundary_mask.shape[0],
                )[0].to(dtype=torch.long)
                per_doc_rank = global_rank - offsets.index_select(0, doc_idx)
                effective_boundary_mask = boundary_mask & (per_doc_rank <= capacity)

            next_hidden_states = hidden_states[effective_boundary_mask]
            next_cu_seqlens = F.pad(
                effective_boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            if self.static_max_seqlen is not None:
                # A configured upper bound is valid for FlashAttention's
                # varlen API and avoids converting a data-dependent CUDA max
                # to a Python integer.  The cumulative lengths still carry
                # the exact token counts for every document.
                next_max_seqlen = int(self.static_max_seqlen)
            else:
                next_max_seqlen = int(
                    (next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max()
                )
            next_mask = None
        else:
            next_cu_seqlens = None
            L = hidden_states.shape[1]
            effective_boundary_mask = boundary_mask
            num_tokens = boundary_mask.sum(dim=-1)
            # CUDA graph capture cannot depend on a host read of a
            # data-dependent max. A configured static bound keeps all inner
            # tensors at one address/shape; the mask carries the true count.
            if self.static_max_seqlen is not None:
                # Evaluation and prompt prefill may use a shorter sequence
                # than the training graph bound.  Do not manufacture a mask
                # longer than the gathered tensor; fixed-shape training still
                # uses the configured bound because L is constant there.
                next_max_seqlen = min(int(self.static_max_seqlen), L)
            else:
                next_max_seqlen = int(num_tokens.max())
            # The causal rank rule below is tensor-only and replay-safe under
            # CUDA graphs while keeping the latent tensor statically shaped.
            capacity = min(int(next_max_seqlen), int(L))
            if self.static_max_seqlen is not None and capacity < L:
                boundary_rank = torch.cumsum(
                    boundary_mask.to(dtype=torch.int32), dim=1
                )
                effective_boundary_mask = boundary_mask & (boundary_rank <= capacity)
                num_tokens = effective_boundary_mask.sum(dim=-1)
            if self.static_max_seqlen is None and int(num_tokens.max()) > next_max_seqlen:
                raise ValueError(
                    "static chunking bound is smaller than the number of input "
                    "boundaries; increase max_seq_len"
                )

            device = hidden_states.device
            token_idx = (
                torch.arange(L, device=device)[None, :]
                + (~effective_boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
            next_max_seqlen = None

        return (
            next_hidden_states,
            next_cu_seqlens,
            next_max_seqlen,
            next_mask,
            effective_boundary_mask,
        )

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        # Just for Mamba2 kernel.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(
                cu_seqlens,
                device=hidden_states.device,
                total_length=hidden_states.shape[0],
            )
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        # B_selected = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        # Routing probabilities are computed in FP32 for numerical stability,
        # while the recurrent/dechunk state follows the model activation dtype
        # (BF16 in production).  Cast before indexed assignment; otherwise
        # generation fails with a dtype mismatch after a valid prefill.
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        ).to(dtype=p.dtype)

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
