from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except (ImportError, ModuleNotFoundError):
    # Configuration and canonical GDN2 construction remain inspectable without
    # importing the optional scan extension. Dechunking still requires the
    # inspected scan kernel and raises explicitly when invoked without it.
    mamba_chunk_scan_combined = None

from hnet.modules.utils import get_seq_idx


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


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
        # Query and key use the same input stream and have identical matrix
        # shapes.  Keep them in one output-wide GEMM: projecting the complete
        # stream once and slicing the two halves removes one launch from every
        # encoder/router stage while doing the same arithmetic as two Linear
        # modules.  The split layout is [q | k], matching the old parameter
        # ordering so checkpoints can be migrated below.
        self.qk_proj_layer = nn.Linear(d_model, 2 * d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.qk_proj_layer.weight.copy_(torch.cat((torch.eye(d_model), torch.eye(d_model)), dim=0))
        self.qk_proj_layer.weight._no_reinit = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Accept checkpoints written before the fused router projection.

        The old router stored ``q_proj_layer.weight`` and
        ``k_proj_layer.weight`` separately.  Concatenating those matrices into
        the new [q | k] projection keeps existing training checkpoints usable
        without retaining duplicate parameters in the live model.
        """
        fused_key = prefix + "qk_proj_layer.weight"
        q_key = prefix + "q_proj_layer.weight"
        k_key = prefix + "k_proj_layer.weight"
        if fused_key not in state_dict and q_key in state_dict and k_key in state_dict:
            state_dict[fused_key] = torch.cat((state_dict[q_key], state_dict[k_key]), dim=0)
            del state_dict[q_key]
            del state_dict[k_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
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
        qk = self.qk_proj_layer(hidden_states)
        q = F.normalize(qk[:, :-1, : self.d_model].float(), dim=-1)
        k = F.normalize(qk[:, 1:, self.d_model :].float(), dim=-1)
        # Elementwise reduction maps to a single fused pointwise/reduction
        # path on CUDA; einsum builds a more general contraction for this
        # diagonal dot product and is measurably slower at byte-model widths.
        cos_sim = (q * k).sum(dim=-1)
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            starts = torch.zeros_like(boundary_prob, dtype=torch.bool)
            starts[cu_seqlens[:-1]] = True
            boundary_prob = torch.where(
                starts,
                torch.full_like(boundary_prob, PAD_PROB),
                boundary_prob,
            )

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
        qk = self.qk_proj_layer(
            torch.cat((inference_params.last_hidden_state, hidden_states), dim=0)
        )
        q = F.normalize(qk[: qk.shape[0] // 2, : self.d_model].float(), dim=-1)
        k = F.normalize(qk[qk.shape[0] // 2 :, self.d_model :].float(), dim=-1)
        cos_sim = (q * k).sum(dim=-1)
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

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"
        self.effective_boundary_mask = boundary_mask

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            # The canonical fixed-shape graph uses a bounded latent stream.
            # Keeping the earliest boundaries preserves causal order while
            # preventing a pathological router batch from turning the inner
            # full-attention block back into an 8K quadratic pass.
            effective_boundary_mask = boundary_mask
            if self.static_max_seqlen is not None and self.static_max_seqlen < hidden_states.shape[1]:
                boundary_rank = boundary_mask.to(torch.int32).cumsum(dim=1) - 1
                effective_boundary_mask = boundary_mask & (
                    boundary_rank < int(self.static_max_seqlen)
                )
            self.effective_boundary_mask = effective_boundary_mask
            num_tokens = effective_boundary_mask.sum(dim=-1)
            # CUDA graph capture cannot depend on a host read of a
            # data-dependent max. A configured static bound keeps all inner
            # tensors at one address/shape; the mask carries the true count.
            device = hidden_states.device
            L = hidden_states.shape[1]
            if self.static_max_seqlen is not None:
                next_max_seqlen = int(self.static_max_seqlen)
                if next_max_seqlen <= 0:
                    raise ValueError(
                        "static_max_seqlen must be positive"
                    )
            else:
                next_max_seqlen = int(num_tokens.max())
            # A static bound can exceed the current input length (for example,
            # a short validation batch captured with the 4096-token production
            # bound). Pad the gather indices so the inner network and its mask
            # have the same fixed shape. Invalid rows are masked below and do
            # not contribute to the dechunked output.
            if hidden_states.shape[0] <= 2:
                # Boundary rows are already in source order. Their destination
                # row is the exclusive prefix rank of the boolean mask;
                # scattering avoids the per-row O(L log L) argsort. The
                # sentinel column receives invalid rows and is discarded.
                boundary_rank = effective_boundary_mask.to(torch.int32).cumsum(dim=1) - 1
                sentinel = next_max_seqlen
                scatter_rank = torch.where(
                    effective_boundary_mask,
                    boundary_rank,
                    torch.full_like(boundary_rank, sentinel),
                )
                packed = hidden_states.new_zeros(
                    (hidden_states.shape[0], next_max_seqlen + 1, hidden_states.shape[-1])
                )
                packed.scatter_(
                    1,
                    scatter_rank[..., None].expand(-1, -1, hidden_states.shape[-1]),
                    hidden_states,
                )
                next_hidden_states = packed[:, :next_max_seqlen]
            else:
                # For larger batches the full-width scatter writes can cost
                # more than the radix sort; retain the gather path there.
                token_idx = (
                    torch.arange(L, device=device)[None]
                    + (~effective_boundary_mask).long() * L
                )
                seq_sorted_indices = torch.argsort(token_idx, dim=1)
                if next_max_seqlen > L:
                    pad = torch.zeros(
                        (hidden_states.shape[0], next_max_seqlen - L),
                        device=device,
                        dtype=seq_sorted_indices.dtype,
                    )
                    seq_sorted_indices = torch.cat((seq_sorted_indices, pad), dim=1)
                next_hidden_states = torch.gather(
                    hidden_states,
                    dim=1,
                    index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                        -1, -1, hidden_states.shape[-1]
                    ),
                )

            next_mask = torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        self.effective_boundary_mask = boundary_mask
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
        static_max_seqlen: int | None = None,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.static_max_seqlen = static_max_seqlen

        # Parameters for the inspected causal scan kernel used by dechunking.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim
        # The decay is fixed for every call. Keep one device-resident copy so
        # dechunking does not allocate and initialize it on every forward or
        # CUDA-graph warmup.
        self.register_buffer(
            "_A",
            -torch.ones(self.nheads, device=device, dtype=torch.float32),
            persistent=True,
        )

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
        if mamba_chunk_scan_combined is None:
            raise RuntimeError(
                "DeChunkLayer requires the pinned dechunk scan extension; "
                "the canonical model does not provide a Mamba fallback"
            )
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
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
            # ChunkLayer's packed path returns (M, D), while the fused scan
            # contract is batched (B, M, D). Keep the document dimension for
            # the scan and remove it only after the recurrent aggregation.
            hidden_states = hidden_states.unsqueeze(0)
        else:
            B, L = boundary_mask.shape
            seq_idx = None
            num_tokens = boundary_mask.sum(dim=-1)

            # Match ChunkLayer's prefix-rank packing without another sort for
            # the production batch sizes. Larger batches retain the gather
            # path because it writes fewer full-width rows than scatter.
            latent_len = hidden_states.shape[1]
            if B <= 2:
                boundary_rank = boundary_mask.to(torch.int32).cumsum(dim=1) - 1
                sentinel = latent_len
                scatter_rank = torch.where(
                    boundary_mask,
                    boundary_rank,
                    torch.full_like(boundary_rank, sentinel),
                )
                packed = p.new_zeros((B, latent_len + 1))
                packed.scatter_(1, scatter_rank, p)
                p = packed[:, :latent_len]
                # Keep static padding at the same minimum decay used below;
                # an exact zero would make dt zero and create inf/nan scan
                # intermediates even though those rows are masked later.
                pad_pos = torch.arange(latent_len, device=p.device)[None, :]
                p = torch.where(
                    pad_pos < num_tokens[:, None],
                    p,
                    torch.full_like(p, 1e-4),
                )
            else:
                token_idx = (
                    torch.arange(L, device=hidden_states.device)[None]
                    + (~boundary_mask).long() * L
                )
                seq_sorted_indices = torch.argsort(token_idx, dim=1)
                if latent_len > L:
                    pad = torch.zeros(
                        (B, latent_len - L),
                        device=hidden_states.device,
                        dtype=seq_sorted_indices.dtype,
                    )
                    seq_sorted_indices = torch.cat((seq_sorted_indices, pad), dim=1)
                p = torch.gather(
                    p, dim=1, index=seq_sorted_indices[:, :latent_len]
                )  # (B, M)

        original_dtype = hidden_states.dtype
        # Use the inspected causal scan kernel for the EMA deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = self._A
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
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
