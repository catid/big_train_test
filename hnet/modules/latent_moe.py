# Base code imported from
# https://github.com/state-spaces/mamba
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hnet.te import TEGroupedLinear, make_linear, te, graph_capture_enabled
from transformer_engine.pytorch.router import fused_compute_score_for_moe_aux_loss


class FusedTokenRouter(nn.Module):
    """Own the complete TE top-1 routing and dispatch boundary.

    TE computes the FP32 softmax scores and the discrete routing map in one
    CUDA operation.  Capacity remains in :class:`LatentMoE` because the
    model applies causal per-sequence capacity, whereas the generic TE router
    only sees the flattened token stream.  The resulting fixed-capacity map is
    dispatched and restored through this object as well, so callers cannot
    silently replace the fused route with a Python or graph-unsafe permutation.
    """

    def __init__(self, num_experts: int, top_k: int = 1):
        super().__init__()
        if int(top_k) != 1:
            raise ValueError("FusedTokenRouter currently requires top_k=1")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.num_shared_experts = 1

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not logits.is_cuda:
            raise RuntimeError("FusedTokenRouter requires CUDA logits")
        if logits.ndim != 2:
            raise ValueError(
                "FusedTokenRouter expects flattened [tokens, experts] logits; "
                f"got shape {tuple(logits.shape)}"
            )
        if logits.shape[-1] != self.num_experts:
            raise ValueError(
                "FusedTokenRouter logits have the wrong expert dimension: "
                f"expected {self.num_experts}, got {logits.shape[-1]}"
            )
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"unsupported fused-router logits dtype: {logits.dtype}")
        # TE's fused score kernel requires a contiguous [tokens, experts]
        # matrix.  Aligned TE projections can return a strided view after
        # packed/recurrent reshaping, so make this boundary explicit.
        logits = logits.contiguous()
        routing_map, probs = fused_compute_score_for_moe_aux_loss(
            logits,
            topk=self.top_k,
            score_function="softmax",
            # Keep the dense boolean map required by the graph-safe TE
            # permutation below.  Do not inherit a future TE default (for
            # example BITMAP_U8), which would silently change the map shape.
            routing_map_format="bytemap",
        )
        if routing_map.shape != logits.shape or routing_map.shape[-1] != self.num_experts:
            raise RuntimeError(
                "Transformer Engine fused router returned an unexpected routing map shape: "
                f"map={tuple(routing_map.shape)} logits={tuple(logits.shape)}"
            )
        if probs.shape != logits.shape or probs.dtype != torch.float32:
            raise RuntimeError(
                "Transformer Engine fused router returned unexpected score output: "
                f"scores={tuple(probs.shape)} dtype={probs.dtype} logits={tuple(logits.shape)}"
            )
        # The pinned TE kernel currently returns bool for BYTEMAP.  Normalize
        # the contract here so a compatible TE rebuild cannot leak uint8 into
        # the int32 map consumed by ``permute_mask_map_fwd``.
        return probs.contiguous(), routing_map.to(dtype=torch.bool).contiguous()

    def permute(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        routing_probs: torch.Tensor | None = None,
        pad_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch routed rows with TE's fixed-count graph-safe primitive.

        ``num_out_tokens`` is a static Python value (experts times capacity),
        never inferred from a device reduction.  This is the only supported
        dispatch path for this router.
        """
        if not tokens.is_cuda:
            raise RuntimeError("FusedTokenRouter dispatch requires CUDA tokens")
        if tokens.ndim != 2:
            raise ValueError(f"dispatch tokens must be [tokens, hidden], got {tuple(tokens.shape)}")
        if routing_map.shape != (tokens.shape[0], self.num_experts):
            raise ValueError(
                "dispatch routing_map must be [tokens, experts], got "
                f"{tuple(routing_map.shape)} for tokens={tokens.shape[0]} experts={self.num_experts}"
            )
        if routing_map.dtype != torch.int32:
            raise TypeError(f"dispatch routing_map must be int32, got {routing_map.dtype}")
        if not isinstance(num_out_tokens, int) or num_out_tokens <= 0:
            raise ValueError(f"dispatch num_out_tokens must be a positive int, got {num_out_tokens!r}")
        if routing_probs is not None:
            if routing_probs.shape != routing_map.shape or routing_probs.dtype != torch.float32:
                raise ValueError(
                    "dispatch routing_probs must be float32 with routing_map shape, got "
                    f"{tuple(routing_probs.shape)} {routing_probs.dtype}"
                )
        if pad_offsets is not None:
            if pad_offsets.shape != (self.num_experts,) or pad_offsets.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "dispatch pad_offsets must be int32/int64 [experts], got "
                    f"{tuple(pad_offsets.shape)} {pad_offsets.dtype}"
                )
        return torch.ops.te_moe.permute_mask_map_fwd(
            tokens, routing_map, num_out_tokens, routing_probs, pad_offsets
        )

    def unpermute(
        self,
        expert_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        restore_shape: torch.Size,
        pad_offsets: torch.Tensor | None = None,
        merging_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore dispatched rows through TE's fused unpermutation primitive."""
        if not expert_tokens.is_cuda:
            raise RuntimeError("FusedTokenRouter unpermute requires CUDA expert tokens")
        return te.moe_unpermute(
            expert_tokens,
            row_id_map,
            merging_probs=merging_probs,
            restore_shape=restore_shape,
            pad_offsets=pad_offsets,
        )


class LatentMoE(nn.Module):
    """Static-shape LatentMoE with eight routed experts and one shared expert.

    Following arXiv:2601.18089, the router reads the full residual width, then
    a shared down projection maps routed tokens to ``d_model / 4``. The routed
    expert width is increased to 1664 so the complete layer stays parameter matched;
    eight experts therefore carry more total neurons than the dense FFN.
    routed experts and their grouped GEMMs operate only in that latent width;
    an up projection restores the residual width before adding the full-width
    shared expert. Dispatch uses a fixed capacity tensor and TE grouped GEMM,
    so it remains CUDA-graph friendly and avoids Python permutation paths.

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
            raise ValueError("LatentMoE requires moe_top_k=1 for 1:8 sparsity")
        if num_experts != 8:
            raise ValueError("LatentMoE requires exactly eight routed experts")
        if capacity_factor < 1.0:
            raise ValueError("moe_capacity_factor must be >= 1.0")
        # Canonical widths are explicit. The d/4 latent matrices reduce each
        # routed FFN matrix, so 1664-wide experts preserve the prior layer
        # parameter budget while increasing total expert neurons.
        if int(expert_intermediate) <= 0 or int(shared_intermediate) <= 0:
            raise ValueError("LatentMoE requires explicit routed/shared intermediate widths")
        expert_intermediate = int(expert_intermediate)
        shared_intermediate = int(shared_intermediate)
        expert_intermediate = (expert_intermediate + multiple_of - 1) // multiple_of * multiple_of
        shared_intermediate = (shared_intermediate + multiple_of - 1) // multiple_of * multiple_of
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = int(d_model)
        self.latent_alpha = 4
        if self.d_model % self.latent_alpha:
            raise ValueError("LatentMoE d_model must be divisible by latent_alpha=4")
        self.latent_dim = self.d_model // self.latent_alpha
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_intermediate = expert_intermediate
        self.shared_intermediate = shared_intermediate
        self.capacity_factor = float(capacity_factor)
        self.capacity_tokens = int(capacity_tokens)
        if self.capacity_tokens < 0:
            raise ValueError("capacity_tokens must be non-negative")
        self.aux_loss_weight = float(aux_loss_weight)

        # W_down/W_up are shared by all experts. Their FP8 GEMMs are kept in
        # the same TE autocast region as the grouped expert projections.
        self.down = make_linear(
            d_model, self.latent_dim, bias=bias, **factory_kwargs
        )
        self.up = make_linear(
            self.latent_dim, d_model, bias=bias, **factory_kwargs
        )
        # [E, 2I, latent] and [E, latent, I] are consumed as one grouped GEMM via
        # einsum.  The layout mirrors TE GroupedLinear's stacked experts.
        self.experts_fc1 = TEGroupedLinear(
            self.num_experts, self.latent_dim, 2 * expert_intermediate, bias=bias,
            params_dtype=dtype, device=device, use_grouped_tensor=True,
        )
        self.experts_fc2 = TEGroupedLinear(
            self.num_experts, expert_intermediate, self.latent_dim, bias=bias,
            params_dtype=dtype, device=device, use_grouped_tensor=True,
        )
        self.shared_fc1 = make_linear(d_model, 2 * shared_intermediate, bias=bias, **factory_kwargs)
        self.shared_fc2 = make_linear(shared_intermediate, d_model, bias=bias, **factory_kwargs)
        # Router decisions are discrete and feed the auxiliary loss. Keep the
        # projection in BF16/FP32 even inside an enclosing FP8 expert region so
        # quantization noise cannot flip near-tied top-1 assignments.
        self.router = make_linear(
            d_model, self.num_experts, bias=False, fp8=False,
            device=device, dtype=torch.float32,
        )
        self.fused_router = FusedTokenRouter(self.num_experts, self.top_k)
        self.shared_gate = make_linear(
            d_model, 1, bias=True, fp8=False, **factory_kwargs
        )
        self._aux_loss = None
        self._aux_loss_count = 0
        self._z_loss = None
        self._shared_stream = None
        # ``expert_starts`` depends only on the static grouped-GEMM capacity.
        # Cache it per device/capacity so every MoE invocation does not launch
        # a tiny arange/multiply pair (there are 20 routed blocks and two FBT
        # passes in the canonical model).
        self._expert_starts_cache = {}
        self._sequence_ids_cache = {}
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
        self._z_loss = None

    @property
    def z_loss(self):
        if self._z_loss is None:
            return self.router.weight.new_zeros((), dtype=torch.float32)
        # Match aux_loss: recurrent loops and FBT invoke the same router
        # several times per model forward.  Average all calls so the z-loss
        # coefficient is independent of loop/pass count.
        return self._z_loss / max(1, self._aux_loss_count)

    def _shared(self, x):
        y, gate = self.shared_fc1(x).chunk(2, dim=-1)
        y = F.silu(gate) * y
        y = self.shared_fc2(y)
        return y * torch.sigmoid(self.shared_gate(x).float()).to(y.dtype)

    def forward(
        self,
        x,
        active_mask=None,
        sequence_ids=None,
        route_counts=None,
        loop_idx=0,
        layer_idx=0,
        decode=False,
    ):
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        shared_result = None
        shared_event = None
        shared_stream = None
        if flat.is_cuda and not graph_capture_enabled():
            current_stream = torch.cuda.current_stream(flat.device)
            if self._shared_stream is None or self._shared_stream.device != flat.device:
                self._shared_stream = torch.cuda.Stream(device=flat.device)
            shared_stream = self._shared_stream
            # Ensure the side stream sees all writes that produced ``flat``.
            shared_stream.wait_stream(current_stream)
            with torch.cuda.stream(shared_stream):
                shared_result = self._shared(flat)
                shared_event = torch.cuda.Event()
                shared_event.record(shared_stream)
            # ``flat`` and the side-stream result participate in autograd on
            # the current stream after the event below.
            flat.record_stream(shared_stream)
        else:
            shared_result = self._shared(flat)
        if active_mask is None:
            active = torch.ones(flat.shape[0], device=flat.device, dtype=torch.bool)
        else:
            if active_mask.shape != x.shape[:-1]:
                raise ValueError(
                    f"active_mask shape {tuple(active_mask.shape)} does not match "
                    f"FFN input {tuple(x.shape[:-1])}"
                )
            active = active_mask.reshape(-1).to(dtype=torch.bool)
        if decode and route_counts is None:
            raise ValueError("MoE decode requires persistent route counters")
        if decode and sequence_ids is None:
            raise ValueError("MoE decode requires one source sequence id per token")
        if route_counts is not None:
            if route_counts.ndim != 4 or route_counts.shape[-1] != self.num_experts:
                raise ValueError(
                    "MoE route counters must have shape "
                    f"(loops, layers, batch, {self.num_experts})"
                )
            if not (0 <= int(loop_idx) < route_counts.shape[0]):
                raise IndexError(f"MoE loop index {loop_idx} is outside route counter state")
            if not (0 <= int(layer_idx) < route_counts.shape[1]):
                raise IndexError(f"MoE layer index {layer_idx} is outside route counter state")
        # Keep routing logits and all gate math in FP32.  This is also the
        # CroissAttn-style safeguard against using a rounded FP8 intermediate
        # for an auxiliary derivative or top-k decision.
        logits = self.router(flat.float())
        # TE fuses FP32 score computation and top-1 selection. Capacity is
        # applied below because it is causal per sequence, outside the generic
        # flattened-token router's contract.
        probs, fused_routing_map = self.fused_router(logits)
        # Keep the router z regularizer in FP32, alongside the load-balancing
        # term. This prevents large logits from destabilizing FP8 expert GEMMs.
        current_z_loss = 1e-3 * torch.logsumexp(logits, dim=-1).square().mean()
        if self._z_loss is None:
            self._z_loss = current_z_loss
        else:
            self._z_loss = self._z_loss + current_z_loss
        # Convert the dense boolean route map once.  Reusing this int32 map
        # for expert selection and the capacity/auxiliary paths avoids a
        # second bool->int64 conversion on every routed invocation.
        routing_map = fused_routing_map.to(dtype=torch.int32)
        expert_idx = routing_map.argmax(dim=-1)

        # Auxiliary loss follows the Mixtral/DeepSeek load-balancing form:
        # mean router probability times dispatched token fraction.  Keep this
        # in FP32 so BF16/FP8 router logits cannot poison the training loss.
        active_f = active.to(dtype=torch.float32)
        active_count = active_f.sum().clamp_min(1.0)
        # Materialize one int32 route map and reuse it for both capacity
        # accounting and dispatch.  The previous path cast the bool map to
        # FP32, multiplied by the active mask, then cast it back to int32;
        # that pair of conversion kernels is on the hot path for every MoE
        # invocation.  Integer masking is exact for the binary route and the
        # single FP32 cast below remains sufficient for the auxiliary loss.
        routing_map = routing_map * active[:, None].to(torch.int32)
        active_one_hot = routing_map.to(torch.float32)
        # ``f_e`` is a fraction of routing *slots*, not token rows.  Keep
        # this normalization explicit so the Switch/GShard term remains
        # invariant if a future configuration allows top-k > 1 (for the
        # canonical 1:8 route ``top_k == 1`` and this is a no-op).
        dispatch_fraction = active_one_hot.sum(dim=0) / (active_count * self.top_k)
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
        # Capacity is per sequence, rather than one cumsum over a flattened
        # microbatch.  A global cumsum makes row 0 consume all expert slots
        # before row 1 is considered and changes the function with batch size.
        # The canonical latent bound is static, so each sequence receives the
        # same fixed number of slots and grouped GEMM keeps one static block per
        # (expert, sequence) while remaining CUDA-graph safe.
        if x.dim() >= 3 and not decode:
            num_sequences = x.shape[0]
            tokens_per_sequence = x.shape[1]
            sequence_key = (
                flat.device.type,
                flat.device.index,
                int(num_sequences),
                int(tokens_per_sequence),
            )
            sequence_ids_flat = self._sequence_ids_cache.get(sequence_key)
            if sequence_ids_flat is None:
                sequence_ids_flat = torch.arange(
                    num_sequences, device=flat.device, dtype=torch.long
                ).repeat_interleave(tokens_per_sequence)
                self._sequence_ids_cache[sequence_key] = sequence_ids_flat
        elif sequence_ids is None:
            num_sequences = 1
            tokens_per_sequence = n_tokens
            sequence_ids_flat = torch.zeros(
                n_tokens, device=flat.device, dtype=torch.long
            )
        else:
            sequence_ids_flat = sequence_ids.reshape(-1).to(
                device=flat.device, dtype=torch.long
            )
            if sequence_ids_flat.numel() != n_tokens:
                raise ValueError(
                    "MoE sequence_ids must contain one id per flattened token"
                )
            # Packed sequence ids are contiguous and sorted by construction.
            # This host read is restricted to eager varlen paths; graph replay
            # always uses the rank-3 padded path above.
            if decode:
                # Decode receives only rows that emitted a boundary. Keep the
                # grouped-GEMM capacity based on the full cached batch so its
                # shape is invariant when a different subset emits a boundary
                # on the next token.
                num_sequences = int(route_counts.shape[-2])
            else:
                num_sequences = int(sequence_ids_flat[-1].item()) + 1 if n_tokens else 1
            tokens_per_sequence = None
        capacity_tokens = max(1, self.capacity_tokens)
        row_capacity = max(
            1, int(math.ceil(self.capacity_factor * capacity_tokens / self.num_experts))
        )
        capacity = row_capacity * num_sequences
        # Keep the same fixed capacity in training and evaluation.  Using a
        # larger inference-only capacity changes which tokens overflow, the
        # amount of expert compute, and the decoded function being evaluated.
        # A single capacity contract also keeps the grouped GEMM shapes and
        # checkpointed model semantics identical across train/eval.
        # Count only active rows. Inactive static-chunk padding receives a
        # zero route and is omitted from the fused map.
        if decode:
            # Decode starts at the count accumulated by prefill and previous
            # decode calls. The selected rows map back to original cache rows
            # through ``sequence_ids_flat``.
            route_state = route_counts[loop_idx, layer_idx]
            if sequence_ids_flat.numel():
                # IDs are produced by HNet's boundary mask and are therefore
                # within the allocated cache batch. Avoid data-dependent host
                # checks here so CUDA graph replay remains capture-safe.
                prior_positions = route_state.index_select(0, sequence_ids_flat)
            else:
                prior_positions = route_state.new_empty((0, self.num_experts))
            token_pos = prior_positions.gather(1, expert_idx[:, None]).squeeze(1)
            valid = (token_pos < row_capacity) & active
        elif tokens_per_sequence is not None:
            positions = torch.cumsum(
                active_one_hot.reshape(num_sequences, tokens_per_sequence, self.num_experts),
                dim=1,
            ).reshape(-1, self.num_experts) - 1
            token_pos = positions.gather(1, expert_idx[:, None]).squeeze(1)
            valid = (token_pos < row_capacity) & active
        else:
            # Packed rows are grouped by sequence. Compute each sequence's
            # local expert rank without a Python loop; sequence_ids are sorted,
            # so subtracting the global count at each document start resets the
            # cumsum for every document.
            global_positions = torch.cumsum(active_one_hot.to(torch.int32), dim=0) - 1
            starts = torch.ones(n_tokens, device=flat.device, dtype=torch.bool)
            if n_tokens > 1:
                starts[1:] = sequence_ids_flat[1:] != sequence_ids_flat[:-1]
            start_idx = starts.nonzero(as_tuple=True)[0]
            start_ids = sequence_ids_flat.index_select(0, start_idx)
            # ``global_positions`` is an inclusive count minus one.  Store the
            # exclusive prefix at each document start so every expert's local
            # rank restarts at zero, including experts that did not receive the
            # first token of the document.
            exclusive_positions = (
                global_positions + 1 - active_one_hot.to(torch.int32)
            )
            start_values = exclusive_positions.index_select(0, start_idx).to(torch.int32)
            offsets = torch.zeros(
                (num_sequences, self.num_experts),
                device=flat.device,
                dtype=torch.int32,
            )
            offsets.index_copy_(0, start_ids, start_values)
            positions = global_positions - offsets.index_select(0, sequence_ids_flat)
            token_pos = positions.gather(1, expert_idx[:, None]).squeeze(1)
            valid = (token_pos < row_capacity) & active
        routing_map = routing_map * valid[:, None].to(torch.int32)
        routing_probs = probs * routing_map.to(probs.dtype)
        if route_counts is not None:
            route_state = route_counts[loop_idx, layer_idx]
            if decode:
                # Count only accepted top-1 routes. Overflow and inactive
                # rows remain on the shared path and consume no future slot.
                route_state.index_add_(0, sequence_ids_flat, routing_map)
            elif tokens_per_sequence is not None:
                accepted_counts = routing_map.reshape(
                    num_sequences, tokens_per_sequence, self.num_experts
                ).sum(dim=1)
                route_state[:num_sequences].copy_(accepted_counts)
            else:
                accepted_counts = route_state.new_zeros((num_sequences, self.num_experts))
                accepted_counts.index_add_(0, sequence_ids_flat, routing_map)
                route_state[:num_sequences].copy_(accepted_counts)
        # ``permute_mask_map_fwd`` first packs routed rows by their actual
        # expert counts.  GroupedLinear, however, receives fixed ``capacity``
        # rows per expert.  Shift each expert's packed segment into its own
        # fixed-capacity block; a zero offset would send the tail of a short
        # expert through the next expert's weights whenever the router loads
        # are uneven.  Counts are derived from the already capacity-trimmed
        # routing map, so this remains a fixed-shape, CUDA-graph-safe tensor
        # computation (the values may vary on each replay).
        counts = routing_map.sum(dim=0, dtype=torch.int64)
        # Inclusive cumsum minus the current count is the exclusive prefix.
        # This avoids the tiny ``cat([zeros, cumsum[:-1]])`` allocation and
        # copy on every routed layer while preserving int64 offsets expected
        # by TE's graph-safe permutation primitive.
        prefix_counts = torch.cumsum(counts, dim=0) - counts
        starts_key = (flat.device.type, flat.device.index, capacity)
        expert_starts = self._expert_starts_cache.get(starts_key)
        if expert_starts is None:
            expert_starts = torch.arange(
                self.num_experts, dtype=torch.int64, device=flat.device
            ) * capacity
            self._expert_starts_cache[starts_key] = expert_starts
        pad_offsets = expert_starts - prefix_counts
        latent = self.down(flat)
        dispatched, row_id_map, permuted_probs = self.fused_router.permute(
            latent,
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
        routed_latent = self.fused_router.unpermute(
            expert_out,
            row_id_map,
            restore_shape=latent.shape,
            pad_offsets=pad_offsets,
        )
        # ``routed_latent`` already has the model activation dtype produced by
        # the grouped expert GEMM.  Feeding it directly to the BF16 up
        # projection avoids an otherwise redundant BF16->FP32->BF16 pair of
        # full-token copies on every routed layer.
        routed = self.up(routed_latent)
        if shared_event is not None:
            current_stream = torch.cuda.current_stream(flat.device)
            current_stream.wait_event(shared_event)
            shared_result.record_stream(current_stream)
        out = routed + shared_result
        return out.view(shape)
