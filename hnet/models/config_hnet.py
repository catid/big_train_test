from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)
    # Virtual Head Attention (VHA) uses a smaller physical Q/KV projection
    # during training and folds to standard GQA-2 for decode.  The value is
    # stage-specific so the single canonical config can describe every stack.
    num_kv_heads: List = field(default_factory=list)
    vha_enabled: bool = False
    vha_rank: int = 0
    # Opt-in CroissAttn FP8 backward correction.  This keeps the auxiliary
    # unrounded O'=P@V_f used by the softmax derivative; see ~/fp8.
    # CroissAttn is stage-specific: the canonical model keeps the SWA path on
    # FlashAttention and enables the patched TE correction only for global FA.
    # Keep this as a list so malformed scalar configuration cannot pass type
    # checking and later be broadcast implicitly by the attention builder.
    croissattn_aux: List[bool] = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    # The fused Mamba-3 SISO kernel is register-safe and fastest at 32 tokens
    # on the production B300 shape.  Keep the tuned geometry in the canonical
    # config so the validator, checkpoint signature, and runtime agree.
    chunk_size: int = 32


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    # Shared learnable RoPE frequency scales.  One FP32 log-scale per rotary
    # band is used by every SWA layer and head; global FA remains NoPE.
    lerope_enabled: bool = True
    # MSAR is enabled only for the innermost main network.  Outer encoder and
    # decoder stacks remain standard residual networks because they implement
    # input/output chunking and must preserve the routing/dechunking identity.
    msar: bool = True
    msar_stream_kernels: List[int] = field(default_factory=lambda: [1, 4, 8, 12])
    # Window width used by sliding-window attention (S/s) layers.
    swa_window_size: int = 128
    # Shared recurrent execution count for the deepest (middle) network.
    inner_loops: int = 2
    # Number of middle layers between MSAR stream boundaries.  This is kept
    # in the canonical configuration rather than hidden in the module so a
    # checkpoint's architecture signature fully describes the recurrence.
    msar_block_size: int = 4
    fbt: bool = True
    fbt_passes: int = 2
    fbt_jitter: float = 0.02
    fbt_prefix_mixin: bool = True
    max_seq_len: int = 4096
    static_chunking: bool = True
    # Maximum number of learned latent boundaries retained by one chunking
    # stage.  The tensor remains statically sized for CUDA graphs; when the
    # router emits more boundaries, the earliest causal boundaries are kept
    # so a future byte cannot change an earlier latent decision.
    chunk_capacity_tokens: int = 512
    # Keep the complete activation graph for the canonical B=16 CUDA-graph
    # shape. Recomputing MSAR or compatible mixers saves memory but adds more
    # work than it removes on B300; the no-checkpoint shape has higher token
    # throughput while remaining below the device memory limit.
    msar_activation_checkpointing: bool = False
    mamba_activation_checkpointing: bool = False
    # Required DeepSeek/Mixtral-style sparse FFN. Eight routed experts with
    # top-1 dispatch give the requested 1:8 active expert sparsity. The shared
    # expert is always evaluated; a disabled dense FFN path is unsupported.
    moe_enabled: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 1
    moe_expert_intermediate: int = 0
    moe_shared_intermediate: int = 0
    moe_capacity_factor: float = 1.25
    # Fixed token bound used to size routed expert capacity.  Keeping this
    # independent of the currently packed token count makes eager, cached
    # prefill, and CUDA-graph execution use identical overflow semantics.
    moe_capacity_tokens: int = 0
    moe_aux_loss_weight: float = 0.01

    def validate_canonical(self) -> None:
        """Reject partial or ablation configurations at model construction."""
        expected_layout = ["s2m3", ["S3T1S3T1S3T1S3T1S3T1"], "s2m3"]
        if self.arch_layout != expected_layout:
            raise ValueError("the canonical model requires the S2M3/(S3T1)x5/S2M3 layout")
        if self.d_model != [1024, 1536] or self.d_intermediate != [0, 4096]:
            raise ValueError("the canonical model requires d_model=[1024,1536] and d_intermediate=[0,4096]")
        if self.vocab_size != 256:
            raise ValueError("the canonical model is byte-level with vocab_size=256")
        if self.swa_window_size != 128 or self.inner_loops != 2:
            raise ValueError("the canonical model requires a 128-token SWA window and two inner loops")
        if int(self.max_seq_len) != 4096:
            raise ValueError("the canonical model requires max_seq_len=4096")
        if int(self.msar_block_size) != 4:
            raise ValueError("the canonical model requires MSAR boundaries every four layers")
        if not (self.msar and self.fbt and self.static_chunking and self.fbt_passes == 2):
            raise ValueError("the canonical model requires MSAR, two-pass FBT, and static chunking")
        if not self.fbt_prefix_mixin or float(self.fbt_jitter) != 0.02:
            raise ValueError(
                "the canonical model requires the enabled FBT prefix mix-in and jitter=0.02"
            )
        if tuple(int(v) for v in self.msar_stream_kernels) != (1, 4, 8, 12):
            raise ValueError("the canonical model requires MSAR stream kernels [1, 4, 8, 12]")
        if int(self.chunk_capacity_tokens) != 512:
            raise ValueError("the canonical model requires a 512-token latent chunk capacity")
        if self.msar_activation_checkpointing or self.mamba_activation_checkpointing:
            raise ValueError("the canonical model disables activation checkpointing for throughput")
        if not (self.moe_enabled and self.moe_num_experts == 8 and self.moe_top_k == 1):
            raise ValueError("the canonical model requires top-1 8-way sparse MoE")
        if not self.attn_cfg.vha_enabled or self.attn_cfg.num_kv_heads != [2, 2]:
            raise ValueError("the canonical model requires VHA with two KV heads per attention stage")
        if int(self.attn_cfg.vha_rank) != 4:
            raise ValueError("the canonical model requires VHA postmix rank 4")
        if self.attn_cfg.window_size != [128, -1]:
            raise ValueError("the canonical model requires 128-token SWA and global FA")
        if self.attn_cfg.num_heads != [16, 16]:
            raise ValueError("the canonical model requires 16 attention heads per stage")
        # Both outer SWA and middle SWA use RoPE. Global FA uses NoPE so its
        # long-range logits are not tied to a rotary phase.
        if self.attn_cfg.rotary_emb_dim != [48, 48]:
            raise ValueError(
                "the canonical model requires stage rotary dimensions [48, 48]; "
                "SWA applies 48-dim RoPE and FA applies NoPE"
            )
        if not self.lerope_enabled:
            raise ValueError("the canonical model requires shared LeRoPE frequencies for SWA")
        if self.attn_cfg.croissattn_aux != [False, True]:
            raise ValueError(
                "the canonical model enables patched CroissAttn only on the global FA stage"
            )
        if (
            int(self.ssm_cfg.chunk_size) != 32
            or int(self.ssm_cfg.d_conv) != 4
            or int(self.ssm_cfg.d_state) != 128
            or int(self.ssm_cfg.expand) != 2
        ):
            raise ValueError(
                "the canonical Mamba-3 model requires SSM settings chunk_size=32, d_conv=4, "
                "d_state=128, expand=2"
            )
        if (
            int(self.moe_expert_intermediate) != 1664
            or int(self.moe_shared_intermediate) != 512
            or float(self.moe_capacity_factor) != 1.25
            or float(self.moe_aux_loss_weight) != 0.01
            or int(self.moe_capacity_tokens)
            != min(int(self.max_seq_len), int(self.chunk_capacity_tokens))
        ):
            raise ValueError(
                "the canonical model requires 1664-wide routed experts, a 512-wide shared "
                "expert, capacity_factor=1.25, aux_loss_weight=0.01, and capacity sized "
                "to the latent chunk bound"
            )
