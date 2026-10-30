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
    croissattn_aux: bool = False


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    # MSAR is enabled only for the innermost main network.  Outer encoder and
    # decoder stacks remain standard residual networks because they implement
    # input/output chunking and must preserve the routing/dechunking identity.
    msar: bool = True
    msar_stream_kernels: List[int] = field(default_factory=lambda: [1, 4, 8, 12])
    # Window width used by sliding-window attention (S/s) layers.
    swa_window_size: int = 128
    # Shared recurrent execution count for the deepest (middle) network.
    inner_loops: int = 2
    fbt: bool = True
    fbt_passes: int = 2
    fbt_jitter: float = 0.02
    fbt_prefix_mixin: bool = True
    max_seq_len: int = 4096
    static_chunking: bool = True
    # Required DeepSeek/Mixtral-style sparse FFN. Eight routed experts with
    # top-1 dispatch give the requested 1:8 active expert sparsity. The shared
    # expert is always evaluated; a disabled dense FFN path is unsupported.
    moe_enabled: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 1
    moe_expert_intermediate: int = 0
    moe_shared_intermediate: int = 0
    moe_capacity_factor: float = 1.25
    moe_aux_loss_weight: float = 0.01

    def validate_canonical(self) -> None:
        """Reject partial or ablation configurations at model construction."""
        expected_layout = ["m4", ["S3T1S3T1S3T1S3T1S3T1"], "m4"]
        if self.arch_layout != expected_layout:
            raise ValueError("the canonical model requires the M4/(S3T1)x5/M4 layout")
        if self.d_model != [1024, 1536] or self.d_intermediate != [0, 4096]:
            raise ValueError("the canonical model requires d_model=[1024,1536] and d_intermediate=[0,4096]")
        if self.vocab_size != 256:
            raise ValueError("the canonical model is byte-level with vocab_size=256")
        if self.swa_window_size != 128 or self.inner_loops != 2:
            raise ValueError("the canonical model requires a 128-token SWA window and two inner loops")
        if not (self.msar and self.fbt and self.static_chunking and self.fbt_passes == 2):
            raise ValueError("the canonical model requires MSAR, two-pass FBT, and static chunking")
        if not (self.moe_enabled and self.moe_num_experts == 8 and self.moe_top_k == 1):
            raise ValueError("the canonical model requires top-1 8-way sparse MoE")
        if not self.attn_cfg.vha_enabled or self.attn_cfg.num_kv_heads != [2, 2]:
            raise ValueError("the canonical model requires VHA with two KV heads per attention stage")
        if int(self.attn_cfg.vha_rank) != 4:
            raise ValueError("the canonical model requires VHA postmix rank 4")
