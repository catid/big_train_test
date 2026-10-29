from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)
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
    tie_embeddings: bool = False
    # MSAR is enabled only for the innermost main network.  Outer encoder and
    # decoder stacks remain standard residual networks because they implement
    # input/output chunking and must preserve the routing/dechunking identity.
    msar: bool = False
    msar_stream_kernels: List[int] = field(default_factory=lambda: [1, 4, 8, 12])
    # Shared recurrent execution count for the deepest (middle) network.
    inner_loops: int = 1
    fbt: bool = False
    fbt_jitter: float = 0.02
    fbt_prefix_mixin: bool = True
    max_seq_len: int = 4096
    static_chunking: bool = False
    # Optional DeepSeek/Mixtral-style sparse FFN.  Eight routed experts with
    # top-1 dispatch give the requested 1:8 active expert sparsity.  The
    # shared expert is always evaluated and is deliberately narrow so it
    # carries common features without duplicating every routed expert.
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 1
    moe_expert_intermediate: int = 0
    moe_shared_intermediate: int = 0
    moe_capacity_factor: float = 1.25
    moe_aux_loss_weight: float = 0.01
