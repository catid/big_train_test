from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


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
    max_seq_len: int = 4096
    static_chunking: bool = False
