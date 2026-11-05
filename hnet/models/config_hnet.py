from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    vocab_size: int = 256
    # Shared recurrent execution count for the deepest (middle) network.
    inner_loops: int = 1
    fbt: bool = False
    fbt_jitter: float = 0.02
    fbt_prefix_mixin: bool = True
    max_seq_len: int = 4096
    # Fixed latent capacity used by the canonical 8K CUDA-graph path.  The
    # chunker keeps the earliest boundaries when a batch exceeds this cap;
    # this prevents the inner FA from silently becoming an 8K quadratic pass.
    latent_max_seq_len: int | None = None
    static_chunking: bool = False
    # The canonical byte stream is a fixed, fully valid sequence.  The outer
    # GDN2/LoL stacks may therefore omit the all-true attention mask and use
    # the fused causal/window kernels.  The latent trunk still receives its
    # validity mask because its 2048-capacity buffer can contain padding.
    static_full_sequence: bool = False
    # Compile the fixed LoL/FA blocks with dynamic=False before DDP and graph
    # capture.  GDN2 remains an external custom-kernel module and is not
    # passed through Inductor.
    compile_blocks: bool = False
    # Every uppercase mixer block uses the production LatentMoE FFN.  The
    # latent width is half the residual width; ``matched`` computes one common
    # expert neuron width from the dense SwiGLU parameter budget, so replacing
    # a dense FFN does not silently change the model size.  The eight routed
    # experts plus the shared expert still expose more total neurons.
    moe_num_experts: int = 8
    moe_top_k: int = 4
    moe_latent_alpha: int = 2
    moe_matched: bool = True

    # The production stack uses explicit GDN2/LoL/FA block names.  The
    # external GDN2 implementation is loaded lazily by
    # ``hnet.modules.gdn2_adapter`` so configuration inspection does not
    # require the NVIDIA/FLa checkout.
    gdn2_module: str = "lit_gpt.gdn2:GatedDeltaNet2"
    gdn2_hidden_size: int = 3072
    gdn2_num_heads: int = 48
    gdn2_head_dim: int = 64
    gdn2_expand_v: float = 2.0
    gdn2_backend: str = "cudnn"
    gdn2_use_short_conv: bool = False
    gdn2_fused_projections: bool = True
    gdn2_disable_recompute: bool = True
    # LoL is represented as a named local-window GQA block. ``fa`` is the same
    # attention primitive with an unbounded causal window and NoPE. Both use
    # the fixed-shape LatentMoE FFN in the canonical block library.
    lol_window_size: int = 128
    lol_hidden_size: int = 1280
    lol_ffn_width: int = 1536
    lol_num_heads: int = 20
    lol_kv_heads: int = 4
    lol_head_dim: int = 64
    fa_num_heads: int = 20
    fa_kv_heads: int = 4
    fa_head_dim: int = 64
    # Metadata consumed by the fixed block library; keeping sharing declarations
    # in the config makes tied middle parameters auditable.
    library_weight_sharing: dict = field(default_factory=dict)
    library_block_widths: dict = field(default_factory=dict)
    external_dependencies: dict = field(default_factory=dict)
    # Canonical fixed-shape training length.
    canonical_seq_len: int = 8192

    def validate_static_shapes(self, *, seq_len: int | None = None) -> None:
        """Validate dimensions needed by the named GDN2/LoL/FA library.

        This is deliberately a pure configuration check: it does not import
        CUDA extensions or instantiate modules.  Trainers can call it before
        spawning DDP workers and report an actionable error on a host that
        lacks the pinned external dependency.
        """
        if self.arch_layout != [
            "GDN2,LoL,GDN2,LoL",
            ["GDN2,LoL,GDN2,LoL,FA"],
            "GDN2,LoL,GDN2,LoL",
        ]:
            raise ValueError("the production model has one fixed GDN2/LoL/FA layout")
        if [int(v) for v in self.d_model] != [3072, 1280]:
            raise ValueError("the production model requires d_model=[3072, 1280]")
        if int(self.max_seq_len) != 8192:
            raise ValueError("the production model requires max_seq_len=8192")
        if int(self.canonical_seq_len) != 8192 or int(self.latent_max_seq_len or 0) != 2048:
            raise ValueError("the production model requires canonical_seq_len=8192 and latent_max_seq_len=2048")
        if not self.fbt or not self.static_full_sequence:
            raise ValueError("the production model requires FBT and static shapes")
        if not self.static_chunking:
            raise ValueError("canonical named-library training requires static_chunking=true")
        if len(self.d_model) < 2:
            raise ValueError("named GDN2/LoL/FA layouts require d_model=[outer, latent]")
        if int(self.d_model[1]) != int(self.lol_hidden_size):
            raise ValueError(
                "canonical latent d_model must equal lol_hidden_size so the "
                f"H-Net trunk boundary is explicit ({self.d_model[1]} != {self.lol_hidden_size})"
            )
        if int(self.inner_loops) != 2:
            raise ValueError("canonical GDN2/LoL/FA trunk requires exactly two inner loops")
        if not self.moe_matched or int(self.moe_num_experts) != 8:
            raise ValueError("canonical LoL/FA FFNs require eight-expert LatentMoE")
        if not 1 <= int(self.moe_top_k) <= int(self.moe_num_experts):
            raise ValueError("moe_top_k must be within the eight-expert LatentMoE range")
        if self.gdn2_hidden_size != self.d_model[0]:
            raise ValueError(
                "gdn2_hidden_size must equal outer d_model[0] for the canonical "
                f"handoff (got {self.gdn2_hidden_size} vs {self.d_model[0]})"
            )
        if self.gdn2_num_heads * self.gdn2_head_dim != self.gdn2_hidden_size:
            raise ValueError(
                "GDN2 shape mismatch: num_heads * head_dim must equal hidden_size "
                f"({self.gdn2_num_heads} * {self.gdn2_head_dim} != {self.gdn2_hidden_size})"
            )
        latent_width = int(self.lol_hidden_size)
        if latent_width <= 0 or int(self.lol_ffn_width) <= 0:
            raise ValueError("LoL hidden and FFN widths must be positive")
        for name, heads in (("LoL", self.lol_num_heads), ("FA", self.fa_num_heads)):
            head_dim = self.lol_head_dim if name == "LoL" else self.fa_head_dim
            if heads * head_dim != latent_width:
                raise ValueError(
                    f"{name} shape mismatch: heads * head_dim must equal latent width "
                    f"({heads} * {head_dim} != {latent_width})"
                )
        for name, query_heads, heads in (
            ("LoL", self.lol_num_heads, self.lol_kv_heads),
            ("FA", self.fa_num_heads, self.fa_kv_heads),
        ):
            if heads <= 0 or query_heads % heads:
                raise ValueError(f"{name} KV heads must divide LoL query heads")
        if self.lol_window_size <= 0:
            raise ValueError("lol_window_size must be positive")
        length = self.canonical_seq_len if seq_len is None else int(seq_len)
        if length <= 0 or length % self.lol_window_size:
            raise ValueError(
                f"static sequence length {length} must be a positive multiple of "
                f"lol_window_size={self.lol_window_size}"
            )
        latent = length if self.latent_max_seq_len is None else int(self.latent_max_seq_len)
        if latent < 1 or latent > length or latent % self.lol_window_size:
            raise ValueError(
                "latent_max_seq_len must be a positive multiple of the LoL "
                f"window and no larger than seq_len ({latent}, {length})"
            )
