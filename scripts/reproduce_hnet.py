#!/usr/bin/env python3
"""Run a small, reproducible H-Net check on raw UTF-8 bytes.

This is deliberately an evaluation/generation harness rather than a claim that a
single workstation can repeat Cartesia's multi-billion-token scaling ladders.
It reports the same basic quantities used in the posts: bits per byte and the
learned byte-to-chunk compression ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `python scripts/reproduce_hnet.py` work from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer


def load_model(checkpoint: Path, config_path: Path) -> HNetForCausalLM:
    with config_path.open() as f:
        raw = json.load(f)
    attn_cfg = AttnConfig(**raw.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**raw.pop("ssm_cfg"))
    cfg = HNetConfig(**raw, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("The released H-Net uses CUDA-only Mamba/FlashAttention kernels; run on a CUDA GPU.")
    model = HNetForCausalLM(cfg, device=device, dtype=torch.bfloat16)
    # Checkpoints are plain state dictionaries.  Keep deserialization on
    # PyTorch's restricted tensor-only path; executing pickle payloads from a
    # downloaded checkpoint would turn inference into arbitrary code execution.
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def score(model: HNetForCausalLM, text: str) -> tuple[float, float, int, int]:
    tok = ByteTokenizer()
    ids = tok.encode([text], add_bos=True)[0]["input_ids"]
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None]
    out = model(x)
    # The model predicts the next raw byte.  Divide nats by ln(2) to obtain
    # the post's bits-per-byte metric.
    nll = F.cross_entropy(
        out.logits[:, :-1].float().reshape(-1, 256), x[:, 1:].reshape(-1)
    )
    # For a 2-stage model this is the outer (byte -> intermediate) hierarchy;
    # the second prediction stream is the inner hierarchy.
    bpred = out.bpred_output[0].boundary_mask
    boundaries = int(bpred.sum().item())
    byte_length = len(ids) - 1  # exclude the synthetic BOS byte
    return float(nll / torch.log(torch.tensor(2.0))), boundaries / byte_length, byte_length, boundaries


@torch.inference_mode()
def greedy(model: HNetForCausalLM, prompt: str, steps: int) -> str:
    tok = ByteTokenizer()
    ids = tok.encode([prompt], add_bos=True)[0]["input_ids"]
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    cache = model.allocate_inference_cache(1, len(ids) + steps, dtype=torch.bfloat16)
    mask = torch.ones_like(x, dtype=torch.bool)
    out = model(x, mask=mask, inference_params=cache)
    generated: list[int] = []
    for _ in range(steps):
        nxt = int(out.logits[:, -1].float().argmax(dim=-1).item())
        if nxt == tok.eos_idx:
            break
        generated.append(nxt)
        out = model.step(torch.tensor([[nxt]], device=device), cache)
    return tok.decode(generated, errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--text", default="Hierarchical networks learn directly from bytes.")
    p.add_argument("--generate", type=int, default=16)
    args = p.parse_args()
    model = load_model(args.checkpoint, args.config)
    params = sum(x.numel() for x in model.parameters())
    bpb, ratio, length, chunks = score(model, args.text)
    print(f"parameters={params:,}")
    print(f"bytes={length} outer_chunks={chunks} bytes_per_outer_chunk={length / chunks:.3f}")
    print(f"bits_per_byte={bpb:.6f}")
    print(f"greedy_completion={greedy(model, args.text, args.generate)!r}")


if __name__ == "__main__":
    main()
