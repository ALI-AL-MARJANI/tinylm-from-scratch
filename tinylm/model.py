"""A minimal Llama-style decoder: RoPE, pre-norm, RMSNorm, SwiGLU, tied embeddings."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Cache = tuple[torch.Tensor, torch.Tensor]  # one layer: (keys, values), each (B, heads, T, head_dim)


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 704  # about 8/3 * d_model, as the gate doubles the input projection
    max_len: int = 256
    pos: str = "rope"  # or "learned", for the ablation


class RMSNorm(nn.Module):
    """LayerNorm without mean subtraction and without bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


class SwiGLU(nn.Module):
    """MLP where one projection gates the other."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE: rotate each pair of dimensions by an angle proportional to the position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rotation: Cache | None,
        mask: torch.Tensor,
        past: Cache | None,
    ) -> tuple[torch.Tensor, Cache]:
        B, T, C = x.shape
        shape = (B, T, self.n_heads, self.head_dim)
        q = self.q(x).view(shape).transpose(1, 2)  # (B, heads, T, head_dim)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        if rotation is not None:
            q, k = rotate(q, *rotation), rotate(k, *rotation)
        if past is not None:  # keys are cached after rotation: their position never changes
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # F.scaled_dot_product_attention fuses the next three lines without storing the matrix.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y), (k, v)


class Block(nn.Module):
    """Pre-norm attention and MLP, each added to the residual stream."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        rotation: Cache | None,
        mask: torch.Tensor,
        past: Cache | None,
    ) -> tuple[torch.Tensor, Cache]:
        y, present = self.attn(self.attn_norm(x), rotation, mask, past)
        x = x + y
        return x + self.mlp(self.mlp_norm(x)), present


class TinyLM(nn.Module):
    """Token ids in, next-token logits out."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model) if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        head_dim = cfg.d_model // cfg.n_heads
        theta = 10000.0 ** (-torch.arange(0, head_dim, 2) / head_dim)
        angles = torch.outer(torch.arange(cfg.max_len).float(), theta)
        mask = torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool).tril()
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("mask", mask, persistent=False)
        for module in self.modules():  # small init: the tied head would otherwise give huge logits
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, cache: list[Cache | None] | None = None) -> torch.Tensor:
        """Return (B, T, vocab) logits. A cache (one slot per layer) is updated in place."""
        past = cache if cache is not None else [None] * len(self.blocks)
        start = 0 if past[0] is None else past[0][0].shape[2]
        end = start + idx.shape[1]
        assert end <= self.cfg.max_len, f"sequence of {end} tokens > max_len={self.cfg.max_len}"
        x, rotation = self.embed(idx), None
        if self.pos_embed is not None:
            x = x + self.pos_embed(torch.arange(start, end, device=idx.device))
        else:
            rotation = self.cos[start:end], self.sin[start:end]
        for i, block in enumerate(self.blocks):
            x, present = block(x, rotation, self.mask[start:end, :end], past[i])
            if cache is not None:
                cache[i] = present
        return self.norm(x) @ self.embed.weight.T  # tied output head

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
