# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def build_sincos_position_embedding(length: int, dim: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32)
    return sinusoidal_embedding(positions, dim).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(gate) * value))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FlowerAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, attn_pdrop: float = 0.0, resid_pdrop: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dropout_p = self.attn_pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.resid_dropout(self.proj(y))


class FlowerCrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, attn_pdrop: float = 0.0, resid_pdrop: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).view(bsz, context.shape[1], 2, self.n_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        k, v = kv.unbind(0)
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask[:, None, None, :].to(dtype=torch.bool)
        dropout_p = self.attn_pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.resid_dropout(self.proj(y))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        emb = sinusoidal_embedding(timesteps.reshape(-1), self.frequency_embedding_size)
        return self.mlp(emb)


class FreqEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, frequency: Tensor) -> Tensor:
        frequency = frequency.reshape(frequency.shape[0], -1).mean(dim=-1)
        emb = sinusoidal_embedding(frequency, self.frequency_embedding_size)
        return self.mlp(emb)


class SharedAdaLNController(nn.Module):
    def __init__(self, cond_dim: int, hidden_size: int, n_layers: int, n_action_spaces: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.action_embedding = nn.Embedding(n_action_spaces, cond_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_size * 6 * n_layers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor, action_type: Tensor) -> Tensor:
        cond = cond + self.action_embedding(action_type)
        params = self.net(cond)
        return params.view(cond.shape[0], self.n_layers, 6, self.hidden_size)


class FlowBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        norm_eps: float = 1e-6,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = FlowerAttention(hidden_size, n_heads, attn_pdrop, resid_pdrop)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), resid_pdrop)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.cross_attn = FlowerCrossAttention(hidden_size, n_heads, attn_pdrop, resid_pdrop)

    def forward(
        self,
        x: Tensor,
        adaln_params: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params.unbind(dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.cross_norm(x), context, context_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
