# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


def stateless_norm(x: Tensor) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-6)


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
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10_000.0) -> tuple[Tensor, Tensor]:
    freqs = torch.arange(0, dim, 2, dtype=torch.float32)
    freqs = theta ** (-freqs / dim)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return angles.cos(), angles.sin()


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    seq_len = q.size(-2)
    position_ids = torch.arange(seq_len, device=q.device)
    cos = cos[position_ids].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin[position_ids].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output * self.weight


class SwiGLU(nn.Module):
    """Reference FLOWER SwiGLU block used inside each DiT block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        n_hidden = find_multiple(int(2 * hidden_dim / 3), 256)
        output_dim = dim if output_dim is None else output_dim
        self.fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.fc1(x)) * self.fc2(x)
        return self.proj(self.dropout(x))


class ActionMlp(nn.Module):
    """Small timm-Mlp-compatible action/proprio encoder."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class FlowerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 120,
        rope_theta: float = 32.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            cos, sin = precompute_freqs_1d(self.head_dim, max_seq_len, theta=rope_theta)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Tensor,
        custom_attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)

        attn_mask = None
        if custom_attn_mask is not None:
            attn_mask = custom_attn_mask
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            if attn_mask.ndim == 4 and attn_mask.shape[1] == 1:
                attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)
            attn_mask = attn_mask.to(device=x.device, dtype=torch.bool)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
            is_causal=is_causal if attn_mask is None else False,
        )
        y = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.resid_dropout(self.proj(y))


class FlowerCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        use_rope: bool = False,
        query_seq_len: int = 64,
        context_seq_len: int = 384,
        rope_theta: float = 32.0,
        context_rope_theta: float = 1000.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            q_cos, q_sin = precompute_freqs_1d(self.head_dim, query_seq_len, theta=rope_theta)
            k_cos, k_sin = precompute_freqs_1d(self.head_dim, context_seq_len, theta=context_rope_theta)
            self.register_buffer("q_cos", q_cos, persistent=False)
            self.register_buffer("q_sin", q_sin, persistent=False)
            self.register_buffer("k_cos", k_cos, persistent=False)
            self.register_buffer("k_sin", k_sin, persistent=False)

    def forward(self, x: Tensor, context: Tensor, custom_attn_mask: Tensor | None = None) -> Tensor:
        bsz, seq_len, dim = x.shape
        context_len = context.shape[1]
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(bsz, context_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(bsz, context_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, _ = apply_rotary_pos_emb(q, q, self.q_cos, self.q_sin)
            k, _ = apply_rotary_pos_emb(k, k, self.k_cos, self.k_sin)

        attn_mask = None
        if custom_attn_mask is not None:
            attn_mask = custom_attn_mask.to(device=x.device, dtype=torch.bool)
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            if attn_mask.ndim == 4 and attn_mask.shape[1] == 1:
                attn_mask = attn_mask.expand(-1, self.n_heads, q.shape[2], -1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.resid_dropout(self.proj(y))


class FlowBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        use_cross_attn: bool = False,
        use_rope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
        lora_dim: int = 256,
        use_global_adaln: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_global_adaln = use_global_adaln
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.norm3 = RMSNorm(dim, eps=norm_eps) if use_cross_attn else None
        self.self_attn = FlowerAttention(
            dim=dim,
            n_heads=heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            use_rope=use_rope,
            max_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )
        if use_cross_attn:
            self.cross_attn = FlowerCrossAttention(
                dim=dim,
                n_heads=heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                use_rope=False,
            )
        self.mlp = SwiGLU(dim, dropout=mlp_pdrop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, lora_dim),
            nn.Linear(lora_dim, 6 * dim),
        )

    def forward(
        self,
        cx: Tensor,
        c: Tensor,
        context: Tensor | None = None,
        custom_attn_mask: Tensor | None = None,
        custom_cross_attn_mask: Tensor | None = None,
        is_causal: bool = False,
        global_adaln: tuple[Tensor, ...] | list[Tensor] | None = None,
    ) -> Tensor:
        modulation = self.adaLN_modulation(c)
        signals = modulation.chunk(6, dim=-1)
        if self.use_global_adaln and global_adaln is not None:
            signals = tuple(signals[i] + global_adaln[i] for i in range(6))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = signals

        x_norm = self.norm1(cx)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        x_self = self.self_attn(x_mod, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        x_out = cx + gate_msa.unsqueeze(1) * x_self

        if self.use_cross_attn:
            if context is None:
                raise ValueError("FLOWER cross-attention requires VLM context.")
            x_cross = self.cross_attn(self.norm2(x_out), context, custom_attn_mask=custom_cross_attn_mask)
            x_out = x_out + x_cross

        norm_layer = self.norm3 if self.use_cross_attn else self.norm2
        if norm_layer is None:
            raise RuntimeError("FLOWER internal norm layer was not initialized.")
        x_mlp = self.mlp(modulate(norm_layer(x_out), shift_mlp, scale_mlp))
        return x_out + gate_mlp.unsqueeze(1) * x_mlp


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: Tensor) -> Tensor:
        emb = self.timestep_embedding(timesteps.reshape(-1), self.frequency_embedding_size)
        return self.mlp(emb.to(dtype=next(self.parameters()).dtype))


class FreqEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 1000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, frequency: Tensor) -> Tensor:
        frequency = frequency.reshape(frequency.shape[0], -1).mean(dim=-1)
        emb = self.timestep_embedding(frequency, self.frequency_embedding_size)
        return self.mlp(emb.to(dtype=next(self.parameters()).dtype))


class SharedAdaLNController(nn.Module):
    def __init__(self, dim: int, global_conddim: int, use_cross_attn: bool = False):
        super().__init__()
        num_mod_signals = 9 if use_cross_attn else 6
        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, num_mod_signals * dim, bias=False),
        )
        nn.init.zeros_(self.modCX[-1].weight)
        self.use_cross_attn = use_cross_attn

    def forward(self, global_cond: Tensor) -> tuple[Tensor, ...]:
        mod_signals = self.modCX(global_cond)
        return mod_signals.chunk(9 if self.use_cross_attn else 6, dim=-1)


class ZeroEncoder(nn.Module):
    def __init__(self, dit_dim: int):
        super().__init__()
        self.dit_dim = dit_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((x.shape[0], self.dit_dim))
