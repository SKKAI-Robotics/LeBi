#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Twin attention primitives for SmolVLA-Twin.

This module ports the algorithm from the reference TwinVLA paper's
`twinvla_attention` (see reference/TwinVLA/twinvla/model/base_models.py L858+)
to the SmolVLA / SmolVLM2 backbone shape:
  - 16 decoder layers (vs reference's hardcoded 24)
  - Multi-query attention: num_attention_heads=15, num_key_value_heads=5, head_dim=64
  - RoPE implementation matches lerobot/policies/smolvla.smolvlm_with_expert.apply_rope

Two top-level forward functions are exposed:

  - `twin_vlm_forward(inputs_embeds, pad_mask, ci_ids, ...)`
        Phase 2: VLM-only (no action expert). Useful for unit-testing the
        bimanual VLM stream in isolation.

  - `twin_vlm_expert_forward(vlm_embeds, vlm_pad_mask, ci_ids,
                              expert_embeds, expert_pad_mask, ...)`
        Phase 3: bimanual VLM stream + shared action expert in a single
        16-layer loop, alternating joint-self-attention layers and
        cross-attention layers like SmolVLA's standard
        `SmolVLMWithExpertModel.forward`.

Convention: callers must construct the VLM prefix in
`[common | left | right]` order so that `ci_ids` is non-decreasing along
the sequence dimension. This makes ci_ids order coincide with our concat
order and avoids any index permutation in the forward loop.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

CI_COMMON = 0
CI_LEFT = 1
CI_RIGHT = 2


# --------------------------------------------------------------------------- #
#  Position ids: common=0..k-1, left and right offset by k
# --------------------------------------------------------------------------- #


def create_twin_position_ids(ci_ids: Tensor) -> Tensor:
    """Reference: BaseTwinVLAMetaModel.create_position_ids

    Returns position ids such that
      ci==0 :  0..count0-1                     (shared prefix)
      ci==1 :  count0 .. count0 + count1 - 1   (left arm sub-sequence)
      ci==2 :  count0 .. count0 + count2 - 1   (right arm sub-sequence, parallel)
    """
    mask0 = ci_ids == CI_COMMON
    mask1 = ci_ids == CI_LEFT
    mask2 = ci_ids == CI_RIGHT

    position_ids = torch.zeros_like(ci_ids, dtype=torch.long)

    pos0 = torch.cumsum(mask0.long(), dim=1) - 1
    position_ids = torch.where(mask0, pos0, position_ids)

    count0 = torch.sum(mask0.long(), dim=1, keepdim=True)
    order_1 = torch.cumsum(mask1.long(), dim=1) - 1
    order_2 = torch.cumsum(mask2.long(), dim=1) - 1
    position_ids = torch.where(mask1, count0 + order_1, position_ids)
    position_ids = torch.where(mask2, count0 + order_2, position_ids)

    return position_ids


# --------------------------------------------------------------------------- #
#  4D causal attention mask with cross-modal L<->R band
# --------------------------------------------------------------------------- #


def create_twin_4d_attention_mask(
    pad_mask: Tensor,
    ci_ids: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Reference: BaseTwinVLAMetaModel.create_4d_causal_mask

    Output shape: [B, 1, T, T] with 0.0 where attention is allowed and
    `torch.finfo(dtype).min` where it must be masked out.

    Rules:
      - Standard lower-triangular causal within the same ci.
      - Cross-modal blocks (query is ci==1, key is ci==2) and the reverse
        get a lower-triangular causal mask of size [modal_len, modal_len]
        applied within the block (so left token i can see right tokens
        0..i of the parallel sub-sequence and vice versa).
      - Padded tokens are masked everywhere.
    """
    device = pad_mask.device
    B, seqlen = pad_mask.shape

    base_causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=device))
    causal_mask = base_causal.unsqueeze(0).expand(B, -1, -1).clone()  # (B, T, T)

    ci_key = ci_ids.unsqueeze(1)  # (B, 1, T)
    ci_query = ci_ids.unsqueeze(2)  # (B, T, 1)

    LR_mask = (ci_query == CI_LEFT) & (ci_key == CI_RIGHT)
    RL_mask = (ci_query == CI_RIGHT) & (ci_key == CI_LEFT)

    # Length of the per-arm sub-sequence (assumed consistent across batch).
    # In TwinVLA paper's setup left and right sub-sequences are constructed
    # symmetrically. We replicate that assumption.
    modal_len = (ci_ids[0] == CI_LEFT).sum().item()
    if modal_len > 0:
        tri_modal = torch.tril(
            torch.ones((B, modal_len, modal_len), dtype=torch.bool, device=device)
        )
        tri_modal_flat = tri_modal.reshape(-1)

        n_LR = LR_mask.sum().item()
        n_RL = RL_mask.sum().item()
        if n_LR == tri_modal_flat.numel() and n_RL == tri_modal_flat.numel():
            causal_mask[LR_mask] = tri_modal_flat
            causal_mask[RL_mask] = tri_modal_flat

    valid = pad_mask.bool().unsqueeze(1).expand(-1, seqlen, -1)
    final_mask = causal_mask & valid

    # Convert to additive mask: 0.0 allowed, -inf disallowed.
    twin_mask = torch.full((B, 1, seqlen, seqlen), torch.finfo(dtype).min, device=device)
    twin_mask = twin_mask.masked_fill(final_mask.unsqueeze(1), 0.0)
    return twin_mask


# --------------------------------------------------------------------------- #
#  fuse / moe / reweighting
# --------------------------------------------------------------------------- #


def fuse_linear(linear_l: nn.Linear, linear_r: nn.Linear, x: Tensor) -> Tensor:
    """Task-arithmetic fuse: average of left and right linear outputs.

    Cast `x` to the linear weight dtype so this works under mixed precision
    (e.g. fp32 input from RMSNorm against bf16 weights).
    """
    dt = linear_l.weight.dtype
    if x.dtype != dt:
        x = x.to(dtype=dt)
    return 0.5 * (linear_l(x) + linear_r(x))


def fuse_layernorm(ln_l: nn.Module, ln_r: nn.Module, x: Tensor) -> Tensor:
    return 0.5 * (ln_l(x) + ln_r(x))


def _safe_proj(linear: nn.Linear, x: Tensor) -> Tensor:
    """Apply `nn.Linear` after casting input to its weight dtype."""
    dt = linear.weight.dtype
    if x.dtype != dt:
        x = x.to(dtype=dt)
    return linear(x)


def moe_mlp_forward(
    gate: nn.Linear,
    mlp_l: nn.Module,
    mlp_r: nn.Module,
    common_inputs: Tensor,
) -> Tensor:
    """STE-based 2-expert MoE selection between mlp_l and mlp_r.

    Reference: BaseTwinVLAMetaModel.moe_mlp.

    Hard argmax routing with a straight-through estimator so that gradients
    still flow through `scores` (and therefore through `gate`).
    """
    # Align input dtype with gate weight (mixed-precision robustness).
    if common_inputs.dtype != gate.weight.dtype:
        common_inputs = common_inputs.to(dtype=gate.weight.dtype)
    logits = gate(common_inputs)  # [B, T, 2]
    scores = F.softmax(logits, dim=-1)
    expert_idx = torch.argmax(scores, dim=-1)

    hard_mask = F.one_hot(expert_idx, num_classes=2).to(scores.dtype)
    ste_mask = (hard_mask - scores).detach() + scores  # [B, T, 2]

    out_l = mlp_l(common_inputs)
    out_r = mlp_r(common_inputs)
    stacked = torch.stack([out_l, out_r], dim=-1)  # [B, T, H, 2]
    return (stacked * ste_mask.unsqueeze(-2)).sum(dim=-1)


def apply_modality_reweighting(
    attn_weights: Tensor,
    ci_ids: Tensor,
    scale_factor: float,
) -> Tensor:
    """Re-weights attention so non-common keys (ci != 0) get a `scale_factor`x boost.

    Reference: BaseTwinVLAMetaModel.apply_modality_mask.

    Returns a tensor that has identical *forward* values as a manually
    re-weighted softmax but pipes gradients through the original `attn_weights`
    (STE). This means the re-weighting acts as a "soft attention prior" rather
    than a hard reroute.
    """
    B, H, T, _ = attn_weights.shape
    k_ids = ci_ids[:, None, :]  # [B, 1, T]
    match_mask = (k_ids != CI_COMMON).unsqueeze(1).expand(-1, H, T, -1)

    with torch.no_grad():
        re_weights = attn_weights * (match_mask.to(attn_weights.dtype) +
                                     scale_factor * (~match_mask).to(attn_weights.dtype))
        re_weights = re_weights / re_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    return attn_weights + (re_weights - attn_weights).detach()


# --------------------------------------------------------------------------- #
#  Eager attention with optional re-weighting
# --------------------------------------------------------------------------- #


def eager_twin_attention(
    query: Tensor,        # [B, T, H_q, D]   H_q = num_attention_heads
    key: Tensor,          # [B, T, H_kv, D]
    value: Tensor,        # [B, T, H_kv, D]
    attention_mask: Tensor,   # [B, 1, T, T]   additive, 0 or -inf
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    ci_ids: Tensor | None = None,
    use_reweighting: bool = False,
    reweight_scale: float = 2.0,
) -> Tensor:
    """Grouped-query eager attention with optional modality re-weighting.

    Mirrors SmolVLA's `eager_attention_forward` but adds the TwinVLA
    paper's attention re-weighting hook (`apply_modality_mask`).
    """
    B, T_q, _, _ = query.shape
    T_k = key.shape[1]
    n_rep = num_attention_heads // num_key_value_heads

    # Expand kv heads to match query heads (multi-query expand).
    # Note: T_k may differ from T_q in cross-attention.
    key = key[:, :, :, None, :].expand(B, T_k, num_key_value_heads, n_rep, head_dim)
    key = key.reshape(B, T_k, num_attention_heads, head_dim)
    value = value[:, :, :, None, :].expand(B, T_k, num_key_value_heads, n_rep, head_dim)
    value = value.reshape(B, T_k, num_attention_heads, head_dim)

    # [B, H, T, D]
    q = query.to(torch.float32).transpose(1, 2)
    k = key.to(torch.float32).transpose(1, 2)
    v = value.transpose(1, 2)

    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scaling   # [B, H, T, T]

    # additive causal mask (already -inf where masked)
    attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    if use_reweighting and ci_ids is not None:
        attn_weights = apply_modality_reweighting(attn_weights, ci_ids, reweight_scale)

    attn_weights = attn_weights.to(v.dtype)
    # [B, H, T_q, D] -> [B, T_q, H*D]
    attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(B, T_q, num_attention_heads * head_dim)
    return attn_output


# --------------------------------------------------------------------------- #
#  16-layer twin VLM forward
# --------------------------------------------------------------------------- #


def twin_vlm_forward(
    inputs_embeds: Tensor,           # [B, T, vlm_hidden]
    pad_mask: Tensor,                # [B, T]
    ci_ids: Tensor,                  # [B, T]
    vlm_l_layers: nn.ModuleList,
    vlm_r_layers: nn.ModuleList,
    vlm_norm: nn.Module,
    moe_gates: nn.ModuleList | None,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    enable_moe: bool,
    enable_joint_attn: bool,
    attn_reweighting: bool,
    reweight_scale: float = 2.0,
) -> Tensor:
    """Main twin VLM forward loop (16 layers).

    Ports `BaseTwinVLAMetaModel.twinvla_attention` to SmolVLM2 layer shape.
    Does NOT touch the action expert; expert integration happens in Phase 3
    via the cross-attention layers (every n-th layer, see SmolVLA's
    `forward_cross_attn_layer`).
    """
    assert len(vlm_l_layers) == len(vlm_r_layers)
    num_layers = len(vlm_l_layers)
    if enable_moe:
        assert moe_gates is not None and len(moe_gates) == num_layers

    # Reference TwinVLA supports two ablations: enable_moe=False and
    # enable_joint_attn=False. Both are wired differently in the paper's code
    # (doubled-common prefix for MoE=False; split-stream attention with no
    # cross-talk for joint=False). We support only the default configuration
    # MoE=True, joint=True at present; the others are paper-ablation-only.
    if not enable_moe:
        raise NotImplementedError(
            "SmolVLA-Twin: enable_moe=False (TwinVLA's doubled-common prefix) "
            "is not implemented. Use enable_moe=True (default)."
        )
    if not enable_joint_attn:
        raise NotImplementedError(
            "SmolVLA-Twin: enable_joint_attn=False (split-stream, no cross-arm) "
            "is paper-ablation-only and not validated. Use enable_joint_attn=True."
        )

    hidden_states = inputs_embeds
    B, T, _ = hidden_states.shape
    device = hidden_states.device

    common_mask = ci_ids == CI_COMMON
    left_mask = ci_ids == CI_LEFT
    right_mask = ci_ids == CI_RIGHT

    position_ids = create_twin_position_ids(ci_ids)
    attention_mask_4d = create_twin_4d_attention_mask(pad_mask, ci_ids, dtype=hidden_states.dtype)

    H = vlm_l_layers[0].self_attn.q_proj.weight.shape[0]
    assert H == num_attention_heads * head_dim

    def _split(x_full: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        common = x_full[common_mask].view(B, -1, x_full.shape[-1])
        left = x_full[left_mask].view(B, -1, x_full.shape[-1])
        right = x_full[right_mask].view(B, -1, x_full.shape[-1])
        return common, left, right

    def _reassemble(common: Tensor | None, left: Tensor, right: Tensor) -> Tensor:
        out = torch.empty_like(hidden_states)
        if enable_moe and common is not None:
            out[common_mask] = common.reshape(-1, common.shape[-1]).to(out.dtype)
        else:
            # When MoE is disabled, common stream is the average of left/right contributions
            # but the reference fuses left/right Q/K/V already; we just leave common slots
            # filled by zeros and rely on attn output assembly below.
            pass
        out[left_mask] = left.reshape(-1, left.shape[-1]).to(out.dtype)
        out[right_mask] = right.reshape(-1, right.shape[-1]).to(out.dtype)
        return out

    for i in range(num_layers):
        residual = hidden_states

        # ----------- 1) Input layernorm ----------- #
        common_in, left_in, right_in = _split(hidden_states)
        if enable_moe:
            common_ln = fuse_layernorm(
                vlm_l_layers[i].input_layernorm,
                vlm_r_layers[i].input_layernorm,
                common_in,
            )
        else:
            common_ln = None
        left_ln = vlm_l_layers[i].input_layernorm(left_in)
        right_ln = vlm_r_layers[i].input_layernorm(right_in)

        # ----------- 2) Q/K/V projections ----------- #
        l_attn = vlm_l_layers[i].self_attn
        r_attn = vlm_r_layers[i].self_attn

        def _to_heads(x: Tensor, n_heads: int) -> Tensor:
            return x.view(B, -1, n_heads, head_dim)

        if enable_moe:
            common_Q = _to_heads(fuse_linear(l_attn.q_proj, r_attn.q_proj, common_ln), num_attention_heads)
            common_K = _to_heads(fuse_linear(l_attn.k_proj, r_attn.k_proj, common_ln), num_key_value_heads)
            common_V = _to_heads(fuse_linear(l_attn.v_proj, r_attn.v_proj, common_ln), num_key_value_heads)
        else:
            common_Q = common_K = common_V = None

        left_Q = _to_heads(_safe_proj(l_attn.q_proj, left_ln), num_attention_heads)
        left_K = _to_heads(_safe_proj(l_attn.k_proj, left_ln), num_key_value_heads)
        left_V = _to_heads(_safe_proj(l_attn.v_proj, left_ln), num_key_value_heads)
        right_Q = _to_heads(_safe_proj(r_attn.q_proj, right_ln), num_attention_heads)
        right_K = _to_heads(_safe_proj(r_attn.k_proj, right_ln), num_key_value_heads)
        right_V = _to_heads(_safe_proj(r_attn.v_proj, right_ln), num_key_value_heads)

        # ----------- 3) Joint attention ----------- #
        if enable_joint_attn:
            if enable_moe:
                Q = torch.cat([common_Q, left_Q, right_Q], dim=1)
                K = torch.cat([common_K, left_K, right_K], dim=1)
                V = torch.cat([common_V, left_V, right_V], dim=1)
            else:
                Q = torch.cat([left_Q, right_Q], dim=1)
                K = torch.cat([left_K, right_K], dim=1)
                V = torch.cat([left_V, right_V], dim=1)

            Q = apply_rope(Q, position_ids)
            K = apply_rope(K, position_ids)

            attn_out_full = eager_twin_attention(
                query=Q,
                key=K,
                value=V,
                attention_mask=attention_mask_4d,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=ci_ids,
                use_reweighting=(enable_moe and attn_reweighting),
                reweight_scale=reweight_scale,
            )  # [B, T, vlm_hidden]
        else:
            # Independent stream attention (no cross-arm). Reference behavior.
            Q = torch.cat([left_Q, right_Q], dim=1)
            K = torch.cat([left_K, right_K], dim=1)
            V = torch.cat([left_V, right_V], dim=1)
            Q = apply_rope(Q, position_ids[left_mask | right_mask].view(B, -1))
            K = apply_rope(K, position_ids[left_mask | right_mask].view(B, -1))
            half = Q.shape[1] // 2
            # Left only
            attn_l = eager_twin_attention(
                Q[:, :half], K[:, :half], V[:, :half],
                attention_mask=attention_mask_4d[:, :, :half, :half],
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=None, use_reweighting=False,
            )
            # Right only
            attn_r = eager_twin_attention(
                Q[:, half:], K[:, half:], V[:, half:],
                attention_mask=attention_mask_4d[:, :, :half, :half],
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=None, use_reweighting=False,
            )
            attn_out_full = torch.cat([attn_l, attn_r], dim=1)

        # ----------- 4) Output projection ----------- #
        # split attn_out_full back along ci_ids
        if enable_joint_attn and enable_moe:
            # attn_out_full ordering: [common | left | right]
            n_c = common_mask[0].sum().item()
            n_l = left_mask[0].sum().item()
            common_attn = attn_out_full[:, :n_c]
            left_attn = attn_out_full[:, n_c:n_c + n_l]
            right_attn = attn_out_full[:, n_c + n_l:]
            common_attn = fuse_linear(l_attn.o_proj, r_attn.o_proj, common_attn)
            left_attn = l_attn.o_proj(left_attn)
            right_attn = r_attn.o_proj(right_attn)
            # Reassemble into full sequence (ci_ids order, NOT cat order)
            new_attn_out = torch.empty_like(hidden_states)
            new_attn_out[common_mask] = common_attn.reshape(-1, common_attn.shape[-1]).to(new_attn_out.dtype)
            new_attn_out[left_mask] = left_attn.reshape(-1, left_attn.shape[-1]).to(new_attn_out.dtype)
            new_attn_out[right_mask] = right_attn.reshape(-1, right_attn.shape[-1]).to(new_attn_out.dtype)
        elif enable_joint_attn and not enable_moe:
            # attn_out_full ordering: [left | right]
            n_l = left_mask[0].sum().item()
            left_attn = l_attn.o_proj(attn_out_full[:, :n_l])
            right_attn = r_attn.o_proj(attn_out_full[:, n_l:])
            new_attn_out = torch.empty_like(hidden_states)
            new_attn_out[left_mask] = left_attn.reshape(-1, left_attn.shape[-1]).to(new_attn_out.dtype)
            new_attn_out[right_mask] = right_attn.reshape(-1, right_attn.shape[-1]).to(new_attn_out.dtype)
        else:
            # not enable_joint_attn case (already half-split)
            half = attn_out_full.shape[1] // 2
            left_attn = l_attn.o_proj(attn_out_full[:, :half])
            right_attn = r_attn.o_proj(attn_out_full[:, half:])
            new_attn_out = torch.empty_like(hidden_states)
            new_attn_out[left_mask] = left_attn.reshape(-1, left_attn.shape[-1]).to(new_attn_out.dtype)
            new_attn_out[right_mask] = right_attn.reshape(-1, right_attn.shape[-1]).to(new_attn_out.dtype)

        hidden_states = residual + new_attn_out

        # ----------- 5) Post-attn layernorm + MLP ----------- #
        residual = hidden_states
        common_in, left_in, right_in = _split(hidden_states)
        if enable_moe:
            common_pln = fuse_layernorm(
                vlm_l_layers[i].post_attention_layernorm,
                vlm_r_layers[i].post_attention_layernorm,
                common_in,
            )
        else:
            common_pln = None
        left_pln = vlm_l_layers[i].post_attention_layernorm(left_in)
        right_pln = vlm_r_layers[i].post_attention_layernorm(right_in)

        if enable_moe:
            common_out = moe_mlp_forward(
                moe_gates[i],
                vlm_l_layers[i].mlp,
                vlm_r_layers[i].mlp,
                common_pln,
            )
        else:
            common_out = None
        left_out = vlm_l_layers[i].mlp(left_pln)
        right_out = vlm_r_layers[i].mlp(right_pln)

        mlp_out = torch.empty_like(hidden_states)
        if enable_moe and common_out is not None:
            mlp_out[common_mask] = common_out.reshape(-1, common_out.shape[-1]).to(mlp_out.dtype)
        mlp_out[left_mask] = left_out.reshape(-1, left_out.shape[-1]).to(mlp_out.dtype)
        mlp_out[right_mask] = right_out.reshape(-1, right_out.shape[-1]).to(mlp_out.dtype)

        hidden_states = residual + mlp_out

    hidden_states = vlm_norm(hidden_states)
    return hidden_states


# --------------------------------------------------------------------------- #
#  Phase 3: integrated twin VLM + shared action expert forward
# --------------------------------------------------------------------------- #


def _expanded_combined_mask(
    twin_vlm_mask_4d: Tensor,
    pad_mask_exp: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build a [B, 1, T_total, T_total] additive attention mask combining
    the VLM twin mask with expert attention rules:
      - VLM can attend only to VLM (twin mask as is)
      - VLM cannot attend to Expert (masked)
      - Expert can attend to all real VLM tokens (subject to pad)
      - Expert can attend to all Expert tokens (bidirectional within suffix)
    """
    B, _, T_vlm, _ = twin_vlm_mask_4d.shape
    T_exp = pad_mask_exp.shape[1]
    T_total = T_vlm + T_exp
    device = twin_vlm_mask_4d.device
    neg_inf = torch.finfo(dtype).min

    combined = torch.full((B, 1, T_total, T_total), neg_inf, dtype=dtype, device=device)
    combined[:, :, :T_vlm, :T_vlm] = twin_vlm_mask_4d

    # Expert -> VLM: allow attending to real (non-pad) VLM positions
    # We have to recover the VLM pad mask from twin_vlm_mask_4d. Instead of
    # decoding it back, derive it directly: a column j is "allowed" by the
    # twin mask in ANY row (i.e., not entirely masked) iff that column is
    # a real (non-pad) token. We use this:
    col_is_real = (twin_vlm_mask_4d[:, 0] > neg_inf).any(dim=1)  # [B, T_vlm]
    allow_e2v = col_is_real.unsqueeze(1).expand(-1, T_exp, -1)  # [B, T_exp, T_vlm]
    combined[:, 0, T_vlm:, :T_vlm] = torch.where(
        allow_e2v, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )

    # Expert -> Expert: bidirectional within real tokens
    exp_real = pad_mask_exp.bool()  # [B, T_exp]
    e2e = exp_real.unsqueeze(2) & exp_real.unsqueeze(1)  # [B, T_exp, T_exp]
    combined[:, 0, T_vlm:, T_vlm:] = torch.where(
        e2e, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )
    return combined


def _build_combined_position_ids(ci_ids: Tensor, T_exp: int) -> Tensor:
    """Combined positions: [vlm_positions | expert_positions starting from max+1]"""
    vlm_positions = create_twin_position_ids(ci_ids)
    max_vlm = vlm_positions.max(dim=1, keepdim=True).values + 1  # [B, 1]
    expert_positions = max_vlm + torch.arange(T_exp, device=ci_ids.device)[None, :]
    return torch.cat([vlm_positions, expert_positions], dim=1)  # [B, T_vlm + T_exp]


def _twin_qkv(
    layer_l: nn.Module,
    layer_r: nn.Module,
    common_ln: Tensor,
    left_ln: Tensor,
    right_ln: Tensor,
    B: int,
    n_attn: int,
    n_kv: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute twin VLM Q/K/V over [common, left, right] in concat order.

    Returns:
      Q [B, n_c + 2*n_arm, n_attn, head_dim]
      K [B, n_c + 2*n_arm, n_kv,   head_dim]
      V [B, n_c + 2*n_arm, n_kv,   head_dim]
    """
    l_attn = layer_l.self_attn
    r_attn = layer_r.self_attn

    def _h(x: Tensor, n: int) -> Tensor:
        return x.view(B, -1, n, head_dim)

    cQ = _h(fuse_linear(l_attn.q_proj, r_attn.q_proj, common_ln), n_attn)
    cK = _h(fuse_linear(l_attn.k_proj, r_attn.k_proj, common_ln), n_kv)
    cV = _h(fuse_linear(l_attn.v_proj, r_attn.v_proj, common_ln), n_kv)
    lQ = _h(_safe_proj(l_attn.q_proj, left_ln), n_attn)
    lK = _h(_safe_proj(l_attn.k_proj, left_ln), n_kv)
    lV = _h(_safe_proj(l_attn.v_proj, left_ln), n_kv)
    rQ = _h(_safe_proj(r_attn.q_proj, right_ln), n_attn)
    rK = _h(_safe_proj(r_attn.k_proj, right_ln), n_kv)
    rV = _h(_safe_proj(r_attn.v_proj, right_ln), n_kv)

    return (
        torch.cat([cQ, lQ, rQ], dim=1),
        torch.cat([cK, lK, rK], dim=1),
        torch.cat([cV, lV, rV], dim=1),
    )


def _apply_twin_o_proj(
    attn_out_vlm: Tensor,         # [B, n_c + 2*n_arm, n_attn * head_dim] in [c, l, r] order
    layer_l: nn.Module,
    layer_r: nn.Module,
    n_c: int,
    n_l: int,
    common_mask: Tensor,
    left_mask: Tensor,
    right_mask: Tensor,
    hidden_vlm: Tensor,
) -> Tensor:
    """Apply o_proj per ci segment, reassembled into ci_ids order.

    eager_twin_attention may return fp32 (it upcasts internally); cast to
    o_proj weight dtype here so the linear projection works for bf16 weights.
    """
    o_dtype = layer_l.self_attn.o_proj.weight.dtype
    if attn_out_vlm.dtype != o_dtype:
        attn_out_vlm = attn_out_vlm.to(dtype=o_dtype)

    c = attn_out_vlm[:, :n_c]
    l = attn_out_vlm[:, n_c : n_c + n_l]
    r = attn_out_vlm[:, n_c + n_l :]
    c_o = fuse_linear(layer_l.self_attn.o_proj, layer_r.self_attn.o_proj, c)
    l_o = layer_l.self_attn.o_proj(l)
    r_o = layer_r.self_attn.o_proj(r)
    out = torch.empty_like(hidden_vlm)
    out[common_mask] = c_o.reshape(-1, c_o.shape[-1]).to(out.dtype)
    out[left_mask] = l_o.reshape(-1, l_o.shape[-1]).to(out.dtype)
    out[right_mask] = r_o.reshape(-1, r_o.shape[-1]).to(out.dtype)
    return out


def _apply_twin_mlp(
    hidden_vlm: Tensor,
    common_pln: Tensor,
    left_pln: Tensor,
    right_pln: Tensor,
    layer_l: nn.Module,
    layer_r: nn.Module,
    moe_gate: nn.Linear,
    common_mask: Tensor,
    left_mask: Tensor,
    right_mask: Tensor,
) -> Tensor:
    """Apply per-ci MLP path with MoE on common."""
    # SmolVLM2 MLP uses gate_proj/up_proj/down_proj nn.Linear chains; ensure input dtype
    # matches the first projection weight to handle mixed precision.
    mlp_dtype = layer_l.mlp.gate_proj.weight.dtype
    if left_pln.dtype != mlp_dtype:
        left_pln = left_pln.to(dtype=mlp_dtype)
    if right_pln.dtype != mlp_dtype:
        right_pln = right_pln.to(dtype=mlp_dtype)
    # common_pln is handled inside moe_mlp_forward via gate dtype alignment;
    # add an explicit cast here in case mlp_l/mlp_r weight dtype differs from gate.
    if common_pln.dtype != mlp_dtype:
        common_pln_for_mlp = common_pln.to(dtype=mlp_dtype)
    else:
        common_pln_for_mlp = common_pln

    c_out = moe_mlp_forward(moe_gate, layer_l.mlp, layer_r.mlp, common_pln_for_mlp)
    l_out = layer_l.mlp(left_pln)
    r_out = layer_r.mlp(right_pln)
    out = torch.empty_like(hidden_vlm)
    out[common_mask] = c_out.reshape(-1, c_out.shape[-1]).to(out.dtype)
    out[left_mask] = l_out.reshape(-1, l_out.shape[-1]).to(out.dtype)
    out[right_mask] = r_out.reshape(-1, r_out.shape[-1]).to(out.dtype)
    return out


def twin_vlm_expert_forward(
    vlm_inputs_embeds: Tensor,       # [B, T_vlm, vlm_hidden]
    vlm_pad_mask: Tensor,            # [B, T_vlm]
    ci_ids: Tensor,                  # [B, T_vlm] in {0,1,2}, non-decreasing
    expert_inputs_embeds: Tensor,    # [B, T_exp, expert_hidden]
    expert_pad_mask: Tensor,         # [B, T_exp]
    vlm_l_layers: nn.ModuleList,
    vlm_r_layers: nn.ModuleList,
    vlm_norm: nn.Module,
    lm_expert_layers: nn.ModuleList,
    expert_norm: nn.Module,
    moe_gates: nn.ModuleList,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    self_attn_every_n_layers: int,
    enable_moe: bool = True,
    enable_joint_attn: bool = True,
    attn_reweighting: bool = True,
    reweight_scale: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Integrated bimanual VLM + shared expert forward.

    Reuses the per-layer pattern from `twin_vlm_forward` for the VLM stream
    and interleaves the action expert as in
    `SmolVLMWithExpertModel.forward_attn_layer / forward_cross_attn_layer`:

      * On layer indices where `i % self_attn_every_n_layers == 0`:
          joint self-attention over `[common | left | right | expert]`.
          Expert Q/K/V come from its own `lm_expert.layers[i].self_attn` projections
          (input dim = expert_hidden=720).
      * On other layer indices:
          twin VLM self-attention as in Phase 2, then expert cross-attends into a
          re-projection of the twin VLM K/V (after RoPE) through the expert's
          `k_proj`/`v_proj` (which were reshaped to input dim = num_kv * head_dim = 320).

    Returns (vlm_outputs, expert_outputs) after the final norms.
    """
    if not (enable_moe and enable_joint_attn):
        raise NotImplementedError(
            "twin_vlm_expert_forward requires enable_moe=True and enable_joint_attn=True. "
            "Ablation paths are not supported (see twin_vlm_forward for the Phase 2 detail)."
        )
    assert len(vlm_l_layers) == len(vlm_r_layers) == len(lm_expert_layers)
    assert moe_gates is not None and len(moe_gates) == len(vlm_l_layers)
    num_layers = len(vlm_l_layers)

    B, T_vlm, vlm_hidden = vlm_inputs_embeds.shape
    _, T_exp, expert_hidden = expert_inputs_embeds.shape
    device = vlm_inputs_embeds.device
    dtype = vlm_inputs_embeds.dtype

    common_mask = ci_ids == CI_COMMON
    left_mask = ci_ids == CI_LEFT
    right_mask = ci_ids == CI_RIGHT
    n_c = int(common_mask[0].sum().item())
    n_l = int(left_mask[0].sum().item())
    n_r = int(right_mask[0].sum().item())
    assert n_l == n_r, f"left/right token counts must match (got {n_l} vs {n_r})"

    # Combined positions and masks
    combined_positions = _build_combined_position_ids(ci_ids, T_exp)
    vlm_positions = combined_positions[:, :T_vlm]
    expert_positions = combined_positions[:, T_vlm:]

    twin_mask_vlm = create_twin_4d_attention_mask(vlm_pad_mask, ci_ids, dtype=dtype)
    combined_mask = _expanded_combined_mask(twin_mask_vlm, expert_pad_mask, dtype)

    # ci-ids extended: expert tokens marked with 3 (treated as "non-common" by reweighting,
    # so they participate in the shared-modality boost like left/right tokens do)
    ci_ext = torch.cat(
        [ci_ids, torch.full((B, T_exp), 3, dtype=ci_ids.dtype, device=device)], dim=1
    )

    # Position vectors in concat order for VLM (already ci_ids order since
    # caller is required to build prefix as [common, left, right])
    c_pos = vlm_positions[common_mask].view(B, n_c)
    l_pos = vlm_positions[left_mask].view(B, n_l)
    r_pos = vlm_positions[right_mask].view(B, n_l)
    vlm_concat_pos = torch.cat([c_pos, l_pos, r_pos], dim=1)

    hidden_vlm = vlm_inputs_embeds
    hidden_exp = expert_inputs_embeds

    for i in range(num_layers):
        residual_vlm = hidden_vlm
        residual_exp = hidden_exp
        l_layer = vlm_l_layers[i]
        r_layer = vlm_r_layers[i]
        e_layer = lm_expert_layers[i]

        # ----- input layernorm -----
        c_in = hidden_vlm[common_mask].view(B, n_c, vlm_hidden)
        l_in = hidden_vlm[left_mask].view(B, n_l, vlm_hidden)
        r_in = hidden_vlm[right_mask].view(B, n_l, vlm_hidden)
        common_ln = fuse_layernorm(l_layer.input_layernorm, r_layer.input_layernorm, c_in)
        left_ln = l_layer.input_layernorm(l_in)
        right_ln = r_layer.input_layernorm(r_in)
        exp_ln = e_layer.input_layernorm(hidden_exp)

        # ----- twin VLM Q/K/V (computed every layer; needed both paths) -----
        vQ, vK, vV = _twin_qkv(
            l_layer, r_layer, common_ln, left_ln, right_ln,
            B, num_attention_heads, num_key_value_heads, head_dim,
        )

        joint_layer = (i % self_attn_every_n_layers == 0)

        if joint_layer:
            # Expert Q/K/V via lm_expert.layers[i] (k_proj/v_proj have input dim
            # = expert_hidden because joint layers are excluded from the reshape).
            e_attn = e_layer.self_attn
            eQ = _safe_proj(e_attn.q_proj, exp_ln).view(B, T_exp, num_attention_heads, head_dim)
            eK = _safe_proj(e_attn.k_proj, exp_ln).view(B, T_exp, num_key_value_heads, head_dim)
            eV = _safe_proj(e_attn.v_proj, exp_ln).view(B, T_exp, num_key_value_heads, head_dim)

            Q_full = torch.cat([vQ, eQ], dim=1)
            K_full = torch.cat([vK, eK], dim=1)
            V_full = torch.cat([vV, eV], dim=1)

            # RoPE with combined positions
            concat_pos = torch.cat([vlm_concat_pos, expert_positions], dim=1)
            Q_full = apply_rope(Q_full, concat_pos)
            K_full = apply_rope(K_full, concat_pos)

            attn_out = eager_twin_attention(
                query=Q_full,
                key=K_full,
                value=V_full,
                attention_mask=combined_mask,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=ci_ext,
                use_reweighting=attn_reweighting,
                reweight_scale=reweight_scale,
            )  # [B, T_total, n_attn * head_dim] = [B, T_total, 960]

            vlm_attn_out = attn_out[:, :T_vlm]   # in [c, l, r] order = ci_ids order
            exp_attn_out = attn_out[:, T_vlm:]

            attn_vlm = _apply_twin_o_proj(
                vlm_attn_out, l_layer, r_layer, n_c, n_l,
                common_mask, left_mask, right_mask, hidden_vlm,
            )
            eo_dtype = e_layer.self_attn.o_proj.weight.dtype
            if exp_attn_out.dtype != eo_dtype:
                exp_attn_out = exp_attn_out.to(dtype=eo_dtype)
            attn_exp = e_layer.self_attn.o_proj(exp_attn_out)

        else:
            # ----- VLM-only twin self-attention -----
            Q_full = apply_rope(vQ, vlm_concat_pos)
            K_full = apply_rope(vK, vlm_concat_pos)
            V_full = vV

            attn_out_vlm = eager_twin_attention(
                query=Q_full,
                key=K_full,
                value=V_full,
                attention_mask=twin_mask_vlm,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=ci_ids,
                use_reweighting=attn_reweighting,
                reweight_scale=reweight_scale,
            )  # [B, T_vlm, n_attn * head_dim]

            attn_vlm = _apply_twin_o_proj(
                attn_out_vlm, l_layer, r_layer, n_c, n_l,
                common_mask, left_mask, right_mask, hidden_vlm,
            )

            # ----- Expert cross-attention into twin VLM K/V -----
            # Take post-RoPE K, V from twin attention (length T_vlm in concat order
            # = ci_ids order). Flatten heads and re-project through expert's
            # k_proj/v_proj (whose input dim was reshaped to 320 = n_kv * head_dim).
            K_flat = K_full.reshape(B, T_vlm, num_key_value_heads * head_dim).to(
                dtype=e_layer.self_attn.k_proj.weight.dtype
            )
            V_flat = V_full.reshape(B, T_vlm, num_key_value_heads * head_dim).to(
                dtype=e_layer.self_attn.v_proj.weight.dtype
            )

            e_attn = e_layer.self_attn
            expert_K = _safe_proj(e_attn.k_proj, K_flat).view(B, T_vlm, num_key_value_heads, head_dim)
            expert_V = _safe_proj(e_attn.v_proj, V_flat).view(B, T_vlm, num_key_value_heads, head_dim)

            # Expert query has its own RoPE in expert position space (relative
            # to 0, matching SmolVLA's behavior where expert_position_id is
            # remapped so the smallest expert position becomes 0).
            exp_pos_norm = expert_positions - expert_positions.min(dim=1, keepdim=True).values
            expert_Q = _safe_proj(e_attn.q_proj, exp_ln).view(B, T_exp, num_attention_heads, head_dim)
            expert_Q = apply_rope(expert_Q, exp_pos_norm)

            # Cross-attention mask: expert (rows T_exp) attends to VLM (cols T_vlm)
            cross_mask = combined_mask[:, :, T_vlm:, :T_vlm]

            cross_out = eager_twin_attention(
                query=expert_Q,
                key=expert_K,
                value=expert_V,
                attention_mask=cross_mask,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                ci_ids=None,
                use_reweighting=False,
            )  # [B, T_exp, n_attn * head_dim]

            eo_dtype = e_attn.o_proj.weight.dtype
            if cross_out.dtype != eo_dtype:
                cross_out = cross_out.to(dtype=eo_dtype)
            attn_exp = e_attn.o_proj(cross_out)  # -> [B, T_exp, expert_hidden]

        # ----- residual after attention -----
        hidden_vlm = residual_vlm + attn_vlm
        hidden_exp = residual_exp + attn_exp

        # ----- post-attention layernorm + MLP (per ci for VLM, plain for expert) -----
        residual_vlm = hidden_vlm
        residual_exp = hidden_exp

        c_in = hidden_vlm[common_mask].view(B, n_c, vlm_hidden)
        l_in = hidden_vlm[left_mask].view(B, n_l, vlm_hidden)
        r_in = hidden_vlm[right_mask].view(B, n_l, vlm_hidden)
        common_pln = fuse_layernorm(l_layer.post_attention_layernorm, r_layer.post_attention_layernorm, c_in)
        left_pln = l_layer.post_attention_layernorm(l_in)
        right_pln = r_layer.post_attention_layernorm(r_in)

        mlp_vlm = _apply_twin_mlp(
            hidden_vlm, common_pln, left_pln, right_pln,
            l_layer, r_layer, moe_gates[i],
            common_mask, left_mask, right_mask,
        )

        exp_pln = e_layer.post_attention_layernorm(hidden_exp)
        emlp_dtype = e_layer.mlp.gate_proj.weight.dtype
        if exp_pln.dtype != emlp_dtype:
            exp_pln = exp_pln.to(dtype=emlp_dtype)
        mlp_exp = e_layer.mlp(exp_pln)

        hidden_vlm = residual_vlm + mlp_vlm
        hidden_exp = residual_exp + mlp_exp

    hidden_vlm = vlm_norm(hidden_vlm)
    hidden_exp = expert_norm(hidden_exp)
    return hidden_vlm, hidden_exp
