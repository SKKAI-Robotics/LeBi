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

"""SmolVLA-Twin model.

Architecture summary (design C-1):

    Inputs (bimanual SO-101, 3 cameras):
        primary_image     ── shared VLM vision encoder
        left_wrist_image  ┘
        right_wrist_image ┘
        proprio_l, proprio_r → state_proj_l, state_proj_r → VLM input
        language          → shared embed_tokens

    VLM stream (16 layers):
        vlm_l.layers[i], vlm_r.layers[i] applied per layer with:
            ci=0 common tokens : fuse(L, R) avg + optional MoE on MLP
            ci=1 left tokens   : vlm_l only
            ci=2 right tokens  : vlm_r only
            + joint attention with cross-modal causal band (ci=1 <-> ci=2)
            + attention re-weighting on shared-modality keys

    Expert stream (16 layers, shared = single lm_expert):
        cross-attends into concat(vlm_l K/V, vlm_r K/V) for the action chunk.
        Action chunk is (left_action_chunk + right_action_chunk) concatenated.

    Output:
        Flow matching velocity prediction → action_out_proj → per-arm action chunks.

This file currently provides the skeleton + surgery routine + a placeholder
forward that delegates to the upstream SmolVLA single-arm flow for shape sanity.
The twin attention / cross-attn integration is implemented in subsequent
phases (see TODO markers tagged PHASE-2 and PHASE-3).
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    pad_vector,
    resize_with_pad,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolvla_twin.configuration_smolvla_twin import SmolVLATwinConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants / helpers
# --------------------------------------------------------------------------- #

# ci (chunk-id) semantics, mirroring the TwinVLA paper:
CI_COMMON = 0
CI_LEFT = 1
CI_RIGHT = 2


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------- #
#  Twin VLM with shared expert  (PHASE-1c scaffold; PHASE-2/3 fills attention)
# --------------------------------------------------------------------------- #


class SmolVLATwinBackbone(nn.Module):
    """Bimanual SmolVLM2 backbone with a shared action expert.

    Builds two duplicate copies of the SmolVLM2 text-model decoder layers
    (vlm_l / vlm_r) while keeping a single shared:
      - vision encoder (vlm_shared.vision_model + connector)
      - input embedding table (vlm_shared.text_model.embed_tokens)
      - final norm (vlm_shared.text_model.norm)
      - action expert (lm_expert) plus flow-matching projections
    """

    def __init__(self, config: SmolVLATwinConfig):
        super().__init__()
        self.config = config

        # We build the SingleVLA wrapper first so we get all the projection
        # modules (state_proj, action_in_proj, action_out_proj, time MLPs) and
        # the vlm_with_expert (vision + text + expert) wired up in their
        # SmolVLA-canonical state. Then we perform the surgery.
        self.flow_matching = VLAFlowMatching(self._build_smolvla_subconfig(config))

        # Now perform the twin surgery in-place on the flow_matching's
        # vlm_with_expert. After this call, vlm_with_expert.vlm.model.text_model.layers
        # is replaced by a structure that holds vlm_l_layers and vlm_r_layers.
        self._apply_twin_surgery()

    # ------------------------------------------------------------------ #
    #  Build a SmolVLA-compatible sub-config from our twin config
    # ------------------------------------------------------------------ #
    def _build_smolvla_subconfig(self, twin: SmolVLATwinConfig):
        """Construct a SmolVLAConfig that VLAFlowMatching can consume.

        We deliberately keep max_state_dim / max_action_dim equal to the twin
        config's values so the projection layers accept zero-padded bimanual
        vectors (length 2 * action_dim, zero-padded to max_action_dim).
        """
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        sub = SmolVLAConfig(
            n_obs_steps=twin.n_obs_steps,
            chunk_size=twin.chunk_size,
            n_action_steps=twin.n_action_steps,
            max_state_dim=twin.max_state_dim,
            max_action_dim=twin.max_action_dim,
            resize_imgs_with_padding=twin.resize_imgs_with_padding,
            empty_cameras=twin.empty_cameras,
            tokenizer_max_length=twin.tokenizer_max_length,
            num_steps=twin.num_steps,
            freeze_vision_encoder=twin.freeze_vision_encoder,
            # NOTE: we always need expert params to be trainable for twin coordination.
            train_expert_only=twin.train_expert_only,
            train_state_proj=twin.train_state_proj,
            vlm_model_name=twin.vlm_model_name,
            load_vlm_weights=twin.load_vlm_weights,
            add_image_special_tokens=twin.add_image_special_tokens,
            attention_mode=twin.attention_mode,
            prefix_length=twin.prefix_length,
            pad_language_to=twin.pad_language_to,
            num_expert_layers=twin.num_expert_layers,
            num_vlm_layers=twin.num_vlm_layers,
            self_attn_every_n_layers=twin.self_attn_every_n_layers,
            expert_width_multiplier=twin.expert_width_multiplier,
            min_period=twin.min_period,
            max_period=twin.max_period,
        )
        # input/output features so SmolVLAConfig.validate_features doesn't choke
        sub.input_features = {
            OBS_STATE: type(
                "F", (), {"type": None, "shape": (twin.max_state_dim,)}
            )(),  # placeholder shape; SmolVLAConfig only reads action_feature for padding
        }
        sub.output_features = {
            ACTION: type("F", (), {"type": None, "shape": (twin.max_action_dim,)})(),
        }
        return sub

    # ------------------------------------------------------------------ #
    #  Surgery: duplicate text_model.layers into vlm_l / vlm_r
    # ------------------------------------------------------------------ #
    def _apply_twin_surgery(self):
        """Replace vlm_with_expert with twin-aware version.

        This is intentionally minimal in Phase 1: we deepcopy text_model.layers
        into two parallel ModuleLists (`vlm_l_layers`, `vlm_r_layers`) attached
        directly on this module. The original `flow_matching.vlm_with_expert`
        still owns the shared pieces (vision_model, connector, embed_tokens,
        norm, lm_expert). Phase 2 will override the forward to use these
        per-arm layers.
        """
        vlm_with_expert: SmolVLMWithExpertModel = self.flow_matching.vlm_with_expert
        text_model = vlm_with_expert.get_vlm_model().text_model
        original_layers = text_model.layers

        # Sanity: SmolVLA's surgery contract assumes a ModuleList of decoder layers.
        if not isinstance(original_layers, nn.ModuleList):
            raise RuntimeError(
                f"Expected text_model.layers to be ModuleList, got {type(original_layers)}"
            )

        # Phase 1: simply hold two deep copies. Both start from the same weights.
        # The original layers ModuleList is kept in place to preserve forward-compat
        # with the upstream SmolVLA forward path (used as a fallback in Phase 1d).
        self.vlm_l_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in original_layers]
        )
        self.vlm_r_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in original_layers]
        )

        # Per-arm state projector (we duplicate the state_proj from flow_matching
        # so each arm has its own state encoder while sharing the action expert)
        if not self.config.share_state_proj:
            shared_state_proj = self.flow_matching.state_proj
            self.state_proj_l = copy.deepcopy(shared_state_proj)
            self.state_proj_r = copy.deepcopy(shared_state_proj)
        else:
            self.state_proj_l = self.flow_matching.state_proj
            self.state_proj_r = self.flow_matching.state_proj

        # MoE gates (one per VLM layer; gates select between vlm_l / vlm_r MLP
        # for common (ci=0) tokens). Allocated only if MoE is enabled.
        vlm_hidden = vlm_with_expert.config.text_config.hidden_size
        if self.config.enable_moe:
            vlm_dtype = original_layers[0].self_attn.q_proj.weight.dtype
            self.moe_gates = nn.ModuleList(
                [nn.Linear(vlm_hidden, 2).to(dtype=vlm_dtype) for _ in range(len(original_layers))]
            )
        else:
            self.moe_gates = None

        # Cast our newly added per-arm state projectors to VLM dtype as well, so
        # mixed-precision plumbing stays consistent across surgery additions.
        if not self.config.share_state_proj:
            vlm_dtype = original_layers[0].self_attn.q_proj.weight.dtype
            self.state_proj_l.to(dtype=vlm_dtype)
            self.state_proj_r.to(dtype=vlm_dtype)

        # Note: we leave `original_layers` in place. Phase 2 will route the
        # forward call through self.vlm_l_layers / self.vlm_r_layers and only
        # touch text_model.layers for the shared norm / embedding lookups.

        logger.info(
            "SmolVLATwinBackbone surgery done: %d twin layers, MoE=%s, joint_attn=%s, reweight=%s",
            len(self.vlm_l_layers),
            self.config.enable_moe,
            self.config.enable_joint_attn,
            self.config.attn_reweighting,
        )

    # ------------------------------------------------------------------ #
    #  Load `lerobot/smolvla_base` checkpoint and re-apply surgery
    # ------------------------------------------------------------------ #

    def load_from_smolvla_policy(self, smolvla_policy: SmolVLAPolicy) -> None:
        """Copy weights from an already-loaded `SmolVLAPolicy` into this backbone.

        Mapping rules:
          - `model.vlm_with_expert.*`                           → `flow_matching.vlm_with_expert.*` (1:1)
          - `model.vlm_with_expert.vlm.model.text_model.layers` → also duplicated into
                                                                   `vlm_l_layers` and `vlm_r_layers`
          - `model.state_proj`                                  → `state_proj_l` and `state_proj_r`
          - other projections (action_in/out_proj, action_time_mlp_*)  → 1:1 copy

        After this call, the backbone holds:
          - identical VLM weights in both arms (will diverge under training)
          - identical state_proj weights in both arms (will diverge under training)
          - shared action expert + projections (= single set of params, MoE gates random init)
        """
        if not isinstance(smolvla_policy, SmolVLAPolicy):
            raise TypeError(f"Expected SmolVLAPolicy, got {type(smolvla_policy).__name__}")

        # 1) Copy the full flow_matching (vlm_with_expert + projections) by
        #    state_dict assignment. Both flow_matching instances should have
        #    identical module structure (we built ours via VLAFlowMatching with
        #    a matching SmolVLAConfig in _build_smolvla_subconfig).
        src_flow = smolvla_policy.model
        dst_flow = self.flow_matching
        missing, unexpected = dst_flow.load_state_dict(src_flow.state_dict(), strict=False)
        if missing:
            logger.warning("load_from_smolvla_policy: missing keys in dst_flow (first 10): %s", missing[:10])
        if unexpected:
            logger.warning("load_from_smolvla_policy: unexpected keys in dst_flow (first 10): %s", unexpected[:10])

        # 2) Re-create vlm_l_layers / vlm_r_layers as deep copies of the now-loaded
        #    text_model.layers (so they start from pretrained weights, not random init).
        original_layers = dst_flow.vlm_with_expert.get_vlm_model().text_model.layers
        new_l = nn.ModuleList([copy.deepcopy(layer) for layer in original_layers])
        new_r = nn.ModuleList([copy.deepcopy(layer) for layer in original_layers])
        # Replace in-place to keep parameter ownership consistent with optimizers
        # constructed before this call.
        self.vlm_l_layers.load_state_dict(new_l.state_dict())
        self.vlm_r_layers.load_state_dict(new_r.state_dict())

        # 3) Re-copy state_proj into state_proj_l / state_proj_r.
        if not self.config.share_state_proj:
            self.state_proj_l.load_state_dict(dst_flow.state_proj.state_dict())
            self.state_proj_r.load_state_dict(dst_flow.state_proj.state_dict())

        logger.info(
            "SmolVLATwinBackbone weights initialized from SmolVLA policy "
            "(vlm_l/vlm_r and state_proj_l/state_proj_r duplicated; MoE gates left at random init)."
        )

    # ------------------------------------------------------------------ #
    #  Twin VLM forward (Phase 2 implemented; Phase 3 wires expert)
    # ------------------------------------------------------------------ #

    def twin_vlm_forward(
        self,
        inputs_embeds: torch.Tensor,
        pad_mask: torch.Tensor,
        ci_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the bimanual VLM-only forward (no expert).

        Phase 2 helper, kept for unit-testing the VLM stream in isolation.
        For full training/inference use `twin_vlm_expert_forward`.
        """
        from lerobot.policies.smolvla_twin.twin_attention import twin_vlm_forward as _twin_fwd

        vlm_with_expert = self.flow_matching.vlm_with_expert
        text_model = vlm_with_expert.get_vlm_model().text_model

        return _twin_fwd(
            inputs_embeds=inputs_embeds,
            pad_mask=pad_mask,
            ci_ids=ci_ids,
            vlm_l_layers=self.vlm_l_layers,
            vlm_r_layers=self.vlm_r_layers,
            vlm_norm=text_model.norm,
            moe_gates=self.moe_gates,
            num_attention_heads=vlm_with_expert.num_attention_heads,
            num_key_value_heads=vlm_with_expert.num_key_value_heads,
            head_dim=vlm_with_expert.config.text_config.head_dim,
            enable_moe=self.config.enable_moe,
            enable_joint_attn=self.config.enable_joint_attn,
            attn_reweighting=self.config.attn_reweighting,
            reweight_scale=self.config.attn_reweighting_scale,
        )

    def twin_vlm_expert_forward(
        self,
        vlm_inputs_embeds: torch.Tensor,
        vlm_pad_mask: torch.Tensor,
        ci_ids: torch.Tensor,
        expert_inputs_embeds: torch.Tensor,
        expert_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrated bimanual VLM + shared action expert forward.

        Returns (vlm_outputs, expert_outputs).
        """
        from lerobot.policies.smolvla_twin.twin_attention import (
            twin_vlm_expert_forward as _twin_exp_fwd,
        )

        vlm_with_expert = self.flow_matching.vlm_with_expert
        text_model = vlm_with_expert.get_vlm_model().text_model

        return _twin_exp_fwd(
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_pad_mask=vlm_pad_mask,
            ci_ids=ci_ids,
            expert_inputs_embeds=expert_inputs_embeds,
            expert_pad_mask=expert_pad_mask,
            vlm_l_layers=self.vlm_l_layers,
            vlm_r_layers=self.vlm_r_layers,
            vlm_norm=text_model.norm,
            lm_expert_layers=vlm_with_expert.lm_expert.layers,
            expert_norm=vlm_with_expert.lm_expert.norm,
            moe_gates=self.moe_gates,
            num_attention_heads=vlm_with_expert.num_attention_heads,
            num_key_value_heads=vlm_with_expert.num_key_value_heads,
            head_dim=vlm_with_expert.config.text_config.head_dim,
            self_attn_every_n_layers=vlm_with_expert.self_attn_every_n_layers,
            enable_moe=self.config.enable_moe,
            enable_joint_attn=self.config.enable_joint_attn,
            attn_reweighting=self.config.attn_reweighting,
            reweight_scale=self.config.attn_reweighting_scale,
        )

    def forward(self, *args, **kwargs):
        # The top-level Policy.forward will assemble inputs_embeds / pad_mask /
        # ci_ids and call `twin_vlm_forward` directly (and in Phase 3 the
        # cross-attention with the action expert). We don't expose a generic
        # forward here.
        raise NotImplementedError(
            "Use SmolVLATwinBackbone.twin_vlm_forward(inputs_embeds, pad_mask, ci_ids)."
        )


# --------------------------------------------------------------------------- #
#  Policy wrapper
# --------------------------------------------------------------------------- #


class SmolVLATwinPolicy(PreTrainedPolicy):
    config_class = SmolVLATwinConfig
    name = "smolvla_twin"

    def __init__(self, config: SmolVLATwinConfig, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        self.backbone = SmolVLATwinBackbone(config)
        self.reset()

    def reset(self):
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    # ------------------------------------------------------------------ #
    #  Construct from a pretrained SmolVLA (lerobot/smolvla_base etc.)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_smolvla_base(
        cls,
        smolvla_path: str = "lerobot/smolvla_base",
        config: SmolVLATwinConfig | None = None,
        **smolvla_from_pretrained_kwargs,
    ) -> "SmolVLATwinPolicy":
        """Instantiate a SmolVLA-Twin policy seeded from an existing SmolVLA checkpoint.

        Args:
            smolvla_path: HF repo id or local path of the pretrained SmolVLA policy
                          (defaults to `lerobot/smolvla_base`).
            config: a `SmolVLATwinConfig`. If None, a default config is used --
                    the caller should override `action_dim`/`state_dim` to match
                    their target embodiment (default is 6, suitable for SO-101).
            **smolvla_from_pretrained_kwargs: forwarded to `SmolVLAPolicy.from_pretrained`.

        Behavior:
            1. Loads SmolVLA via `SmolVLAPolicy.from_pretrained`.
            2. Builds an empty `SmolVLATwinPolicy` (this triggers the twin surgery
               on a fresh, randomly-initialized SmolVLA shell with `load_vlm_weights=False`).
            3. Replaces all weights with the ones from the loaded SmolVLA policy
               (duplicating VLM and state_proj into the per-arm modules).
        """
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SmolVLAPolicy

        if config is None:
            config = SmolVLATwinConfig(
                load_vlm_weights=False,
                smolvla_pretrained_path=None,
            )
        else:
            # Avoid double-downloading the VLM. We'll get weights from the SmolVLA
            # checkpoint below.
            config = copy.copy(config)
            config.load_vlm_weights = False
            config.smolvla_pretrained_path = None

        logger.info("from_smolvla_base: loading %s ...", smolvla_path)
        smolvla = _SmolVLAPolicy.from_pretrained(smolvla_path, **smolvla_from_pretrained_kwargs)

        logger.info("from_smolvla_base: building twin shell ...")
        twin = cls(config)

        logger.info("from_smolvla_base: copying weights into twin shell ...")
        twin.backbone.load_from_smolvla_policy(smolvla)

        return twin

    # ------------------------------------------------------------------ #
    #  Convenience views into the backbone modules
    # ------------------------------------------------------------------ #

    @property
    def _flow(self) -> VLAFlowMatching:
        return self.backbone.flow_matching

    @property
    def _vlm_we(self) -> SmolVLMWithExpertModel:
        return self.backbone.flow_matching.vlm_with_expert

    # ------------------------------------------------------------------ #
    #  Input preparation
    # ------------------------------------------------------------------ #

    def _prepare_image(self, img: Tensor) -> Tensor:
        """Apply SmolVLA's standard image preprocessing (resize+pad + [-1,1])."""
        if img.ndim == 5:
            img = img[:, -1]  # take last obs step if a time-history window was passed
        if self.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
        img = img * 2.0 - 1.0
        return img

    def _get_camera_or_zeros(self, batch: dict[str, Tensor], key: str, ref_shape: Tensor) -> Tensor:
        """Fetch a camera from batch or fall back to a zero image (mask handled separately)."""
        if key in batch:
            return self._prepare_image(batch[key])
        # mask-aware fallback: -1 fill (SmolVLA conv: empty cams are -1)
        return torch.full_like(ref_shape, -1.0)

    def _prepare_per_arm_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Split the bimanual state into per-arm vectors padded to max_state_dim."""
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1]
        d = self.config.state_dim
        state_l = state[:, :d]
        state_r = state[:, d : 2 * d]
        return pad_vector(state_l, self.config.max_state_dim), pad_vector(
            state_r, self.config.max_state_dim
        )

    def _prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Zero-pad the bimanual action chunk to max_action_dim."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # ------------------------------------------------------------------ #
    #  Prefix embedding (ci_ids = [common | left | right])
    # ------------------------------------------------------------------ #

    def _embed_prefix_twin(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the bimanual VLM prefix in [common | left | right] order.

        Returns:
          vlm_embs: [B, T_vlm, vlm_hidden]
          vlm_pad : [B, T_vlm] bool
          ci_ids  : [B, T_vlm] in {0,1,2}, non-decreasing along T
        """
        cfg = self.config
        vlm_we = self._vlm_we
        device = next(self.parameters()).device
        vlm_dtype = self.backbone.vlm_l_layers[0].self_attn.q_proj.weight.dtype

        # Cameras -- assume all three are present; processor pipeline ensures
        # they were normalized (Identity for visual) and on device.
        primary_img = self._prepare_image(batch[cfg.primary_camera])
        left_wrist_img = self._prepare_image(batch[cfg.left_wrist_camera])
        right_wrist_img = self._prepare_image(batch[cfg.right_wrist_camera])

        # Embed each through the shared vision encoder + connector
        def _embed_img(img: Tensor) -> Tensor:
            emb = vlm_we.embed_image(img)  # [B, n_img, vlm_hidden]
            # SigLIP-style scale (matches SmolVLA's embed_prefix)
            d = emb.shape[-1]
            return emb * math.sqrt(d)

        primary_emb = _embed_img(primary_img)
        left_wrist_emb = _embed_img(left_wrist_img)
        right_wrist_emb = _embed_img(right_wrist_img)

        # Language
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        lang_emb = vlm_we.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        # Per-arm states (project to vlm_hidden)
        state_l, state_r = self._prepare_per_arm_state(batch)
        state_l = state_l.to(dtype=vlm_dtype)
        state_r = state_r.to(dtype=vlm_dtype)
        # Use the per-arm state projectors
        state_emb_l = self.backbone.state_proj_l(state_l)[:, None, :]
        state_emb_r = self.backbone.state_proj_r(state_r)[:, None, :]

        B = primary_emb.shape[0]
        device = primary_emb.device

        # Common block: [primary_image | language]
        common_emb = torch.cat([primary_emb, lang_emb], dim=1)
        common_pad = torch.cat(
            [
                torch.ones(B, primary_emb.shape[1], dtype=torch.bool, device=device),
                lang_masks,
            ],
            dim=1,
        )

        # Left block: [left_wrist_image | left_state]
        left_emb = torch.cat([left_wrist_emb, state_emb_l], dim=1)
        left_pad = torch.ones(B, left_emb.shape[1], dtype=torch.bool, device=device)

        # Right block: [right_wrist_image | right_state]
        right_emb = torch.cat([right_wrist_emb, state_emb_r], dim=1)
        right_pad = torch.ones(B, right_emb.shape[1], dtype=torch.bool, device=device)

        assert left_emb.shape[1] == right_emb.shape[1], (
            "left and right blocks must have equal length (twin attention assumption); "
            f"got {left_emb.shape[1]} vs {right_emb.shape[1]}"
        )

        n_common = common_emb.shape[1]
        n_arm = left_emb.shape[1]

        vlm_embs = torch.cat([common_emb, left_emb, right_emb], dim=1)
        vlm_pad = torch.cat([common_pad, left_pad, right_pad], dim=1)
        ci_ids = (
            torch.cat(
                [
                    torch.zeros(n_common, dtype=torch.long),
                    torch.ones(n_arm, dtype=torch.long),
                    torch.full((n_arm,), 2, dtype=torch.long),
                ]
            )
            .unsqueeze(0)
            .expand(B, -1)
            .to(device)
        )

        return vlm_embs, vlm_pad, ci_ids

    # ------------------------------------------------------------------ #
    #  Suffix embedding (action chunk + time MLP)
    # ------------------------------------------------------------------ #

    def _embed_suffix(self, noisy_actions: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        """Embed (noisy_actions, time) into the action expert input space.

        Returns (action_time_emb [B, chunk_size, expert_hidden], pad_mask [B, chunk_size]).
        Uses SmolVLA's projection modules (kept shared across arms in C-1).
        """
        flow = self._flow
        vlm_we = self._vlm_we

        action_emb = flow.action_in_proj(noisy_actions)  # [B, T_exp, expert_hidden]
        dtype = action_emb.dtype
        device = action_emb.device

        time_emb = create_sinusoidal_pos_embedding(
            time,
            vlm_we.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)

        action_time = torch.cat([action_emb, time_emb], dim=2)
        action_time = flow.action_time_mlp_in(action_time)
        action_time = F.silu(action_time)
        action_time = flow.action_time_mlp_out(action_time)

        B, T_exp = action_time.shape[:2]
        pad = torch.ones(B, T_exp, dtype=torch.bool, device=device)
        return action_time, pad

    # ------------------------------------------------------------------ #
    #  Flow-matching helpers
    # ------------------------------------------------------------------ #

    def _sample_noise(self, shape, device: torch.device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def _sample_time(self, bsize: int, device: torch.device) -> Tensor:
        beta = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        t = beta.sample((bsize,)).to(device=device, dtype=torch.float32)
        return t * 0.999 + 0.001

    # ------------------------------------------------------------------ #
    #  Forward (training)
    # ------------------------------------------------------------------ #

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        """Training forward pass: bimanual flow-matching loss."""
        cfg = self.config
        vlm_dtype = self.backbone.vlm_l_layers[0].self_attn.q_proj.weight.dtype
        expert_dtype = self._flow.action_in_proj.weight.dtype

        actions = self._prepare_action(batch)  # [B, T_exp, max_action_dim]
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)
        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        t_exp = time[:, None, None].to(actions.dtype)
        x_t = t_exp * noise.to(actions.dtype) + (1.0 - t_exp) * actions
        u_t_target = noise.to(actions.dtype) - actions

        # Embed prefix (VLM input)
        vlm_embs, vlm_pad, ci_ids = self._embed_prefix_twin(batch)

        # Embed suffix (action chunk + time)
        x_t_cast = x_t.to(dtype=expert_dtype)
        time_cast = time.to(dtype=expert_dtype)
        suffix_embs, suffix_pad = self._embed_suffix(x_t_cast, time_cast)

        # Run twin VLM + Expert
        _, suffix_out = self.backbone.twin_vlm_expert_forward(
            vlm_inputs_embeds=vlm_embs,
            vlm_pad_mask=vlm_pad,
            ci_ids=ci_ids,
            expert_inputs_embeds=suffix_embs,
            expert_pad_mask=suffix_pad,
        )  # suffix_out: [B, T_exp, expert_hidden]

        # Velocity prediction
        v_t = self._flow.action_out_proj(suffix_out)  # [B, T_exp, max_action_dim]

        # Loss: MSE between predicted velocity and (noise - action)
        # Limit to real action dims (2 * action_dim) to avoid training the pad bins
        d_real = cfg.action_dim * 2
        per_elem = (v_t[:, :, :d_real].float() - u_t_target[:, :, :d_real].float()) ** 2

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_ep = (~actions_is_pad).to(per_elem.dtype)
            per_elem = per_elem * in_ep.unsqueeze(-1)

        loss = per_elem.mean()
        return loss, {"loss": loss.detach().item()}

    # ------------------------------------------------------------------ #
    #  Inference (denoising)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Flow-matching denoising loop, bimanual.

        Performs `num_steps` Euler integration steps backward in time from
        t=1 (pure noise) to t≈0 (action). At each step we run a full
        `twin_vlm_expert_forward` (KV-cache optimization is left to Phase 4).

        Returns:
          actions: [B, chunk_size, 2 * action_dim] -- the bimanual action chunk
                   with padding dims stripped.
        """
        self.eval()
        cfg = self.config
        device = next(self.parameters()).device
        expert_dtype = self._flow.action_in_proj.weight.dtype

        first = next(iter(batch.values()))
        bsize = first.shape[0]

        if noise is None:
            noise = self._sample_noise(
                (bsize, cfg.chunk_size, cfg.max_action_dim), device
            )
        x_t = noise.clone()

        vlm_embs, vlm_pad, ci_ids = self._embed_prefix_twin(batch)

        num_steps = cfg.num_steps
        dt = -1.0 / num_steps

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            x_cast = x_t.to(dtype=expert_dtype)
            t_cast = time_tensor.to(dtype=expert_dtype)
            suffix_embs, suffix_pad = self._embed_suffix(x_cast, t_cast)

            _, suffix_out = self.backbone.twin_vlm_expert_forward(
                vlm_inputs_embeds=vlm_embs,
                vlm_pad_mask=vlm_pad,
                ci_ids=ci_ids,
                expert_inputs_embeds=suffix_embs,
                expert_pad_mask=suffix_pad,
            )
            v_t = self._flow.action_out_proj(suffix_out).to(dtype=x_t.dtype)
            x_t = x_t + dt * v_t

        d_real = cfg.action_dim * 2
        return x_t[:, :, :d_real]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a single action via an internal action queue."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]
            # Queue is consumed step-by-step; store as [n_action_steps, B, action_dim]
            actions = actions.transpose(0, 1)
            self._action_queue.extend(actions)
        return self._action_queue.popleft()
