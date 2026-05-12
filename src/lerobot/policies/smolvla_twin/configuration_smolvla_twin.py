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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("smolvla_twin")
@dataclass
class SmolVLATwinConfig(PreTrainedConfig):
    # ----- Per-arm dims (total = 2 * action_dim / 2 * state_dim) -----
    action_dim: int = 6
    state_dim: int = 6

    # ----- Padding dims used for SmolVLA projection layers -----
    # Inherited from SmolVLA (max_state_dim=32, max_action_dim=32).
    # Per-arm action/state will be zero-padded to these dims before action_in_proj/state_proj.
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ----- Time / chunk -----
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # ----- Backbone (SmolVLA / SmolVLM2) -----
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    # If non-None, surgery starts from this pretrained SmolVLA checkpoint.
    smolvla_pretrained_path: str | None = "lerobot/smolvla_base"
    # SmolVLA-Twin native checkpoint (after our surgery + finetuning). Mutually exclusive
    # with smolvla_pretrained_path for resume scenarios.
    smolvla_twin_pretrained_path: str | None = None
    load_vlm_weights: bool = True

    # SmolVLA backbone layer setup (kept identical to SmolVLA defaults for weight compat)
    num_vlm_layers: int = 16
    num_expert_layers: int = -1  # <= 0 means same as num_vlm_layers
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    attention_mode: str = "cross_attn"  # interleaved CA / SA same as SmolVLA

    # ----- Twin architecture toggles -----
    share_vision: bool = True       # share vision_model + connector across arms
    share_embed_tokens: bool = True # share text_model.embed_tokens
    share_decoder: bool = True      # share lm_expert + flow-matching projections (C-1)
    share_state_proj: bool = False  # per-arm state projector (encoded separately)

    enable_moe: bool = True
    enable_joint_attn: bool = True
    attn_reweighting: bool = True
    attn_reweighting_scale: float = 2.0

    # ----- Flow matching settings (from SmolVLA) -----
    num_steps: int = 10              # inference denoising steps
    min_period: float = 4e-3
    max_period: float = 4.0
    prefix_length: int = -1
    add_image_special_tokens: bool = False
    pad_language_to: str = "longest"
    tokenizer_max_length: int = 48

    # ----- Image preprocessing -----
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0

    # ----- LeRobot camera key mapping -----
    primary_camera: str = "observation.images.primary"
    left_wrist_camera: str = "observation.images.left_wrist"
    right_wrist_camera: str = "observation.images.right_wrist"

    # ----- Training / freeze policy -----
    freeze_vision_encoder: bool = True   # SmolVLA default
    train_expert_only: bool = False      # bimanual surgery → unfreeze twin LLMs
    train_state_proj: bool = True

    gradient_checkpointing: bool = False
    device: str | None = None
    dtype: str = "bfloat16"

    # ----- Optimizer / Scheduler -----
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # ----- Normalization -----
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_dim * 2 > self.max_action_dim:
            raise ValueError(
                f"2*action_dim ({2 * self.action_dim}) exceeds max_action_dim ({self.max_action_dim})."
            )
        if self.state_dim * 2 > self.max_state_dim:
            raise ValueError(
                f"2*state_dim ({2 * self.state_dim}) exceeds max_state_dim ({self.max_state_dim})."
            )

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim * 2,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim * 2,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
