#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_TWINVLA_IMAGE_SIZE = 448


@PreTrainedConfig.register_subclass("twinvla")
@dataclass
class TwinVLAConfig(PreTrainedConfig):
    # Per-arm action dimensionality (total action = 2 * action_dim)
    action_dim: int = 6
    # Per-arm proprioceptive state dimensionality (total state = 2 * state_dim)
    state_dim: int = 6

    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 20

    # TwinVLA backbone
    singlevla_pretrained_path: str = "jellyho/TwinVLA"
    model_type_name: str = "Eagle2_1BTwinVLA"
    dtype: str = "bfloat16"

    # TwinVLA architecture options
    share_vision: bool = True
    share_decoder: bool = True
    share_embed_tokens: bool = True
    attn_reweighting: bool = True
    enable_moe: bool = True
    enable_joint_attn: bool = True

    # DiT action head
    dit_size: str = "DiT-L"
    action_head: str = "DiT"
    num_readouts: int = 1
    train_denoising_steps: int = 100
    test_denoising_steps: int = 10
    denoiser: str = "DDIM"
    normalization_type: str = "quantile"

    # Image
    image_resolution: tuple[int, int] = (DEFAULT_TWINVLA_IMAGE_SIZE, DEFAULT_TWINVLA_IMAGE_SIZE)

    # Camera key mapping: LeRobot camera key → TwinVLA role
    primary_camera: str = "observation.images.primary"
    left_wrist_camera: str = "observation.images.left_wrist"
    right_wrist_camera: str = "observation.images.right_wrist"

    # Training
    gradient_checkpointing: bool = False
    freeze_vision_backbone: bool = False
    device: str | None = None

    # Optimizer
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 50_000
    scheduler_decay_lr: float = 1e-6

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
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
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
