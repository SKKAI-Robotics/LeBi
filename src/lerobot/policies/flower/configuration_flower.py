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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerConfig(PreTrainedConfig):
    """FLOWER VLA policy configuration for LeRobot.

    The default contract targets the Task1 dual-arm dataset:
    three RGB cameras and 12D action chunks. The action registry also creates
    7D and 16D heads so later fine-tuning can reuse the same checkpoint format.
    """

    n_obs_steps: int = 1
    chunk_size: int = 10
    n_action_steps: int = 10

    camera_keys: tuple[str, ...] = (
        f"{OBS_IMAGES}.left_wrist_cam",
        f"{OBS_IMAGES}.right_wrist_cam",
        f"{OBS_IMAGES}.top_view",
    )

    action_space: str = "auto"
    default_action_space: str = "bimanual_12d"
    supported_action_spaces: dict[str, int] = field(
        default_factory=lambda: {
            "single_arm_7d": 7,
            "bimanual_12d": 12,
            "bimanual_16d": 16,
        }
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Vision/language model.
    vlm_path: str = "microsoft/Florence-2-large"
    trust_remote_code: bool = True
    freeze_florence: bool = False
    freeze_vision_tower: bool = True
    tokenizer_max_length: int = 64
    num_prompt_tokens: int = 1
    token_dropout: float = 0.1
    default_task: str | None = None

    # Image preprocessing.
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    normalize_images: bool = True
    validate_image_range: bool = True

    # FLOWER/DiT head.
    dit_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 18
    mlp_ratio: float = 4.0
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    norm_eps: float = 1e-6
    timestep_embed_dim: int = 256
    frequency_embed_dim: int = 256
    num_sampling_steps: int = 4
    sampling_type: str = "uniform"
    use_cross_attn: bool = True
    use_proprio: bool = False
    proprio_dim: int | None = None
    action_clip_value: float | None = 1.0

    # Training presets.
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.05
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError("FLOWER currently expects `n_obs_steps=1`.")
        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps <= 0:
            raise ValueError("`n_action_steps` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"`dit_dim` ({self.dit_dim}) must be divisible by `n_heads` ({self.n_heads}).")
        if self.action_space != "auto" and self.action_space not in self.supported_action_spaces:
            raise ValueError(
                f"Unsupported `action_space={self.action_space}`. "
                f"Expected 'auto' or one of {sorted(self.supported_action_spaces)}."
            )
        if self.default_action_space not in self.supported_action_spaces:
            raise ValueError(
                f"`default_action_space={self.default_action_space}` is not in "
                f"{sorted(self.supported_action_spaces)}."
            )
        if self.sampling_type not in {"uniform", "logit_normal", "pi_zero"}:
            raise ValueError(
                f"`sampling_type` must be one of 'uniform', 'logit_normal', or 'pi_zero', got {self.sampling_type}."
            )

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.supported_action_spaces[self.default_action_space],),
            )

        image_features = self.image_features
        missing_cameras = [key for key in self.camera_keys if key not in image_features]
        if missing_cameras:
            raise ValueError(
                "FLOWER requires the configured camera keys in `input_features`. "
                f"Missing: {missing_cameras}. Available visual keys: {sorted(image_features)}."
            )

        for key in self.camera_keys:
            shape = image_features[key].shape
            if len(shape) != 3:
                raise ValueError(f"Camera feature '{key}' must have shape (C, H, W), got {shape}.")
            if shape[0] != 3:
                raise ValueError(f"Camera feature '{key}' must be RGB with C=3, got {shape}.")

        action_dim = self.dataset_action_dim
        self.action_space_name_from_dim(action_dim)
        if self.action_space != "auto" and action_dim != self.supported_action_spaces[self.action_space]:
            raise ValueError(
                f"Configured `action_space={self.action_space}` expects action dim "
                f"{self.supported_action_spaces[self.action_space]}, but dataset action dim is {action_dim}."
            )

        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires `observation.state` in `input_features`.")

    @property
    def dataset_action_dim(self) -> int:
        action_feature = self.action_feature
        if action_feature is None:
            return self.supported_action_spaces[self.default_action_space]
        if not action_feature.shape:
            raise ValueError("Action feature must have a non-empty shape.")
        return int(action_feature.shape[-1])

    @property
    def state_dim(self) -> int:
        if self.proprio_dim is not None:
            return self.proprio_dim
        if self.robot_state_feature is None:
            return 0
        return int(self.robot_state_feature.shape[-1])

    def action_space_name_from_dim(self, action_dim: int) -> str:
        matches = [name for name, dim in self.supported_action_spaces.items() if dim == action_dim]
        if not matches:
            raise ValueError(
                f"Unsupported FLOWER action dimension {action_dim}. "
                f"Supported dimensions: {sorted(set(self.supported_action_spaces.values()))}."
            )
        if self.default_action_space in matches:
            return self.default_action_space
        return matches[0]

    def resolve_action_space_name(self, action_dim: int | None = None) -> str:
        if self.action_space != "auto":
            if action_dim is not None and self.supported_action_spaces[self.action_space] != action_dim:
                raise ValueError(
                    f"Configured `action_space={self.action_space}` expects action dim "
                    f"{self.supported_action_spaces[self.action_space]}, got {action_dim}."
                )
            return self.action_space

        if action_dim is None:
            if self.action_feature is None:
                return self.default_action_space
            action_dim = self.dataset_action_dim
        return self.action_space_name_from_dim(action_dim)

    def resolve_action_dim(self, action_space_name: str | None = None) -> int:
        name = action_space_name or self.resolve_action_space_name()
        if name not in self.supported_action_spaces:
            raise ValueError(f"Unknown FLOWER action space '{name}'.")
        return self.supported_action_spaces[name]

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
