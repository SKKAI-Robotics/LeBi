# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _channel_stats(stats: tuple[float, float, float], tensor: torch.Tensor) -> torch.Tensor:
    shape = [1] * tensor.ndim
    shape[-3] = 3
    return torch.tensor(stats, device=tensor.device, dtype=tensor.dtype).view(*shape)


@dataclass
@ProcessorStepRegistry.register(name="flower_image_to_float")
class FlowerImageToFloatProcessorStep(ProcessorStep):
    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        observation = observation.copy()
        keys = self.image_keys or [key for key in observation if key.startswith(f"{OBS_IMAGES}.")]
        for key in keys:
            if key not in observation or not isinstance(observation[key], torch.Tensor):
                continue
            image = observation[key]
            if image.numel() == 0:
                observation[key] = image.float()
                continue
            min_val = image.min().item()
            max_val = image.max().item()
            if max_val <= 1.0:
                observation[key] = image.float()
                continue
            if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                raise ValueError(
                    f"Image '{key}' has values outside [0, 255]: min={min_val:.4f}, max={max_val:.4f}."
                )
            observation[key] = image.float() / 255.0

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys, "validate_range": self.validate_range}


@dataclass
@ProcessorStepRegistry.register(name="flower_clip_normalize")
class FlowerClipNormalizeProcessorStep(ProcessorStep):
    image_keys: list[str] | None = None
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        observation = observation.copy()
        keys = self.image_keys or [key for key in observation if key.startswith(f"{OBS_IMAGES}.")]
        for key in keys:
            if key not in observation or not isinstance(observation[key], torch.Tensor):
                continue
            image = observation[key]
            if image.numel() == 0:
                continue
            min_val = image.min().item()
            max_val = image.max().item()
            if self.validate_range and (min_val < 0.0 or max_val > 1.0):
                raise ValueError(
                    f"Image '{key}' has values outside [0, 1] before CLIP normalization: "
                    f"min={min_val:.4f}, max={max_val:.4f}."
                )
            mean = _channel_stats(self.mean, image)
            std = _channel_stats(self.std, image)
            observation[key] = (image - mean) / std

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "mean": self.mean,
            "std": self.std,
            "validate_range": self.validate_range,
        }


def make_flower_pre_post_processors(
    config: FlowerConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    features = {**config.input_features, **config.output_features}
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        FlowerImageToFloatProcessorStep(
            image_keys=list(config.camera_keys),
            validate_range=config.validate_image_range,
        ),
    ]
    if config.normalize_images:
        input_steps.append(
            FlowerClipNormalizeProcessorStep(
                image_keys=list(config.camera_keys),
                mean=config.image_mean,
                std=config.image_std,
                validate_range=config.validate_image_range,
            )
        )
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features=features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
