from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.processor_flower import make_flower_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


CAMERA_KEYS = (
    f"{OBS_IMAGES}.left_left",
    f"{OBS_IMAGES}.left_top",
    f"{OBS_IMAGES}.right_right",
)


def make_config() -> FlowerConfig:
    return FlowerConfig(
        device="cpu",
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for key in CAMERA_KEYS},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
        image_size=16,
        chunk_size=3,
        n_action_steps=2,
        load_pretrained=False,
    )


def test_flower_processor_normalizes_images_and_preserves_task():
    config = make_config()
    stats = {
        ACTION: {
            "min": torch.full((12,), -2.0),
            "max": torch.full((12,), 2.0),
        }
    }
    preprocessor, postprocessor = make_flower_pre_post_processors(config, dataset_stats=stats)
    batch = {key: torch.full((3, 16, 16), 255, dtype=torch.uint8) for key in CAMERA_KEYS}
    batch[ACTION] = torch.zeros(3, 12)
    batch["task"] = "arrange the flower"

    processed = preprocessor(batch)

    assert processed["task"] == ["arrange the flower"]
    for key in CAMERA_KEYS:
        assert processed[key].shape == (1, 3, 16, 16)
        assert processed[key].dtype == torch.float32
        assert torch.isfinite(processed[key]).all()
    assert processed[ACTION].shape == (3, 12)

    action = torch.zeros(1, 12)
    restored = postprocessor(action)
    assert restored.shape == (1, 12)
