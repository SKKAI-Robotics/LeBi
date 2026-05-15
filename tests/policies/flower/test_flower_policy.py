from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.modeling_flower import FlowerModel, FlowerPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES


CAMERA_KEYS = (
    f"{OBS_IMAGES}.left_left",
    f"{OBS_IMAGES}.left_top",
    f"{OBS_IMAGES}.right_right",
)


class FakeTokenizer:
    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        batch_size = len(prompts)
        seq_len = min(max_length, 6)
        input_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def add_tokens(self, tokens):
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return 1

    def __len__(self):
        return 128


class FakeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs_embeds, attention_mask=None, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


class FakeVLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(d_model=hidden_dim))
        self.embeddings = nn.Embedding(128, hidden_dim)
        self.image_proj = nn.Linear(3, hidden_dim)
        self.encoder = FakeEncoder(hidden_dim)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, length):
        return None

    def get_encoder(self):
        return self.encoder

    def _encode_image(self, pixels):
        pooled = pixels.mean(dim=(-1, -2))
        token = self.image_proj(pooled).unsqueeze(1)
        return token.repeat(1, 4, 1)


@pytest.fixture(autouse=True)
def patch_florence(monkeypatch):
    def fake_setup_vlm(self, *args, **kwargs):
        self.vlm_hidden_dim = self.config.dit_dim
        self.vlm = FakeVLM(self.vlm_hidden_dim)
        self.tokenizer = FakeTokenizer()
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.prompt_embeds = nn.Parameter(torch.zeros(1, self.config.num_prompt_tokens, self.vlm_hidden_dim))
        self.vlm_token_dropout = nn.Dropout(self.config.token_dropout)

    monkeypatch.setattr(FlowerModel, "_setup_vlm", fake_setup_vlm)


def make_config(action_dim: int, **overrides) -> FlowerConfig:
    kwargs = {
        "device": "cpu",
        "input_features": {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for key in CAMERA_KEYS
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        "chunk_size": 3,
        "n_action_steps": 2,
        "image_size": 16,
        "dit_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "num_sampling_steps": 2,
        "tokenizer_max_length": 8,
        "optimizer_lr": 1e-4,
        "scheduler_warmup_steps": 1,
        "scheduler_decay_steps": 10,
        "load_pretrained": False,
    }
    kwargs.update(overrides)
    return FlowerConfig(**kwargs)


def make_batch(action_dim: int, batch_size: int = 2) -> dict:
    batch = {
        key: torch.rand(batch_size, 3, 16, 16)
        for key in CAMERA_KEYS
    }
    batch[ACTION] = torch.randn(batch_size, 3, action_dim)
    batch["action_is_pad"] = torch.zeros(batch_size, 3, dtype=torch.bool)
    batch["task"] = ["put the flower in the vase", "move the flower to the target"][:batch_size]
    return batch


@pytest.mark.parametrize("action_dim,expected_space", [(7, "single_arm_7d"), (12, "bimanual_12d"), (16, "bimanual_16d")])
def test_flower_forward_and_action_chunk_shapes(action_dim, expected_space):
    policy = FlowerPolicy(make_config(action_dim))
    batch = make_batch(action_dim)

    loss, info = policy.forward(batch)
    chunk = policy.predict_action_chunk(batch)

    assert torch.isfinite(loss)
    assert info["action_dim"] == action_dim
    assert policy.config.resolve_action_space_name(action_dim) == expected_space
    assert chunk.shape == (2, 3, action_dim)


def test_flower_task1_default_12d_batch():
    policy = FlowerPolicy(make_config(12, chunk_size=10, n_action_steps=10))
    batch = {
        key: torch.rand(1, 3, 16, 16)
        for key in CAMERA_KEYS
    }
    batch[ACTION] = torch.randn(1, 10, 12)
    batch["action_is_pad"] = torch.zeros(1, 10, dtype=torch.bool)
    batch["task"] = ["arrange the flower"]

    loss, info = policy.forward(batch)

    assert torch.isfinite(loss)
    assert info["action_dim"] == 12


def test_flower_missing_camera_has_clear_error():
    policy = FlowerPolicy(make_config(12))
    batch = make_batch(12)
    batch.pop(f"{OBS_IMAGES}.right_right")

    with pytest.raises(ValueError, match="right_right"):
        policy.forward(batch)


def test_flower_unsupported_action_dim_has_clear_error():
    with pytest.raises(ValueError, match="Unsupported FLOWER action dimension 13"):
        FlowerPolicy(make_config(13))


def test_flower_missing_task_has_clear_error():
    policy = FlowerPolicy(make_config(12))
    batch = make_batch(12)
    batch.pop("task")

    with pytest.raises(ValueError, match="requires a `task`"):
        policy.forward(batch)


def test_flower_forward_does_not_mutate_batch():
    policy = FlowerPolicy(make_config(12))
    batch = make_batch(12)
    before = copy.deepcopy(batch)

    policy.forward(batch)

    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(batch[key], value)
        else:
            assert batch[key] == value
