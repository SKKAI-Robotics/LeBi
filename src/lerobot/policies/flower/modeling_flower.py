# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.modeling_flower_blocks import (
    FlowBlock,
    FreqEmbedder,
    RMSNorm,
    SharedAdaLNController,
    TimestepEmbedder,
    build_sincos_position_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available


def generate_policy_prompt(task: str, action_space_name: str) -> str:
    if action_space_name == "single_arm_7d":
        action_desc = "single-arm 7D end-effector delta action"
        arm_desc = "one robot arm"
    elif action_space_name == "bimanual_16d":
        action_desc = "dual-arm 16D action"
        arm_desc = "two robot arms"
    else:
        action_desc = "dual-arm 12D action"
        arm_desc = "two robot arms"

    return (
        "You are controlling a robot policy. "
        f"The robot has {arm_desc}. "
        f"Predict the next {action_desc} chunk for the instruction: {task}"
    )


@dataclass(frozen=True)
class FlowerActionSpaceRegistry:
    spaces: dict[str, int]
    default_space: str

    @property
    def names(self) -> list[str]:
        return list(self.spaces.keys())

    @property
    def dim_to_name(self) -> dict[int, str]:
        mapping = {dim: name for name, dim in self.spaces.items()}
        if self.spaces[self.default_space] in mapping:
            mapping[self.spaces[self.default_space]] = self.default_space
        return mapping

    def index(self, name: str) -> int:
        if name not in self.spaces:
            raise ValueError(f"Unknown FLOWER action space '{name}'.")
        return self.names.index(name)

    def dim(self, name: str) -> int:
        if name not in self.spaces:
            raise ValueError(f"Unknown FLOWER action space '{name}'.")
        return self.spaces[name]

    def name_from_dim(self, action_dim: int) -> str:
        if action_dim not in self.dim_to_name:
            raise ValueError(f"Unsupported FLOWER action dimension {action_dim}.")
        return self.dim_to_name[action_dim]


class FlowerModel(nn.Module):
    def __init__(self, config: FlowerConfig):
        super().__init__()
        self.config = config
        self.action_registry = FlowerActionSpaceRegistry(
            spaces=dict(config.supported_action_spaces),
            default_space=config.default_action_space,
        )

        self._setup_vlm(config.vlm_path, config.freeze_vision_tower, config.freeze_florence)
        self.vlm_to_dit = nn.Linear(self.vlm_hidden_dim, config.dit_dim)
        self.t_embedder = TimestepEmbedder(config.dit_dim, config.timestep_embed_dim)
        self.frequency_embedder = FreqEmbedder(config.dit_dim, config.frequency_embed_dim)
        self.cond_norm = RMSNorm(config.dit_dim, eps=config.norm_eps)

        self.action_encoders = nn.ModuleDict(
            {name: nn.Linear(dim, config.dit_dim) for name, dim in config.supported_action_spaces.items()}
        )
        self.action_decoders = nn.ModuleDict(
            {name: nn.Linear(config.dit_dim, dim) for name, dim in config.supported_action_spaces.items()}
        )
        for decoder in self.action_decoders.values():
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

        self.action_adaln = nn.ModuleDict(
            {
                name: SharedAdaLNController(
                    cond_dim=config.dit_dim,
                    hidden_size=config.dit_dim,
                    n_layers=config.n_layers,
                    n_action_spaces=len(config.supported_action_spaces),
                )
                for name in config.supported_action_spaces
            }
        )

        if config.use_proprio:
            if config.state_dim <= 0:
                raise ValueError("`use_proprio=True` requires a positive state dimension.")
            self.proprio_encoders = nn.ModuleDict(
                {name: nn.Linear(config.state_dim, config.dit_dim) for name in config.supported_action_spaces}
            )
        else:
            self.proprio_encoders = None

        self.dit = nn.ModuleList(
            [
                FlowBlock(
                    hidden_size=config.dit_dim,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    norm_eps=config.norm_eps,
                    use_cross_attn=config.use_cross_attn,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.dit_dim, eps=config.norm_eps)
        self.register_buffer(
            "action_pos_embed",
            build_sincos_position_embedding(config.chunk_size, config.dit_dim),
            persistent=False,
        )

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool) -> None:
        if not _transformers_available:
            raise ImportError(
                "FLOWER requires transformers. Install with `pip install 'lerobot[flower]'` "
                "or `pip install 'lerobot[transformers-dep]'`."
            )

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            vlm_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        self.vlm_hidden_dim = self._infer_vlm_hidden_dim()

        if hasattr(self.tokenizer, "add_tokens"):
            self.tokenizer.add_tokens(["<Flow>"])
        if hasattr(self.vlm, "resize_token_embeddings") and hasattr(self.tokenizer, "__len__"):
            self.vlm.resize_token_embeddings(len(self.tokenizer))

        prompt_embed = torch.zeros(1, self.config.num_prompt_tokens, self.vlm_hidden_dim)
        if hasattr(self.tokenizer, "convert_tokens_to_ids") and hasattr(self.vlm, "get_input_embeddings"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids("<Flow>")
                token = torch.tensor([[token_id]], dtype=torch.long)
                prompt_embed = self.vlm.get_input_embeddings()(token).detach()
                if self.config.num_prompt_tokens > 1:
                    prompt_embed = prompt_embed.repeat(1, self.config.num_prompt_tokens, 1)
            except Exception:
                pass
        self.prompt_embeds = nn.Parameter(prompt_embed)
        self.vlm_token_dropout = nn.Dropout(self.config.token_dropout)

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_vision_tower:
            for module_name in ("vision_tower", "vision_model"):
                module = getattr(self.vlm, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False

    def _infer_vlm_hidden_dim(self) -> int:
        config = getattr(self.vlm, "config", None)
        candidates = [
            getattr(config, "hidden_size", None),
            getattr(config, "d_model", None),
            getattr(getattr(config, "text_config", None), "hidden_size", None),
            getattr(getattr(config, "text_config", None), "d_model", None),
        ]
        for value in candidates:
            if value is not None:
                return int(value)
        if hasattr(self.vlm, "get_input_embeddings"):
            return int(self.vlm.get_input_embeddings().embedding_dim)
        raise ValueError("Could not infer Florence hidden dimension for FLOWER.")

    def encode_observations(self, batch: dict[str, Any], action_space_name: str) -> dict[str, Tensor]:
        images = self._prepare_images(batch)
        image_features = [self._encode_image_view(image) for image in images]
        vlm_image_features = torch.cat(image_features, dim=1)

        bsz = vlm_image_features.shape[0]
        prompts = [
            generate_policy_prompt(task, action_space_name)
            for task in self._get_task_list(batch, batch_size=bsz)
        ]
        text_embeds, text_attention_mask = self._get_text_embeddings(prompts)
        prompt_embeds = self.prompt_embeds.to(device=vlm_image_features.device, dtype=vlm_image_features.dtype)
        prompt_embeds = prompt_embeds.expand(bsz, -1, -1)

        encoder_inputs = torch.cat([vlm_image_features, prompt_embeds, text_embeds], dim=1)
        image_prompt_mask = torch.ones(
            bsz,
            vlm_image_features.shape[1] + prompt_embeds.shape[1],
            dtype=text_attention_mask.dtype,
            device=text_attention_mask.device,
        )
        attention_mask = torch.cat([image_prompt_mask, text_attention_mask], dim=1)
        encoder_outputs = self._run_vlm_encoder(encoder_inputs, attention_mask)
        vlm_features = self.vlm_to_dit(self.vlm_token_dropout(encoder_outputs))
        vlm_features = self.cond_norm(vlm_features)

        frequency = torch.full(
            (bsz, 1, 1),
            3.0,
            dtype=vlm_features.dtype,
            device=vlm_features.device,
        )
        action_type = torch.full(
            (bsz,),
            self.action_registry.index(action_space_name),
            dtype=torch.long,
            device=vlm_features.device,
        )

        cond = {
            "vlm_features": vlm_features,
            "vlm_attention_mask": attention_mask.to(device=vlm_features.device, dtype=torch.bool),
            "frequency": self.frequency_embedder(frequency),
            "action_type": action_type,
        }
        if self.config.use_proprio:
            if OBS_STATE not in batch:
                raise ValueError("FLOWER `use_proprio=True` requires `observation.state` in the batch.")
            state = batch[OBS_STATE]
            if state.ndim == 3:
                state = state[:, -1]
            cond["proprio"] = state.to(device=vlm_features.device, dtype=vlm_features.dtype)
        return cond

    def _prepare_images(self, batch: dict[str, Any]) -> list[Tensor]:
        images: list[Tensor] = []
        for key in self.config.camera_keys:
            if key not in batch:
                raise ValueError(f"FLOWER missing required camera key '{key}' in batch.")
            image = batch[key]
            if not isinstance(image, Tensor):
                image = torch.as_tensor(image)
            image = image.to(device=self.prompt_embeds.device, dtype=self.prompt_embeds.dtype)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim == 4:
                image = image.unsqueeze(1)
            if image.ndim != 5:
                raise ValueError(f"Camera '{key}' must be BCHW or BTCHW, got shape {tuple(image.shape)}.")
            bsz, timesteps, channels, height, width = image.shape
            if channels != 3:
                raise ValueError(f"Camera '{key}' must be channel-first RGB, got shape {tuple(image.shape)}.")
            image = image.reshape(bsz * timesteps, channels, height, width)
            if self.config.image_size and (height != self.config.image_size or width != self.config.image_size):
                image = F.interpolate(
                    image,
                    size=(self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            image = image.reshape(bsz, timesteps, channels, self.config.image_size, self.config.image_size)
            images.append(image)
        return images

    def _encode_image_view(self, image: Tensor) -> Tensor:
        bsz, timesteps = image.shape[:2]
        flat = image.flatten(0, 1)
        if not hasattr(self.vlm, "_encode_image"):
            raise AttributeError("FLOWER expects the Florence model to expose `_encode_image`.")
        encoded = self.vlm._encode_image(flat)
        if isinstance(encoded, tuple | list):
            encoded = encoded[0]
        if encoded.ndim == 2:
            encoded = encoded.unsqueeze(1)
        return encoded.reshape(bsz, timesteps * encoded.shape[1], encoded.shape[2])

    def _get_task_list(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        if "task" not in batch or batch["task"] is None:
            if self.config.default_task is None:
                raise ValueError("FLOWER requires a `task` string or list of strings in the batch.")
            return [self.config.default_task for _ in range(batch_size)]

        task = batch["task"]
        if isinstance(task, str):
            return [task for _ in range(batch_size)]
        if isinstance(task, tuple):
            task = list(task)
        if isinstance(task, list) and all(isinstance(item, str) for item in task):
            if len(task) == 1 and batch_size > 1:
                return task * batch_size
            if len(task) != batch_size:
                raise ValueError(f"FLOWER got {len(task)} task strings for batch size {batch_size}.")
            return task
        raise ValueError(f"FLOWER `task` must be a string or list of strings, got {type(task)}.")

    def _get_text_embeddings(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        device = self.prompt_embeds.device
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.tokenizer_max_length,
        )
        if hasattr(tokenized, "to"):
            tokenized = tokenized.to(device)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)
        embeds = self.vlm.get_input_embeddings()(input_ids).to(dtype=self.prompt_embeds.dtype)
        return embeds, attention_mask

    def _run_vlm_encoder(self, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        if hasattr(self.vlm, "get_encoder"):
            encoder = self.vlm.get_encoder()
            output = encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        elif hasattr(self.vlm, "language_model") and hasattr(self.vlm.language_model, "model"):
            output = self.vlm.language_model.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            output = SimpleNamespace(last_hidden_state=inputs_embeds)
        return output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]

    def dit_forward(self, z: Tensor, t: Tensor, cond: dict[str, Tensor], action_space_name: str) -> Tensor:
        bsz, seq_len, _ = z.shape
        x = self.action_encoders[action_space_name](z)
        time_embed = self.t_embedder(t.reshape(bsz, -1)[:, 0]).unsqueeze(1)
        x = x + time_embed + self.action_pos_embed[:, :seq_len].to(device=x.device, dtype=x.dtype)

        context = cond["vlm_features"]
        cond_summary = context.mean(dim=1) + cond["frequency"]
        if self.proprio_encoders is not None and "proprio" in cond:
            cond_summary = cond_summary + self.proprio_encoders[action_space_name](cond["proprio"])

        adaln = self.action_adaln[action_space_name](cond_summary, cond["action_type"])
        context_mask = cond.get("vlm_attention_mask")
        for idx, block in enumerate(self.dit):
            x = block(x, adaln[:, idx], context=context, context_mask=context_mask)
        x = self.final_norm(x)
        return self.action_decoders[action_space_name](x)

    def rf_loss(
        self,
        actions: Tensor,
        cond: dict[str, Tensor],
        action_space_name: str,
        *,
        action_is_pad: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[:, : self.config.chunk_size]
        noise = torch.randn_like(actions)
        t = self._sample_timestep(actions.shape[0], actions.device, actions.dtype)
        zt = (1 - t) * noise + t * actions
        target = actions - noise
        pred = self.dit_forward(zt, t, cond, action_space_name)
        loss = (pred - target).pow(2).mean(dim=-1)

        if action_is_pad is not None:
            mask = ~action_is_pad[:, : loss.shape[1]].bool()
            if reduction == "none":
                denom = mask.sum(dim=1).clamp_min(1)
                return (loss * mask).sum(dim=1) / denom, {"loss": float(loss.detach().mean().cpu())}
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        elif reduction == "none":
            return loss.mean(dim=1), {"loss": float(loss.detach().mean().cpu())}
        else:
            loss = loss.mean()

        return loss, {"loss": float(loss.detach().cpu())}

    def _sample_timestep(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.config.sampling_type == "logit_normal":
            t = torch.sigmoid(torch.randn(batch_size, 1, 1, device=device, dtype=dtype))
        elif self.config.sampling_type == "pi_zero":
            t = torch.distributions.Beta(1.5, 1.0).sample((batch_size, 1, 1)).to(device=device, dtype=dtype)
        else:
            t = torch.rand(batch_size, 1, 1, device=device, dtype=dtype)
        return t

    @torch.no_grad()
    def sample_actions(
        self,
        noise: Tensor,
        cond: dict[str, Tensor],
        action_space_name: str,
    ) -> Tensor:
        z = noise
        num_steps = self.config.num_sampling_steps
        for step in range(num_steps):
            t_value = step / num_steps
            t = torch.full((z.shape[0], 1, 1), t_value, dtype=z.dtype, device=z.device)
            velocity = self.dit_forward(z, t, cond, action_space_name)
            z = z + velocity / num_steps
        if self.config.action_clip_value is not None:
            z = z.clamp(-self.config.action_clip_value, self.config.action_clip_value)
        return z


class FlowerPolicy(PreTrainedPolicy):
    config_class = FlowerConfig
    name = "flower"

    def __init__(self, config: FlowerConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowerModel(config)
        self._queues: dict[str, deque] | None = None
        self.reset()

    def get_optim_params(self):
        return [param for param in self.parameters() if param.requires_grad]

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _action_space_from_batch(self, batch: dict[str, Any]) -> str:
        if ACTION in batch:
            action = batch[ACTION]
            return self.config.resolve_action_space_name(int(action.shape[-1]))
        return self.config.resolve_action_space_name()

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        if ACTION not in batch:
            raise ValueError("FLOWER training requires `action` in the batch.")
        action_space_name = self._action_space_from_batch(batch)
        cond = self.model.encode_observations(batch, action_space_name)
        loss, stats = self.model.rf_loss(
            batch[ACTION],
            cond,
            action_space_name,
            action_is_pad=batch.get("action_is_pad"),
            reduction="mean",
        )
        stats["action_dim"] = self.config.supported_action_spaces[action_space_name]
        return loss, stats

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        action_space_name = self._action_space_from_batch(batch)
        cond = self.model.encode_observations(batch, action_space_name)
        action_dim = self.config.resolve_action_dim(action_space_name)
        bsz = cond["vlm_features"].shape[0]
        if noise is None:
            dtype = cond["vlm_features"].dtype
            device = cond["vlm_features"].device
            noise = torch.randn(bsz, self.config.chunk_size, action_dim, dtype=dtype, device=device)
        return self.model.sample_actions(noise, cond, action_space_name)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        if self._queues is None:
            self.reset()
        obs_batch = {key: value for key, value in batch.items() if key != ACTION}
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(obs_batch, noise=noise)[:, : self.config.n_action_steps]
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()
