# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.modeling_flower_blocks import (
    ActionMlp,
    FlowBlock,
    FreqEmbedder,
    RMSNorm,
    SharedAdaLNController,
    TimestepEmbedder,
    ZeroEncoder,
    stateless_norm,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

logger = logging.getLogger(__name__)


def generate_policy_prompt(task: str, action_space_name: str, prompt_style: str = "default") -> str:
    if action_space_name == "single_arm_7d":
        robot_name = "Franka Panda"
        action_desc = "Delta End-Effector"
        arm_count = 1
    elif action_space_name == "bimanual_16d":
        robot_name = "Dual Arm Robot"
        action_desc = "16D Bimanual Action"
        arm_count = 2
    else:
        robot_name = "Dual Arm Robot"
        action_desc = "12D Bimanual Action"
        arm_count = 2

    meta = f"Agent Type: {arm_count}-arm {robot_name}, Action Space: {action_desc}, "
    if prompt_style == "combined":
        return (
            f"{meta}. </od>Task Instruction: {task}</od>"
            "<grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
        )
    if prompt_style == "visual":
        return (
            f"<od>Task Instruction: {task}</od> "
            "<grounding>identify key objects and their spatial relationships</grounding> "
            "<region_cap>analyze motion paths and collision-free trajectories</region_cap> "
            f"<cap>{meta}</cap>"
        )
    if prompt_style == "structured":
        return (
            f"<od>ROBOT CONFIGURATION: {meta}. TASK OBJECTIVE: {task}. "
            "ANALYSIS REQUIREMENTS: identify target objects and obstacles; "
            "determine spatial relationships; plan manipulation sequence.</od>"
        )
    if prompt_style in {"minimal", "default"}:
        return f"{meta}. Task Instruction: {task}"
    return (
        "You are controlling a robot policy. "
        f"The robot has {arm_count} arm(s). "
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
        self.cond_linear = nn.Linear(self.vlm_hidden_dim, config.dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(config.dit_dim, config.timestep_embed_dim)
        self.frequency_embedder = FreqEmbedder(config.dit_dim, config.frequency_embed_dim)
        self.cond_norm = RMSNorm(self.vlm_hidden_dim, eps=config.norm_eps)

        self.action_encoders = nn.ModuleDict(
            {
                name: ActionMlp(in_features=dim, hidden_features=config.dit_dim, out_features=config.dit_dim)
                for name, dim in config.supported_action_spaces.items()
            }
        )
        self.action_decoders = nn.ModuleDict(
            {name: nn.Linear(config.dit_dim, dim) for name, dim in config.supported_action_spaces.items()}
        )

        if config.action_type_adaln:
            self.action_adaln = nn.ModuleDict(
                {
                    name: SharedAdaLNController(
                        dim=config.dit_dim,
                        global_conddim=config.dit_dim,
                        use_cross_attn=config.use_cross_attn,
                    )
                    for name in config.supported_action_spaces
                }
            )
            self.global_adaln = None
        else:
            self.action_adaln = None
            self.global_adaln = SharedAdaLNController(
                dim=config.dit_dim,
                global_conddim=config.dit_dim,
                use_cross_attn=config.use_cross_attn,
            )

        if config.use_proprio:
            if config.state_dim <= 0:
                raise ValueError("`use_proprio=True` requires a positive state dimension.")
            self.proprio_encoders = nn.ModuleDict(
                {
                    name: (
                        ActionMlp(
                            in_features=config.state_dim,
                            hidden_features=config.dit_dim,
                            out_features=config.dit_dim,
                            drop=0.2,
                        )
                        if name == "bimanual_16d"
                        else ZeroEncoder(config.dit_dim)
                    )
                    for name in config.supported_action_spaces
                }
            )
        else:
            self.proprio_encoders = None

        self.dit = nn.ModuleList(
            [
                FlowBlock(
                    dim=config.dit_dim,
                    heads=config.n_heads,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    mlp_pdrop=config.mlp_pdrop,
                    use_cross_attn=config.use_cross_attn,
                    use_rope=config.use_rope and not config.use_nope,
                    query_seq_len=config.query_seq_len,
                    rope_theta=config.rope_theta,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        if not config.use_rope and not config.use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, config.chunk_size, config.dit_dim) * 0.1)
        else:
            self.register_buffer(
                "positional_encoding", torch.zeros(1, config.chunk_size, config.dit_dim), persistent=False
            )
        if config.load_pretrained and config.pretrained_model_path is not None:
            self.load_pretrained_weights(
                config.pretrained_model_path,
                use_ema=config.pretrained_use_ema,
                ignore_mismatched_sizes=config.pretrained_ignore_mismatched_sizes,
            )

    def load_pretrained_weights(
        self,
        pretrained_model_path: str,
        *,
        use_ema: bool = True,
        ignore_mismatched_sizes: bool = True,
    ) -> dict[str, Any]:
        path = Path(pretrained_model_path)
        logger.info("Loading raw FLOWER pretrained weights from %s", path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint: dict[str, Any] = {"state_dict": load_file(str(path), device=str(self.prompt_embeds.device))}
        else:
            try:
                checkpoint = torch.load(path, map_location=self.prompt_embeds.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=self.prompt_embeds.device)

        state_dict = checkpoint.get("state_dict", checkpoint)
        if use_ema:
            state_dict = self._maybe_extract_ema_state_dict(checkpoint, state_dict)

        remapped = self._remap_reference_state_dict_keys(state_dict)
        target_state = self.state_dict()
        loadable: dict[str, Tensor] = {}
        skipped_mismatched: list[str] = []
        unexpected: list[str] = []
        for key, value in remapped.items():
            if key not in target_state:
                unexpected.append(key)
                continue
            if tuple(target_state[key].shape) != tuple(value.shape):
                if ignore_mismatched_sizes:
                    skipped_mismatched.append(key)
                    continue
                raise ValueError(
                    f"Shape mismatch while loading FLOWER checkpoint key '{key}': "
                    f"checkpoint {tuple(value.shape)} vs model {tuple(target_state[key].shape)}."
                )
            loadable[key] = value

        missing, incompatible_unexpected = self.load_state_dict(loadable, strict=False)
        report = {
            "loaded": len(loadable),
            "missing": list(missing),
            "unexpected": unexpected + list(incompatible_unexpected),
            "skipped_mismatched": skipped_mismatched,
        }
        logger.info(
            "Raw FLOWER checkpoint load report: loaded=%s missing=%s unexpected=%s skipped_mismatched=%s",
            report["loaded"],
            len(report["missing"]),
            len(report["unexpected"]),
            len(report["skipped_mismatched"]),
        )
        return report

    def _maybe_extract_ema_state_dict(
        self, checkpoint: dict[str, Any], fallback_state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        callbacks = checkpoint.get("callbacks")
        if not isinstance(callbacks, dict):
            return fallback_state_dict
        ema_block = callbacks.get("EMA")
        if not isinstance(ema_block, dict) or "ema_weights" not in ema_block:
            return fallback_state_dict

        ema_weights = ema_block["ema_weights"]
        if not isinstance(ema_weights, list):
            return fallback_state_dict

        ema_state_dict: dict[str, Tensor] = {}
        ema_idx = 0
        for name, tensor in fallback_state_dict.items():
            if ema_idx >= len(ema_weights):
                ema_state_dict[name] = tensor
                continue
            ema_tensor = ema_weights[ema_idx]
            if hasattr(ema_tensor, "shape") and tuple(ema_tensor.shape) == tuple(tensor.shape):
                ema_state_dict[name] = ema_tensor
                ema_idx += 1
            else:
                ema_state_dict[name] = tensor
        return ema_state_dict

    def _remap_reference_state_dict_keys(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        action_name_remap = {
            "eef_delta": "single_arm_7d",
            "bimanual_nav": "bimanual_16d",
        }
        remapped: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix("agent.").removeprefix("model.")
            new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_key = new_key.replace("action_pos_embed", "positional_encoding")
            if new_key.startswith("adaln."):
                new_key = f"action_{new_key}"
            for old_name, new_name in action_name_remap.items():
                new_key = new_key.replace(f"action_encoders.{old_name}.", f"action_encoders.{new_name}.")
                new_key = new_key.replace(f"action_decoders.{old_name}.", f"action_decoders.{new_name}.")
                new_key = new_key.replace(f"proprio_encoders.{old_name}.", f"proprio_encoders.{new_name}.")
                new_key = new_key.replace(f"action_adaln.{old_name}.", f"action_adaln.{new_name}.")
            remapped[new_key] = value
        return remapped

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

        if hasattr(self.tokenizer, "add_special_tokens"):
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<Flow>"]})
        elif hasattr(self.tokenizer, "add_tokens"):
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
        self.prompt_embeds = nn.Parameter(prompt_embed, requires_grad=False)
        self.vlm_token_dropout = nn.Dropout(self.config.token_dropout)

        try:
            del self.vlm.language_model.model.decoder
        except AttributeError:
            pass
        try:
            del self.vlm.language_model.lm_head
        except AttributeError:
            pass

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
            generate_policy_prompt(task, action_space_name, self.config.vlm_prompt_style)
            for task in self._get_task_list(batch, batch_size=bsz)
        ]
        text_embeds, _ = self._get_text_embeddings(prompts)
        prompt_embeds = self.prompt_embeds.to(device=vlm_image_features.device, dtype=vlm_image_features.dtype)
        prompt_embeds = prompt_embeds.expand(bsz, -1, -1)

        encoder_inputs = torch.cat([vlm_image_features, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.ones(encoder_inputs.shape[:2], dtype=torch.long, device=encoder_inputs.device)
        encoder_outputs = self._run_vlm_encoder(encoder_inputs, attention_mask)
        features = self.vlm_token_dropout(encoder_outputs)

        frequency = torch.full(
            (bsz, 1, 1),
            3.0,
            dtype=features.dtype,
            device=features.device,
        )
        action_type = torch.full(
            (bsz,),
            self.action_registry.index(action_space_name),
            dtype=torch.long,
            device=features.device,
        )

        cond = {
            "features": features,
            "attention_mask": attention_mask.to(device=features.device, dtype=torch.bool),
            "frequency_embeds": self.frequency_embedder(frequency),
            "action_type": action_type,
        }
        if self.config.use_proprio:
            if OBS_STATE not in batch:
                raise ValueError("FLOWER `use_proprio=True` requires `observation.state` in the batch.")
            state = batch[OBS_STATE]
            if state.ndim == 3:
                state = state[:, -1]
            cond["proprio"] = state.to(device=features.device, dtype=features.dtype)
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
            if self.training and self.config.random_shift_aug:
                pad = (
                    self.config.random_shift_top_pad
                    if key in self.config.top_camera_keys or key.endswith("top_view")
                    else self.config.random_shift_wrist_pad
                )
                image = self._random_shift_images(image, pad)
            image = image.reshape(bsz, timesteps, channels, self.config.image_size, self.config.image_size)
            images.append(image)
        return images

    def _random_shift_images(self, images: Tensor, pad: int) -> Tensor:
        if pad <= 0:
            return images
        _, _, height, width = images.shape
        if height != width:
            return images
        padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
        eps = 1.0 / (height + 2 * pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            height + 2 * pad,
            device=images.device,
            dtype=images.dtype,
        )[:height]
        arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
        shift = torch.randint(
            0,
            2 * pad + 1,
            size=(images.shape[0], 1, 1, 2),
            device=images.device,
        ).to(dtype=images.dtype)
        shift *= 2.0 / (height + 2 * pad)
        return F.grid_sample(padded, base_grid + shift, padding_mode="zeros", align_corners=False)

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
        default_dtype = next(self.parameters()).dtype
        bsz, seq_len, _ = z.shape
        z = z.to(dtype=default_dtype)
        features = cond["features"].to(dtype=default_dtype)
        frequency_embeds = cond["frequency_embeds"].to(dtype=default_dtype)
        while frequency_embeds.ndim > 2:
            frequency_embeds = frequency_embeds.squeeze(1)

        x = self.action_encoders[action_space_name](z)
        if not self.config.use_rope and not self.config.use_nope:
            x = x + self.positional_encoding[:, :seq_len].to(device=x.device, dtype=x.dtype)

        t_emb = self.t_embedder(t.reshape(bsz, -1)[:, 0])
        proprio_embeds = torch.zeros_like(frequency_embeds)
        if self.proprio_encoders is not None and "proprio" in cond:
            proprio = cond["proprio"].to(dtype=default_dtype)
            proprio_embeds = self.proprio_encoders[action_space_name](proprio)

        global_cond = (
            stateless_norm(t_emb)
            + stateless_norm(frequency_embeds)
            + stateless_norm(proprio_embeds)
        )

        context = self.cond_linear(self.cond_norm(features))
        if self.config.use_adaln_cond:
            vlm_token = context[:, 0] if self.config.use_readout_token else context.mean(dim=1)
            global_cond = global_cond + vlm_token

        if self.config.action_type_adaln:
            global_adaln = self.action_adaln[action_space_name](global_cond)
        else:
            global_adaln = self.global_adaln(global_cond)

        context_mask = cond.get("attention_mask")
        context = context if self.config.use_cross_attn else None
        for block in self.dit:
            x = block(
                x,
                global_cond,
                context=context,
                custom_cross_attn_mask=context_mask,
                is_causal=self.config.use_causal_attention,
                global_adaln=global_adaln,
            )
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
        zt = (1 - t) * actions + t * noise
        target = noise - actions
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
        if self.config.sampling_type in {"logit_normal", "ln"}:
            t = torch.sigmoid(torch.randn(batch_size, 1, 1, device=device, dtype=dtype))
        elif self.config.sampling_type == "pi_zero":
            t = torch.distributions.Beta(1.5, 1.0).sample((batch_size, 1, 1)).to(device=device, dtype=dtype)
        else:
            eps = 1e-5
            t = (torch.rand(1, device=device, dtype=dtype) + torch.arange(batch_size, device=device, dtype=dtype) / batch_size) % (1 - eps)
            t = t.view(batch_size, 1, 1)
        return t.clamp(max=0.999)

    @torch.no_grad()
    def sample_actions(
        self,
        noise: Tensor,
        cond: dict[str, Tensor],
        action_space_name: str,
    ) -> Tensor:
        z = noise
        num_steps = self.config.num_sampling_steps
        for step in range(num_steps, 0, -1):
            t_value = step / num_steps
            t = torch.full((z.shape[0], 1, 1), t_value, dtype=z.dtype, device=z.device)
            velocity = self.dit_forward(z, t, cond, action_space_name)
            z = z - velocity / num_steps
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
        no_decay = ("bias", "layernorm", "layer_norm", "ln", "norm")
        decay_group = []
        no_decay_group = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name.lower() for token in no_decay):
                no_decay_group.append(param)
            else:
                decay_group.append(param)
        return [
            {"params": decay_group, "weight_decay": self.config.optimizer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0},
        ]

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
        bsz = cond["features"].shape[0]
        if noise is None:
            dtype = cond["features"].dtype
            device = cond["features"].device
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
