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

from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.twinvla_v2.configuration_twinvla_v2 import TwinVLAV2Config
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


def _build_model_args(config: TwinVLAV2Config) -> SimpleNamespace:
    """Translate LeRobot TwinVLAConfig into TwinVLA's model_args namespace."""
    return SimpleNamespace(
        model_type=config.model_type_name,
        singlevla_pretrained_path=config.singlevla_pretrained_path,
        action_dim=config.action_dim,
        state_dim=config.state_dim,
        action_len=config.chunk_size,
        action_head=config.action_head,
        dit_size=config.dit_size,
        num_readouts=config.num_readouts,
        train_denoising_steps=config.train_denoising_steps,
        test_denoising_steps=config.test_denoising_steps,
        denoiser=config.denoiser,
        normalization=config.normalization_type,
        share_vision=config.share_vision,
        share_decoder=config.share_decoder,
        share_embed_tokens=config.share_embed_tokens,
        attn_reweighting=config.attn_reweighting,
        enable_moe=config.enable_moe,
        enable_joint_attn=config.enable_joint_attn,
    )


def _pad_and_stack(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of single-sample dicts into a batched dict with padding."""
    keys = items[0].keys()
    collated: dict[str, Tensor] = {}
    for key in keys:
        vals = [item[key] for item in items]
        if isinstance(vals[0], torch.Tensor):
            if vals[0].dim() >= 1:
                max_len = max(v.shape[0] for v in vals)
                if any(v.shape[0] != max_len for v in vals):
                    padded = []
                    for v in vals:
                        pad_size = max_len - v.shape[0]
                        if pad_size > 0:
                            pad_shape = (pad_size, *v.shape[1:])
                            padded.append(torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)], dim=0))
                        else:
                            padded.append(v)
                    vals = padded
            collated[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], np.ndarray):
            collated[key] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            collated[key] = vals
    return collated


class TwinVLAV2Policy(PreTrainedPolicy):
    config_class = TwinVLAV2Config
    name = "twinvla_v2"

    def __init__(self, config: TwinVLAV2Config, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        from twinvla.model.twinvla import TwinVLA

        if config.twinvla_pretrained_path:
            logger.info("Loading raw TwinVLA checkpoint: %s", config.twinvla_pretrained_path)
            self.twinvla = TwinVLA(
                pretrained_path=config.twinvla_pretrained_path,
                device="cpu",
                dtype=self._resolve_dtype(),
            )
        else:
            logger.info("Building TwinVLA from SingleVLA: %s", config.singlevla_pretrained_path)
            model_args = _build_model_args(config)
            self.twinvla = TwinVLA(
                model_args=model_args,
                device="cpu",
                dtype=self._resolve_dtype(),
            )

        self.twinvla.model.to(dtype=self._resolve_dtype())

        if config.freeze_vision_backbone:
            self._freeze_vision()

        self.reset()

    def _resolve_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def _freeze_vision(self):
        model = self.twinvla.model
        for attr in ("vision_c", "vision_l", "vision_r"):
            module = getattr(model, attr, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
        logger.info("Vision backbone frozen.")

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    # ------------------------------------------------------------------ #
    #  Batch conversion helpers
    # ------------------------------------------------------------------ #

    def _extract_images(self, batch: dict[str, Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 3 camera views from batch → numpy [B, H, W, 3] uint8."""
        img_keys = [
            self.config.primary_camera,
            self.config.left_wrist_camera,
            self.config.right_wrist_camera,
        ]
        arrays = []
        for key in img_keys:
            tensor = batch[key]  # [B, 3, H, W] in [0, 1]
            tensor = tensor.float().clamp(0, 1) * 255.0
            arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            arrays.append(arr)
        return arrays[0], arrays[1], arrays[2]

    def _extract_language(self, batch: dict) -> list[str]:
        task = batch.get("task", None)
        if task is None:
            return [""] * next(iter(batch.values())).shape[0]
        if isinstance(task, str):
            return [task]
        return list(task)

    def _preprocess_batch_for_twinvla(
        self,
        batch: dict[str, Tensor],
        include_action: bool = True,
    ) -> "BatchFeature":  # noqa: F821
        """Convert a LeRobot batch into TwinVLA's expected BatchFeature format.

        Handles per-sample tokenisation (instruction-dependent sequence lengths)
        with padding to the longest sequence in the batch.
        """
        from transformers.feature_extraction_utils import BatchFeature

        img_primary, img_wrist_l, img_wrist_r = self._extract_images(batch)
        instructions = self._extract_language(batch)
        proprio = batch[OBS_STATE]  # [B, 2*state_dim]

        bsz = proprio.shape[0]
        model = self.twinvla.model

        all_items: list[dict] = []
        for i in range(bsz):
            single = model.preprocess_inputs(
                img_primary[i : i + 1],
                img_wrist_r[i : i + 1],
                img_wrist_l[i : i + 1],
                instructions[i],
                action=None,
            )
            all_items.append(single)

        collated = _pad_and_stack(all_items)
        collated["proprio"] = proprio.unsqueeze(1)  # [B, 1, 2*state_dim]

        if include_action and batch.get(ACTION) is not None:
            collated["action"] = batch[ACTION]  # [B, chunk_size, 2*action_dim]

        twinvla_batch = BatchFeature(collated)
        device = next(model.parameters()).device
        dtype = self._resolve_dtype()
        twinvla_batch = twinvla_batch.to(device=device, dtype=dtype)
        return twinvla_batch

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        twinvla_batch = self._preprocess_batch_for_twinvla(batch, include_action=True)
        # TwinVLAMetaModel.base_forward → (output_dict, attn_weights | None)
        outputs = self.twinvla.model.base_forward(twinvla_batch)[0]
        loss = outputs["loss"]
        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.twinvla.model.eval()
        twinvla_batch = self._preprocess_batch_for_twinvla(batch, include_action=False)

        action_dim = self.config.action_dim
        action_len = self.config.chunk_size

        with torch.inference_mode():
            # TwinVLAMetaModel.inference → (normalized_action, val)
            # normalized_action: [B, action_len, 2*action_dim] (left+right concat)
            normalized_action, _ = self.twinvla.model.inference(
                twinvla_batch,
                action_len=action_len,
                action_dim=action_dim,
            )
        return normalized_action

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # actions: [B, chunk_size, 2*action_dim]
            actions = actions[:, : self.config.n_action_steps]
            # (n_action_steps, B, 2*action_dim) for queuing
            actions = actions.transpose(0, 1)
            self._action_queue.extend(actions)

        return self._action_queue.popleft()
