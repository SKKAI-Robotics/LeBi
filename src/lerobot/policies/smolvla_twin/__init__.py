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

"""SmolVLA-Twin policy.

A bimanual VLA built by applying the TwinVLA paper's "twin single-arm" surgery
on top of the SmolVLA (lerobot/smolvla_base) architecture.

Design (C-1):
  - VLM stream is duplicated into vlm_l / vlm_r (deepcopy of SmolVLM2 text_model layers)
  - Vision encoder, embed_tokens, final norm are shared (vlm_shared)
  - Action expert (lm_expert) and flow-matching projections are SHARED across arms
  - Per-layer common-tokens use fuse(left, right) avg with optional 2-expert MoE
  - Joint attention with cross-modal causal band (left <-> right)
  - Attention re-weighting boosting shared-modality keys
"""

from .configuration_smolvla_twin import SmolVLATwinConfig
from .modeling_smolvla_twin import SmolVLATwinPolicy
from .processor_smolvla_twin import make_smolvla_twin_pre_post_processors

__all__ = [
    "SmolVLATwinConfig",
    "SmolVLATwinPolicy",
    "make_smolvla_twin_pre_post_processors",
]
