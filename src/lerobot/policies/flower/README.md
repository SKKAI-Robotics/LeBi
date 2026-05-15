# FLOWER Policy

`flower` is a LeRobot port of the reference FLOWER VLA model in `reference/flower_vla_calvin`.

The default setup targets the collected local Task1 dataset described by the
repository-level `info.json` and `stats.json` files:

- `observation.images.left_left`
- `observation.images.left_top`
- `observation.images.right_right`
- `action` with dimension 12

The model always builds action heads for 7D, 12D, and 16D action spaces. With
`action_space="auto"`, the active head is selected from the dataset action
dimension during training or from `default_action_space` during action selection.

Reference FLOWER checkpoints are loaded through `policy.pretrained_model_path`.
The default raw checkpoint path is:

- `./checkpoints/flower_vla_pret/360000_model_weights.pt`

The LeRobot `pretrained_path` field is still reserved for LeRobot-format policy
checkpoints containing `config.json` and model weights. Raw FLOWER
`.pt`/`.safetensors` weights should use `policy.pretrained_model_path`.

Fine-tune on the first 250 Task1 episodes (`0..249`) from the local LeRobot
dataset at `./datasets/Task1`:

```bash
DATASET_ROOT=../datasets/Task1
PRETRAINED_MODEL=../checkpoints/flower_vla_pret/360000_model_weights.pt
EPISODES="$(printf '['; seq -s, 0 249; printf ']')"

lerobot-train \
  --policy.type=flower \
  --policy.load_pretrained=true \
  --policy.pretrained_model_path="$PRETRAINED_MODEL" \
  --policy.pretrained_use_ema=true \
  --policy.pretrained_ignore_mismatched_sizes=true \
  --dataset.repo_id=Task1 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episodes="$EPISODES" \
  --dataset.use_imagenet_stats=true \
  --policy.camera_keys='["observation.images.left_left","observation.images.left_top","observation.images.right_right"]' \
  --policy.top_camera_keys='["observation.images.left_top"]' \
  --policy.action_space=auto \
  --policy.default_action_space=bimanual_12d \
  --policy.action_dim=12 \
  --policy.lowdim_obs_dim=7 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.act_window_size=10 \
  --policy.multistep=10 \
  --policy.vlm_path=microsoft/Florence-2-large \
  --policy.trust_remote_code=true \
  --policy.freeze_florence=false \
  --policy.freeze_vision_tower=false \
  --policy.vlm_prompt_style=default \
  --policy.tokenizer_max_length=64 \
  --policy.num_prompt_tokens=1 \
  --policy.token_dropout=0.1 \
  --policy.image_size=224 \
  --policy.normalize_images=true \
  --policy.validate_image_range=true \
  --policy.random_shift_aug=true \
  --policy.random_shift_top_pad=10 \
  --policy.random_shift_wrist_pad=4 \
  --policy.dit_dim=1024 \
  --policy.n_heads=16 \
  --policy.n_layers=18 \
  --policy.mlp_ratio=4.0 \
  --policy.attn_pdrop=0.1 \
  --policy.resid_pdrop=0.1 \
  --policy.mlp_pdrop=0.1 \
  --policy.norm_eps=1e-6 \
  --policy.timestep_embed_dim=256 \
  --policy.frequency_embed_dim=256 \
  --policy.num_sampling_steps=4 \
  --policy.sampling_type=uniform \
  --policy.use_cross_attn=true \
  --policy.use_second_view=true \
  --policy.second_view_key=image_wrist \
  --policy.action_type_adaln=true \
  --policy.use_causal_attention=true \
  --policy.use_adaln_cond=false \
  --policy.use_readout_token=false \
  --policy.use_proprio=false \
  --policy.return_act_chunk=false \
  --policy.action_clip_value=1.0 \
  --policy.use_rope=true \
  --policy.use_nope=false \
  --policy.query_seq_len=100 \
  --policy.rope_theta=32.0 \
  --policy.optimizer_type=adamw \
  --policy.optimizer_lr=2e-5 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=0.05 \
  --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_init_lr_scale=0.1 \
  --policy.scheduler_final_lr_scale=0.5 \
  --policy.scheduler_total_steps=50000 \
  --policy.scheduler_phase_ratio='[0.05,0.1,0.85]' \
  --seed=242 \
  --batch_size=8 \
  --steps=50000 \
  --num_workers=12 \
  --use_policy_training_preset=true
  --
```

`policy.action_dim=12` follows Task1's `info.json`. The reference CALVIN config
uses `action_dim=7`, but using 7D here would not match the 12D Task1 actions.
