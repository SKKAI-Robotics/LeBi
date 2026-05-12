# TwinVLA Policy for LeBi

이 문서는 LeBi의 `twinvla` policy를 현재 검증된 환경에서 재현하고 확장하기 위한 내부 작업 노트입니다.

현재 구현은 LeRobot 표준 `PreTrainedPolicy` 인터페이스 위에서 Toy TwinVLA의 원본 모델인 `twinvla.model.twinvla.TwinVLA`를 감싸는 wrapper 구조입니다. 즉, LeBi의 dataset/processor/training loop는 그대로 사용하되, 실제 VLA backbone과 action head는 Toy TwinVLA 쪽 구현을 호출합니다.

## 현재 상태 요약

검증 완료된 baseline:

```text
policy: twinvla
backbone: Eagle2_1BTwinVLA
dataset: jellyho/aloha_handover_box
device: cuda
dtype: float32
batch_size: 1
steps tested: 1, 20
checkpoint save/load: verified
```

현재 LeBi TwinVLA finetuning은 `cuda + float32 + batch_size=1`에서 20-step smoke test까지 완료되었습니다. 20-step에서 생성된 LeBi-native checkpoint는 `TwinVLAPolicy.from_pretrained(...)`로 재로드되는 것까지 확인했습니다.

중요한 caveat:

- 이번 성공 결과는 SmolVLM 또는 SmolVLA가 아니라 `Eagle2_1BTwinVLA` 기준입니다.
- `bfloat16`은 아직 TwinVLA 내부 projection layer에서 `Float` / `BFloat16` dtype mismatch로 실패합니다.
- 20-step smoke test는 학습 파이프라인 검증이며, 실제 task 성공률이나 장시간 학습 안정성을 보장하지 않습니다.

## Architecture

고수준 흐름:

```text
LeRobot batch
  -> TwinVLA preprocessor
  -> TwinVLAPolicy
  -> Toy TwinVLA model
  -> loss or action chunk
```

학습 흐름:

```text
TwinVLAPolicy.forward()
  -> _preprocess_batch_for_twinvla()
  -> twinvla.model.base_forward()
  -> outputs["loss"]
```

추론 흐름:

```text
TwinVLAPolicy.select_action()
  -> predict_action_chunk()
  -> twinvla.model.inference()
  -> action queue
  -> one action step at a time
```

LeRobot batch에서 TwinVLA 입력으로 변환되는 핵심 요소:

```text
observation.images.*  -> numpy uint8 images [B, H, W, 3]
task                  -> language instruction
observation.state     -> proprio [B, 2 * state_dim]
action                -> action chunk [B, chunk_size, 2 * action_dim]
```

## Files

```text
configuration_twinvla.py
modeling_twinvla.py
processor_twinvla.py
```

`configuration_twinvla.py`:

- LeBi policy config를 정의합니다.
- 기본 action/state dim, chunk size, camera key, optimizer/scheduler preset을 관리합니다.
- `twinvla_pretrained_path`는 raw TwinVLA checkpoint를 직접 로드하기 위한 field입니다.
- `pretrained_path`는 LeBi-native policy checkpoint 로딩 용도로 남겨둡니다.

`modeling_twinvla.py`:

- `TwinVLAPolicy(PreTrainedPolicy)` 구현입니다.
- raw TwinVLA checkpoint가 있으면 `TwinVLA(pretrained_path=...)`로 로드합니다.
- raw checkpoint가 없으면 `singlevla_pretrained_path`와 `model_type_name`으로 Toy TwinVLA 모델을 구성합니다.
- LeRobot batch를 Toy TwinVLA의 `BatchFeature` 형식으로 변환합니다.

`processor_twinvla.py`:

- LeRobot 표준 pre/post processor를 만듭니다.
- 입력 pipeline은 batch dimension 추가, device 이동, quantile normalization을 수행합니다.
- 출력 pipeline은 action unnormalization 후 CPU로 이동합니다.

## Environment

검증된 conda env:

```text
env name: lebi
```

확인된 핵심 패키지와 CUDA 상태:

```text
torch: 2.10.0+cu128
CUDA runtime: 12.8
Driver CUDA: 13.0
torch.cuda.is_available: True
GPU: NVIDIA GB10
GPU total memory seen by PyTorch: 121.69 GiB
numpy: 2.2.6
transformers: 4.57.6
huggingface_hub: 0.36.2
```

Toy TwinVLA import를 위해 필요한 최소 추가 의존성:

```text
twinvla editable install
scipy
matplotlib
peft
```

예상 설치 방향:

```bash
conda activate lebi
cd /home/skkai/Robotics/LeBi
pip install -e .

cd /home/skkai/Robotics/toy/TwinVLA
pip install -e .
pip install scipy matplotlib peft
```

주의:

- `lebi` env는 CUDA-enabled PyTorch wheel이 필요합니다.
- CPU-only PyTorch가 설치되어 있으면 `torch.cuda.is_available()`이 `False`가 됩니다.
- 현재 `huggingface_hub 0.36.2`는 일부 LeBi declared dependency와 다를 수 있지만, dataset load, train, checkpoint save/load는 검증되었습니다.

## Dataset

사용한 Hugging Face dataset:

```text
jellyho/aloha_handover_box
```

원본 다운로드 root:

```text
/home/skkai/Robotics/datasets/jellyho__aloha_handover_box
```

LeBi TwinVLA용 변환 root:

```text
/home/skkai/Robotics/datasets/jellyho__aloha_handover_box_lebi_twinvla
```

변환 스크립트:

```text
/home/skkai/Robotics/prepare_lebi_twinvla_dataset.py
```

변환이 필요한 이유:

- 원본 dataset root에는 서로 다른 schema의 parquet shard가 섞여 있습니다.
- LeBi TwinVLA policy는 canonical key인 `observation.state`와 `action`을 기대합니다.
- 원본 TwinVLA ALOHA mapping은 `observation.state.ee_6d_pos`, `action.ee_6d_pos`를 사용합니다.

변환 후 canonical key:

```text
observation.state
action
```

카메라 key:

```text
observation.images.agentview
observation.images.wrist_left
observation.images.wrist_right
```

검증된 dataset 정보:

```text
frames: 11829
episodes: 50
state shape: (20,)
single action shape: (20,)
chunked action shape: (20, 20)
```

현재 성공한 LeBi command에서는 다음 camera mapping을 명시합니다.

```text
--policy.primary_camera observation.images.agentview
--policy.left_wrist_camera observation.images.wrist_left
--policy.right_wrist_camera observation.images.wrist_right
```

## Checkpoints

raw TwinVLA task checkpoint:

```text
/home/skkai/Robotics/checkpoints/jellyho__TwinVLA-aloha_handover_box
```

성공한 LeBi GPU 20-step checkpoint:

```text
/home/skkai/Robotics/checkpoints/lebi_twinvla_gpu_ft_20steps_fp32/checkpoints/000020/pretrained_model
```

checkpoint field 구분:

```text
twinvla_pretrained_path
  raw Toy TwinVLA / Hugging Face checkpoint를 로드할 때 사용

pretrained_path
  LeBi-native policy checkpoint를 로드할 때 사용
```

중요:

- raw TwinVLA checkpoint config는 LeBi policy config 형식이 아닙니다.
- raw checkpoint를 `policy.pretrained_path`로 직접 넣으면 LeBi의 `PreTrainedConfig.from_pretrained()` 계약과 맞지 않습니다.
- 그래서 raw checkpoint는 `--policy.twinvla_pretrained_path`로 전달합니다.

## Finetuning Command

현재 GPU 20-step smoke test가 성공한 command:

```bash
conda run -n lebi python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id jellyho/aloha_handover_box \
  --dataset.root /home/skkai/Robotics/datasets/jellyho__aloha_handover_box_lebi_twinvla \
  --policy.type twinvla \
  --policy.twinvla_pretrained_path /home/skkai/Robotics/checkpoints/jellyho__TwinVLA-aloha_handover_box \
  --policy.action_dim 10 \
  --policy.state_dim 10 \
  --policy.device cuda \
  --policy.dtype float32 \
  --policy.primary_camera observation.images.agentview \
  --policy.left_wrist_camera observation.images.wrist_left \
  --policy.right_wrist_camera observation.images.wrist_right \
  --policy.push_to_hub false \
  --batch_size 1 \
  --steps 20 \
  --num_workers 0 \
  --eval_freq 0 \
  --save_checkpoint true \
  --save_freq 20 \
  --wandb.enable false \
  --output_dir /home/skkai/Robotics/checkpoints/lebi_twinvla_gpu_ft_20steps_fp32
```

메모리 측정용으로 checkpoint 저장 없이 실행한 command:

```bash
conda run -n lebi python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id jellyho/aloha_handover_box \
  --dataset.root /home/skkai/Robotics/datasets/jellyho__aloha_handover_box_lebi_twinvla \
  --policy.type twinvla \
  --policy.twinvla_pretrained_path /home/skkai/Robotics/checkpoints/jellyho__TwinVLA-aloha_handover_box \
  --policy.action_dim 10 \
  --policy.state_dim 10 \
  --policy.device cuda \
  --policy.dtype float32 \
  --policy.primary_camera observation.images.agentview \
  --policy.left_wrist_camera observation.images.wrist_left \
  --policy.right_wrist_camera observation.images.wrist_right \
  --policy.push_to_hub false \
  --batch_size 1 \
  --steps 20 \
  --num_workers 0 \
  --eval_freq 0 \
  --save_checkpoint false \
  --wandb.enable false \
  --output_dir /home/skkai/Robotics/checkpoints/lebi_twinvla_gpu_memcheck_internal_20steps_fp32
```

## Hyperparameters and GPU Memory

검증된 TwinVLA hyperparameters:

```text
action_dim: 10
state_dim: 10
chunk_size: 20
n_action_steps: 20
n_obs_steps: 1
image_resolution: 448 x 448
model_type_name: Eagle2_1BTwinVLA
dit_size: DiT-L
action_head: DiT
train_denoising_steps: 100
test_denoising_steps: 10
denoiser: DDIM
share_vision: true
share_decoder: true
share_embed_tokens: true
enable_moe: true
enable_joint_attn: true
```

optimizer / scheduler:

```text
optimizer: AdamW
lr: 1e-4
weight_decay: 0.01
betas: [0.9, 0.95]
eps: 1e-8
grad_clip_norm: 1.0

scheduler: cosine_decay_with_warmup
warmup_steps: 500
decay_steps: 50000
decay_lr: 1e-6
```

GPU memory 측정 조건:

```text
device: cuda
dtype: float32
batch_size: 1
steps: 20
save_checkpoint: false
```

측정 결과:

```text
total parameters: 1,357,359,914
PyTorch max allocated: 26.09 GiB
PyTorch max reserved: 28.67 GiB
external used-memory increase: about 32.11 GiB
```

메모리 분해 추정:

```text
model weights: about 5.06 GiB
gradients: about 5.06 GiB
AdamW states: about 10.11 GiB
activation/batch/temp: about 5.86 GiB
allocator/cache headroom: about 2.58 GiB
```

실무적 판단:

```text
batch_size=1, float32 finetuning: minimum 32GB-class GPU
recommended: 40GB or more
batch_size > 1: not measured
```

20-step runtime:

```text
without checkpoint save: about 37 seconds
with checkpoint save: about 54 seconds
```

## Known Issues

### `bfloat16` dtype mismatch

현재 `--policy.dtype bfloat16`은 실패합니다.

대표 에러:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

현재 안정 설정은 다음입니다.

```text
--policy.dtype float32
```

### 현재 성공은 Smol 계열이 아님

이번에 검증된 LeBi TwinVLA 경로는 SmolVLM/SmolVLA 기반이 아닙니다.

현재 baseline:

```text
model_type_name: Eagle2_1BTwinVLA
singlevla_pretrained_path: jellyho/TwinVLA
twinvla_pretrained_path: /home/skkai/Robotics/checkpoints/jellyho__TwinVLA-aloha_handover_box
```

SmolVLM2 기반 TwinVLA를 사용하려면 Toy TwinVLA repo의 SingleVLA adapter뿐 아니라 `twinvla/model/twinvlas/`의 dual-arm wrapper/config도 별도로 맞춰야 합니다.

### GB10 capability warning

PyTorch 실행 시 다음 경고가 출력됩니다.

```text
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

GPU 학습은 성공했지만 장시간 학습 안정성은 별도 확인이 필요합니다.

### `nvidia-smi` memory unsupported

GB10 환경에서 `nvidia-smi`가 다음처럼 memory usage를 표시하지 못했습니다.

```text
Memory-Usage: Not Supported
```

따라서 memory 수치는 `torch.cuda.max_memory_allocated()`와 `torch.cuda.max_memory_reserved()` 기준을 우선 사용했습니다.

### FlashAttention2 미설치

실행 중 다음 메시지가 출력됩니다.

```text
FlashAttention2 is not installed.
```

기능상 20-step 학습은 성공했습니다. 속도나 메모리 최적화가 필요할 때만 별도 설치를 검토하세요.

### torchcodec fallback

실행 중 다음 메시지가 출력됩니다.

```text
'torchcodec' is not available in your platform, falling back to 'pyav'
```

현재는 `pyav` fallback으로 dataset video decoding이 동작합니다.

### 성능 검증은 아직 아님

20-step smoke test는 다음을 확인한 것입니다.

```text
dataset load
forward/backward
optimizer step
checkpoint save
checkpoint reload
GPU execution
```

하지만 실제 robot task 성공률은 아직 평가하지 않았습니다. 실제 행동 성능을 확인하려면 충분한 step 수로 학습한 뒤 rollout/evaluation을 별도로 수행해야 합니다.

## Quick Validation

CUDA 확인:

```bash
conda run -n lebi python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

TwinVLA import 확인:

```bash
conda run -n lebi python -c "from twinvla.model.twinvla import TwinVLA; print('TwinVLA import ok')"
```

LeBi-native checkpoint reload 확인:

```bash
conda run -n lebi python -c "from lerobot.policies.twinvla.modeling_twinvla import TwinVLAPolicy; path='/home/skkai/Robotics/checkpoints/lebi_twinvla_gpu_ft_20steps_fp32/checkpoints/000020/pretrained_model'; policy=TwinVLAPolicy.from_pretrained(path, local_files_only=True); print('loaded', policy.config.type, policy.config.device, policy.config.dtype, policy.config.twinvla_pretrained_path, policy.config.action_dim, policy.config.state_dim)"
```

Dataset 경로 확인:

```bash
ls -lh /home/skkai/Robotics/datasets/jellyho__aloha_handover_box_lebi_twinvla
```

README 기준 successful checkpoint:

```text
/home/skkai/Robotics/checkpoints/lebi_twinvla_gpu_ft_20steps_fp32/checkpoints/000020/pretrained_model
```

