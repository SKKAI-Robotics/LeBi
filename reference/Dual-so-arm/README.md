
https://www.notion.so/Physical-AI-294938610919803d8e36f11d15433898

# 준비

`lerobot/src/lerobot/robots` 폴더에 `bi_so101_follower` 폴더를 넣어주세요.

`lerobot/src/lerobot/teleoperators` 폴더에 `bi_so101_leader` 폴더를 넣어주세요.

# 1. 새로운 로봇/텔레오퍼레이터 등록 

`bi_so101_follower`와 `bi_so101_leader`를 시스템에 등록

다음 두 파일을 수정해야 합니다:

### `src/lerobot/robots/__init__.py`

`bi_so101_follower`를 import

```python
from .bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
```

### `src/lerobot/teleoperators/__init__.py`

`bi_so101_leader`를 import

```python
from .bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
```


# lerobot_teleoperate.py 수정

`src/lerobot/scripts/lerobot_teleoperate.py` 파일을 수정

1. `lerobot.robots` import에 `bi_so101_follower` 추가
2. `lerobot.teleoperators` import에 `bi_so101_leader` 추가

변경 후:

```python
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    bi_so101_follower,  # 추가
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    bi_so101_leader,  # 추가
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
```


# 텔레오퍼레이션 실행 

## 1. 카메라 없이

```bash
lerobot-teleoperate \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/so101_follower \
  --robot.right_arm_port=/dev/so101_follower_right \
  --robot.id=bimanual_so101_follower \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/so101_leader \
  --teleop.right_arm_port=/dev/so101_leader_right \
  --teleop.id=bimanual_so101_leader \
  --display_data=false

```

## 2. 카메라와 함께

```bash
lerobot-teleoperate \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/so101_follower \
  --robot.right_arm_port=/dev/so101_follower_right \
  --robot.id=bimanual_so101_follower \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/so101_leader \
  --teleop.right_arm_port=/dev/so101_leader_right \
  --teleop.id=bimanual_so101_leader \
  --display_data=true \
  --robot.cameras='{    
    left:  {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
    right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 25}
  }'

```