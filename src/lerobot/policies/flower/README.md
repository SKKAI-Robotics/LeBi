# FLOWER Policy

`flower` is a LeRobot port of the reference FLOWER VLA model in `reference/flower_vla_calvin`.

The default setup targets the collected Task1 dataset:

- `observation.images.left_wrist_cam`
- `observation.images.right_wrist_cam`
- `observation.images.top_view`
- `action` with dimension 12

The model always builds action heads for 7D, 12D, and 16D action spaces. With
`action_space="auto"`, the active head is selected from the dataset action
dimension during training or from `default_action_space` during action selection.

Example:

```bash
lerobot-train --policy.type=flower --dataset.repo_id=Task1 --steps=1 --batch_size=1
```
