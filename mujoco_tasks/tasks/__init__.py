"""Registers the Booster K1 mjlab tasks. Importing this package requires mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mujoco_tasks.velocity.cfg import booster_k1_ppo_runner_cfg

from .k1_velocity import (
  booster_k1_flat_env_cfg,
  booster_k1_rough_env_cfg,
  booster_k1_stairs_env_cfg,
)

for _name, _fn in (
  ("Rough", booster_k1_rough_env_cfg),
  ("Flat", booster_k1_flat_env_cfg),
  ("Stairs", booster_k1_stairs_env_cfg),
):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-{_name}-Booster-K1",
    env_cfg=_fn(),
    play_env_cfg=_fn(play=True),
    rl_cfg=booster_k1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
