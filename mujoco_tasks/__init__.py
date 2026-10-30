"""MuJoCo/mjlab tasks for the Booster K1.

Importing this package does not import mjlab; call load_tasks() to register the tasks.
"""

from .load_mjlab import init_mjlab, load_config, load_tasks, make_env_cfg, make_rl_cfg

__all__ = ["init_mjlab", "load_config", "load_tasks", "make_env_cfg", "make_rl_cfg"]
