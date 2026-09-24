"""K1 task definitions."""
import gymnasium as gym
from . import velocity  # noqa: F401
from . import basic     # noqa: F401 — P1 BASIC family; register_tasks also imports it, but package users (train_student.py) rely on this
from . import kick     # noqa: F401 — P4 chase & kick family
from . import head     # noqa: F401 — P3 head tracking family (detection-based)
