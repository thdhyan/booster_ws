# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Gym-registration callback for the unified Isaac Lab 3.0 CLI.

The unified entrypoint (``isaaclab train --rl_library rsl_rl``) only imports
``isaaclab_tasks`` / ``isaaclab_tasks_experimental``; third-party task packages
must register themselves through the ``--external_callback`` hook, which is
invoked before the preset CLI reads Gym metadata::

    isaaclab train --rl_library rsl_rl \
        --task Isaac-Velocity-Rough-K1-v0 \
        --external_callback k1_velocity.register_tasks.register_tasks ...
"""


def register_tasks() -> list[str]:
    """Import task packages so their ``gym.register`` calls run.

    Returns:
        Additional CLI args to merge (none — all args come from the command line).
    """
    import k1_velocity.tasks.velocity  # noqa: F401 — registers velocity task family
    return []
