#!/usr/bin/env python3
"""
Train K1 velocity locomotion policy with RSL-RL PPO.

Usage:
  # From booster_ws root, with IsaacLab env active:
  python isaac_tasks/k1_velocity/scripts/train.py \
      --task Isaac-Velocity-Rough-K1-v0 \
      --num_envs 4096 \
      --headless

  # Smoke test (5 iterations):
  python ... --task Isaac-Velocity-Rough-K1-v0 --max_iterations 5 --headless
"""
import argparse
import sys
import os

# Add k1_velocity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

def parse_args():
    parser = argparse.ArgumentParser(description="Train K1 velocity policy")
    parser.add_argument("--task", default="Isaac-Velocity-Rough-K1-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_dir", default="logs/k1_velocity")
    return parser.parse_args()

def main():
    args = parse_args()

    # Isaac Lab imports (must happen after SimulationApp init in real run)
    try:
        import isaaclab  # noqa
    except ImportError:
        print("ERROR: Isaac Lab not found. Activate your IsaacLab conda/venv environment.")
        sys.exit(1)

    # Import task to register it
    import k1_velocity.tasks.velocity  # noqa: F401

    # Use booster_train train.py pattern if available, otherwise Isaac Lab default
    try:
        from isaaclab_rl.rsl_rl.runners import OnPolicyRunner
        print(f"Task: {args.task}")
        print(f"Envs: {args.num_envs}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Device: {args.device}")
        print("TODO: wire up full RSL-RL training loop")
        print("Reference: isaac_tasks/booster_train_ref/scripts/rsl_rl/train.py")
    except ImportError as e:
        print(f"RSL-RL not available: {e}")
        print("Install: pip install rsl-rl")

if __name__ == "__main__":
    main()
