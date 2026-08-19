#!/usr/bin/env python3
"""
Evaluate K1 velocity policy and optionally export to TorchScript.

Usage:
  python isaac_tasks/k1_velocity/scripts/play.py \
      --task Isaac-Velocity-Rough-K1-Play-v0 \
      --checkpoint logs/k1_velocity/model_5000.pt \
      --export models/k1_velocity_policy.pt
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Velocity-Rough-K1-Play-v0")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--export", default="", help="Export TorchScript to this path")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    import k1_velocity.tasks.velocity  # noqa

    if args.export:
        print(f"TODO: export policy to TorchScript -> {args.export}")
        print("Exported policy goes to models/ (tracked via Git LFS)")

if __name__ == "__main__":
    main()
