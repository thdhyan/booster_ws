# K1 Training Containers (Docker + Apptainer)

Reproducible, self-contained training for the Booster K1 locomotion policies.
The base reproduces the **validated local stack exactly** (python 3.12,
torch 2.11.0+cu128, isaacsim 6.0.1 `[all,extscache]`, isaaclab 3.0.0b2,
rsl-rl-lib 5.0.1 — all pip, same recipe as the workstation venv). The
workspace source is **cloned at runtime** — the image contains only the
slow-to-build simulator + pinned dependencies.

> **Verified end-to-end on this machine**: container boot → git clone with
> submodules → editable installs → Isaac Sim headless → PPO iteration →
> checkpoint into mounted `logs/` (RTX 4060, 16 envs, 1 iter).

> ⚠️ `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` was pulled and inspected:
> despite the tag it ships Isaac Lab **6.1.16**, incompatible with the
> workspace 3.0.0b2 task code. It is kept as `docker/Dockerfile.base-nvcr`
> for a future 6.x migration only.

```
docker/
  Dockerfile.base      stage 1: parity base (isaacsim 6.0.1 + isaaclab 3.0.0b2
                       + rsl-rl 5.0.1) + optional ROS 2 Jazzy/teleop + mjlab
  Dockerfile           stage 2: thin retag of the pushed base (hub workflow)
  entrypoint.sh        runtime: clone repo (submodules+LFS) -> pip -e pkgs -> train.py
  run.sh               host-side runner (env-file, GPUs, mounts, CLI passthrough)
  .env.example         secrets template — copy to .env, NEVER commit .env
  apptainer/k1_train.def  Apptainer/Singularity definition for HPC clusters
```

## 0. Prerequisites

- NVIDIA driver + `nvidia-container-toolkit` (docker) / working GPU in apptainer
- ~40 GB free disk for the base image (nvcr isaaclab is ~25 GB unpacked)
- Accounts/tokens: [wandb](https://wandb.ai/authorize), [HF](https://huggingface.co/settings/tokens),
  GitHub PAT **only if** the repo/submodules are private

## 1. Secrets (never baked into images)

```bash
cp docker/.env.example docker/.env   # then edit: WANDB_API_KEY, HF_TOKEN, GITHUB_TOKEN
```

`docker/.env` is gitignored. It is injected at **runtime** via
`docker run --env-file` / `apptainer --env-file`. Nothing is stored in any
image layer — safe to push to Docker Hub.

## 2. Docker — staged build & push (recommended)

```bash
# stage 1: base (once; ~30-60 min on first build, ~25 GB)
docker build -f docker/Dockerfile.base \
    --build-arg INSTALL_ROS=1 --build-arg INSTALL_MJLAB=0 \
    -t thdhyan/k1-isaac-base:3.0.0b2 .
docker push thdhyan/k1-isaac-base:3.0.0b2

# stage 2: thin runtime tag
docker build -f docker/Dockerfile \
    --build-arg BASE_IMAGE=docker.io/thdhyan/k1-isaac-base:3.0.0b2 \
    -t thdhyan/k1-isaac-train:3.0.0b2 .
docker push thdhyan/k1-isaac-train:3.0.0b2
```

Build args on the base: `INSTALL_ROS=0|1` (ROS 2 Jazzy + teleop pkgs; default 1),
`INSTALL_MJLAB=0|1` (mjlab from git; default 0), `RSL_RL_VERSION` (default 5.0.1).

### Run training

```bash
# stage-1 teacher (privileged: height scan + noise-free obs)
docker/run.sh --task Isaac-Velocity-Rough-K1-Teacher-v0 --num-envs 4096 --iters 5000

# stage-2 student distillation (teacher ckpt from stage 1)
docker/run.sh --branch dev/phase-0 -- \
  python_dummy_ignored --task Isaac-Velocity-Distill-K1-v0 \
  --teacher_checkpoint logs/rsl_rl/k1_velocity_teacher/<run>/model_4999.pt
# (for the student, also set: -e K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/train_student.py)

# iterate on local uncommitted code (bind-mount this checkout instead of clone)
docker/run.sh --source --task Isaac-Velocity-Rough-K1-Teacher-v0
```

- Checkpoints stream into `./logs/` (bind mount), models into `./models/`.
- Branch via `--branch` (default `main`); private repos via `GITHUB_TOKEN` in `.env`.
- WandB runs appear under `booster_k1_locomotion` / `thakk100-dhyan-home`.

## 3. Docker — one-stage alternative (no Docker Hub)

```bash
docker build -f docker/Dockerfile.base -t k1-isaac-train:local .
K1_IMAGE=k1-isaac-train:local docker/run.sh --task Isaac-Velocity-Rough-K1-Teacher-v0
```

## 4. Apptainer / Singularity (university cluster)

Already installed locally (`apptainer 1.5.3` via `ppa:apptainer/ppa`). On the
cluster, either build from the `.def` or pull the pushed docker image:

```bash
# option A: build from def (needs nvcr pull rights on that machine)
apptainer build k1-train.sif docker/apptainer/k1_train.def

# option B: pull from docker hub (recommended for clusters)
apptainer pull k1-train.sif docker://docker.io/thdhyan/k1-isaac-train:3.0.0b2
```

Run (from a dir containing `logs/`):

```bash
apptainer run --nv --cleanenv \
  --bind "$PWD/logs:/workspace/mounts/logs" \
  --bind "$PWD/models:/workspace/mounts/models" \
  --env K1_GIT_BRANCH=main \
  --env-file docker/.env \
  k1-train.sif \
  --task Isaac-Velocity-Rough-K1-Teacher-v0 --num_envs 4096 --headless
```

Slurm wrapper:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00
apptainer run --nv --cleanenv \
  --bind "$SLURM_SUBMIT_DIR/logs:/workspace/mounts/logs" \
  --env-file "$SLURM_SUBMIT_DIR/docker/.env" \
  k1-train.sif --task Isaac-Velocity-Rough-K1-Teacher-v0 --num_envs 4096 --headless
```

Notes:
- The clone lands in `/tmp/booster_ws` (cluster homes can be read-only in the
  container); bind a scratch dir over `/tmp` to cache it.
- `--nv` passes the host NVIDIA driver; `--cleanenv` keeps host env out.
- If pulls fail with `invalid username/password`, your `~/.docker/config.json`
  has stale credentials apptainer picks up — run with `HOME=<clean dir>` or
  `apptainer registry logout`.

## 5. Pitfalls (learned here, also encoded in the `/isaac-container` skill)

- **Default branch is `dev/phase-0`** — `main` predates the training code.
  Commit + push your working branch so the container can clone it; use
  `--source` for uncommitted local work.
- **Never source ROS into Isaac processes** (`K1_SOURCE_ROS=1` is for
  non-Isaac workflows only — RMW double-init).
- rsl-rl is pinned to **5.0.1** and the sim stack to isaacsim 6.0.1 /
  isaaclab 3.0.0b2 to match local runs — don't bump casually.
- Submodules use SSH URLs; the entrypoint rewrites them to HTTPS (token-aware)
  and fails loudly if any submodule is missing.
- USD/kit writes go to the clone dir — keep it on a writable, non-NFS path
  when possible (`/tmp`, `/scratch`).
- First wandb run per container prompts for login if `WANDB_API_KEY` is unset —
  set it in `.env` or `WANDB_MODE=offline`.
- Editing `docker/entrypoint.sh` requires rebuilding the stage-2 image
  (`docker build -f docker/Dockerfile -t thdhyan/k1-isaac-train:3.0.0b2 .`).
