"""Train a locomotion policy with rsl_rl PPO in Isaac Lab.

Examples (from the repo root, inside the Isaac Lab python environment):

    # Phase 1: blind flat-ground baseline
    python source/policies/train.py --task Go2-Flat-v0 \
        --agent_cfg configs/ppo_baseline.yaml --env_cfg configs/env_flat.yaml --headless

    # Phase 3: rough terrain + curriculum
    python source/policies/train.py --task Go2-Rough-v0 \
        --agent_cfg configs/ppo_baseline.yaml --env_cfg configs/env_rough.yaml --headless

    # Phase 4: full domain randomization
    python source/policies/train.py --task Go2-Rough-DR-v0 \
        --agent_cfg configs/ppo_domain_randomized.yaml --headless

    # Extreme version: depth-vision locomotion (cameras must be enabled)
    python source/policies/train.py --task Go2-Vision-v0 \
        --agent_cfg configs/ppo_vision.yaml --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + app launch. Nothing from isaaclab.* (besides AppLauncher) or from this
# project may be imported before the simulation app exists.
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train a Go2 locomotion policy with rsl_rl PPO.")
parser.add_argument("--task", type=str, default="Go2-Rough-v0", help="Registered task id.")
parser.add_argument("--agent_cfg", type=str, default="configs/ppo_baseline.yaml",
                    help="Path to the rsl_rl agent YAML.")
parser.add_argument("--env_cfg", type=str, default=None,
                    help="Optional YAML of dotted-path env cfg overrides.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs.")
parser.add_argument("--seed", type=int, default=None, help="Override the seed.")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override training iterations.")
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override the log folder name (logs/rsl_rl/<experiment_name>). "
                         "Needed when one agent cfg is reused for several tasks -- e.g. "
                         "ppo_baseline.yaml drives both Go2-Flat-v0 and Go2-Rough-v0, "
                         "whose policies have different observation dimensions and must "
                         "not share a checkpoint directory.")
parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory name to resume from (default: latest).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit checkpoint path to resume from.")
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="Distillation only: PPO checkpoint whose actor becomes the "
                         "frozen teacher. rsl_rl routes an 'actor_state_dict' into the "
                         "teacher automatically, so this is a normal PPO model_*.pt.")
parser.add_argument("--video", action="store_true", help="Record rollout videos while training.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in env steps.")
parser.add_argument("--video_interval", type=int, default=5000,
                    help="Env steps between video recordings.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# video recording needs offscreen rendering even in headless mode
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import copy  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402  (registers the Go2-* gym tasks)
from source.policies.utils import (  # noqa: E402
    apply_env_overrides,
    load_agent_cfg,
    resolve_checkpoint,
)


def main():
    agent_cfg = load_agent_cfg(args_cli.agent_cfg)
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg["max_iterations"] = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg["experiment_name"] = args_cli.experiment_name

    # -- env config: task defaults -> YAML overrides -> CLI overrides
    env_cfg = tasks.get_env_cfg(args_cli.task)
    apply_env_overrides(env_cfg, args_cli.env_cfg)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = agent_cfg["seed"]

    # -- logging directory: logs/rsl_rl/<experiment>/<timestamp>_<run_name>
    log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", agent_cfg["experiment_name"])
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.get("run_name"):
        run_name += f"_{agent_cfg['run_name']}"
    log_dir = os.path.join(log_root, run_name)

    # -- build the environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))

    # Snapshot the config BEFORE building the runner: rsl_rl's construct_algorithm
    # pops "class_name" out of the actor/critic/algorithm sections in place, so a dump
    # taken afterwards would not be re-runnable.
    agent_cfg_snapshot = copy.deepcopy(agent_cfg)

    # -- build the runner. rsl_rl >= 5.0 resolves the model classes by name from the
    # agent cfg ("MLPModel" for the blind tasks, "CNNModel" for vision), so no custom
    # policy classes need registering.
    device = agent_cfg.get("device", "cuda:0")
    runner_name = agent_cfg.get("class_name", "OnPolicyRunner")
    if runner_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg, log_dir=log_dir, device=device)
    elif runner_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_name}")

    # -- distillation needs a trained teacher before learn() will start. Loading a PPO
    # checkpoint (one containing "actor_state_dict") makes rsl_rl copy that actor into
    # the teacher and leave the student randomly initialized, which is exactly what we
    # want: the blind DR policy teaches the depth-vision student.
    if args_cli.teacher_checkpoint is not None:
        teacher_path = args_cli.teacher_checkpoint
        if not os.path.isabs(teacher_path):
            teacher_path = os.path.join(REPO_ROOT, teacher_path)
        print(f"[INFO] Loading teacher from: {teacher_path}")
        runner.load(teacher_path)
    elif runner_name == "DistillationRunner" and not args_cli.resume:
        raise ValueError(
            "Distillation requires --teacher_checkpoint (a PPO model_*.pt to distil from)."
        )

    if args_cli.resume:
        resume_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)  # loads actor/critic (or student/teacher) + optimizer

    # -- snapshot the exact configs used, for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_snapshot)

    # init_at_random_ep_len desynchronizes episode resets across envs so the rollout
    # buffer is not dominated by correlated post-reset states
    runner.learn(
        num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
