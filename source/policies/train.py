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
parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory name to resume from (default: latest).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit checkpoint path to resume from.")
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

from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402  (registers the Go2-* gym tasks)
from source.policies.models import register_custom_modules  # noqa: E402
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
    env = RslRlVecEnvWrapper(env)

    # -- build the runner (custom policy classes must be registered first)
    register_custom_modules()
    runner = OnPolicyRunner(
        env, agent_cfg, log_dir=log_dir, device=agent_cfg.get("device", "cuda:0")
    )
    if args_cli.resume:
        resume_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # -- snapshot the exact configs used, for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # init_at_random_ep_len desynchronizes episode resets across envs so the rollout
    # buffer is not dominated by correlated post-reset states
    runner.learn(
        num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
