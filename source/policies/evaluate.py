"""Evaluation harness: runs a trained policy for N episodes and reports the metrics
that matter for locomotion quality and sim-to-real readiness.

Per-episode metrics (aggregated as means over completed episodes):
    success            episode reached timeout without falling
    fall               episode ended in a non-timeout termination
    distance_m         planar distance integrated over the episode
    lin_vel_err_mps    mean ||v_cmd - v||_xy per step
    ang_vel_err_rps    mean |yaw_rate_cmd - yaw_rate| per step
    energy_J           integral of sum_j |tau_j * qdot_j| dt
    energy_per_m       energy_J / distance_m
    cost_of_transport  energy_J / (m * g * distance_m)  (dimensionless, comparable
                       across robots; healthy quadruped trot ~ 0.4-1.5)

Push-recovery suite (--suite push): at t = push_time every robot receives a lateral
velocity impulse of magnitude --push_speed (random direction); recovery = no fall
within --recovery_window seconds after the push.

Robustness sweeps: pass --push_speed at increasing magnitudes and plot success rate
vs perturbation strength (see reports/final_report.md).

Examples:
    python source/policies/evaluate.py --task Go2-Rough-Play-v0 \
        --agent_cfg configs/ppo_baseline.yaml --num_episodes 100 --headless

    python source/policies/evaluate.py --task Go2-Rough-Play-v0 \
        --agent_cfg configs/ppo_domain_randomized.yaml --suite push \
        --push_speed 1.5 --num_episodes 100 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained locomotion policy.")
parser.add_argument("--task", type=str, default="Go2-Rough-Play-v0",
                    help="Registered task id (use a -Play variant).")
parser.add_argument("--agent_cfg", type=str, default="configs/ppo_baseline.yaml",
                    help="Agent YAML the checkpoint was trained with.")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to evaluate (default: latest).")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Completed episodes to collect before reporting.")
parser.add_argument("--suite", type=str, default="nominal", choices=["nominal", "push"],
                    help="'nominal': undisturbed episodes. 'push': velocity impulse mid-episode.")
parser.add_argument("--push_speed", type=float, default=1.0,
                    help="Push impulse magnitude in m/s (push suite).")
parser.add_argument("--push_time", type=float, default=5.0,
                    help="Episode time of the push in seconds (push suite).")
parser.add_argument("--recovery_window", type=float, default=3.0,
                    help="Seconds after the push within which a fall counts as failed recovery.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs.")
parser.add_argument("--output_dir", type=str, default="reports",
                    help="Directory for the JSON results file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402
from source.policies.models import register_custom_modules  # noqa: E402
from source.policies.utils import load_agent_cfg, resolve_checkpoint  # noqa: E402

GRAVITY = 9.81


class EpisodeRecorder:
    """Accumulates per-env, per-step quantities and finalizes them into episode records
    when envs terminate. All state lives on the GPU; only finished episodes cross to CPU."""

    def __init__(self, num_envs: int, device: str, robot_mass_kg: torch.Tensor):
        self.device = device
        self.mass = robot_mass_kg  # (num_envs,)
        zeros = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.steps = zeros()
        self.lin_vel_err_sum = zeros()
        self.ang_vel_err_sum = zeros()
        self.energy = zeros()
        self.distance = zeros()
        self.pushed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.push_step = zeros()
        self.episodes: list[dict] = []

    def step(self, base_env, prev_pos_xy: torch.Tensor):
        robot = base_env.scene["robot"]
        command = base_env.command_manager.get_command("base_velocity")
        dt = base_env.step_dt

        self.steps += 1
        self.lin_vel_err_sum += torch.norm(
            command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1
        )
        self.ang_vel_err_sum += torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])
        self.energy += (
            torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1) * dt
        )
        # integrate displacement stepwise (robust to teleports at reset, which are
        # handled by finalize before the next step call)
        self.distance += torch.norm(robot.data.root_pos_w[:, :2] - prev_pos_xy, dim=1)

    def finalize(self, done_ids: torch.Tensor, fell: torch.Tensor, recovery_failed: torch.Tensor):
        """Convert finished env rollouts into episode records and reset accumulators."""
        for i, env_id in enumerate(done_ids.tolist()):
            steps = max(int(self.steps[env_id].item()), 1)
            dist = self.distance[env_id].item()
            energy = self.energy[env_id].item()
            record = {
                "success": not bool(fell[i].item()),
                "fall": bool(fell[i].item()),
                "distance_m": dist,
                "lin_vel_err_mps": self.lin_vel_err_sum[env_id].item() / steps,
                "ang_vel_err_rps": self.ang_vel_err_sum[env_id].item() / steps,
                "energy_J": energy,
                "energy_per_m": energy / dist if dist > 0.05 else float("nan"),
                "cost_of_transport": (
                    energy / (self.mass[env_id].item() * GRAVITY * dist)
                    if dist > 0.05
                    else float("nan")
                ),
                "was_pushed": bool(self.pushed[env_id].item()),
                "recovery_failed": bool(recovery_failed[i].item()),
            }
            self.episodes.append(record)
        # reset accumulators for the envs that just restarted
        for buf in (self.steps, self.lin_vel_err_sum, self.ang_vel_err_sum,
                    self.energy, self.distance, self.push_step):
            buf[done_ids] = 0.0
        self.pushed[done_ids] = False


def apply_push(base_env, push_speed: float):
    """Lateral velocity impulse in a random planar direction, applied to every env."""
    robot = base_env.scene["robot"]
    num_envs = base_env.num_envs
    angle = torch.rand(num_envs, device=base_env.device) * 2 * math.pi
    vel = robot.data.root_vel_w.clone()
    vel[:, 0] += push_speed * torch.cos(angle)
    vel[:, 1] += push_speed * torch.sin(angle)
    robot.write_root_velocity_to_sim(vel)


def aggregate(episodes: list[dict]) -> dict:
    """Mean of each numeric field over episodes (nan-safe)."""
    result = {"num_episodes": len(episodes)}
    keys = ["success", "fall", "distance_m", "lin_vel_err_mps", "ang_vel_err_rps",
            "energy_J", "energy_per_m", "cost_of_transport"]
    for key in keys:
        values = [e[key] for e in episodes if not math.isnan(float(e[key]))]
        result[key] = sum(float(v) for v in values) / max(len(values), 1)
    pushed = [e for e in episodes if e["was_pushed"]]
    if pushed:
        result["push_recovery_rate"] = sum(
            0.0 if e["recovery_failed"] else 1.0 for e in pushed
        ) / len(pushed)
    return result


def main():
    agent_cfg = load_agent_cfg(args_cli.agent_cfg)
    env_cfg = tasks.get_env_cfg(args_cli.task)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = base_env.device

    # load policy
    register_custom_modules()
    log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", agent_cfg["experiment_name"])
    ckpt_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Evaluating checkpoint: {ckpt_path}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)

    robot = base_env.scene["robot"]
    total_mass = robot.data.default_mass.sum(dim=1).to(device)
    recorder = EpisodeRecorder(base_env.num_envs, device, total_mass)

    push_step_idx = int(args_cli.push_time / base_env.step_dt)
    recovery_steps = int(args_cli.recovery_window / base_env.step_dt)
    push_deadline = torch.zeros(base_env.num_envs, device=device)  # step index limit

    obs, _ = env.get_observations()
    prev_pos_xy = robot.data.root_pos_w[:, :2].clone()

    with torch.inference_mode():
        while len(recorder.episodes) < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            recorder.step(base_env, prev_pos_xy)

            # push suite: shove all envs whose episode clock hits push_time
            if args_cli.suite == "push":
                at_push = recorder.steps == push_step_idx
                if at_push.any():
                    apply_push(base_env, args_cli.push_speed)
                    recorder.pushed |= at_push
                    push_deadline[at_push] = recorder.steps[at_push] + recovery_steps

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                # non-timeout termination = fall (illegal base contact)
                fell = base_env.termination_manager.terminated[done_ids]
                # failed recovery = fell within the window after being pushed
                recovery_failed = (
                    fell
                    & recorder.pushed[done_ids]
                    & (recorder.steps[done_ids] <= push_deadline[done_ids])
                )
                recorder.finalize(done_ids, fell, recovery_failed)

            prev_pos_xy = robot.data.root_pos_w[:, :2].clone()

    results = aggregate(recorder.episodes[: args_cli.num_episodes])
    results["suite"] = args_cli.suite
    results["task"] = args_cli.task
    results["checkpoint"] = ckpt_path
    if args_cli.suite == "push":
        results["push_speed_mps"] = args_cli.push_speed

    # -- report
    print("\n===== Evaluation results =====")
    print(f"| metric | value |\n|---|---|")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"| {key} | {value:.4f} |")
        else:
            print(f"| {key} | {value} |")

    out_dir = os.path.join(REPO_ROOT, args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"eval_{args_cli.task}_{args_cli.suite}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args_cli), "results": results,
                   "episodes": recorder.episodes[: args_cli.num_episodes]}, f, indent=2)
    print(f"\n[INFO] Saved results to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
