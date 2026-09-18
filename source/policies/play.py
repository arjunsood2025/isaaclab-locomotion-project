"""Interactive viewer for visual inspection of the environment and a trained policy.

This is the "look at it with your own eyes" tool. Metrics tell you *whether* a policy
works; the viewport tells you *how* — skating feet, toe dragging, a hopping gait, or a
camera pointed at the sky are all obvious in one second of video and invisible in a
success-rate column.

Three action sources, chosen with ``--policy``:
    checkpoint  load a trained policy (default; needs a run in logs/rsl_rl/<experiment>)
    zero        hold the default standing pose -- the right choice for inspecting
                terrain generation and camera placement before any policy exists
    random      sample uniform actions, to see the safety envelope of the action scale

Examples:
    # Inspect the terrain grid and the robot at rest (no checkpoint required)
    python source/policies/play.py --task Go2-Rough-Play-v0 --policy zero --num_envs 16

    # Watch a trained policy walk
    python source/policies/play.py --task Go2-Flat-Play-v0 \
        --agent_cfg configs/ppo_baseline.yaml --experiment_name go2_flat --num_envs 16

    # Verify what the depth camera actually sees, and save frames for the writeup
    python source/policies/play.py --task Go2-Vision-Play-v0 --policy zero \
        --num_envs 4 --enable_cameras --save_depth media/depth

Note: this script deliberately does NOT pass --headless. Add it only if you want to run
the depth dump without a window.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visually inspect an env or a trained policy.")
parser.add_argument("--task", type=str, default="Go2-Rough-Play-v0",
                    help="Registered task id (use a -Play variant).")
parser.add_argument("--policy", type=str, default="checkpoint",
                    choices=["checkpoint", "zero", "random"],
                    help="Action source. 'zero' and 'random' need no trained model.")
parser.add_argument("--agent_cfg", type=str, default="configs/ppo_baseline.yaml",
                    help="Agent YAML the checkpoint was trained with.")
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override the log folder name to load from.")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=16,
                    help="Keep this small: a GUI session shares the GPU with training.")
parser.add_argument("--steps", type=int, default=100000, help="Control steps to run.")
parser.add_argument("--save_depth", type=str, default=None,
                    help="Directory to save a strip of depth-camera frames as PNGs "
                         "(vision tasks only; requires --enable_cameras).")
parser.add_argument("--depth_frames", type=int, default=6, help="How many depth frames to save.")
parser.add_argument("--depth_panels", type=int, default=4,
                    help="Environments shown side by side in each saved frame.")
parser.add_argument("--depth_interval", type=int, default=25,
                    help="Control steps between saved depth frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.save_depth:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402
from source.policies.utils import load_agent_cfg, resolve_checkpoint  # noqa: E402


def save_depth_frames(image: torch.Tensor, out_dir: str, index: int, num_envs: int = 4) -> None:
    """Save a strip of depth observations, one panel per environment.

    ``image`` is (N, 1, H, W) normalized inverse depth in [0, 1]: near = bright.
    Panelling several envs matters because the play terrain spreads robots across the
    difficulty grid -- one env alone may be standing on a flat patch and tell you
    nothing about whether the camera resolves stair edges or gaps.

    Fixed vmin/vmax keeps panels comparable; an autoscaled colormap would make a
    featureless frame look as detailed as a rich one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = min(num_envs, image.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), dpi=150)
    axes = [axes] if n == 1 else list(axes)
    for env_id, ax in enumerate(axes):
        frame = image[env_id, 0].detach().cpu().numpy()
        im = ax.imshow(frame, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"env {env_id}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"depth observation, t={index}", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, label="far ←→ near")
    path = os.path.join(out_dir, f"depth_{index:04d}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved {path}")


def main():
    env_cfg = tasks.get_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    # terrain and command arrows are the whole point of looking, so turn them on
    env_cfg.scene.terrain.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = base_env.device
    num_actions = base_env.action_manager.total_action_dim

    policy = None
    if args_cli.policy == "checkpoint":
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_agent_cfg(args_cli.agent_cfg)
        experiment = args_cli.experiment_name or agent_cfg["experiment_name"]
        log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", experiment)
        ckpt_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Playing checkpoint: {ckpt_path}")
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=device)
    else:
        print(f"[INFO] No checkpoint: using '{args_cli.policy}' actions.")

    has_camera = "tiled_camera" in base_env.scene.sensors
    if args_cli.save_depth and not has_camera:
        print("[WARN] --save_depth given but this task has no camera; ignoring.")

    obs = env.get_observations()
    saved = 0
    with torch.inference_mode():
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            if policy is not None:
                actions = policy(obs)
            elif args_cli.policy == "random":
                actions = torch.rand(args_cli.num_envs, num_actions, device=device) * 2 - 1
            else:
                actions = torch.zeros(args_cli.num_envs, num_actions, device=device)
            obs, _, _, _ = env.step(actions)

            if (
                args_cli.save_depth
                and has_camera
                and saved < args_cli.depth_frames
                and step % args_cli.depth_interval == 0
            ):
                save_depth_frames(obs["depth"], args_cli.save_depth, step, args_cli.depth_panels)
                saved += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
