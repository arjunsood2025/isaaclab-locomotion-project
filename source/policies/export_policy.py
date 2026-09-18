"""Export a trained policy to TorchScript (and optionally ONNX) for deployment, verify
numerical parity with the eager model, and benchmark inference latency.

Exported signatures (produced by rsl_rl's own exporters, so they stay in sync with the
training-time model definition):

    blind policy  (MLPModel):  forward(obs: Tensor[1, N]) -> Tensor[1, 12]
    vision policy (CNNModel):  forward(obs_1d: Tensor[1, 48],
                                       obs_2d: List[Tensor[1, 1, 64, 64]]) -> Tensor[1, 12]

The blind signature is exactly what ``source/deployment/`` consumes: one flat float
vector in, twelve joint position offsets out. The vision policy keeps the image as a
separate input rather than concatenating it, because the CNN needs the spatial layout.

Latency matters because the control loop runs at 50 Hz: the *entire* loop (state read,
obs assembly, inference, publish) has a 20 ms budget; inference should use well under
half of it. The benchmark reports mean / p50 / p95 / p99 on both CPU and (if available)
GPU so the deployment target can be chosen with data.

Example:
    python source/policies/export_policy.py --task Go2-Rough-Play-v0 \
        --agent_cfg configs/ppo_domain_randomized.yaml --onnx --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a trained policy for deployment.")
parser.add_argument("--task", type=str, default="Go2-Rough-Play-v0",
                    help="Task the checkpoint was trained on (used to get obs/action dims).")
parser.add_argument("--agent_cfg", type=str, default="configs/ppo_baseline.yaml")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override the log folder name to load from "
                         "(must match what training used).")
parser.add_argument("--output_dir", type=str, default="exported")
parser.add_argument("--onnx", action="store_true", help="Also export ONNX.")
parser.add_argument("--bench_iters", type=int, default=1000,
                    help="Iterations for the latency benchmark.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402
from source.policies.utils import load_agent_cfg, resolve_checkpoint  # noqa: E402


def build_example_inputs(policy: nn.Module, obs, device: str) -> tuple:
    """Assemble a batch-1 input tuple in the exact order the exported module expects.

    rsl_rl models expose the observation groups they were built from: ``obs_groups``
    (the 1-D groups, concatenated in order) and, for CNN models, ``obs_groups_2d`` (one
    image tensor per CNN encoder). Reproducing that ordering here is what guarantees the
    exported artifact is fed the layout it was trained on -- the same contract the
    deployment code has to honour.
    """
    obs_1d = torch.cat([obs[group] for group in policy.obs_groups], dim=-1)[:1]
    obs_1d = obs_1d.to(device).contiguous()
    groups_2d = getattr(policy, "obs_groups_2d", [])
    if groups_2d:
        obs_2d = [obs[group][:1].to(device).contiguous() for group in groups_2d]
        return (obs_1d, obs_2d)
    return (obs_1d,)


def benchmark(module: nn.Module, example: tuple, iters: int, label: str) -> dict:
    """Fixed-input latency benchmark; reports mean/p50/p95/p99 in milliseconds."""
    device = example[0].device
    with torch.inference_mode():
        for _ in range(50):  # warmup (JIT optimization passes, cuDNN autotune)
            module(*example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            module(*example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = sum(times) / len(times)
    p = lambda q: times[min(int(q * len(times)), len(times) - 1)]  # noqa: E731
    print(f"[BENCH] {label}: mean {mean:.3f} ms | p50 {p(0.50):.3f} | "
          f"p95 {p(0.95):.3f} | p99 {p(0.99):.3f}  (n={iters})")
    return {"mean_ms": mean, "p50_ms": p(0.50), "p95_ms": p(0.95), "p99_ms": p(0.99)}


def main():
    agent_cfg = load_agent_cfg(args_cli.agent_cfg)

    # A tiny env instance is spun up only to recover the observation layout and build the
    # runner exactly as it was during training -- this is what guarantees dim agreement
    # between the checkpoint and the exported artifact.
    env_cfg = tasks.get_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 4
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    sim_device = env.unwrapped.device

    experiment = args_cli.experiment_name or agent_cfg["experiment_name"]
    log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", experiment)
    ckpt_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Exporting checkpoint: {ckpt_path}")
    runner_cls = (
        DistillationRunner
        if agent_cfg.get("class_name") == "DistillationRunner"
        else OnPolicyRunner
    )
    runner = runner_cls(env, agent_cfg, log_dir=None, device=sim_device)
    runner.load(ckpt_path)

    out_dir = os.path.join(REPO_ROOT, args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -- TorchScript. rsl_rl scripts (not traces) a deterministic-inference copy of the
    # model: the sampling distribution is replaced by its deterministic output and the
    # observation normalizer is folded in, so the artifact is self-contained.
    runner.export_policy_to_jit(path=out_dir, filename="policy.pt")
    jit_path = os.path.join(out_dir, "policy.pt")
    print(f"[INFO] TorchScript saved: {jit_path}")

    if args_cli.onnx:
        runner.export_policy_to_onnx(path=out_dir, filename="policy.onnx")
        print(f"[INFO] ONNX saved: {os.path.join(out_dir, 'policy.onnx')}")

    # -- parity check: the exported graph must match the eager model to float precision
    obs = env.get_observations()
    policy = runner.get_inference_policy(device="cpu")
    example_cpu = build_example_inputs(policy, obs, "cpu")
    traced_cpu = torch.jit.load(jit_path, map_location="cpu").eval()

    with torch.inference_mode():
        eager_out = policy(obs[:1].to("cpu"))
        jit_out = traced_cpu(*example_cpu)
        max_err = (jit_out - eager_out).abs().max().item()
    print(f"[INFO] Exported-vs-eager max abs error: {max_err:.2e}")
    assert max_err < 1e-5, "Exported policy diverges from the eager model!"

    # -- latency benchmark, batch size 1 (the deployment case)
    results = {"checkpoint": ckpt_path, "task": args_cli.task,
               "parity_max_abs_err": max_err}
    results["cpu"] = benchmark(traced_cpu, example_cpu, args_cli.bench_iters,
                               "TorchScript CPU (batch=1)")
    if torch.cuda.is_available():
        traced_gpu = torch.jit.load(jit_path, map_location="cuda").eval()
        example_gpu = build_example_inputs(policy, obs, "cuda")
        results["cuda"] = benchmark(traced_gpu, example_gpu, args_cli.bench_iters,
                                    "TorchScript CUDA (batch=1)")

    bench_path = os.path.join(out_dir, "latency.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Latency results saved: {bench_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
