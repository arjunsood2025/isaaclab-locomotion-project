"""Export a trained policy to TorchScript (and optionally ONNX) for deployment, verify
numerical parity with the eager model, and benchmark inference latency.

The exported artifact is a self-contained ``forward(obs) -> actions`` module:
    - blind policy:  MLP actor (+ observation normalizer if training used one)
    - vision policy: depth CNN encoder + MLP actor fused into one graph

Latency matters because the control loop runs at 50 Hz: the *entire* loop (state read,
obs assembly, inference, publish) has a 20 ms budget; inference should use well under
half of it. The benchmark reports mean / p50 / p95 / p99 on both CPU and (if available)
GPU so the deployment target can be chosen with data.

Example:
    python source/policies/export_policy.py --task Go2-Rough-Play-v0 \
        --agent_cfg configs/ppo_domain_randomized.yaml --headless
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
parser.add_argument("--output_dir", type=str, default="exported")
parser.add_argument("--onnx", action="store_true", help="Also export ONNX (opset 17).")
parser.add_argument("--bench_iters", type=int, default=1000,
                    help="Iterations for the latency benchmark.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------

import copy  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import source.tasks as tasks  # noqa: E402
from source.policies.models import ActorCriticVision, register_custom_modules  # noqa: E402
from source.policies.utils import load_agent_cfg, resolve_checkpoint  # noqa: E402


class PolicyExporter(nn.Module):
    """Deployment wrapper: obs normalizer (optional) + encoder (vision only) + actor.

    Deep-copies the trained submodules so tracing cannot mutate runner state, and
    reimplements only the *deterministic inference path* (the distribution machinery
    in ActorCritic does not need to ship).
    """

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
        if isinstance(actor_critic, ActorCriticVision):
            self.encoder = copy.deepcopy(actor_critic.encoder)
            self.prop_dim = actor_critic.prop_dim
            self.image_shape = actor_critic.image_shape
        else:
            self.encoder = None
            self.prop_dim = -1
            self.image_shape = (0,)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        if self.encoder is not None:
            prop = obs[:, : self.prop_dim]
            image = obs[:, self.prop_dim :].reshape(-1, *self.image_shape)
            obs = torch.cat([prop, self.encoder(image)], dim=-1)
        return self.actor(obs)


def benchmark(module: nn.Module, example: torch.Tensor, iters: int, label: str):
    """Fixed-input latency benchmark; reports mean/p50/p95/p99 in milliseconds."""
    device = example.device
    with torch.inference_mode():
        for _ in range(50):  # warmup (JIT optimization passes, cuDNN autotune)
            module(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            module(example)
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

    # a tiny env instance is spun up only to recover obs/action dims and build the
    # runner exactly as it was during training -- guarantees dim agreement
    env_cfg = tasks.get_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = 2
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    register_custom_modules()
    log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl", agent_cfg["experiment_name"])
    ckpt_path = resolve_checkpoint(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Exporting checkpoint: {ckpt_path}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cpu")
    runner.load(ckpt_path, load_optimizer=False)

    normalizer = runner.obs_normalizer if agent_cfg.get("empirical_normalization") else None
    exporter = PolicyExporter(runner.alg.actor_critic, normalizer).eval()

    obs, _ = env.get_observations()
    example = obs[:1].detach().cpu().clone()

    out_dir = os.path.join(REPO_ROOT, args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -- TorchScript (trace: our graph is control-flow-free at inference)
    traced = torch.jit.trace(exporter, example)
    jit_path = os.path.join(out_dir, "policy.pt")
    traced.save(jit_path)
    print(f"[INFO] TorchScript saved: {jit_path}")

    # -- parity check: traced graph must match eager to float precision
    with torch.inference_mode():
        max_err = (traced(example) - exporter(example)).abs().max().item()
    print(f"[INFO] Traced-vs-eager max abs error: {max_err:.2e}")
    assert max_err < 1e-5, "Traced policy diverges from eager model!"

    # -- ONNX (for TensorRT / onnxruntime deployment)
    if args_cli.onnx:
        onnx_path = os.path.join(out_dir, "policy.onnx")
        torch.onnx.export(
            exporter, example, onnx_path, opset_version=17,
            input_names=["obs"], output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        print(f"[INFO] ONNX saved: {onnx_path}")

    # -- latency benchmark, batch size 1 (the deployment case)
    benchmark(traced, example, args_cli.bench_iters, "TorchScript CPU (batch=1)")
    if torch.cuda.is_available():
        traced_gpu = torch.jit.trace(exporter.to("cuda"), example.to("cuda"))
        benchmark(traced_gpu, example.to("cuda"), args_cli.bench_iters,
                  "TorchScript CUDA (batch=1)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
