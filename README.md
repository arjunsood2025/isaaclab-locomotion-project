# Sim-to-Real Reinforcement Learning for Robust Quadruped Locomotion 🐕🤖

**PPO-trained locomotion policies for a Unitree Go2 in NVIDIA Isaac Lab — from a blind
flat-ground baseline to depth-vision rough-terrain traversal — with terrain curriculum,
full domain randomization, a quantitative robustness harness, and a deployment path
(TorchScript → ROS2 / C++ @ 50 Hz).**

<!-- [IMAGE-1: hero GIF — robot traversing stairs/rough terrain with a stumble-and-recovery moment.
     Create: train, then record with `train.py --task Go2-Rough-Play-v0 --video`;
     ffmpeg -i rollout.mp4 -vf "fps=15,scale=640:-1" -loop 0 hero.gif -->

## Highlights

- **4,096 parallel environments** on a single GPU; the blind rough-terrain policy trains in hours.
- **Vision in the loop:** a 64×64 egocentric depth camera replaces privileged height maps;
  a compact CNN encoder is trained end-to-end inside PPO, with an asymmetric actor-critic
  (the training-only critic keeps the privileged terrain map).
- **Sim-to-real engineering, not just a demo:** randomized mass, friction, motor strength,
  PD gains, sensor noise, *and* sensor/actuator latency; push-recovery training; every
  claim backed by a 100-episode evaluation suite.
- **Deployment-shaped:** TorchScript export with numerical parity checks and latency
  percentiles; a ROS2 inference node and a C++ fixed-rate control loop with watchdogs,
  safety clamps, and deadline-miss accounting.

<!-- [IMAGE-2: architecture diagram — recreate GUIDE.md §3.0 block diagram in draw.io/Excalidraw,
     or use a Mermaid graph directly (GitHub renders it natively)] -->

## Task ladder

| Task | Terrain | Actor observes | Randomization |
|---|---|---|---|
| `Go2-Flat-v0` | plane | proprioception (48) | mild |
| `Go2-Rough-v0` | procedural grid + curriculum | proprio + height scan (235) | mild |
| `Go2-Rough-DR-v0` | procedural grid + curriculum | proprio + height scan | **full DR + latency** |
| `Go2-Vision-v0` | procedural grid + curriculum | proprio + 64×64 depth (4144) | full DR |

## Results

| Policy | Success % | Fall % | Vel err (m/s) | Energy/m | Push recovery @1.5 m/s |
|---|---|---|---|---|---|
| A — baseline (no DR) | – | – | – | – | – |
| D — full DR | – | – | – | – | – |
| E — vision | – | – | – | – | – |

*(Fill from `reports/` after training; the story to tell: A wins slightly in clean sim
and collapses under perturbation, D dominates everywhere it matters.)*

<!-- [IMAGE-3: robustness curve — success rate vs push speed, one line per policy;
     matplotlib over the evaluate.py JSONs in reports/] -->
<!-- [IMAGE-4: training curves — mean reward + mean terrain level vs iteration, from TensorBoard] -->
<!-- [IMAGE-5: "what the robot sees" — depth image strip next to viewport screenshots] -->

## Repository layout

```
configs/            PPO + env-override YAMLs (reward ablations are one line here)
source/tasks/       Isaac Lab env: rewards, observations, terrain curriculum, DR events
source/policies/    train / evaluate / export + custom vision actor-critic (rsl_rl)
source/deployment/  ROS2 inference node + C++ libtorch 50 Hz loop
reports/            evaluation JSONs, ablation tables, final report
GUIDE.md            full write-up: ground-up explanation, tech stack, design deep-dive
```

## Reproduce

Prereqs: Isaac Sim 4.5+, Isaac Lab 2.x, an NVIDIA GPU (≥12 GB VRAM blind, ≥24 GB vision),
`rsl-rl-lib==2.2.4`. All commands run through Isaac Lab's python (`./isaaclab.sh -p` /
`isaaclab.bat -p`) from the repo root.

```bash
# Phase 1 — blind flat baseline
./isaaclab.sh -p source/policies/train.py --task Go2-Flat-v0 \
    --agent_cfg configs/ppo_baseline.yaml --env_cfg configs/env_flat.yaml --headless

# Rough terrain + curriculum, then full domain randomization
./isaaclab.sh -p source/policies/train.py --task Go2-Rough-v0 \
    --agent_cfg configs/ppo_baseline.yaml --env_cfg configs/env_rough.yaml --headless
./isaaclab.sh -p source/policies/train.py --task Go2-Rough-DR-v0 \
    --agent_cfg configs/ppo_domain_randomized.yaml --headless

# Extreme version — depth-vision locomotion
./isaaclab.sh -p source/policies/train.py --task Go2-Vision-v0 \
    --agent_cfg configs/ppo_vision.yaml --headless --enable_cameras

# Evaluate (nominal + push-robustness sweep)
./isaaclab.sh -p source/policies/evaluate.py --task Go2-Rough-Play-v0 \
    --agent_cfg configs/ppo_domain_randomized.yaml --num_episodes 100 --headless
./isaaclab.sh -p source/policies/evaluate.py --task Go2-Rough-Play-v0 \
    --agent_cfg configs/ppo_domain_randomized.yaml --suite push --push_speed 1.5 \
    --num_episodes 100 --headless

# Export + deployment latency benchmark
./isaaclab.sh -p source/policies/export_policy.py --task Go2-Rough-Play-v0 \
    --agent_cfg configs/ppo_domain_randomized.yaml --onnx --headless
```

## Key design choices (short version)

- **PPO over SAC** — on-policy + massive parallelism beats replay-buffer sample
  efficiency when samples are nearly free; stable under curriculum non-stationarity.
- **Position targets + PD (kp 25 / kd 0.5) over torques** — feedback between policy
  steps, a bounded safety envelope, and it is literally the Go2's hardware API.
- **Depth over RGB** — geometry is the quantity foot placement needs; no appearance
  domain gap to randomize away.
- **Asymmetric actor-critic** — the throwaway critic sees privileged clean state + true
  height map; the deployable actor sees only realistic sensors.
- **Latency randomization** — 0–20 ms actuator delay + 0–40 ms sensor delay: the DR axis
  most sim-only projects skip and a leading real-world failure cause.

Full rationale, file-by-file walkthrough, and interview-grade Q&A: **[GUIDE.md](GUIDE.md)**.

<!-- [IMAGE-6: top-down screenshot of the terrain difficulty grid, rows/columns annotated] -->

## References

- Rudin et al., 2022 — *Learning to Walk in Minutes Using Massively Parallel Deep RL*
- Lee et al., 2020 — *Learning Quadrupedal Locomotion over Challenging Terrain*
- Agarwal et al., 2022 — *Legged Locomotion in Challenging Terrains using Egocentric Vision*
- NVIDIA Isaac Lab · ETH Zürich rsl_rl
