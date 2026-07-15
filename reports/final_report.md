# Final report — Sim-to-Real RL for Robust Quadruped Locomotion

> Template: fill each table/plot from `evaluate.py` JSON outputs in this folder.
> Every number should be reproducible from a committed config + seed + checkpoint.

## 1. Policies compared

| Policy | Task | Config | Checkpoint |
|---|---|---|---|
| A: baseline (blind, no DR) | Go2-Rough-v0 | ppo_baseline.yaml | |
| B: + dynamics randomization | Go2-Rough-DR-v0 (obs delay terms off) | ppo_domain_randomized.yaml | |
| C: + sensor noise randomization | Go2-Rough-DR-v0 (actuator delay off) | ppo_domain_randomized.yaml | |
| D: + latency randomization (full DR) | Go2-Rough-DR-v0 | ppo_domain_randomized.yaml | |
| E: vision (depth camera) | Go2-Vision-v0 | ppo_vision.yaml | |

## 2. Headline results (100 episodes each, `--suite nominal`)

| Policy | Success % | Fall % | Distance (m) | Vel err (m/s) | Energy/m (J/m) | CoT |
|---|---|---|---|---|---|---|
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |
| D | | | | | | |
| E | | | | | | |

Expected pattern: A wins slightly in clean sim; D wins everywhere once perturbations
are applied — the core sim-to-real lesson of the project.

## 3. Robustness sweep (`--suite push`, increasing `--push_speed`)

x-axis: push speed 0.5 / 1.0 / 1.5 / 2.0 / 2.5 m/s — y-axis: recovery success rate.

| Push (m/s) | A | B | C | D | E |
|---|---|---|---|---|---|
| 0.5 | | | | | |
| 1.0 | | | | | |
| 1.5 | | | | | |
| 2.0 | | | | | |
| 2.5 | | | | | |

## 4. Terrain breakdown

Run `evaluate.py` with the play cfg and note per-terrain-type outcomes (the play grid
spawns robots across all terrain columns).

| Terrain | Success % (blind D) | Success % (vision E) | Notes |
|---|---|---|---|
| flat | | | |
| random rough | | | |
| slopes | | | |
| boxes | | | |
| stairs up / down | | | |
| gaps | | | |

## 5. Failure analysis

Document, with videos/timestamps:
- **When it falls:** terrain type, speed, phase of gait at failure.
- **When it slips:** friction values that break it (sweep `physics_material` ranges).
- **Which reward terms mattered:** cross-reference `ablation_tables.md`.
- **Which randomizations helped:** B vs C vs D deltas per suite.
- **Latency limits:** increase actuator `max_delay` at eval until tracking degrades;
  report the knee point in ms.
- **Sim-to-real risks:** unmodeled effects (foot deformation, motor heating,
  depth-sensor artifacts in sunlight) and how they would be mitigated.

## 6. Deployment measurements

From `export_policy.py` and `cpp_inference_loop`:

| Metric | Blind policy | Vision policy |
|---|---|---|
| TorchScript CPU latency p95 (ms) | | |
| TorchScript GPU latency p95 (ms) | | |
| C++ loop deadline misses @50 Hz | | n/a |
