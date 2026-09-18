# Reward-term ablations

Each row is **one full retrain** with a single reward weight zeroed via a config override —
no code edit — on the same seed (42), the same budget (1500 iterations × 4096 environments)
and the same task as the unablated baseline. Every row is then scored through the identical
seeded 100-episode evaluation, so a difference between rows is the reward function and not
the dice.

```yaml
# configs/env_rough_ablate_feet_slide.yaml — the entire experiment
scene.num_envs: 4096
episode_length_s: 20.0
rewards.feet_slide.weight: 0.0
```

## Results

| Removed | Success % | Distance (m) | Vel err | **foot slip (m/s)** | **action rate** | **torque (N·m)** | **CoT** |
|---|---|---|---|---|---|---|---|
| *(none — full reward)* | 100.0 | 16.80 | 0.112 | 0.046 | 0.884 | 2.87 | 0.378 |
| `feet_slide` | 98.0 | 16.74 | 0.110 | **0.124** | 0.850 | 2.92 | 0.375 |
| `action_rate` + `action_smoothness` | 97.0 | 16.17 | 0.135 | 0.067 | **1.051** | 3.01 | **0.439** |
| `energy` + `joint_torques` | 97.0 | **17.78** | **0.096** | 0.077 | 1.007 | **3.11** | **0.420** |
| terrain curriculum | **0.0** | **0.84** | **0.933** | 0.685 | 1.994 | 3.81 | 0.496 |

`foot_slip`, `action_rate` and `mean_torque` were added to the evaluation harness
specifically for this experiment. Zeroing a reward weight makes that term log as `0.0`
during training, so the reward telemetry cannot reveal whether the behaviour it suppressed
returned — it has to be measured independently at evaluation time. Without these columns
every row here would have read "success rate barely moved."

**Success rates of 97–100% are inside evaluation noise** (~2 points at n=100, measured by
re-running one checkpoint twice) and should not be read as a trend. The signal is in the
targeted diagnostics, where the effects are far larger.

## What each ablation showed

### `feet_slide` — confirmed, and narrower than assumed

Foot slip rose **2.7×** (0.046 → 0.124 m/s) while success, distance, velocity error and cost
of transport were unchanged. The term does exactly what it claims and *only* that: it buys
gait quality, not task performance.

That distinction matters for the sim-to-real story rather than for a benchmark table.
Skating feet abrade real rubber, inject noise into leg-odometry state estimation, and behave
completely differently on a friction coefficient the simulator never sampled. A policy that
scores identically while dragging its feet is not equally deployable.

### `action_rate` + `action_smoothness` — confirmed, but via a different mechanism

Action rate rose 19% (0.884 → 1.051) — real, but less dramatic than "twitchy motion"
suggests. The larger effect was efficiency: **cost of transport rose 16%** (0.378 → 0.439),
velocity tracking degraded (0.112 → 0.135), and distance covered dropped.

So the smoothness penalties earn their place mainly by producing an efficient gait, not by
suppressing visible jitter. The hypothesised failure mode was partly wrong while the term's
value held up — worth recording, because the reason a term helps is as much a claim as
whether it helps.

### `energy` + `joint_torques` — a genuine trade-off, not a failure

This is the most interesting row. Removing the effort penalties made the policy **better at
the task**: it travelled further than the baseline (17.78 m vs 16.80) with *lower* velocity
error (0.096 vs 0.112). It paid for that with 8% more torque (3.11 vs 2.87 N·m) and **11%
worse cost of transport** (0.420 vs 0.378).

That is precisely what an effort penalty is supposed to do. It is not free: it deliberately
trades a little tracking performance for efficiency and actuator life. Reporting this as
"ablation degrades performance" would be false — the correct statement is that the penalty
buys efficiency at a measurable and intentional cost, and the weight (`-1e-3`) sets the
exchange rate.

### Terrain curriculum — load-bearing

**Success 100.0% → 0.0%. Distance 16.80 m → 0.84 m.** Velocity error 0.112 → 0.933, which
exceeds the maximum commanded speed. Removing the curriculum did not degrade the policy; it
prevented one from being learned at all.

The experiment was designed to make this conclusion airtight. Disabling promotion alone would
pin robots to the easy rows they spawn on, conflating "no curriculum" with "never saw hard
terrain" — two different claims. So the override *also* spreads the initial terrain level
across all 10 rows:

```yaml
curriculum.terrain_levels: null
scene.terrain.max_init_terrain_level: null   # identical terrain exposure, no ordering
```

Both conditions therefore see the same distribution of terrain; only the *ordering* differs.
The result isolates the curriculum's actual claim — that encountering difficulty in
increasing order beats encountering it all at once — and shows it is not a refinement but a
precondition. Thrown onto stairs and gaps from iteration zero, the policy fails so
consistently that it never accumulates the experience needed to learn basic locomotion.

## Related controlled experiments

Two further comparisons in this project follow the same one-variable discipline and are
reported in `final_report.md`:

| Comparison | Variable isolated | Result |
|---|---|---|
| A vs D | domain randomization stack (identical hyperparameters, seed) | 34-point recovery gap at 2.5 m/s |
| E0 vs E | training signal only (PPO vs distillation; identical student architecture) | 1.57 m → 14.53 m distance travelled |

## Limitations

- **n = 1 per row.** Each ablation is a single training run. Effects at or below the ~2-point
  evaluation noise floor (all the success-rate differences) are not resolvable; the 2.7×,
  19%, 16% and 100→0 effects are well clear of it. A 3-seed sweep would put error bars on all
  of them and is the obvious next experiment.
- **Terms were ablated in the groups they were designed as** (`action_rate` with
  `action_smoothness`, `energy` with `joint_torques`), so these rows do not separate the
  members of each pair.
- **Not every term was ablated.** Eleven of the eighteen reward terms remain untested,
  including `flat_orientation`, `feet_air_time` and `undesired_contacts`.
