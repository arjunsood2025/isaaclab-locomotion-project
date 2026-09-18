# Generated results

<!-- Produced by source/policies/make_report.py -- do not hand-edit; re-run after adding evaluation JSONs. -->

## Headline results (100 episodes, nominal suite)

| Policy | Success % | Fall % | Distance (m) | Vel err (m/s) | Energy/m (J/m) | CoT |
|---|---|---|---|---|---|---|
| A: baseline (no DR) | 100.0 | 0.0 | 16.80 | 0.112 | 55.74 | 0.378 |
| D: full DR | 99.0 | 1.0 | 15.33 | 0.151 | 55.72 | 0.378 |
| E: vision (depth, distilled) | 90.0 | 10.0 | 14.53 | 0.210 | 71.54 | 0.486 |
| E0: vision, end-to-end RL | 92.0 | 8.0 | 1.57 | 0.594 | 166.21 | 1.128 |

## Push-recovery robustness (recovery rate %)

| Push (m/s) | A: baseline (no DR) | D: full DR | E: vision (depth, distilled) |
|---|---|---|---|
| 0.5 | 100.0 | 99.0 | 96.9 |
| 1.0 | 99.0 | 98.0 | 95.9 |
| 1.5 | 86.0 | 97.0 | 96.9 |
| 2.0 | 58.0 | 93.0 | 86.7 |
| 2.5 | 53.0 | 87.0 | 83.7 |

## Reward-term ablations

| Removed | Success % | Distance (m) | Vel err | Foot slip (m/s) | Action rate | Torque (N·m) | CoT |
|---|---|---|---|---|---|---|---|
| *(none: full reward)* | 100.0 | 16.80 | 0.112 | 0.046 | 0.884 | 2.87 | 0.378 |
| `feet_slide` | 98.0 | 16.74 | 0.110 | 0.124 | 0.850 | 2.92 | 0.375 |
| `action_rate` + `action_smoothness` | 97.0 | 16.17 | 0.135 | 0.067 | 1.051 | 3.01 | 0.439 |
| `energy` + `joint_torques` | 97.0 | 17.78 | 0.096 | 0.077 | 1.007 | 3.11 | 0.420 |
| terrain curriculum | 0.0 | 0.84 | 0.933 | 0.685 | 1.994 | 3.81 | 0.496 |
