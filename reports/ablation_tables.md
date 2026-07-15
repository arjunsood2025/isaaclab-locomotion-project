# Reward-term ablations

Each row = one training run with a single reward weight zeroed via an override YAML
(e.g. `rewards.action_rate.weight: 0.0` in a copy of `configs/env_rough.yaml`),
evaluated with `evaluate.py --num_episodes 100` on the same seed and terrain.

| Reward removed | Expected failure | Observed (fill in) | Success % | Vel err | Energy/m |
|---|---|---|---|---|---|
| (none — full reward) | — | | | | |
| action_rate + action_smoothness | twitchy motion | | | | |
| energy + joint_torques | unnatural high-torque gait | | | | |
| feet_slide | skating feet | | | | |
| flat_orientation | unstable body posture | | | | |
| feet_air_time | rapid paddling gait | | | | |
| foot_clearance | toe-dragging, trips on rough terrain | | | | |
| terrain curriculum (curriculum.terrain_levels: null) | poor rough-terrain learning | | | | |

Tip: keep the seed fixed (42) across rows so differences are attributable to the
ablation, and record the TensorBoard curves for each run — reward-term episodic sums
are logged individually, so you can show *which* term the policy traded away.
