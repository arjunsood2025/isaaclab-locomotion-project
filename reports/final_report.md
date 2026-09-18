# Final report — Sim-to-Real RL for Robust Quadruped Locomotion

All numbers below were produced on one machine with the committed configs and seed 42.
Every table is regenerated from raw evaluation JSONs by `source/policies/make_report.py`;
nothing here is hand-entered.

**Evaluation is seeded** (`evaluate.py --seed 42`, the default). Every policy therefore faces
an identical sequence of spawn poses, velocity commands and physics-material draws, so a
difference between two rows is the policy and not the dice — and the numbers reproduce from
the committed commands. This matters more than it sounds: re-running one checkpoint through
the *unseeded* harness moved its success rate by 2 points and its velocity error by 10%.

**Stack:** Isaac Sim 5.1 · Isaac Lab 0.54.4 · rsl-rl-lib 5.0.1 · PyTorch 2.7.0+cu128
**Hardware:** single NVIDIA RTX 5070 (12 GB), Windows 11

---

## 1. Policies compared

| Policy | Task | Actor observes | Envs | Iters | Wall clock |
|---|---|---|---|---|---|
| Flat | `Go2-Flat-v0` | proprio (48) | 2048 | 500 | 16 min |
| **A — baseline** | `Go2-Rough-v0` | proprio + height scan (235) | 4096 | 1500 | 2.3 h |
| **D — full DR** | `Go2-Rough-DR-v0` | proprio + height scan (235) | 4096 | 2000 | 3.3 h |
| E0 — vision, end-to-end | `Go2-Vision-v0` | proprio (48) + 64×64 depth | 512 | 3000 | 6.6 h |
| **E — vision, distilled** | `Go2-Vision-Distill-v0` | proprio (48) + 64×64 depth | 512 | 1500 | 3.4 h |

A and D differ **only** in the randomization stack — identical hyperparameters, identical
seed — so any difference between them is attributable to domain randomization rather than
tuning. E0 and E share an identical student architecture and differ only in the training
signal (PPO reward vs. behaviour cloning from D), which isolates the effect of the
learning method from the effect of the network.

The flat policy is a Phase-1 sanity gate, not a comparison point: its 48-dim observation
is incompatible with the rough evaluation environment.

## 2. Headline results — nominal suite

100 episodes per policy, `Go2-Rough-Play-v0` (or its vision equivalent), no perturbation,
corruption and curriculum disabled.

| Policy | Success % | Fall % | Distance (m) | Vel err (m/s) | Energy/m (J/m) | CoT |
|---|---|---|---|---|---|---|
| A — baseline (no DR) | 100.0 | 0.0 | 16.80 | 0.112 | 55.74 | 0.378 |
| D — full DR | 99.0 | 1.0 | 15.33 | 0.151 | 55.72 | 0.378 |
| E — vision (distilled) | 90.0 | 10.0 | 14.53 | 0.210 | 71.54 | 0.486 |
| E0 — vision (end-to-end) | 92.0 | 8.0 | **1.57** | 0.594 | 166.21 | 1.128 |

**A wins in clean simulation.** It tracks velocity best (0.112 m/s), travels furthest, and
never falls. That is the expected result and it is *not* the point — see §3.

**E0 is the reason success rate alone is a worthless metric.** It scores 92% — *higher than
the working vision policy's 90%* — while travelling **1.57 m in a 20-second episode** against
that policy's 14.53 m. It "succeeded" by standing still: `success` means only "did not fall",
and a robot that never moves never falls. Distance, velocity error and cost of transport are
what expose it; CoT makes it unmistakable at 1.128 against 0.38–0.49 for every policy that
actually walks. Ranking these two policies by success rate would get the answer exactly
backwards.

## 3. Robustness — the central experiment

Recovery rate after a lateral velocity impulse of the given magnitude applied mid-episode,
100 episodes each, measured over a 3 s recovery window.

| Push (m/s) | A — baseline | D — full DR | E — vision (distilled) |
|---|---|---|---|
| 0.5 | 100.0 | 99.0 | 96.9 |
| 1.0 | 99.0 | 98.0 | 95.9 |
| 1.5 | 86.0 | 97.0 | 96.9 |
| 2.0 | 58.0 | 93.0 | 86.7 |
| 2.5 | **53.0** | **87.0** | **83.7** |

![robustness curve](../media/robustness_curve.png)

Three findings:

1. **Domain randomization inverts the ranking under perturbation.** A leads unperturbed and
   at 0.5 m/s; by 1.5 m/s D is ahead, and at 2.5 m/s A has degraded to a coin flip (53.0%)
   while D holds 87.0% — a **34-point gap**. This is the core sim-to-real lesson: the policy
   that looks better in clean simulation is the one that fails first off-distribution. Note
   where the curves separate — around 1.0–1.5 m/s, right where the perturbation leaves the
   distribution A was trained on.

2. **Randomization matters more than sensor quality.** E sees only a forward-facing depth
   camera — a sensor a real robot actually carries — while A gets a privileged 187-ray height
   map of the ground all around it, including behind. Under a 2.5 m/s shove E recovers 83.7%
   of the time against A's 53.0%, a **31-point advantage for the worse sensor**. Robustness
   came from *how* D was trained, and distillation carried it across.

3. **Distillation transfers nearly all of the teacher's robustness.** E is statistically tied
   with D at 1.5 m/s (96.9 vs 97.0) and trails by only 3.3 points at 2.5 m/s (83.7 vs 87.0),
   despite losing the privileged height map entirely. Whatever the teacher learned about
   staying upright under an impulse is evidently encoded mostly in the proprioceptive policy
   rather than in the terrain map — which is a reassuring result for deployment, because
   proprioception is the sensing a real robot has most reliably.

## 4. Terrain curriculum

Mean terrain level reached (0–9, 10 difficulty rows), and the training curves behind it.

| Policy | Final terrain level |
|---|---|
| A — baseline | 5.58 |
| D — full DR | 5.38 |
| E — vision (distilled) | 3.18 |
| E0 — vision (end-to-end) | **0.00** |

![training curves](../media/training_curves.png)

The plateau at ~5.5 for the blind policies is not a bug and not a ceiling of the method: the
curriculum promotes on >4 m travelled and demotes on failure, so it settles at the level
where promotion and demotion balance. That level *is* the policy's competence at this
training budget; a longer run would push it higher.

E's lower plateau (3.18) reflects the weaker observation. E0's flat 0.00 across all 3000
iterations is the failure analysed in §7.

## 5. Reward-term ablations

Each row is a full retrain with one reward weight zeroed — same seed, same budget, same
evaluation protocol as the baseline. Full analysis in [`ablation_tables.md`](ablation_tables.md).

| Removed | Success % | Distance (m) | Foot slip (m/s) | Action rate | Torque (N·m) | CoT |
|---|---|---|---|---|---|---|
| *(none — full reward)* | 100.0 | 16.80 | 0.046 | 0.884 | 2.87 | 0.378 |
| `feet_slide` | 98.0 | 16.74 | **0.124** | 0.850 | 2.92 | 0.375 |
| `action_rate` + `action_smoothness` | 97.0 | 16.17 | 0.067 | **1.051** | 3.01 | **0.439** |
| `energy` + `joint_torques` | 97.0 | **17.78** | 0.077 | 1.007 | **3.11** | **0.420** |
| terrain curriculum | **0.0** | **0.84** | 0.685 | 1.994 | 3.81 | 0.496 |

`foot_slip`, `action_rate` and `mean_torque` were added to the harness *for* this experiment:
zeroing a weight makes that term log as 0.0 during training, so the reward telemetry cannot
show whether the suppressed behaviour returned. Without those columns every row would have
read "success rate barely moved."

Success rates of 97–100% sit inside evaluation noise and mean nothing. The results that do:

- **The terrain curriculum is load-bearing, not a refinement.** Removing it takes success from
  100% to **0%** and distance from 16.80 m to **0.84 m** — no policy is learned at all. The
  override deliberately also spreads the initial terrain level across all 10 rows, so both
  conditions see identical terrain and only the *ordering* differs. Progressive difficulty is
  a precondition here, not a nicety.
- **`energy` + `joint_torques` is a trade-off, not a failure mode.** Removing the effort
  penalties made the policy *better at the task* — further (17.78 m) with lower velocity error
  (0.096) — while costing 8% more torque and 11% worse cost of transport. That is exactly what
  an effort penalty is for; the weight sets the exchange rate between performance and efficiency.
- **`feet_slide` buys gait quality, not task performance.** Foot slip rose 2.7× with every
  headline metric unchanged. That distinction matters for sim-to-real — skating abrades real
  feet and corrupts leg odometry — and is invisible to a benchmark table.
- **The smoothness terms work through efficiency, not jitter.** Action rate rose only 19%,
  but cost of transport rose 16% and tracking degraded. The hypothesised mechanism was partly
  wrong while the terms' value held.

## 6. Deployment

TorchScript export via rsl-rl's own exporter, benchmarked at batch size 1 — the deployment
case — over 500 iterations after warm-up.

| Metric | Blind DR policy | Vision student |
|---|---|---|
| Export-vs-eager max abs error | **0.00e+00** | **0.00e+00** |
| TorchScript CPU p50 (ms) | 0.096 | 0.597 |
| TorchScript CPU **p95** (ms) | **0.105** | **0.753** |
| TorchScript CPU p99 (ms) | 0.110 | 1.045 |
| TorchScript CUDA p95 (ms) | 0.633 | 1.535 |

At 50 Hz the whole control loop has a 20 ms budget. Blind inference uses **0.5%** of it and
the vision student **3.8%**, so the policy is nowhere near the bottleneck — sensor
acquisition and the depth render dominate on real hardware.

**CPU beats GPU at batch size 1** (0.105 ms vs 0.633 ms for the blind policy). For a network
this small, kernel-launch overhead and host↔device transfer exceed the arithmetic; the GPU
only wins in the massively-batched training regime. That is an argument for running the
deployed policy on the robot's CPU and leaving the GPU for perception.

Exported artifacts also verify **exact** numerical parity with the eager model, not merely
close agreement, because rsl-rl scripts a deterministic-inference copy rather than tracing a
stochastic one.

## 7. Failure analysis

### 7.1 `foot_clearance` measured against the wrong reference frame

**Symptom.** Two single-iteration collapses to −260 mean reward in the baseline learning curve.

**Diagnosis.** Read from the per-term episodic breakdown at the spike: `foot_clearance` went
−0.103 → **−14.578** (140×) while the tracking rewards *improved* and episode length rose.
Not a policy collapse — one reward term detonating.

**Root cause.** The term used absolute world-frame foot height, `(body_pos_w.z - 0.08)²`. The
terrain generator builds pyramid stairs and slopes reaching roughly ±2.3 m, so a robot on top
of the stairs was charged `(2.3 - 0.08)² ≈ 4.9` per foot instead of ~0.

**Fix.** Measure clearance against the height-scanner ray-cast hit nearest each foot, falling
back to world z on the flat task where it is exact. Verified by forcing spawns onto terrain
level 9: the term reads **−0.0000 to −0.0024** where it previously read −14.578.

**Honest outcome.** Retraining changed less than expected. The baseline's final reward
(37.28 → 37.34) and terrain level (5.62 → 5.58) were unchanged — the bug fired only when
robots reached high elevation, which was rare, so it produced occasional spikes rather than a
systematic drag. The DR policy did improve (reward 25.36 → 30.33, terrain 5.24 → 5.38,
episode length 954 → 982), but with one run per condition that cannot be cleanly separated
from run-to-run variance. The fix was still correct and necessary; its measured benefit was
smaller than the initial hypothesis predicted.

The pre-fix evidence is committed rather than summarised: the learning curve showing both
−260 spikes is at
[`media/before_foot_clearance_fix/training_curves.png`](../media/before_foot_clearance_fix/training_curves.png),
and the pre-fix results table at
[`before_foot_clearance_fix/results_tables.md`](before_foot_clearance_fix/results_tables.md).
Those numbers came from an earlier, unseeded evaluation harness and are therefore not
directly comparable to the tables above; they are kept so the before/after claim in this
section can be checked rather than taken on trust.

### 7.2 Residual reward spikes are falls, not bugs

After the fix a −155 spike remained, traced to `lin_vel_z`: −0.069 → −13.420, with velocity
tracking error tripling and the value loss collapsing. These are robots falling off high
terrain — at level 9 the inverted pyramid stairs are a ~2.3 m pit, so a fall reaches ~6.7 m/s
vertically and `v² × 2.0 ≈ 90` per step. Both the pre-fix and post-fix spikes occur at nearly
the same iteration (1318 and 1304) under the same seed: the same physical event, previously
amplified by the `foot_clearance` bug and now correctly attributed. Documented rather than
suppressed — the transient lasts a few iterations and final performance is unaffected.

### 7.3 End-to-end vision RL collapsed into a stand-and-turn optimum

**Symptom.** 3000 iterations, terrain level pinned at exactly 0.00 throughout, velocity error
plateaued at 1.4–1.8 m/s — larger than the maximum commanded speed.

**Diagnosis** from the reward-term breakdown against the blind DR policy:

| Term | E0 (vision) | D (blind DR) |
|---|---|---|
| track_lin_vel_xy | **0.388** | **1.450** |
| track_ang_vel_z | 0.844 | 0.758 |
| alive | 0.250 (max) | 0.246 |
| energy | −0.038 | −0.103 |
| termination | 0.000 | −0.003 |

The policy maximised the survival bonus and yaw tracking — it turns in place better than the
blind policy — while avoiding every effort penalty by not translating. It never fell.

**Why it was stuck.** The failure is self-reinforcing: no translation → never travels >4 m →
curriculum never promotes → camera only ever sees flat ground → image is uninformative → the
encoder collapses. Measured directly on the trained checkpoint: the CNN latent has L2 norm
**0.87** against proprioception's **7.03**, i.e. the vision pathway had been driven to ~1/8
the magnitude of the proprioceptive one. An initial hypothesis that the 256-dim latent was
*swamping* the 48-dim proprioception was tested and found to be exactly backwards.

**Fix — teacher–student distillation.** Learning to walk and to read terrain from pixels
simultaneously, under full domain randomization, is an exploration problem. Distillation
removes it: the teacher supplies the correct action at every visited state and the student
only learns perception. This is what the egocentric-vision locomotion literature does
(Lee et al. 2020; Agarwal et al. 2022), and the privileged teacher already existed — policy D.

**Result.** Terrain level 0.00 → **3.18**; distance travelled 1.57 m → **14.53 m**; cost of
transport 1.128 → **0.486** — at half the iterations of the failed run. Behaviour loss fell
0.958 → 0.174 and plateaued by iteration ~750. Under perturbation the distilled student ends
up within 3.3 points of its privileged teacher (§3).

### 7.4 Survivorship bias in the evaluation harness

The original harness collected "the first N episodes to finish" across parallel environments.
That is biased: a robot that falls at t=200 finishes, resets, and finishes *again* before a
survivor completes one 1000-step episode, so failures are over-represented and fall rate
inflated. Fixed to sample exactly one episode per environment — environments are i.i.d. over
terrain patch, spawn pose and command, so one episode from each is unbiased. Every number in
this report comes from the corrected harness.

## 8. Known limitations

- **Single seed per condition.** All comparisons are n=1. The baseline's 2.5 m/s recovery
  differed by 26 points between two training runs with near-identical nominal metrics, which
  shows push recovery — an out-of-distribution metric nobody optimises directly — has real
  variance. A 3-seed sweep with error bars is the obvious next experiment.
- **Reward ablations cover 7 of 18 terms.** Four ablations were run (see
  `ablation_tables.md`); eleven terms remain untested, and the two grouped ablations do not
  separate the members of each pair.
- **No real hardware.** The sim-to-real argument rests on robustness under perturbation and
  latency randomization in simulation, which is a proxy, not a substitute.
- **Vision is single-frame.** No memory, so terrain that has passed out of view is forgotten;
  a recurrent student (rsl-rl ships `RNNModel`) is the natural extension.
- **Teacher ceiling.** The student cannot exceed its teacher, and D itself plateaued at
  terrain level 5.38. DAgger-style iterative distillation would let the student improve on
  states its own policy visits rather than only those the teacher does.
