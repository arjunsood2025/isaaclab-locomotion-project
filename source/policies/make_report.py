"""Turn raw evaluation JSONs and TensorBoard logs into the tables and figures the
report and README need. Pure analysis -- no Isaac Sim, no GPU, safe to run while
training is in flight.

Inputs
    reports/eval_*.json          written by evaluate.py, one per (policy, suite, push)
    logs/rsl_rl/<exp>/<run>/     TensorBoard event files written by train.py

Outputs
    reports/results_tables.md    markdown tables, ready to paste into final_report.md
    media/robustness_curve.png   IMAGE-3: recovery rate vs push strength, one line/policy
    media/training_curves.png    IMAGE-4: mean reward + terrain level vs iteration

Usage:
    python source/policies/make_report.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# experiment folder name -> label used in every table and plot legend
POLICY_LABELS = {
    "go2_flat": "Flat baseline",
    "go2_rough_baseline": "A: baseline (no DR)",
    "go2_rough_dr": "D: full DR",
    "go2_vision": "E0: vision, end-to-end RL",
    "go2_vision_distill": "E: vision (depth, distilled)",
}
# ablation experiment name -> the reward term(s) that were zeroed
ABLATION_LABELS = {
    "go2_rough_baseline": "*(none: full reward)*",
    "go2_ablate_feet_slide": "`feet_slide`",
    "go2_ablate_smoothness": "`action_rate` + `action_smoothness`",
    "go2_ablate_energy": "`energy` + `joint_torques`",
    "go2_ablate_curriculum": "terrain curriculum",
}
ABLATION_ORDER = list(ABLATION_LABELS)
PLOT_ORDER = ["go2_rough_baseline", "go2_rough_dr", "go2_vision_distill", "go2_vision"]


def policy_key(checkpoint_path: str) -> str:
    """Recover the experiment name from a checkpoint path.

    The checkpoint path is the only field that ties a result file back to the run that
    produced it, which is why evaluate.py records it verbatim.
    """
    parts = checkpoint_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "rsl_rl" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def load_results() -> list[dict]:
    """Load every eval JSON, newest-last so later runs win on duplicate keys."""
    records = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "reports", "eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        res = blob["results"]
        res["_policy"] = policy_key(res.get("checkpoint", ""))
        res["_file"] = os.path.basename(path)
        records.append(res)
    return records


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_tables(records: list[dict]) -> str:
    """Nominal comparison table + push-recovery table, as markdown."""
    lines: list[str] = []

    nominal = {r["_policy"]: r for r in records if r.get("suite") == "nominal"}
    lines.append("## Headline results (100 episodes, nominal suite)\n")
    lines.append("| Policy | Success % | Fall % | Distance (m) | Vel err (m/s) | "
                 "Energy/m (J/m) | CoT |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in PLOT_ORDER:
        if key not in nominal:
            continue
        r = nominal[key]
        lines.append(
            f"| {POLICY_LABELS.get(key, key)} | {r['success'] * 100:.1f} | "
            f"{r['fall'] * 100:.1f} | {fmt(r['distance_m'], 2)} | "
            f"{fmt(r['lin_vel_err_mps'])} | {fmt(r['energy_per_m'], 2)} | "
            f"{fmt(r['cost_of_transport'])} |"
        )

    # push suite: policy -> push speed -> recovery rate
    push: dict[str, dict[float, dict]] = defaultdict(dict)
    for r in records:
        if r.get("suite") == "push":
            push[r["_policy"]][float(r["push_speed_mps"])] = r
    speeds = sorted({s for p in push.values() for s in p})

    if speeds:
        lines.append("\n## Push-recovery robustness (recovery rate %)\n")
        header = "| Push (m/s) | " + " | ".join(
            POLICY_LABELS.get(k, k) for k in PLOT_ORDER if k in push
        ) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (1 + sum(1 for k in PLOT_ORDER if k in push)))
        for speed in speeds:
            row = [f"| {speed:.1f} "]
            for key in PLOT_ORDER:
                if key not in push:
                    continue
                r = push[key].get(speed)
                rate = r.get("push_recovery_rate") if r else None
                row.append(f"| {rate * 100:.1f} " if rate is not None else "| — ")
            lines.append("".join(row) + "|")

    return "\n".join(lines) + "\n"


def build_ablation_table(records: list[dict]) -> str:
    """Reward-ablation table, keyed on the gait diagnostics the ablations target."""
    nominal = {r["_policy"]: r for r in records if r.get("suite") == "nominal"}
    if not any(k.startswith("go2_ablate") for k in nominal):
        return ""
    lines = ['\n## Reward-term ablations\n',
             "| Removed | Success % | Distance (m) | Vel err | Foot slip (m/s) | "
             "Action rate | Torque (N·m) | CoT |",
             "|---|---|---|---|---|---|---|---|"]
    for key in ABLATION_ORDER:
        if key not in nominal:
            continue
        r = nominal[key]
        lines.append(
            f"| {ABLATION_LABELS[key]} | {r['success'] * 100:.1f} | "
            f"{fmt(r['distance_m'], 2)} | {fmt(r['lin_vel_err_mps'])} | "
            f"{fmt(r.get('foot_slip_mps'))} | {fmt(r.get('action_rate'))} | "
            f"{fmt(r.get('mean_torque_Nm'), 2)} | {fmt(r['cost_of_transport'])} |"
        )
    return '\n'.join(lines) + '\n'


def plot_robustness(records: list[dict], out_path: str) -> bool:
    """IMAGE-3: recovery rate vs push strength. The project's central figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    push: dict[str, dict[float, float]] = defaultdict(dict)
    for r in records:
        if r.get("suite") == "push" and r.get("push_recovery_rate") is not None:
            push[r["_policy"]][float(r["push_speed_mps"])] = r["push_recovery_rate"] * 100
    if not push:
        print("[WARN] no push-suite results yet; skipping robustness curve")
        return False

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for key in PLOT_ORDER:
        if key not in push:
            continue
        speeds = sorted(push[key])
        ax.plot(speeds, [push[key][s] for s in speeds], marker="o",
                label=POLICY_LABELS.get(key, key))
    ax.set_xlabel("Push impulse (m/s)")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_title("Push-recovery robustness vs perturbation strength")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")
    return True


def plot_training_curves(out_path: str) -> bool:
    """IMAGE-4: mean reward and terrain level vs iteration, read from event files.

    Reading the event files directly (rather than screenshotting TensorBoard) keeps the
    figure reproducible and lets both panels share one style.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs: dict[str, str] = {}
    for exp in PLOT_ORDER + ["go2_flat"]:
        candidates = sorted(glob.glob(os.path.join(REPO_ROOT, "logs", "rsl_rl", exp, "*")))
        candidates = [c for c in candidates if os.path.isdir(c)]
        if candidates:
            runs[exp] = candidates[-1]  # latest run for this experiment
    if not runs:
        print("[WARN] no training logs found; skipping training curves")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for exp, run_dir in runs.items():
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags().get("scalars", []))
        label = POLICY_LABELS.get(exp, exp)

        for tag in ("Train/mean_reward", "Train/mean_reward/time"):
            if tag in tags:
                events = acc.Scalars(tag)
                axes[0].plot([e.step for e in events], [e.value for e in events], label=label)
                break
        for tag in ("Curriculum/terrain_levels", "Curriculum/terrain_levels/time"):
            if tag in tags:
                events = acc.Scalars(tag)
                axes[1].plot([e.step for e in events], [e.value for e in events], label=label)
                break

    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("mean episodic reward")
    axes[0].set_title("Learning curves")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mean terrain level")
    axes[1].set_title("Terrain curriculum progression")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")
    return True


def main() -> None:
    records = load_results()
    print(f"[INFO] loaded {len(records)} evaluation result files")

    tables = build_tables(records) + build_ablation_table(records)
    out_md = os.path.join(REPO_ROOT, "reports", "results_tables.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Generated results\n\n")
        f.write("<!-- Produced by source/policies/make_report.py -- do not hand-edit; "
                "re-run after adding evaluation JSONs. -->\n\n")
        f.write(tables)
    print(f"[INFO] wrote {out_md}")
    print("\n" + tables)

    plot_robustness(records, os.path.join(REPO_ROOT, "media", "robustness_curve.png"))
    plot_training_curves(os.path.join(REPO_ROOT, "media", "training_curves.png"))


if __name__ == "__main__":
    main()
