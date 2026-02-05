import argparse
import csv
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation results CSV")
    parser.add_argument("--csv", default="eval_results.csv", help="CSV file from eval_runs.py")
    parser.add_argument("--out", default="eval_plots.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--summary-table", action="store_true", help="Write a single summary table figure")
    parser.add_argument("--summary-out", default="summary_table.png", help="Summary table output path")
    return parser.parse_args()


def _read_csv(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError("No rows found in CSV")

    numeric_int = {
        "episode",
        "success",
        "speed_viol",
        "separation_viol",
        "force_viol",
        "steps",
        "cbf_calls",
        "cbf_modified",
        "cbf_fallback",
        "cbf_force_stop",
    }
    numeric_float = {"time"}
    data = {key: [] for key in rows[0].keys() if key in numeric_int or key in numeric_float or key.startswith("time_")}
    for row in rows:
        for key, value in row.items():
            if key not in data:
                continue
            if value is None or value == "":
                data[key].append(np.nan)
                continue
            if key in numeric_int:
                data[key].append(int(float(value)))
            else:
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(np.nan)
    for key, values in data.items():
        data[key] = np.array(values, dtype=float)
    return data


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    data = _read_csv(csv_path)
    import matplotlib.pyplot as plt  # lazy import

    episodes = np.arange(len(data["episode"]))
    success = data["success"]
    times = data["time"]
    speed_v = data["speed_viol"]
    sep_v = data["separation_viol"]
    force_v = data["force_viol"]

    phase_keys = [k for k in data.keys() if k.startswith("time_")]
    phase_keys.sort()
    phase_means = [np.nanmean(data[k]) for k in phase_keys]

    has_cbf = "cbf_calls" in data
    if has_cbf:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("VLM-CBF Evaluation Summary", fontsize=14)

    ax = axes[0, 0]
    running = np.cumsum(success) / np.maximum(1, np.arange(1, len(success) + 1))
    ax.plot(episodes, running, color="#2a6fdb", linewidth=2)
    ax.set_title("Running Success Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")

    ax = axes[0, 1]
    ax.plot(episodes, times, color="#2a9d8f")
    ax.set_title("Episode Time (s)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Time (s)")

    if has_cbf:
        ax = axes[0, 2]
        cbf_calls = data.get("cbf_calls", np.zeros_like(episodes))
        cbf_modified = data.get("cbf_modified", np.zeros_like(episodes))
        cbf_rate = cbf_modified / np.maximum(1, cbf_calls)
        ax.plot(episodes, cbf_rate, color="#5f0f40")
        ax.set_title("CBF Intervention Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.05)

    ax = axes[1, 0]
    ax.bar(episodes, speed_v, label="speed", color="#e76f51")
    ax.bar(episodes, sep_v, bottom=speed_v, label="separation", color="#f4a261")
    ax.bar(episodes, force_v, bottom=speed_v + sep_v, label="force", color="#264653")
    ax.set_title("Safety Violations per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 1]
    if phase_keys:
        ax.barh(phase_keys, phase_means, color="#6d597a")
        ax.set_title("Mean Phase Time (s)")
        ax.set_xlabel("Time (s)")
    else:
        ax.text(0.5, 0.5, "No phase timing columns", ha="center", va="center")

    if has_cbf:
        ax = axes[1, 2]
        cbf_fallback = data.get("cbf_fallback", np.zeros_like(episodes))
        cbf_force = data.get("cbf_force_stop", np.zeros_like(episodes))
        ax.bar(episodes, cbf_modified, label="modified", color="#7b2cbf")
        ax.bar(
            episodes,
            cbf_fallback,
            bottom=cbf_modified,
            label="fallback",
            color="#f72585",
        )
        ax.bar(
            episodes,
            cbf_force,
            bottom=cbf_modified + cbf_fallback,
            label="force_stop",
            color="#ffb703",
        )
        ax.set_title("CBF Outcomes per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")

    if args.summary_table:
        total_episodes = len(success)
        success_rate = float(np.mean(success)) if total_episodes else 0.0
        mean_time = float(np.mean(times)) if total_episodes else 0.0
        mean_steps = float(np.mean(data.get("steps", np.array([0])))) if total_episodes else 0.0

        speed_viol = float(np.nansum(speed_v))
        sep_viol = float(np.nansum(sep_v))
        force_viol = float(np.nansum(force_v))

        rows = [
            ("Episodes", f"{total_episodes:d}"),
            ("Success Rate", f"{success_rate * 100:.1f}%"),
            ("Mean Time (s)", f"{mean_time:.2f}"),
            ("Mean Steps", f"{mean_steps:.1f}"),
            ("Speed Violations", f"{speed_viol:.0f}"),
            ("Separation Violations", f"{sep_viol:.0f}"),
            ("Force Violations", f"{force_viol:.0f}"),
        ]

        if "cbf_calls" in data:
            cbf_calls = float(np.nansum(data.get("cbf_calls", np.zeros_like(success))))
            cbf_modified = float(np.nansum(data.get("cbf_modified", np.zeros_like(success))))
            cbf_fallback = float(np.nansum(data.get("cbf_fallback", np.zeros_like(success))))
            cbf_force = float(np.nansum(data.get("cbf_force_stop", np.zeros_like(success))))
            cbf_rate = cbf_modified / max(cbf_calls, 1.0)
            rows.extend(
                [
                    ("CBF Calls", f"{cbf_calls:.0f}"),
                    ("CBF Modified", f"{cbf_modified:.0f}"),
                    ("CBF Fallback", f"{cbf_fallback:.0f}"),
                    ("CBF Force Stop", f"{cbf_force:.0f}"),
                    ("CBF Rate", f"{cbf_rate:.3f}"),
                ]
            )

        fig2, ax2 = plt.subplots(figsize=(6.5, 0.4 * len(rows) + 1.2))
        ax2.axis("off")
        table = ax2.table(
            cellText=[[label, value] for label, value in rows],
            colLabels=["Metric", "Value"],
            colLoc="left",
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.2)
        summary_path = Path(args.summary_out)
        fig2.tight_layout()
        fig2.savefig(summary_path, dpi=160)
        print(f"Saved summary table: {summary_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
