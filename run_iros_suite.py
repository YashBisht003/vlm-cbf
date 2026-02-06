from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IROS-style training/evaluation suite for VLM-CBF")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--out-dir", default="iros_artifacts", help="Artifact output directory")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run evaluation/report")
    parser.add_argument("--resume-train", action="store_true", help="Resume training from checkpoint if available")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--episodes", type=int, default=500, help="Evaluation episodes per method")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max steps per episode")
    parser.add_argument("--updates", type=int, default=1600, help="Training updates")
    parser.add_argument("--steps-per-update", type=int, default=512, help="Rollout steps per update")
    parser.add_argument("--epochs", type=int, default=5, help="PPO epochs")
    parser.add_argument("--minibatch", type=int, default=32, help="PPO minibatch size")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint cadence")
    parser.add_argument("--log-interval", type=int, default=10, help="Train logging cadence")
    parser.add_argument(
        "--verify-checkpoints",
        dest="verify_checkpoints",
        action="store_true",
        help="Enable checkpoint verification during training (default: enabled)",
    )
    parser.add_argument(
        "--no-verify-checkpoints",
        dest="verify_checkpoints",
        action="store_false",
        help="Disable checkpoint verification during training",
    )
    parser.add_argument("--torch-threads", type=int, default=0, help="Torch CPU threads (0=default)")
    parser.add_argument(
        "--carry-mode",
        choices=("auto", "constraint", "kinematic"),
        default="auto",
        help="Carry mode for all evaluations",
    )
    parser.add_argument(
        "--robot-size-mode",
        choices=("base", "full"),
        default="base",
        help="Robot size reference mode for object scaling",
    )
    parser.add_argument("--size-ratio-min", type=float, default=1.0, help="Min object size ratio")
    parser.add_argument("--size-ratio-max", type=float, default=3.0, help="Max object size ratio")
    parser.add_argument(
        "--constraint-force-scale",
        type=float,
        default=1.5,
        help="Vacuum constraint force scale against payload",
    )
    parser.add_argument("--vacuum-attach-dist", type=float, default=0.1, help="Vacuum attach distance")
    parser.add_argument("--vacuum-break-dist", type=float, default=0.2, help="Vacuum break distance")
    parser.add_argument("--vacuum-force-margin", type=float, default=1.05, help="Vacuum force margin")
    parser.add_argument("--deterministic-eval", action="store_true", help="Deterministic policy eval")
    parser.add_argument(
        "--run-neural-ablation",
        action="store_true",
        help="Also evaluate learned policy with neural CBF disabled",
    )
    parser.add_argument("--run-vlm-eval", action="store_true", help="Also run VLM formation evaluation")
    parser.add_argument("--vlm-model-id", default="llava-hf/llava-1.5-7b-hf", help="VLM model id for eval")
    parser.add_argument("--vlm-adapter", default="", help="LoRA adapter for VLM eval")
    parser.add_argument("--vlm-model-path", default="", help="Merged VLM model path alternative")
    parser.add_argument("--vlm-val-jsonl", default="vlm_dataset/val.jsonl", help="Validation JSONL for VLM eval")
    parser.add_argument("--vlm-image-root", default="vlm_dataset", help="Image root for VLM eval")
    parser.add_argument("--checkpoint-dir", default="", help="Checkpoint directory (default: <out-dir>/checkpoints)")
    parser.add_argument("--policy-out", default="", help="Final policy output path (default: <out-dir>/mappo_policy.pt)")
    parser.set_defaults(verify_checkpoints=True)
    return parser.parse_args()


def _run(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _num(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _summarize_csv(path: Path, method: str) -> Dict[str, float | str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    n = len(rows)
    success = sum(_num(r, "success") for r in rows)
    mean_time = sum(_num(r, "time") for r in rows) / n
    mean_steps = sum(_num(r, "steps") for r in rows) / n

    speed_viol = sum(_num(r, "speed_viol") for r in rows)
    sep_viol = sum(_num(r, "separation_viol") for r in rows)
    force_viol = sum(_num(r, "force_viol") for r in rows)

    cbf_calls = sum(_num(r, "cbf_calls") for r in rows)
    cbf_modified = sum(_num(r, "cbf_modified") for r in rows)
    cbf_fallback = sum(_num(r, "cbf_fallback") for r in rows)
    cbf_force_stop = sum(_num(r, "cbf_force_stop") for r in rows)
    cbf_rate = cbf_modified / max(cbf_calls, 1.0)

    grasp_attempts = sum(_num(r, "grasp_attach_attempts") for r in rows)
    grasp_success = sum(_num(r, "grasp_attach_success") for r in rows)
    grasp_detach = sum(_num(r, "grasp_detach_events") for r in rows)
    grasp_overload = sum(_num(r, "grasp_overload_drop") for r in rows)
    grasp_stretch = sum(_num(r, "grasp_stretch_drop") for r in rows)
    grasp_attach_rate = grasp_success / max(grasp_attempts, 1.0)

    return {
        "method": method,
        "episodes": float(n),
        "success_rate": success / n,
        "mean_time_s": mean_time,
        "mean_steps": mean_steps,
        "speed_viol_total": speed_viol,
        "separation_viol_total": sep_viol,
        "force_viol_total": force_viol,
        "cbf_calls": cbf_calls,
        "cbf_modified": cbf_modified,
        "cbf_fallback": cbf_fallback,
        "cbf_force_stop": cbf_force_stop,
        "cbf_rate": cbf_rate,
        "grasp_attempts": grasp_attempts,
        "grasp_success": grasp_success,
        "grasp_attach_rate": grasp_attach_rate,
        "grasp_detach_events": grasp_detach,
        "grasp_overload_drop": grasp_overload,
        "grasp_stretch_drop": grasp_stretch,
    }


def _write_summary_table(summary_rows: List[Dict[str, float | str]], out_csv: Path, out_md: Path) -> None:
    fields = [
        "method",
        "episodes",
        "success_rate",
        "mean_time_s",
        "mean_steps",
        "speed_viol_total",
        "separation_viol_total",
        "force_viol_total",
        "cbf_calls",
        "cbf_modified",
        "cbf_fallback",
        "cbf_force_stop",
        "cbf_rate",
        "grasp_attempts",
        "grasp_success",
        "grasp_attach_rate",
        "grasp_detach_events",
        "grasp_overload_drop",
        "grasp_stretch_drop",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("| Method | Success (%) | Mean Time (s) | Mean Steps | CBF Rate | Grasp Attach Rate |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} | {100.0 * float(row['success_rate']):.2f} | "
                f"{float(row['mean_time_s']):.2f} | {float(row['mean_steps']):.1f} | "
                f"{float(row['cbf_rate']):.3f} | {float(row['grasp_attach_rate']):.3f} |\n"
            )


def main() -> None:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    policy_out = Path(args.policy_out) if args.policy_out else out_dir / "mappo_policy.pt"

    train_csv = out_dir / "train_metrics.csv"
    eval_policy_csv = out_dir / "eval_learned_policy.csv"
    eval_policy_no_neural_csv = out_dir / "eval_learned_policy_no_neural.csv"
    eval_heuristic_cbf_csv = out_dir / "eval_heuristic_cbf.csv"
    eval_heuristic_no_cbf_csv = out_dir / "eval_heuristic_no_cbf.csv"
    vlm_eval_json = out_dir / "vlm_eval_metrics.json"
    vlm_eval_csv = out_dir / "vlm_eval_samples.csv"

    shared_env_args = [
        "--carry-mode",
        args.carry_mode,
        "--robot-size-mode",
        args.robot_size_mode,
        "--size-ratio-min",
        str(args.size_ratio_min),
        "--size-ratio-max",
        str(args.size_ratio_max),
        "--constraint-force-scale",
        str(args.constraint_force_scale),
        "--vacuum-attach-dist",
        str(args.vacuum_attach_dist),
        "--vacuum-break-dist",
        str(args.vacuum_break_dist),
        "--vacuum-force-margin",
        str(args.vacuum_force_margin),
    ]
    if args.seed is not None:
        shared_env_args.extend(["--seed", str(args.seed)])
    if args.headless:
        shared_env_args.append("--headless")

    if not args.skip_train:
        train_cmd = [
            args.python,
            "train_mappo.py",
            "--device",
            args.device,
            "--updates",
            str(args.updates),
            "--steps-per-update",
            str(args.steps_per_update),
            "--epochs",
            str(args.epochs),
            "--minibatch",
            str(args.minibatch),
            "--save-every",
            str(args.save_every),
            "--log-interval",
            str(args.log_interval),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--out",
            str(policy_out),
            "--log-csv",
            str(train_csv),
        ]
        if args.torch_threads > 0:
            train_cmd.extend(["--torch-threads", str(args.torch_threads)])
        if args.resume_train:
            train_cmd.append("--resume")
        if args.verify_checkpoints:
            train_cmd.append("--verify-checkpoints")
        else:
            train_cmd.append("--no-verify-checkpoints")
        train_cmd.extend(shared_env_args)
        _run(train_cmd, repo_dir)

    checkpoint_candidates = [
        checkpoint_dir / "mappo_policy_best.pt",
        checkpoint_dir / "mappo_policy_latest.pt",
        policy_out,
    ]
    policy_ckpt = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            policy_ckpt = candidate
            break
    if policy_ckpt is None:
        raise FileNotFoundError(
            "No policy checkpoint found. Expected one of: "
            + ", ".join(str(p) for p in checkpoint_candidates)
        )

    eval_policy_cmd = [
        args.python,
        "eval_policy_runs.py",
        "--model",
        str(policy_ckpt),
        "--device",
        args.device,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--out",
        str(eval_policy_csv),
    ]
    if args.deterministic_eval:
        eval_policy_cmd.append("--deterministic")
    eval_policy_cmd.extend(shared_env_args)
    _run(eval_policy_cmd, repo_dir)

    if args.run_neural_ablation:
        eval_policy_no_neural_cmd = [
            args.python,
            "eval_policy_runs.py",
            "--model",
            str(policy_ckpt),
            "--device",
            args.device,
            "--episodes",
            str(args.episodes),
            "--max-steps",
            str(args.max_steps),
            "--out",
            str(eval_policy_no_neural_csv),
            "--no-neural-cbf",
        ]
        if args.deterministic_eval:
            eval_policy_no_neural_cmd.append("--deterministic")
        eval_policy_no_neural_cmd.extend(shared_env_args)
        _run(eval_policy_no_neural_cmd, repo_dir)

    eval_heuristic_cbf_cmd = [
        args.python,
        "eval_runs.py",
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--out",
        str(eval_heuristic_cbf_csv),
    ]
    eval_heuristic_cbf_cmd.extend(shared_env_args)
    _run(eval_heuristic_cbf_cmd, repo_dir)

    eval_heuristic_no_cbf_cmd = [
        args.python,
        "eval_runs.py",
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--no-cbf",
        "--out",
        str(eval_heuristic_no_cbf_csv),
    ]
    eval_heuristic_no_cbf_cmd.extend(shared_env_args)
    _run(eval_heuristic_no_cbf_cmd, repo_dir)

    plot_paths = [eval_policy_csv, eval_heuristic_cbf_csv, eval_heuristic_no_cbf_csv]
    if args.run_neural_ablation:
        plot_paths.append(eval_policy_no_neural_csv)
    for csv_path in plot_paths:
        plot_cmd = [
            args.python,
            "plot_results.py",
            "--csv",
            str(csv_path),
            "--out",
            str(out_dir / f"{csv_path.stem}_plots.png"),
            "--summary-table",
            "--summary-out",
            str(out_dir / f"{csv_path.stem}_summary.png"),
        ]
        _run(plot_cmd, repo_dir)

    if args.run_vlm_eval:
        vlm_cmd = [
            args.python,
            "eval_vlm_formations.py",
            "--jsonl",
            args.vlm_val_jsonl,
            "--image-root",
            args.vlm_image_root,
            "--model-id",
            args.vlm_model_id,
            "--out-csv",
            str(vlm_eval_csv),
            "--out-json",
            str(vlm_eval_json),
        ]
        if args.vlm_adapter:
            vlm_cmd.extend(["--adapter", args.vlm_adapter])
        if args.vlm_model_path:
            vlm_cmd.extend(["--model-path", args.vlm_model_path])
        _run(vlm_cmd, repo_dir)

    summary_rows = [_summarize_csv(eval_policy_csv, "learned_policy")]
    if args.run_neural_ablation:
        summary_rows.append(_summarize_csv(eval_policy_no_neural_csv, "learned_policy_no_neural"))
    summary_rows.extend(
        [
            _summarize_csv(eval_heuristic_cbf_csv, "heuristic_cbf"),
            _summarize_csv(eval_heuristic_no_cbf_csv, "heuristic_no_cbf"),
        ]
    )
    summary_csv = out_dir / "suite_summary.csv"
    summary_md = out_dir / "suite_summary.md"
    _write_summary_table(summary_rows, summary_csv, summary_md)

    print(f"Suite complete. Artifacts in: {out_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD: {summary_md}")
    if args.run_vlm_eval:
        print(f"VLM eval JSON: {vlm_eval_json}")
        print(f"VLM eval CSV: {vlm_eval_csv}")


if __name__ == "__main__":
    main()
