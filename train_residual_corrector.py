from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from residual_corrector import FEATURE_NAMES, synthetic_residual_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight residual waypoint corrector (MLP)")
    parser.add_argument("--out", default="residual_corrector.joblib", help="Output model path")
    parser.add_argument("--dataset-csv", default="", help="Optional CSV with residual training data")
    parser.add_argument("--samples", type=int, default=40000, help="Synthetic samples if no dataset CSV provided")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--hidden", default="64,64", help="MLP hidden sizes, comma-separated")
    parser.add_argument("--max-iter", type=int, default=300, help="MLP max iterations")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--gain", type=float, default=0.12, help="Synthetic target gain")
    parser.add_argument("--max-shift", type=float, default=0.12, help="Synthetic target clip")
    return parser.parse_args()


def _parse_hidden(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    dims = tuple(int(p) for p in parts)
    if not dims:
        raise ValueError("Invalid hidden layer spec")
    return dims


def main() -> None:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib

    args = _parse_args()
    hidden = _parse_hidden(args.hidden)
    if args.dataset_csv:
        import csv

        rows = []
        with Path(args.dataset_csv).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No rows in dataset CSV: {args.dataset_csv}")
        x = np.array(
            [[float(r[name]) for name in FEATURE_NAMES] for r in rows],
            dtype=np.float32,
        )
        y = np.array(
            [[float(r["target_dx"]), float(r["target_dy"])] for r in rows],
            dtype=np.float32,
        )
    else:
        x, y = synthetic_residual_dataset(
            samples=max(2000, int(args.samples)),
            seed=int(args.seed),
            gain=float(args.gain),
            max_shift=float(args.max_shift),
        )

    rng = np.random.default_rng(args.seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    n = len(idx)
    if n < 20:
        raise RuntimeError(f"Dataset too small for train/val split: {n} rows")
    split = int((1.0 - float(args.test_ratio)) * n)
    min_train = max(8, int(0.6 * n))
    max_train = n - max(4, int(0.1 * n))
    split = max(min_train, min(split, max_train))
    tr, va = idx[:split], idx[split:]
    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(x_tr_s, y_tr)
    pred = model.predict(x_va_s)

    mse = float(mean_squared_error(y_va, pred))
    mae = float(mean_absolute_error(y_va, pred))
    r2 = float(r2_score(y_va, pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(FEATURE_NAMES),
        "metrics": {"mse": mse, "mae": mae, "r2": r2},
        "config": vars(args),
    }
    joblib.dump(payload, out_path)

    print(f"Saved residual model: {out_path}")
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation MAE: {mae:.6f}")
    print(f"Validation R2: {r2:.4f}")


if __name__ == "__main__":
    main()
