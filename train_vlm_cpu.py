import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise ImportError("imageio is required. Install with: pip install imageio") from exc

try:
    from skimage.color import rgb2gray
    from skimage.feature import hog
    from skimage.transform import resize
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-image is required. Install with: pip install scikit-image") from exc

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise ImportError("joblib is required. Install with: pip install joblib") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-friendly waypoint regressor training")
    parser.add_argument("--data", default="vlm_dataset", help="Dataset directory")
    parser.add_argument("--train", default="train.jsonl", help="Train JSONL file")
    parser.add_argument("--val", default="val.jsonl", help="Validation JSONL file")
    parser.add_argument("--out", default="cpu_vlm_model.joblib", help="Output model file")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--max-train", type=int, default=0, help="Max training samples (0 = all)")
    parser.add_argument("--max-val", type=int, default=0, help="Max validation samples (0 = all)")
    parser.add_argument("--hog-cells", type=int, default=8, help="Pixels per cell for HOG")
    parser.add_argument("--hog-orient", type=int, default=9, help="HOG orientations")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    return parser.parse_args()


def _load_jsonl(path: Path, max_rows: int = 0):
    rows = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if max_rows and len(rows) >= max_rows:
                break
    if bad_lines:
        print(f"Skipped {bad_lines} malformed JSON lines in {path.name}")
    return rows


def _extract_target(sample: dict) -> np.ndarray:
    output = sample.get("output") or sample.get("conversations", [{}])[-1].get("value")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return None
    waypoints = output.get("waypoints", [])
    if len(waypoints) < 4:
        return None
    coords = []
    for wp in waypoints[:4]:
        coords.extend([float(wp.get("x", 0.0)), float(wp.get("y", 0.0))])
    return np.array(coords, dtype=np.float32)


def _feature_from_image(
    image_path: Path, size: int, hog_cells: int, hog_orient: int
) -> Optional[np.ndarray]:
    try:
        img = imageio.imread(image_path)
    except Exception:
        return None
    if img.ndim == 2:
        gray = img.astype(np.float32) / 255.0
    else:
        gray = rgb2gray(img)
    gray = resize(gray, (size, size), anti_aliasing=True)
    feat = hog(
        gray,
        orientations=hog_orient,
        pixels_per_cell=(hog_cells, hog_cells),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def _build_dataset(rows, data_dir: Path, size: int, hog_cells: int, hog_orient: int):
    X = []
    y = []
    skipped_images = 0
    for sample in rows:
        image_rel = sample.get("image")
        if not image_rel:
            continue
        image_path = data_dir / image_rel
        if not image_path.exists():
            continue
        target = _extract_target(sample)
        if target is None:
            continue
        feat = _feature_from_image(image_path, size, hog_cells, hog_orient)
        if feat is None:
            skipped_images += 1
            continue
        X.append(feat)
        y.append(target)
    if skipped_images:
        print(f"Skipped {skipped_images} unreadable images")
    if not X:
        return None, None
    return np.vstack(X), np.vstack(y)


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data)
    train_path = data_dir / args.train
    val_path = data_dir / args.val
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    train_rows = _load_jsonl(train_path, args.max_train)
    val_rows = _load_jsonl(val_path, args.max_val)

    X_train, y_train = _build_dataset(
        train_rows, data_dir, args.image_size, args.hog_cells, args.hog_orient
    )
    X_val, y_val = _build_dataset(val_rows, data_dir, args.image_size, args.hog_cells, args.hog_orient)

    if X_train is None:
        raise RuntimeError("No training data found.")
    if X_val is None:
        raise RuntimeError("No validation data found.")

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=args.alpha)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"Validation MAE (avg over 8 coords): {mae:.4f}")

    meta = {
        "image_size": args.image_size,
        "hog_cells": args.hog_cells,
        "hog_orient": args.hog_orient,
        "alpha": args.alpha,
        "mae": mae,
    }
    out_path = Path(args.out)
    joblib.dump({"model": model, "meta": meta}, out_path)
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()
