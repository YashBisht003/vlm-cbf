import argparse
import json
from pathlib import Path

import numpy as np

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
    parser = argparse.ArgumentParser(description="CPU VLM-like inference")
    parser.add_argument("--model", default="cpu_vlm_model.joblib", help="Model file")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="formation.json", help="Output JSON path")
    return parser.parse_args()


def _feature_from_image(image_path: Path, size: int, hog_cells: int, hog_orient: int) -> np.ndarray:
    img = imageio.imread(image_path)
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


def _labels_for_waypoints(waypoints):
    # Simple default: first two are high, last two low
    labels = ["high", "high", "low", "low"]
    if len(waypoints) < 4:
        return labels[: len(waypoints)]
    return labels


def infer_image(model_path: Path, image_path: Path) -> dict:
    model_data = joblib.load(model_path)
    model = model_data["model"]
    meta = model_data.get("meta", {})
    size = int(meta.get("image_size", 128))
    hog_cells = int(meta.get("hog_cells", 8))
    hog_orient = int(meta.get("hog_orient", 9))
    mae = float(meta.get("mae", 0.2))

    feat = _feature_from_image(image_path, size, hog_cells, hog_orient)
    pred = model.predict(feat.reshape(1, -1))[0]

    waypoints = []
    labels = _labels_for_waypoints(pred)
    for i in range(0, len(pred), 2):
        waypoints.append({"x": float(pred[i]), "y": float(pred[i + 1]), "load": labels[i // 2]})

    confidence = max(0.05, 1.0 - mae)
    return {"confidence": float(confidence), "waypoints": waypoints}


def main() -> None:
    args = _parse_args()
    output = infer_image(Path(args.model), Path(args.image))
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
