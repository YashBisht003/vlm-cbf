import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise ImportError("imageio is required. Install with: pip install imageio") from exc

from vlm_cbf_env import ObjectSpec, TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VLM training dataset from PyBullet")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--out", default="vlm_dataset", help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--distance", type=float, default=1.6, help="Camera distance")
    parser.add_argument("--pitch", type=float, default=-25.0, help="Camera pitch (deg)")
    parser.add_argument("--yaw-range", type=float, default=180.0, help="Yaw random range (+/- deg)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV")
    parser.add_argument("--views", type=int, default=1, help="Number of views per sample")
    parser.add_argument(
        "--multi-camera",
        action="store_true",
        help="Use top/oblique/side camera banks in addition to yaw sweep",
    )
    parser.add_argument(
        "--random-lighting",
        action="store_true",
        help="Randomize light direction and color per sample",
    )
    parser.add_argument(
        "--random-color",
        action="store_true",
        help="Randomize object color per sample",
    )
    parser.add_argument("--pitch-min", type=float, default=-45.0, help="Minimum pitch (deg)")
    parser.add_argument("--pitch-max", type=float, default=-15.0, help="Maximum pitch (deg)")
    parser.add_argument("--distance-min", type=float, default=1.2, help="Minimum camera distance")
    parser.add_argument("--distance-max", type=float, default=2.2, help="Maximum camera distance")
    parser.add_argument("--distractors", type=int, default=0, help="Number of distractor objects per scene")
    parser.add_argument(
        "--distractor-random",
        action="store_true",
        help="Randomize distractor count from 0..N each sample",
    )
    parser.add_argument(
        "--robot-size-mode",
        choices=("base", "full"),
        default="base",
        help="Robot size mode for object scaling (base or full)",
    )
    parser.add_argument("--size-ratio-min", type=float, default=1.0, help="Min object size ratio to robot")
    parser.add_argument("--size-ratio-max", type=float, default=3.0, help="Max object size ratio to robot")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for JSONL output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output directory (append JSONL, continue sample index)",
    )
    parser.add_argument("--format", choices=("llava", "simple"), default="llava")
    return parser.parse_args()


def _capture_image(
    width: int,
    height: int,
    target,
    distance,
    yaw_deg,
    pitch_deg,
    fov,
    light_direction=None,
    light_color=None,
) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw_deg,
        pitch=pitch_deg,
        roll=0.0,
        upAxisIndex=2,
    )
    aspect = width / float(height)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=0.05, farVal=5.0)
    kwargs = {}
    if light_direction is not None:
        kwargs["lightDirection"] = light_direction
    if light_color is not None:
        kwargs["lightColor"] = light_color

    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        **kwargs,
    )
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    return rgb


def _llava_sample(image_path: str, prompt: str, output: dict, sample_id: int) -> dict:
    return {
        "id": f"sample_{sample_id:06d}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": json.dumps(output, separators=(",", ":"))},
        ],
    }


def _simple_sample(image_path: str, prompt: str, output: dict, sample_id: int) -> dict:
    return {"id": f"sample_{sample_id:06d}", "image": image_path, "prompt": prompt, "output": output}


def _spawn_distractors(env: VlmCbfEnv, rng: np.random.Generator, count: int) -> list[int]:
    if count <= 0:
        return []
    distractors = []
    obj_pos, _ = p.getBasePositionAndOrientation(env.object_id)
    obj_pos = np.array(obj_pos)
    taken = [obj_pos[:2]]
    for _ in range(count):
        placed = False
        for _attempt in range(50):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.7, 1.4)
            pos_xy = obj_pos[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
            if all(np.linalg.norm(pos_xy - t) > 0.4 for t in taken):
                placed = True
                taken.append(pos_xy)
                break
        if not placed:
            continue

        shape = rng.choice(["cuboid", "l_shape", "t_shape"])
        dims = env._sample_dims(shape)
        mass = float(rng.uniform(*env.cfg.object_mass_range))
        friction = float(rng.uniform(*env.cfg.friction_range))
        com_offset = env._sample_com_offset(dims)
        spec = ObjectSpec(shape=shape, dims=dims, mass=mass, friction=friction, com_offset=com_offset)
        spawn_z = dims[2] * 0.5 + 0.02
        body_id = env._create_object_body(spec, base_pos=(float(pos_xy[0]), float(pos_xy[1]), spawn_z))
        color = rng.uniform(0.2, 0.95, size=3).tolist() + [1.0]
        p.changeVisualShape(body_id, -1, rgbaColor=color)
        distractors.append(body_id)
    return distractors

def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotations.jsonl"
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    cfg = TaskConfig(
        gui=False,
        random_seed=args.seed,
        object_size_ratio=(args.size_ratio_min, args.size_ratio_max),
        robot_size_mode=args.robot_size_mode,
    )
    env = VlmCbfEnv(cfg)
    prompt = (
        "Given the object geometry and dimensions, output 4 robot waypoints in the object frame. "
        "Return JSON with fields: waypoints[{x,y,load}], confidence."
    )

    try:
        mode = "a" if args.resume else "w"
        start_idx = 0
        banks_count = 3 if args.multi_camera else 1
        per_sample_views = max(1, int(args.views)) * banks_count
        if args.resume:
            existing_images = 0
            if img_dir.exists():
                existing_images = len(list(img_dir.glob("*.png")))
            start_idx = existing_images // per_sample_views
            print(
                f"Resuming from sample index {start_idx} (existing images: {existing_images}, views/sample: {per_sample_views})"
            )

        with out_path.open(mode, encoding="utf-8") as handle, (
            train_path.open(mode, encoding="utf-8")
        ) as train_handle, (
            val_path.open(mode, encoding="utf-8")
        ) as val_handle:
            for idx in range(start_idx, args.samples):
                env.reset()
                # Move robots far away to keep object visible
                for robot in env.robots:
                    p.resetBasePositionAndOrientation(
                        robot.base_id, [10.0, 10.0, 0.0], [0.0, 0.0, 0.0, 1.0]
                    )

                dims = env.object_spec.dims
                waypoints, labels = env._geometric_fallback(dims)
                base_output = {
                    "confidence": 1.0,
                    "waypoints": [
                        {"x": float(x), "y": float(y), "load": labels[i]} for i, (x, y) in enumerate(waypoints)
                    ],
                    "dimensions": {"length": dims[0], "width": dims[1], "height": dims[2]},
                    "shape": env.object_spec.shape,
                }

                if args.random_color:
                    color = rng.uniform(0.2, 0.95, size=3).tolist() + [1.0]
                    p.changeVisualShape(env.object_id, -1, rgbaColor=color)

                distractor_count = int(args.distractors)
                if args.distractor_random and distractor_count > 0:
                    distractor_count = int(rng.integers(0, distractor_count + 1))
                distractors = _spawn_distractors(env, rng, distractor_count)
                base_output["scene"] = {
                    "distractors": len(distractors),
                    "random_color": bool(args.random_color),
                    "random_lighting": bool(args.random_lighting),
                }

                obj_pos, _ = p.getBasePositionAndOrientation(env.object_id)
                views = max(1, int(args.views))

                if views == 1:
                    yaw_list = [float(rng.uniform(-args.yaw_range, args.yaw_range))]
                else:
                    base_yaw = float(rng.uniform(-180.0, 180.0))
                    yaw_list = [base_yaw + y for y in np.linspace(-args.yaw_range, args.yaw_range, views)]

                if args.multi_camera:
                    camera_banks = [
                        {"pitch": rng.uniform(args.pitch_min, args.pitch_max), "dist": rng.uniform(args.distance_min, args.distance_max)},
                        {"pitch": -70.0, "dist": rng.uniform(args.distance_min, args.distance_max)},  # top-down-ish
                        {"pitch": -10.0, "dist": rng.uniform(args.distance_min, args.distance_max)},  # side-ish
                    ]
                else:
                    camera_banks = [
                        {"pitch": rng.uniform(args.pitch_min, args.pitch_max), "dist": rng.uniform(args.distance_min, args.distance_max)}
                    ]

                view_idx = 0
                for bank_idx, bank in enumerate(camera_banks):
                    for yaw_deg in yaw_list:
                        light_dir = None
                        light_col = None
                        if args.random_lighting:
                            light_dir = rng.uniform(-1.0, 1.0, size=3).tolist()
                            light_col = rng.uniform(0.7, 1.0, size=3).tolist()

                        rgb = _capture_image(
                            width=args.width,
                            height=args.height,
                            target=obj_pos,
                            distance=float(bank["dist"]),
                            yaw_deg=float(yaw_deg),
                            pitch_deg=float(bank["pitch"]),
                            fov=args.fov,
                            light_direction=light_dir,
                            light_color=light_col,
                        )

                        image_name = f"sample_{idx:06d}_view_{view_idx:02d}.png"
                        image_path = img_dir / image_name
                        imageio.imwrite(image_path, rgb)

                        output = dict(base_output)
                        output["view"] = {
                            "index": view_idx,
                            "yaw_deg": float(yaw_deg),
                            "pitch_deg": float(bank["pitch"]),
                            "distance": float(bank["dist"]),
                            "bank": bank_idx,
                        }

                        rel_image = str(Path("images") / image_name)
                        sample_id = idx * len(yaw_list) * len(camera_banks) + view_idx
                        if args.format == "llava":
                            sample = _llava_sample(rel_image, prompt, output, sample_id)
                        else:
                            sample = _simple_sample(rel_image, prompt, output, sample_id)

                        handle.write(json.dumps(sample) + "\n")
                        if rng.uniform(0.0, 1.0) < args.val_ratio:
                            val_handle.write(json.dumps(sample) + "\n")
                        else:
                            train_handle.write(json.dumps(sample) + "\n")
                        view_idx += 1
    finally:
        env.close()

    print(f"Wrote {args.samples} samples to {out_path}")


if __name__ == "__main__":
    main()
