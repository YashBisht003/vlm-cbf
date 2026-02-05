import argparse
import json
from pathlib import Path
import time

import pybullet as p

from vlm_cbf_env import TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM-CBF PyBullet demo runner")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--carry-mode",
        choices=("auto", "constraint", "kinematic"),
        default="auto",
        help="Select carry mode (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Max steps to run (0 = until done)",
    )
    parser.add_argument("--no-sleep", action="store_true", help="Run as fast as possible")
    parser.add_argument(
        "--vlm-json",
        default=None,
        help="Path to VLM output JSON with waypoints and load labels",
    )
    parser.add_argument("--pos-noise", type=float, default=0.0, help="Position noise std (m)")
    parser.add_argument("--yaw-noise", type=float, default=0.0, help="Yaw noise std (rad)")
    parser.add_argument("--force-noise", type=float, default=0.0, help="Force noise std (N)")
    parser.add_argument("--noisy-obs", action="store_true", help="Apply noise to observations")
    parser.add_argument(
        "--noisy-control",
        action="store_true",
        help="Use noisy object pose for contact control",
    )
    parser.add_argument(
        "--noisy-plan",
        action="store_true",
        help="Use noisy object pose for formation planning",
    )
    parser.add_argument("--heavy-urdf", default=None, help="URDF path for heavy robots")
    parser.add_argument("--agile-urdf", default=None, help="URDF path for agile robots")
    parser.add_argument("--heavy-ee-link", type=int, default=None, help="End effector link index for heavy robots")
    parser.add_argument("--agile-ee-link", type=int, default=None, help="End effector link index for agile robots")
    parser.add_argument(
        "--heavy-ee-link-name",
        default=None,
        help="End effector link name for heavy robots (e.g. tool0, flange)",
    )
    parser.add_argument(
        "--agile-ee-link-name",
        default=None,
        help="End effector link name for agile robots (e.g. tool0, flange)",
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
        "--auto-vlm",
        action="store_true",
        help="Capture an image and run CPU VLM inference before planning",
    )
    parser.add_argument(
        "--vlm-model",
        default="cpu_vlm_model.joblib",
        help="CPU VLM model file for auto inference",
    )
    parser.add_argument(
        "--vlm-out",
        default="vlm_auto.json",
        help="Output JSON path for auto VLM inference",
    )
    parser.add_argument("--video", action="store_true", help="Record MP4 video of the demo")
    parser.add_argument("--video-path", default="demo.mp4", help="Output MP4 path")
    parser.add_argument("--screenshots", action="store_true", help="Capture screenshots on phase changes")
    parser.add_argument("--screenshots-dir", default="screenshots", help="Output directory for screenshots")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Screenshot width")
    parser.add_argument("--screenshot-height", type=int, default=720, help="Screenshot height")
    return parser.parse_args()


def _capture_vlm_image(
    env: VlmCbfEnv,
    out_path: Path,
    width: int = 640,
    height: int = 480,
    distance: float = 1.6,
    yaw_deg: float = 40.0,
    pitch_deg: float = -25.0,
    fov: float = 60.0,
    renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
) -> None:
    if env.object_id is None:
        raise RuntimeError("Object not spawned; cannot capture VLM image.")
    obj_pos, _ = p.getBasePositionAndOrientation(env.object_id, physicsClientId=env.client_id)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=obj_pos,
        distance=distance,
        yaw=yaw_deg,
        pitch=pitch_deg,
        roll=0.0,
        upAxisIndex=2,
    )
    aspect = width / float(height)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=0.05, farVal=5.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
    )
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise ImportError("imageio is required for auto VLM capture.") from exc
    rgb = (rgba[:, :, :3]).astype("uint8")
    imageio.imwrite(out_path, rgb)


def _capture_scene(
    env: VlmCbfEnv,
    out_path: Path,
    width: int,
    height: int,
    renderer: int,
) -> None:
    _capture_vlm_image(
        env,
        out_path,
        width=width,
        height=height,
        distance=1.8,
        yaw_deg=35.0,
        pitch_deg=-25.0,
        fov=60.0,
        renderer=renderer,
    )


def _run_cpu_vlm(model_path: Path, image_path: Path) -> dict:
    try:
        from infer_vlm_cpu import infer_image
    except Exception as exc:  # pragma: no cover
        raise ImportError("infer_vlm_cpu.py is required for auto VLM inference.") from exc
    return infer_image(model_path, image_path)


def main() -> None:
    args = _parse_args()
    cfg = TaskConfig(
        gui=not args.headless,
        random_seed=args.seed,
        carry_mode=args.carry_mode,
        sensor_pos_noise=args.pos_noise,
        sensor_yaw_noise=args.yaw_noise,
        sensor_force_noise=args.force_noise,
        use_noisy_obs=args.noisy_obs,
        use_noisy_control=args.noisy_control,
        use_noisy_plan=args.noisy_plan,
        vlm_json_path=args.vlm_json or (args.vlm_out if args.auto_vlm else None),
        heavy_urdf=args.heavy_urdf,
        agile_urdf=args.agile_urdf,
        heavy_ee_link=args.heavy_ee_link,
        agile_ee_link=args.agile_ee_link,
        heavy_ee_link_name=args.heavy_ee_link_name,
        agile_ee_link_name=args.agile_ee_link_name,
        object_size_ratio=(args.size_ratio_min, args.size_ratio_max),
        robot_size_mode=args.robot_size_mode,
    )
    env = VlmCbfEnv(cfg)
    env.reset()
    log_id = None
    if args.video:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.video_path)
    screenshot_dir = None
    if args.screenshots:
        screenshot_dir = Path(args.screenshots_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_vlm:
        image_path = Path("vlm_auto.png")
        renderer = p.ER_TINY_RENDERER if args.headless else p.ER_BULLET_HARDWARE_OPENGL
        _capture_vlm_image(env, image_path, renderer=renderer)
        output = _run_cpu_vlm(Path(args.vlm_model), image_path)
        Path(args.vlm_out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    steps = 0
    last_phase = None
    try:
        while True:
            _obs, info = env.step()
            steps += 1
            if screenshot_dir is not None and info["phase"] != last_phase:
                renderer = p.ER_TINY_RENDERER if args.headless else p.ER_BULLET_HARDWARE_OPENGL
                image_path = screenshot_dir / f"phase_{info['phase']}_{steps:06d}.png"
                _capture_scene(
                    env,
                    image_path,
                    width=args.screenshot_width,
                    height=args.screenshot_height,
                    renderer=renderer,
                )
                last_phase = info["phase"]
            if info["phase"] == "done":
                break
            if args.steps and steps >= args.steps:
                break
            if not args.no_sleep:
                time.sleep(env.control_dt)
    finally:
        if log_id is not None:
            p.stopStateLogging(log_id)
        env.close()
    print(f"Finished after {steps} steps | phase={info['phase']} | carry_mode={info['carry_mode']}")
    print(f"Violations: {info['violations']}")


if __name__ == "__main__":
    main()
