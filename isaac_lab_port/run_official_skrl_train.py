from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "IsaacLab"
        / "scripts"
        / "reinforcement_learning"
        / "skrl"
        / "train.py"
    )
    if not script.exists():
        raise FileNotFoundError(f"Official IsaacLab SKRL train.py not found: {script}")

    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
