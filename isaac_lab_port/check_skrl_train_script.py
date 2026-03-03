from __future__ import annotations

import ast
import json
import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    rel = Path("scripts/reinforcement_learning/skrl/train.py")

    for env_key in ("ISAACLAB_PATH", "ISAAC_LAB_PATH"):
        root = os.environ.get(env_key, "").strip()
        if root:
            candidates.append(Path(root).expanduser() / rel)

    cwd = Path.cwd().resolve()
    candidates.append(cwd / rel)
    for parent in cwd.parents:
        candidates.append(parent / rel)

    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(c)
    return uniq


def _name_error_algorithm_suspect(source: str) -> bool:
    """Heuristic check for the known '--algorithm MAPPO' NameError bug."""
    try:
        tree = ast.parse(source)
    except Exception:
        # If parsing fails, treat script as suspicious.
        return True

    assigned: set[str] = set()
    loaded: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            self.generic_visit(node)

    Visitor().visit(tree)
    # Suspicious case: variable 'algorithm' is read but never assigned.
    return ("algorithm" in loaded) and ("algorithm" not in assigned)


def check_script() -> dict:
    for path in _candidate_paths():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        suspect = _name_error_algorithm_suspect(text)
        return {
            "found": True,
            "path": str(path),
            "algorithm_nameerror_suspect": bool(suspect),
        }
    return {
        "found": False,
        "path": "",
        "algorithm_nameerror_suspect": None,
    }


def main() -> None:
    print(json.dumps(check_script(), indent=2))


if __name__ == "__main__":
    main()

