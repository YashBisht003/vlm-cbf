from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


class TrainLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False
        self._fieldnames: Optional[list[str]] = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header:
                    self._fieldnames = list(header)
                    self._header_written = True

    def write(self, row: Dict[str, float]) -> None:
        fieldnames = self._fieldnames or list(row.keys())
        if self._fieldnames is None:
            self._fieldnames = fieldnames
        if not self._header_written:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            self._header_written = True
            return
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)
