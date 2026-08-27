"""Low-overhead runtime performance diagnostics for Arnesis."""
from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    operation: str
    samples: int
    average_ms: float
    maximum_ms: float
    latest_ms: float


class PerformanceDiagnostics:
    """Collect bounded timing samples without blocking inference or UI threads."""

    def __init__(self, output_root: str | Path = "D:/Arnesis/Data/Diagnostics") -> None:
        self.output_root = Path(output_root)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._lock = threading.RLock()
        self._last_export_monotonic = 0.0

    def record(self, operation: str, elapsed_ms: float) -> None:
        with self._lock:
            self._samples[operation].append(float(elapsed_ms))

    def measure(self, operation: str):
        return _Measurement(self, operation)

    def snapshots(self) -> tuple[PerformanceSnapshot, ...]:
        with self._lock:
            copied = {name: tuple(values) for name, values in self._samples.items()}
        return tuple(
            PerformanceSnapshot(
                operation=name,
                samples=len(values),
                average_ms=round(sum(values) / len(values), 3),
                maximum_ms=round(max(values), 3),
                latest_ms=round(values[-1], 3),
            )
            for name, values in sorted(copied.items())
            if values
        )

    def export_if_due(self, interval_seconds: float = 60.0) -> Path | None:
        now = time.monotonic()
        if now - self._last_export_monotonic < interval_seconds:
            return None
        self._last_export_monotonic = now
        snapshots = self.snapshots()
        if not snapshots:
            return None
        self.output_root.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = self.output_root / f"performance_diagnostics_{day}.csv"
        exists = path.exists() and path.stat().st_size > 0
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with path.open("a", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(("TimestampUTC", "Operation", "Samples", "AverageMs", "MaximumMs", "LatestMs"))
            for item in snapshots:
                writer.writerow((timestamp, item.operation, item.samples, item.average_ms, item.maximum_ms, item.latest_ms))
        return path


class _Measurement:
    def __init__(self, diagnostics: PerformanceDiagnostics, operation: str) -> None:
        self.diagnostics = diagnostics
        self.operation = operation
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        elapsed_ms = (time.perf_counter() - self.started) * 1000.0
        self.diagnostics.record(self.operation, elapsed_ms)
