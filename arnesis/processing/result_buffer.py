"""Thread-safe latest-result buffer for RT consumers."""
from __future__ import annotations

import threading


class LatestResultBuffer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest = None
        self._closed = False

    def publish(self, result) -> None:
        with self._lock:
            if not self._closed:
                self._latest = result

    def get_latest(self, after_sequence: int | None = None):
        with self._lock:
            result = self._latest
            if result is None:
                return None
            if after_sequence is not None and result.sequence <= after_sequence:
                return None
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._latest = None
