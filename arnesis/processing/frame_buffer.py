"""Thread-safe latest-frame buffer for real-time camera processing."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Immutable metadata wrapper around one BGR image."""

    sequence: int
    captured_at: datetime
    frame: np.ndarray


@dataclass(frozen=True, slots=True)
class FrameBufferSnapshot:
    capacity: int
    current_size: int
    frames_published: int
    frames_consumed: int
    frames_dropped: int
    latest_sequence: int


class LatestFrameBuffer:
    """Bounded buffer that discards stale frames instead of accumulating delay."""

    def __init__(self, capacity: int = 2, copy_frames: bool = False) -> None:
        if capacity < 1:
            raise ValueError("Frame buffer capacity must be at least one.")
        self.capacity = capacity
        self.copy_frames = copy_frames
        self._items: list[FramePacket] = []
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._frames_published = 0
        self._frames_consumed = 0
        self._frames_dropped = 0
        self._latest_sequence = 0

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def publish(
        self,
        frame: np.ndarray,
        captured_at: datetime | None = None,
    ) -> FramePacket:
        self._validate_frame(frame)
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed frame buffer.")

            self._latest_sequence += 1
            packet = FramePacket(
                sequence=self._latest_sequence,
                captured_at=captured_at or datetime.now(timezone.utc),
                frame=frame.copy() if self.copy_frames else frame,
            )
            if len(self._items) >= self.capacity:
                overflow = len(self._items) - self.capacity + 1
                del self._items[:overflow]
                self._frames_dropped += overflow

            self._items.append(packet)
            self._frames_published += 1
            self._condition.notify_all()
            return packet

    def get_latest(
        self,
        timeout_seconds: float | None = None,
        after_sequence: int | None = None,
    ) -> FramePacket | None:
        """Return the newest available frame, optionally waiting for a newer one."""
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("Timeout cannot be negative.")

        with self._condition:
            ready = lambda: (
                self._closed
                or bool(self._items)
                and (
                    after_sequence is None
                    or self._items[-1].sequence > after_sequence
                )
            )
            if not ready():
                self._condition.wait_for(ready, timeout=timeout_seconds)

            if not self._items:
                return None
            latest = self._items[-1]
            if after_sequence is not None and latest.sequence <= after_sequence:
                return None

            stale_count = len(self._items) - 1
            if stale_count > 0:
                self._frames_dropped += stale_count
            self._items.clear()
            self._frames_consumed += 1
            return latest

    def clear(self) -> None:
        with self._condition:
            self._frames_dropped += len(self._items)
            self._items.clear()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self) -> FrameBufferSnapshot:
        with self._condition:
            return FrameBufferSnapshot(
                capacity=self.capacity,
                current_size=len(self._items),
                frames_published=self._frames_published,
                frames_consumed=self._frames_consumed,
                frames_dropped=self._frames_dropped,
                latest_sequence=self._latest_sequence,
            )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("A camera frame must be a NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("A camera frame must use HxWx3 BGR format.")
        if frame.dtype != np.uint8:
            raise ValueError("A camera frame must use uint8 values.")
