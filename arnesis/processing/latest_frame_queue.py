"""Bounded latest-frame queue for low-latency real-time processing."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Any


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Typed video frame transported through the processing pipeline."""

    camera_id: int
    sequence: int
    captured_at: float
    frame: Any


class LatestFrameQueue:
    """Keep only the newest typed frames instead of accumulating latency."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("Frame queue capacity must be positive.")
        self._capacity = capacity
        self._items: list[FramePacket] = []
        self._condition = Condition()
        self.dropped_frames = 0
        self.closed = False
        self._last_packet: FramePacket | None = None

    def put(self, packet: FramePacket) -> None:
        if not isinstance(packet, FramePacket):
            raise TypeError(
                "LatestFrameQueue.put() requires FramePacket; "
                f"received {type(packet).__name__}."
            )
        with self._condition:
            if self.closed:
                return
            while len(self._items) >= self._capacity:
                self._items.pop(0)
                self.dropped_frames += 1
            self._items.append(packet)
            self._last_packet = packet
            self._condition.notify()

    def get_latest(self, timeout: float | None = None) -> FramePacket | None:
        """Consume and return the newest queued packet for inference."""
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while not self._items and not self.closed:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._items:
                return None
            packet = self._items[-1]
            if len(self._items) > 1:
                self.dropped_frames += len(self._items) - 1
            self._items.clear()
            return packet

    def peek_latest(self, copy_frame: bool = True) -> FramePacket | None:
        """Return the last packet without consuming the inference queue."""
        with self._condition:
            packet = self._last_packet
            if packet is None:
                return None
            frame = (
                packet.frame.copy()
                if copy_frame and hasattr(packet.frame, "copy")
                else packet.frame
            )
            return FramePacket(
                camera_id=packet.camera_id,
                sequence=packet.sequence,
                captured_at=packet.captured_at,
                frame=frame,
            )

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._items.clear()
            self._condition.notify_all()
