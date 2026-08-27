"""Thread-safe bounded cache for privacy-safe rendered preview frames."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class PreviewRenderKey:
    camera_id: int
    frame_token: Any
    inference_sequence: int | None
    width: int
    height: int


class PreviewRenderCache:
    """Keep a small number of rendered BGR frames to avoid repeated work."""

    def __init__(self, capacity: int = 12) -> None:
        if capacity < 1:
            raise ValueError("Preview render cache capacity must be positive.")
        self._capacity = capacity
        self._items: OrderedDict[PreviewRenderKey, Any] = OrderedDict()
        self._lock = RLock()

    def get(self, key: PreviewRenderKey):
        with self._lock:
            frame = self._items.get(key)
            if frame is None:
                return None
            self._items.move_to_end(key)
            return frame

    def put(self, key: PreviewRenderKey, frame: Any) -> None:
        with self._lock:
            self._items[key] = frame
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def discard_camera(self, camera_id: int) -> None:
        with self._lock:
            keys = [key for key in self._items if key.camera_id == camera_id]
            for key in keys:
                self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
