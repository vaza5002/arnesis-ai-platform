"""Canonical owner of all active Arnesis camera sessions."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from arnesis.processing.camera_session import (
    CameraMetrics,
    CameraSession,
    CameraSessionConfig,
    StateCallback,
)
from arnesis.processing.latest_frame_queue import FramePacket


class CameraSessionNotFoundError(KeyError):
    """Raised when a command references a camera that is not in the runtime."""


class CameraRuntime:
    """Owns unique camera sessions and exposes group-aware lifecycle commands."""

    def __init__(self, state_callback: StateCallback | None = None) -> None:
        self._sessions: dict[int, CameraSession] = {}
        self._state_callback = state_callback
        self._lock = threading.RLock()

    def start_camera(self, config: CameraSessionConfig) -> CameraMetrics:
        with self._lock:
            existing = self._sessions.get(config.camera_id)
            if existing is not None:
                if existing.config != config and existing.is_alive:
                    raise RuntimeError(
                        "Stop the active camera before changing its configuration."
                    )
                if existing.is_alive:
                    return existing.snapshot()
            session = CameraSession(config, state_callback=self._state_callback)
            self._sessions[config.camera_id] = session
        try:
            return session.start()
        except Exception:
            with self._lock:
                if self._sessions.get(config.camera_id) is session:
                    self._sessions.pop(config.camera_id, None)
            raise

    def start_cameras(self, configs: Iterable[CameraSessionConfig]) -> list[CameraMetrics]:
        started_ids: list[int] = []
        results: list[CameraMetrics] = []
        try:
            for config in configs:
                results.append(self.start_camera(config))
                started_ids.append(config.camera_id)
            return results
        except Exception:
            for camera_id in reversed(started_ids):
                try:
                    self.stop_camera(camera_id)
                except Exception:
                    pass
            raise

    def pause_camera(self, camera_id: int) -> CameraMetrics:
        return self._require(camera_id).pause()

    def resume_camera(self, camera_id: int) -> CameraMetrics:
        return self._require(camera_id).resume()

    def stop_camera(self, camera_id: int) -> CameraMetrics:
        session = self._require(camera_id)
        result = session.stop()
        with self._lock:
            if not session.is_alive:
                self._sessions.pop(camera_id, None)
        return result

    def pause_group(self, group_id: int) -> list[CameraMetrics]:
        return [session.pause() for session in self._sessions_for_group(group_id)]

    def resume_group(self, group_id: int) -> list[CameraMetrics]:
        return [session.resume() for session in self._sessions_for_group(group_id)]

    def stop_group(self, group_id: int) -> list[CameraMetrics]:
        with self._lock:
            camera_ids = [
                camera_id
                for camera_id, session in self._sessions.items()
                if session.config.group_id == group_id
            ]
        return [self.stop_camera(camera_id) for camera_id in camera_ids]

    def stop_all(self) -> list[CameraMetrics]:
        with self._lock:
            camera_ids = list(self._sessions)
        results: list[CameraMetrics] = []
        failures: list[str] = []
        for camera_id in camera_ids:
            try:
                results.append(self.stop_camera(camera_id))
            except Exception as exc:
                failures.append(f"camera {camera_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("Unable to stop all camera sessions: " + "; ".join(failures))
        return results

    def subscribe_preview(self, camera_id: int) -> int:
        """Subscribe one viewer to a camera preview cache."""
        return self._require(camera_id).subscribe_preview()

    def unsubscribe_preview(self, camera_id: int) -> int:
        """Release one viewer and free preview memory when no viewers remain."""
        return self._require(camera_id).unsubscribe_preview()

    def preview_frame(self, camera_id: int, copy_frame: bool = True):
        """Return a cached preview without consuming inference queue frames."""
        return self._require(camera_id).get_preview_frame(copy_frame=copy_frame)

    def latest_frame(
        self,
        camera_id: int,
        timeout_seconds: float | None = 0.0,
        after_sequence: int | None = None,
    ) -> FramePacket | None:
        return self._require(camera_id).latest_frame(timeout_seconds, after_sequence)

    def next_frame(
        self,
        camera_id: int,
        timeout_seconds: float | None = None,
    ) -> FramePacket | None:
        """Consume the newest queued frame exclusively for CUDA inference."""
        return self._require(camera_id).next_frame(
            timeout_seconds=timeout_seconds,
        )

    def get_camera(self, camera_id: int) -> CameraMetrics:
        return self._require(camera_id).snapshot()

    def list_cameras(self, group_id: int | None = None) -> list[CameraMetrics]:
        with self._lock:
            sessions = list(self._sessions.values())
        if group_id is not None:
            sessions = [session for session in sessions if session.config.group_id == group_id]
        return sorted(
            (session.snapshot() for session in sessions),
            key=lambda item: (item.group_id, item.camera_name.casefold()),
        )

    def contains(self, camera_id: int) -> bool:
        with self._lock:
            return camera_id in self._sessions

    def _sessions_for_group(self, group_id: int) -> list[CameraSession]:
        with self._lock:
            sessions = [
                session
                for session in self._sessions.values()
                if session.config.group_id == group_id
            ]
        if not sessions:
            raise CameraSessionNotFoundError(
                f"No active camera sessions exist for group id {group_id}."
            )
        return sessions

    def _require(self, camera_id: int) -> CameraSession:
        with self._lock:
            session = self._sessions.get(camera_id)
        if session is None:
            raise CameraSessionNotFoundError(
                f"No active camera session exists for camera id {camera_id}."
            )
        return session
