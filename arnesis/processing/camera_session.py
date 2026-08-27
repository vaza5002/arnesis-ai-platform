"""Canonical camera session for Arnesis real-time acquisition.

A CameraSession owns exactly one OpenCV capture worker and one latest-frame
queue. It is independent from the UI and from CUDA inference so cameras can be
started, paused, resumed, reconnected, and stopped without blocking other
cameras in the same group.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Callable

import cv2

from arnesis.processing.latest_frame_queue import FramePacket, LatestFrameQueue


class CameraState(StrEnum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CameraSessionConfig:
    camera_id: int
    group_id: int
    camera_name: str
    source_uri: str
    target_fps: float = 15.0
    reconnect_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    backend: int = cv2.CAP_FFMPEG

    def __post_init__(self) -> None:
        if self.camera_id <= 0:
            raise ValueError("Camera id must be greater than zero.")
        if self.group_id <= 0:
            raise ValueError("Group id must be greater than zero.")
        if not self.camera_name.strip():
            raise ValueError("Camera name is required.")
        if not self.source_uri.strip():
            raise ValueError("Camera source URI is required.")
        if self.target_fps <= 0 or self.target_fps > 120:
            raise ValueError("Target FPS must be between 0 and 120.")
        if self.reconnect_seconds < 0 or self.reconnect_seconds > 300:
            raise ValueError("Reconnect interval must be between 0 and 300 seconds.")
        if self.read_timeout_seconds <= 0:
            raise ValueError("Read timeout must be greater than zero.")

    @property
    def safe_source(self) -> str:
        """Return a display-safe source without exposing RTSP credentials."""
        source = self.source_uri
        if "://" not in source or "@" not in source:
            return source
        scheme, remainder = source.split("://", 1)
        authority, suffix = remainder.split("@", 1)
        username = authority.split(":", 1)[0]
        return f"{scheme}://{username}:*****@{suffix}"


@dataclass(frozen=True, slots=True)
class CameraMetrics:
    camera_id: int
    group_id: int
    camera_name: str
    state: str
    frames_captured: int
    read_failures: int
    reconnect_attempts: int
    frames_dropped: int
    measured_fps: float
    frame_width: int
    frame_height: int
    last_frame_monotonic: float | None
    last_error: str | None
    safe_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(frozen=True, slots=True)
class PreviewFramePacket:
    """Non-destructive camera preview frame cached outside inference queues."""

    camera_id: int
    captured_at: float
    frame: Any

StateCallback = Callable[[CameraMetrics], None]


class CameraSession:
    """Thread-safe lifecycle owner for one camera and its latest-frame queue."""

    def __init__(
        self,
        config: CameraSessionConfig,
        *,
        queue: LatestFrameQueue | None = None,
        state_callback: StateCallback | None = None,
    ) -> None:
        self.config = config
        self.queue = queue or LatestFrameQueue()
        self._state_callback = state_callback
        self._state = CameraState.STOPPED
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._started_event = threading.Event()
        self._lock = threading.RLock()
        self._frames_captured = 0
        self._read_failures = 0
        self._reconnect_attempts = 0
        self._frame_width = 0
        self._frame_height = 0
        self._last_frame_monotonic: float | None = None
        self._last_error: str | None = None
        self._fps_window_started = time.monotonic()
        self._fps_window_frames = 0
        self._measured_fps = 0.0
        self._preview_packet: PreviewFramePacket | None = None
        self._preview_subscribers = 0

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, timeout_seconds: float = 12.0) -> CameraMetrics:
        with self._lock:
            if self.is_alive:
                return self.snapshot()
            self._stop_event.clear()
            self._pause_event.clear()
            self._started_event.clear()
            self._last_error = None
            self._set_state(CameraState.CONNECTING)
            self._thread = threading.Thread(
                target=self._run,
                name=f"arnesis-camera-{self.config.camera_id}",
                daemon=False,
            )
            self._thread.start()

        if not self._started_event.wait(timeout_seconds):
            self.stop(timeout_seconds=timeout_seconds)
            raise TimeoutError(
                f"Camera '{self.config.camera_name}' did not initialize within "
                f"{timeout_seconds:.1f} seconds."
            )
        if self.state == CameraState.ERROR:
            raise RuntimeError(self._last_error or "Camera initialization failed.")
        return self.snapshot()

    def pause(self) -> CameraMetrics:
        with self._lock:
            if self._state == CameraState.PAUSED:
                return self.snapshot()
            if self._state not in {CameraState.RUNNING, CameraState.RECONNECTING}:
                raise RuntimeError(f"Camera cannot pause from {self._state.value}.")
            self._pause_event.set()
            self._set_state(CameraState.PAUSED)
            return self.snapshot()

    def resume(self) -> CameraMetrics:
        with self._lock:
            if self._state == CameraState.RUNNING:
                return self.snapshot()
            if self._state != CameraState.PAUSED:
                raise RuntimeError(f"Camera cannot resume from {self._state.value}.")
            self._pause_event.clear()
            self._set_state(CameraState.RUNNING)
            return self.snapshot()

    def stop(self, timeout_seconds: float = 15.0) -> CameraMetrics:
        with self._lock:
            if not self.is_alive:
                self._release_capture()
                self._set_state(CameraState.STOPPED)
                return self.snapshot()
            self._set_state(CameraState.STOPPING)
            self._stop_event.set()
            self._pause_event.clear()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(
                    f"Camera '{self.config.camera_name}' did not stop within "
                    f"{timeout_seconds:.1f} seconds."
                )
        with self._lock:
            self._thread = None
            self._release_capture()
            if self._state != CameraState.ERROR:
                self._set_state(CameraState.STOPPED)
            return self.snapshot()

    def latest_frame(
        self,
        timeout_seconds: float | None = 0.0,
        after_sequence: int | None = None,
    ) -> FramePacket | None:
        """Return a non-consuming frame copy for UI previews."""
        del timeout_seconds
        peeker = getattr(self.queue, "peek_latest", None)
        if peeker is None:
            raise RuntimeError(
                "LatestFrameQueue does not expose peek_latest() for previews."
            )
        packet = peeker(copy_frame=True)
        if packet is None:
            return None
        if after_sequence is not None and packet.sequence <= after_sequence:
            return None
        return packet

    def next_frame(
        self,
        timeout_seconds: float | None = None,
    ) -> FramePacket | None:
        """Consume the newest queued frame for future CUDA inference."""
        getter = getattr(self.queue, "get_latest", None)
        if getter is None:
            raise RuntimeError(
                "LatestFrameQueue does not expose get_latest() for inference."
            )
        return getter(timeout=timeout_seconds)


    @property
    def preview_subscriber_count(self) -> int:
        """Return the number of active preview consumers."""
        with self._lock:
            return self._preview_subscribers

    def subscribe_preview(self) -> int:
        """Enable preview caching for one additional consumer."""
        with self._lock:
            self._preview_subscribers += 1
            return self._preview_subscribers

    def unsubscribe_preview(self) -> int:
        """Release one consumer and free the cached preview when unused."""
        with self._lock:
            if self._preview_subscribers > 0:
                self._preview_subscribers -= 1
            if self._preview_subscribers == 0:
                self._preview_packet = None
            return self._preview_subscribers

    def get_preview_frame(self, copy_frame: bool = True) -> PreviewFramePacket | None:
        """Return the latest frame without consuming an inference queue item."""
        with self._lock:
            if self._preview_subscribers <= 0:
                return None
            packet = self._preview_packet
            if packet is None:
                return None
            frame = (
                packet.frame.copy()
                if copy_frame and hasattr(packet.frame, "copy")
                else packet.frame
            )
            return PreviewFramePacket(
                camera_id=packet.camera_id,
                captured_at=packet.captured_at,
                frame=frame,
            )

    def snapshot(self) -> CameraMetrics:
        with self._lock:
            dropped = self._queue_dropped_count()
            return CameraMetrics(
                camera_id=self.config.camera_id,
                group_id=self.config.group_id,
                camera_name=self.config.camera_name,
                state=self._state.value,
                frames_captured=self._frames_captured,
                read_failures=self._read_failures,
                reconnect_attempts=self._reconnect_attempts,
                frames_dropped=dropped,
                measured_fps=round(self._measured_fps, 2),
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                last_frame_monotonic=self._last_frame_monotonic,
                last_error=self._last_error,
                safe_source=self.config.safe_source,
            )

    def _run(self) -> None:
        first_attempt = True
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self._stop_event.wait(0.10)
                    continue

                if self._capture is None or not self._capture.isOpened():
                    self._set_state(
                        CameraState.CONNECTING if first_attempt else CameraState.RECONNECTING
                    )
                    if not first_attempt:
                        self._reconnect_attempts += 1
                    if not self._open_capture():
                        self._started_event.set()
                        first_attempt = False
                        if not self._stop_event.wait(self.config.reconnect_seconds):
                            continue
                        break
                    first_attempt = False
                    self._set_state(CameraState.RUNNING)
                    self._started_event.set()

                cycle_started = time.monotonic()
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._read_failures += 1
                    self._last_error = "Camera stream returned no valid frame."
                    self._release_capture()
                    continue

                self._frame_height, self._frame_width = frame.shape[:2]
                self._last_frame_monotonic = time.monotonic()
                self._publish_frame(frame, self._last_frame_monotonic)
                self._frames_captured += 1
                self._fps_window_frames += 1
                self._last_error = None
                self._update_fps()
                self._throttle(cycle_started)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._set_state(CameraState.ERROR)
            self._started_event.set()
        finally:
            self._release_capture()
            with self._lock:
                if self._state != CameraState.ERROR:
                    self._set_state(CameraState.STOPPED)

    def _open_capture(self) -> bool:
        capture = cv2.VideoCapture(self.config.source_uri, self.config.backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            self._last_error = "OpenCV could not open the camera stream."
            return False
        self._capture = capture
        return True

    def _publish_frame(self, frame: Any, captured_at: float) -> None:
        """Publish a typed packet and update the independent preview cache."""
        sequence = self._frames_captured + 1
        packet = FramePacket(
            camera_id=self.config.camera_id,
            sequence=sequence,
            captured_at=captured_at,
            frame=frame,
        )
        with self._lock:
            preview_enabled = self._preview_subscribers > 0
        if preview_enabled:
            preview_frame = frame.copy() if hasattr(frame, "copy") else frame
            with self._lock:
                if self._preview_subscribers > 0:
                    self._preview_packet = PreviewFramePacket(
                        camera_id=self.config.camera_id,
                        captured_at=captured_at,
                        frame=preview_frame,
                    )
        self.queue.put(packet)


    def _queue_dropped_count(self) -> int:
        snapshot_method = getattr(self.queue, "snapshot", None)
        if snapshot_method is not None:
            snapshot = snapshot_method()
            for name in ("frames_dropped", "dropped", "dropped_frames"):
                value = getattr(snapshot, name, None)
                if value is not None:
                    return int(value)
        for name in ("frames_dropped", "dropped", "dropped_frames"):
            value = getattr(self.queue, name, None)
            if value is not None:
                return int(value)
        return 0

    def _update_fps(self) -> None:
        now = time.monotonic()
        elapsed = now - self._fps_window_started
        if elapsed >= 1.0:
            self._measured_fps = self._fps_window_frames / elapsed
            self._fps_window_started = now
            self._fps_window_frames = 0

    def _throttle(self, cycle_started: float) -> None:
        remaining = (1.0 / self.config.target_fps) - (time.monotonic() - cycle_started)
        if remaining > 0:
            self._stop_event.wait(remaining)

    def _set_state(self, state: CameraState) -> None:
        self._state = state
        callback = self._state_callback
        if callback is not None:
            callback(self.snapshot())

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
