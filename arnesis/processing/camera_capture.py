"""Resilient OpenCV camera capture worker for Arnesis."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2

from arnesis.processing.frame_buffer import LatestFrameBuffer


class CaptureState(str, Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CameraCaptureConfiguration:
    camera_id: int
    camera_name: str
    source_type: str
    connection_uri: str
    target_fps: float = 15.0
    reconnect_seconds: float = 5.0
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if self.camera_id <= 0:
            raise ValueError("Camera id must be greater than zero.")
        if not self.camera_name.strip():
            raise ValueError("Camera name is required.")
        if self.target_fps <= 0:
            raise ValueError("Target FPS must be greater than zero.")
        if self.reconnect_seconds < 0:
            raise ValueError("Reconnect interval cannot be negative.")


@dataclass(frozen=True, slots=True)
class CameraCaptureSnapshot:
    camera_id: int
    camera_name: str
    state: str
    source_type: str
    frames_read: int
    read_failures: int
    reconnect_attempts: int
    actual_width: int
    actual_height: int
    last_frame_at: str | None
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CameraCaptureWorker:
    """Captures BGR frames without allowing one camera to block other cameras."""

    def __init__(
        self,
        configuration: CameraCaptureConfiguration,
        buffer: LatestFrameBuffer,
    ) -> None:
        self.configuration = configuration
        self.buffer = buffer
        self._state = CaptureState.STOPPED
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._started_event = threading.Event()
        self._lock = threading.RLock()
        self._frames_read = 0
        self._read_failures = 0
        self._reconnect_attempts = 0
        self._actual_width = 0
        self._actual_height = 0
        self._last_frame_at: datetime | None = None
        self._last_error: str | None = None

    @property
    def state(self) -> CaptureState:
        with self._lock:
            return self._state

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout_seconds: float = 10.0) -> CameraCaptureSnapshot:
        with self._lock:
            if self.is_alive:
                return self.snapshot()
            self._stop_event.clear()
            self._pause_event.clear()
            self._started_event.clear()
            self._last_error = None
            self._state = CaptureState.CONNECTING
            self._thread = threading.Thread(
                target=self._run,
                name=f"arnesis-camera-{self.configuration.camera_id}",
                daemon=False,
            )
            self._thread.start()

        if not self._started_event.wait(timeout_seconds):
            self.stop()
            raise TimeoutError(
                f"Camera {self.configuration.camera_name} did not initialize in time."
            )
        if self.state == CaptureState.ERROR:
            raise RuntimeError(self._last_error or "Camera initialization failed.")
        return self.snapshot()

    def pause(self) -> CameraCaptureSnapshot:
        with self._lock:
            if self._state == CaptureState.PAUSED:
                return self.snapshot()
            if self._state not in {CaptureState.RUNNING, CaptureState.RECONNECTING}:
                raise RuntimeError(f"Camera cannot pause from {self._state.value}.")
            self._pause_event.set()
            self._state = CaptureState.PAUSED
            return self.snapshot()

    def resume(self) -> CameraCaptureSnapshot:
        with self._lock:
            if self._state == CaptureState.RUNNING:
                return self.snapshot()
            if self._state != CaptureState.PAUSED:
                raise RuntimeError(f"Camera cannot resume from {self._state.value}.")
            self._pause_event.clear()
            self._state = CaptureState.RUNNING
            return self.snapshot()

    def stop(self, timeout_seconds: float = 15.0) -> CameraCaptureSnapshot:
        with self._lock:
            if not self.is_alive:
                self._release_capture()
                self._state = CaptureState.STOPPED
                return self.snapshot()
            self._state = CaptureState.STOPPING
            self._stop_event.set()
            self._pause_event.clear()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(
                    f"Camera {self.configuration.camera_name} did not stop in time."
                )
        with self._lock:
            self._thread = None
            self._release_capture()
            self._state = CaptureState.STOPPED
            return self.snapshot()

    def snapshot(self) -> CameraCaptureSnapshot:
        with self._lock:
            return CameraCaptureSnapshot(
                camera_id=self.configuration.camera_id,
                camera_name=self.configuration.camera_name,
                state=self._state.value,
                source_type=self.configuration.source_type.upper(),
                frames_read=self._frames_read,
                read_failures=self._read_failures,
                reconnect_attempts=self._reconnect_attempts,
                actual_width=self._actual_width,
                actual_height=self._actual_height,
                last_frame_at=(
                    self._last_frame_at.isoformat(timespec="seconds")
                    if self._last_frame_at
                    else None
                ),
                last_error=self._last_error,
            )

    def _run(self) -> None:
        try:
            first_attempt = True
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self._stop_event.wait(0.10)
                    continue

                if self._capture is None or not self._capture.isOpened():
                    self._state = (
                        CaptureState.CONNECTING if first_attempt else CaptureState.RECONNECTING
                    )
                    if not first_attempt:
                        self._reconnect_attempts += 1
                    if not self._open_capture():
                        self._started_event.set()
                        first_attempt = False
                        if not self._wait_for_reconnect():
                            break
                        continue
                    self._state = CaptureState.RUNNING
                    self._started_event.set()
                    first_attempt = False

                started = time.perf_counter()
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._read_failures += 1
                    self._release_capture()
                    if self.configuration.source_type.upper() == "FILE":
                        break
                    if not self._wait_for_reconnect():
                        break
                    continue

                self._actual_height, self._actual_width = frame.shape[:2]
                self._last_frame_at = datetime.now(timezone.utc)
                self.buffer.publish(frame, self._last_frame_at)
                self._frames_read += 1
                self._last_error = None
                self._throttle(started)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._state = CaptureState.ERROR
            self._started_event.set()
        finally:
            self._release_capture()
            with self._lock:
                if self._state != CaptureState.ERROR:
                    self._state = CaptureState.STOPPED

    def _open_capture(self) -> bool:
        source = self._resolve_source()
        capture = cv2.VideoCapture(source)
        if self.configuration.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.configuration.width)
        if self.configuration.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configuration.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            self._last_error = f"Unable to open camera source: {source}"
            return False
        self._capture = capture
        return True

    def _resolve_source(self) -> int | str:
        source_type = self.configuration.source_type.strip().upper()
        uri = self.configuration.connection_uri.strip()
        if source_type == "USB":
            try:
                return int(uri)
            except ValueError as exc:
                raise ValueError("A USB camera URI must be an integer index.") from exc
        if source_type == "FILE":
            path = Path(uri).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Video file was not found: {path}")
            return str(path)
        if source_type == "RTSP":
            if not uri.lower().startswith(("rtsp://", "rtsps://")):
                raise ValueError("An RTSP camera URI must start with rtsp:// or rtsps://.")
            return uri
        raise ValueError(f"Unsupported camera source type: {source_type}")

    def _wait_for_reconnect(self) -> bool:
        self._state = CaptureState.RECONNECTING
        return not self._stop_event.wait(self.configuration.reconnect_seconds)

    def _throttle(self, started: float) -> None:
        frame_interval = 1.0 / self.configuration.target_fps
        remaining = frame_interval - (time.perf_counter() - started)
        if remaining > 0:
            self._stop_event.wait(remaining)

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
