"""Secure RTSP configuration and connection testing for canonical CameraSession."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2

from arnesis.infrastructure.credential_store import CameraCredentialStore
from arnesis.infrastructure.rtsp_endpoint import RtspEndpoint
from arnesis.processing.camera_session import CameraSessionConfig


@dataclass(frozen=True, slots=True)
class CameraSnapshotResult:
    """One static frame captured without starting a runtime session."""

    success: bool
    camera_id: int
    camera_name: str
    masked_url: str
    elapsed_ms: int
    width: int
    height: int
    frame: object | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CameraConnectionTestResult:
    success: bool
    camera_id: int
    camera_name: str
    masked_url: str
    elapsed_ms: int
    width: int
    height: int
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CameraConnectionService:
    """Creates secure CameraSessionConfig objects without persisting passwords."""

    def __init__(self, credential_store: CameraCredentialStore | None = None) -> None:
        self.credential_store = credential_store or CameraCredentialStore()

    def save_password(self, camera_id: int, password: str) -> None:
        self.credential_store.set_password(
            CameraCredentialStore.camera_target(camera_id), password
        )

    def delete_password(self, camera_id: int) -> bool:
        return self.credential_store.delete_password(
            CameraCredentialStore.camera_target(camera_id)
        )

    def password_is_configured(self, camera_id: int) -> bool:
        return self.credential_store.contains(
            CameraCredentialStore.camera_target(camera_id)
        )

    def build_session_config(
        self,
        *,
        camera_id: int,
        group_id: int,
        camera_name: str,
        endpoint: RtspEndpoint,
        target_fps: float = 15.0,
        reconnect_seconds: float = 5.0,
        read_timeout_seconds: float = 10.0,
    ) -> CameraSessionConfig:
        password = self._get_password(camera_id)
        return CameraSessionConfig(
            camera_id=camera_id,
            group_id=group_id,
            camera_name=camera_name,
            source_uri=endpoint.build_url(password),
            target_fps=target_fps,
            reconnect_seconds=reconnect_seconds,
            read_timeout_seconds=read_timeout_seconds,
            backend=cv2.CAP_FFMPEG,
        )

    def build_capture_configuration(self, **kwargs) -> CameraSessionConfig:
        """Compatibility alias retained while callers migrate to build_session_config."""
        if "group_id" not in kwargs:
            raise ValueError(
                "group_id is required by the canonical CameraSession configuration."
            )
        return self.build_session_config(**kwargs)

    def capture_snapshot(
        self,
        *,
        camera_id: int,
        camera_name: str,
        endpoint: RtspEndpoint,
        read_attempts: int = 30,
    ) -> CameraSnapshotResult:
        """Open RTSP temporarily, capture one frame, and release immediately."""
        if read_attempts < 1:
            raise ValueError("Read attempts must be greater than zero.")

        started = time.perf_counter()
        capture: cv2.VideoCapture | None = None
        try:
            password = self._get_password(camera_id)
            capture = cv2.VideoCapture(
                endpoint.build_url(password),
                cv2.CAP_FFMPEG,
            )
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise ConnectionError("OpenCV could not open the RTSP stream.")

            frame = None
            for _ in range(read_attempts):
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size > 0:
                    frame = candidate.copy()
                    break
            if frame is None:
                raise ConnectionError(
                    "The RTSP stream opened but returned no valid frame."
                )

            height, width = frame.shape[:2]
            return CameraSnapshotResult(
                success=True,
                camera_id=camera_id,
                camera_name=camera_name,
                masked_url=endpoint.masked_url(),
                elapsed_ms=self._elapsed_ms(started),
                width=width,
                height=height,
                frame=frame,
                error=None,
            )
        except Exception as exc:
            return CameraSnapshotResult(
                success=False,
                camera_id=camera_id,
                camera_name=camera_name,
                masked_url=endpoint.masked_url(),
                elapsed_ms=self._elapsed_ms(started),
                width=0,
                height=0,
                frame=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if capture is not None:
                capture.release()

    def test_connection(
        self,
        *,
        camera_id: int,
        camera_name: str,
        endpoint: RtspEndpoint,
        read_attempts: int = 30,
    ) -> CameraConnectionTestResult:
        if read_attempts < 1:
            raise ValueError("Read attempts must be greater than zero.")

        started = time.perf_counter()
        capture: cv2.VideoCapture | None = None
        try:
            password = self._get_password(camera_id)
            capture = cv2.VideoCapture(endpoint.build_url(password), cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise ConnectionError("OpenCV could not open the RTSP stream.")

            frame = None
            for _ in range(read_attempts):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    frame = candidate
                    break
            if frame is None:
                raise ConnectionError("The RTSP stream opened but returned no valid frame.")

            height, width = frame.shape[:2]
            return CameraConnectionTestResult(
                success=True,
                camera_id=camera_id,
                camera_name=camera_name,
                masked_url=endpoint.masked_url(),
                elapsed_ms=self._elapsed_ms(started),
                width=width,
                height=height,
                error=None,
            )
        except Exception as exc:
            return CameraConnectionTestResult(
                success=False,
                camera_id=camera_id,
                camera_name=camera_name,
                masked_url=endpoint.masked_url(),
                elapsed_ms=self._elapsed_ms(started),
                width=0,
                height=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if capture is not None:
                capture.release()

    def _get_password(self, camera_id: int) -> str:
        return self.credential_store.get_password(
            CameraCredentialStore.camera_target(camera_id)
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
