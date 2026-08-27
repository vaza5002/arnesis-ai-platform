"""Camera orchestration for Arnesis real-time group processing.

This service is the application boundary between persistent camera
configuration, protected RTSP credentials, and the canonical CameraRuntime.
It does not allocate GPUs and does not own GroupSession state. Those concerns
remain in ProcessingService.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from arnesis.application.camera_connection_service import CameraConnectionService
from arnesis.core.database import DatabaseManager
from arnesis.domain.entities import Camera, Group
from arnesis.infrastructure.rtsp_endpoint import RtspEndpoint
from arnesis.processing.camera_runtime import CameraRuntime
from arnesis.processing.camera_session import CameraMetrics, CameraSessionConfig
from arnesis.processing.latest_frame_queue import FramePacket


class RealtimeProcessingError(RuntimeError):
    """Raised when real-time camera orchestration cannot complete safely."""


@dataclass(frozen=True, slots=True)
class PreparedCameraGroup:
    group_id: int
    group_code: str
    camera_configs: tuple[CameraSessionConfig, ...]


class RealtimeProcessingService:
    """Prepares and controls canonical camera sessions for one or more groups."""

    def __init__(
        self,
        database: DatabaseManager,
        connection_service: CameraConnectionService | None = None,
        camera_runtime: CameraRuntime | None = None,
    ) -> None:
        self.database = database
        self.connection_service = connection_service or CameraConnectionService()
        self.camera_runtime = camera_runtime or CameraRuntime()

    def prepare_group(self, group_id: int) -> PreparedCameraGroup:
        """Validate a group and build protected in-memory camera configurations."""
        with self.database.session_scope() as session:
            group = session.scalar(
                select(Group)
                .options(selectinload(Group.cameras))
                .where(Group.id == group_id)
            )
            if group is None:
                raise RealtimeProcessingError(f"Group id {group_id} was not found.")
            if not group.enabled:
                raise RealtimeProcessingError(f"Group {group.code} is disabled.")

            cameras = sorted(
                (camera for camera in group.cameras if camera.enabled),
                key=lambda camera: (camera.display_order, camera.id),
            )
            if not cameras:
                raise RealtimeProcessingError(
                    f"Group {group.code} has no enabled cameras. Configure and enable "
                    "at least one camera before starting real-time processing."
                )
            if len(cameras) > group.max_concurrent_streams:
                raise RealtimeProcessingError(
                    f"Group {group.code} has {len(cameras)} enabled cameras but allows "
                    f"only {group.max_concurrent_streams} concurrent streams."
                )

            values = [
                (
                    camera.id,
                    camera.group_id,
                    camera.name,
                    camera.connection_uri,
                    camera.target_fps,
                    camera.reconnect_seconds,
                )
                for camera in cameras
            ]
            group_code = group.code

        configs: list[CameraSessionConfig] = []
        missing_credentials: list[str] = []
        invalid_cameras: list[str] = []
        for camera_id, camera_group_id, name, connection_uri, fps, reconnect in values:
            if not self.connection_service.password_is_configured(camera_id):
                missing_credentials.append(name)
                continue
            try:
                endpoint = RtspEndpoint.from_database_uri(connection_uri)
                configs.append(
                    self.connection_service.build_session_config(
                        camera_id=camera_id,
                        group_id=camera_group_id,
                        camera_name=name,
                        endpoint=endpoint,
                        target_fps=fps,
                        reconnect_seconds=float(reconnect),
                    )
                )
            except Exception as exc:
                invalid_cameras.append(f"{name}: {type(exc).__name__}: {exc}")

        if missing_credentials:
            raise RealtimeProcessingError(
                "Protected credentials are missing for: "
                + ", ".join(missing_credentials)
                + ". Open Camera Management and save each password before starting."
            )
        if invalid_cameras:
            raise RealtimeProcessingError(
                "One or more camera configurations are invalid: "
                + "; ".join(invalid_cameras)
            )

        return PreparedCameraGroup(
            group_id=group_id,
            group_code=group_code,
            camera_configs=tuple(configs),
        )

    def start_prepared_group(
        self, prepared: PreparedCameraGroup
    ) -> list[dict[str, object]]:
        try:
            metrics = self.camera_runtime.start_cameras(prepared.camera_configs)
            return [item.to_dict() for item in metrics]
        except Exception as exc:
            try:
                self.camera_runtime.stop_group(prepared.group_id)
            except Exception:
                pass
            raise RealtimeProcessingError(
                f"Unable to start cameras for group {prepared.group_code}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def pause_group(self, group_id: int) -> list[dict[str, object]]:
        return self._camera_command("pause", group_id)

    def resume_group(self, group_id: int) -> list[dict[str, object]]:
        return self._camera_command("resume", group_id)

    def stop_group(self, group_id: int) -> list[dict[str, object]]:
        if not self.camera_runtime.list_cameras(group_id):
            return []
        return self._camera_command("stop", group_id)

    def stop_all(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.camera_runtime.stop_all()]

    def group_snapshot(self, group_id: int) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.camera_runtime.list_cameras(group_id)]

    def subscribe_preview(self, group_id: int, camera_id: int) -> int:
        """Enable preview caching after validating group ownership."""
        metrics = self.camera_runtime.get_camera(camera_id)
        if metrics.group_id != group_id:
            raise KeyError(
                f"Camera id {camera_id} is not active in group id {group_id}."
            )
        return self.camera_runtime.subscribe_preview(camera_id)

    def unsubscribe_preview(self, group_id: int, camera_id: int) -> int:
        """Release preview caching after validating group ownership."""
        metrics = self.camera_runtime.get_camera(camera_id)
        if metrics.group_id != group_id:
            raise KeyError(
                f"Camera id {camera_id} is not active in group id {group_id}."
            )
        return self.camera_runtime.unsubscribe_preview(camera_id)

    def preview_frame(self, group_id: int, camera_id: int):
        """Return a cached preview without consuming an inference frame."""
        snapshots = self.group_snapshot(group_id)
        camera_ids = {
            int(item.get("camera_id"))
            for item in snapshots
            if item.get("camera_id") is not None
        }
        if camera_id not in camera_ids:
            raise KeyError(
                f"Camera id {camera_id} is not active in group id {group_id}."
            )
        runtime = (
            getattr(self, "camera_runtime", None)
            or getattr(self, "_camera_runtime", None)
            or getattr(self, "runtime", None)
        )
        if runtime is None or not hasattr(runtime, "preview_frame"):
            raise RuntimeError(
                "RealtimeProcessingService has no preview-capable camera runtime."
            )
        return runtime.preview_frame(camera_id)

    def latest_frame(
        self,
        group_id: int,
        camera_id: int,
        timeout_seconds: float | None = 0.0,
        after_sequence: int | None = None,
    ) -> FramePacket | None:
        metrics = self.camera_runtime.get_camera(camera_id)
        if metrics.group_id != group_id:
            raise RealtimeProcessingError(
                f"Camera id {camera_id} does not belong to active group id {group_id}."
            )
        return self.camera_runtime.latest_frame(
            camera_id,
            timeout_seconds=timeout_seconds,
            after_sequence=after_sequence,
        )

    def next_inference_frame(
        self,
        group_id: int,
        camera_id: int,
        timeout_seconds: float | None = None,
    ) -> FramePacket | None:
        """Consume the newest queued frame for one CUDA inference worker."""
        metrics = self.camera_runtime.get_camera(camera_id)
        if metrics.group_id != group_id:
            raise RealtimeProcessingError(
                f"Camera id {camera_id} does not belong to active group id {group_id}."
            )
        return self.camera_runtime.next_frame(
            camera_id=camera_id,
            timeout_seconds=timeout_seconds,
        )

    def _camera_command(self, command: str, group_id: int) -> list[dict[str, object]]:
        try:
            method = getattr(self.camera_runtime, f"{command}_group")
            metrics: list[CameraMetrics] = method(group_id)
            return [item.to_dict() for item in metrics]
        except Exception as exc:
            raise RealtimeProcessingError(
                f"Unable to {command} camera sessions for group id {group_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
