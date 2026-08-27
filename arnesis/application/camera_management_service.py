"""Camera CRUD and secure RTSP configuration use cases for Arnesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select

from arnesis.application.camera_connection_service import (
    CameraConnectionService,
    CameraConnectionTestResult,
    CameraSnapshotResult,
)
from arnesis.core.database import DatabaseManager
from arnesis.domain.entities import Camera, Group
from arnesis.infrastructure.repositories import ValidationError
from arnesis.infrastructure.rtsp_endpoint import RtspEndpoint, RtspEndpointError


@dataclass(frozen=True, slots=True)
class CameraRecord:
    id: int
    group_id: int
    group_code: str
    name: str
    host: str
    port: int
    username: str
    stream_path: str
    target_fps: float
    reconnect_seconds: int
    width: int | None
    height: int | None
    enabled: bool
    credential_configured: bool
    masked_url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CameraManagementService:
    """Provides validated camera CRUD without exposing plain-text credentials."""

    def __init__(
        self,
        database: DatabaseManager,
        connection_service: CameraConnectionService | None = None,
    ) -> None:
        self.database = database
        self.connection_service = connection_service or CameraConnectionService()

    def list_groups(self) -> list[dict[str, object]]:
        with self.database.session_scope() as session:
            groups = session.scalars(select(Group).order_by(Group.code)).all()
            return [
                {"id": group.id, "code": group.code, "name": group.name}
                for group in groups
            ]

    def list_cameras(self, group_id: int | None = None) -> list[CameraRecord]:
        with self.database.session_scope() as session:
            statement = select(Camera, Group).join(Group, Camera.group_id == Group.id)
            if group_id is not None:
                statement = statement.where(Camera.group_id == group_id)
            statement = statement.order_by(Group.code, Camera.display_order, Camera.name)
            rows = session.execute(statement).all()
            return [self._to_record(camera, group.code) for camera, group in rows]

    def get_camera(self, camera_id: int) -> CameraRecord:
        with self.database.session_scope() as session:
            row = session.execute(
                select(Camera, Group)
                .join(Group, Camera.group_id == Group.id)
                .where(Camera.id == camera_id)
            ).one_or_none()
            if row is None:
                raise ValidationError(f"Camera id {camera_id} was not found.")
            return self._to_record(row[0], row[1].code)

    def create_camera(
        self,
        *,
        group_id: int,
        name: str,
        host: str,
        port: int,
        username: str,
        password: str,
        stream_path: str = "/Streaming/Channels/101",
        target_fps: float = 15.0,
        reconnect_seconds: int = 5,
        width: int | None = None,
        height: int | None = None,
    ) -> CameraRecord:
        endpoint = self._validate_input(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            stream_path=stream_path,
            target_fps=target_fps,
            reconnect_seconds=reconnect_seconds,
            width=width,
            height=height,
        )

        camera_id: int | None = None
        try:
            with self.database.session_scope() as session:
                group = session.get(Group, group_id)
                if group is None:
                    raise ValidationError("Select a valid group before saving the camera.")
                self._ensure_unique_camera(session, group_id, name, endpoint.host)
                camera = Camera(
                    group_id=group_id,
                    name=name.strip(),
                    source_type="RTSP",
                    connection_uri=endpoint.database_uri(),
                    enabled=True,
                    target_fps=target_fps,
                    reconnect_seconds=reconnect_seconds,
                    width=width,
                    height=height,
                    display_order=len(group.cameras),
                )
                session.add(camera)
                session.flush()
                camera_id = camera.id
                self.connection_service.save_password(camera.id, password)
        except Exception:
            if camera_id is not None:
                self.connection_service.delete_password(camera_id)
            raise
        return self.get_camera(camera_id)

    def update_camera(
        self,
        camera_id: int,
        *,
        group_id: int,
        name: str,
        host: str,
        port: int,
        username: str,
        password: str | None,
        stream_path: str,
        target_fps: float,
        reconnect_seconds: int,
        width: int | None,
        height: int | None,
        enabled: bool,
    ) -> CameraRecord:
        endpoint = self._validate_input(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            stream_path=stream_path,
            target_fps=target_fps,
            reconnect_seconds=reconnect_seconds,
            width=width,
            height=height,
            password_required=False,
        )
        with self.database.session_scope() as session:
            camera = session.get(Camera, camera_id)
            if camera is None:
                raise ValidationError(f"Camera id {camera_id} was not found.")
            if session.get(Group, group_id) is None:
                raise ValidationError("Select a valid group before saving the camera.")
            self._ensure_unique_camera(
                session, group_id, name, endpoint.host, excluded_camera_id=camera_id
            )
            camera.group_id = group_id
            camera.name = name.strip()
            camera.connection_uri = endpoint.database_uri()
            camera.target_fps = target_fps
            camera.reconnect_seconds = reconnect_seconds
            camera.width = width
            camera.height = height
            camera.enabled = enabled
            if password:
                self.connection_service.save_password(camera_id, password)
        return self.get_camera(camera_id)

    def delete_camera(self, camera_id: int) -> None:
        with self.database.session_scope() as session:
            camera = session.get(Camera, camera_id)
            if camera is None:
                raise ValidationError(f"Camera id {camera_id} was not found.")
            session.delete(camera)
        self.connection_service.delete_password(camera_id)

    def capture_snapshot(self, camera_id: int) -> CameraSnapshotResult:
        """Capture one static frame without starting the processing group."""
        record = self.get_camera(camera_id)
        endpoint = RtspEndpoint(
            host=record.host,
            port=record.port,
            username=record.username,
            stream_path=record.stream_path,
        )
        return self.connection_service.capture_snapshot(
            camera_id=record.id,
            camera_name=record.name,
            endpoint=endpoint,
        )

    def test_connection(self, camera_id: int) -> CameraConnectionTestResult:
        record = self.get_camera(camera_id)
        endpoint = RtspEndpoint(
            host=record.host,
            port=record.port,
            username=record.username,
            stream_path=record.stream_path,
        )
        return self.connection_service.test_connection(
            camera_id=record.id,
            camera_name=record.name,
            endpoint=endpoint,
        )

    def _to_record(self, camera: Camera, group_code: str) -> CameraRecord:
        endpoint = RtspEndpoint.from_database_uri(camera.connection_uri)
        return CameraRecord(
            id=camera.id,
            group_id=camera.group_id,
            group_code=group_code,
            name=camera.name,
            host=endpoint.host,
            port=endpoint.port,
            username=endpoint.username,
            stream_path=endpoint.stream_path,
            target_fps=camera.target_fps,
            reconnect_seconds=camera.reconnect_seconds,
            width=camera.width,
            height=camera.height,
            enabled=camera.enabled,
            credential_configured=self.connection_service.password_is_configured(camera.id),
            masked_url=endpoint.masked_url(),
        )

    @staticmethod
    def _validate_input(
        *,
        name: str,
        host: str,
        port: int,
        username: str,
        password: str | None,
        stream_path: str,
        target_fps: float,
        reconnect_seconds: int,
        width: int | None,
        height: int | None,
        password_required: bool = True,
    ) -> RtspEndpoint:
        if not name.strip():
            raise ValidationError("Camera name is required.")
        if password_required and not password:
            raise ValidationError("Camera password is required for a new camera.")
        if target_fps <= 0 or target_fps > 120:
            raise ValidationError("Target FPS must be between 0 and 120.")
        if reconnect_seconds < 0 or reconnect_seconds > 300:
            raise ValidationError("Reconnect interval must be between 0 and 300 seconds.")
        if width is not None and width <= 0:
            raise ValidationError("Camera width must be greater than zero.")
        if height is not None and height <= 0:
            raise ValidationError("Camera height must be greater than zero.")
        try:
            return RtspEndpoint(host, port, username, stream_path)
        except RtspEndpointError as exc:
            raise ValidationError(str(exc)) from exc

    @staticmethod
    def _ensure_unique_camera(
        session: Any,
        group_id: int,
        name: str,
        host: str,
        excluded_camera_id: int | None = None,
    ) -> None:
        cameras = session.scalars(select(Camera).where(Camera.group_id == group_id)).all()
        for camera in cameras:
            if excluded_camera_id is not None and camera.id == excluded_camera_id:
                continue
            if camera.name.casefold() == name.strip().casefold():
                raise ValidationError("Another camera in this group already uses that name.")
            try:
                existing_host = urlsplit(camera.connection_uri).hostname
            except Exception:
                existing_host = None
            if existing_host == host:
                raise ValidationError("This IP or hostname is already used by another camera in the group.")
