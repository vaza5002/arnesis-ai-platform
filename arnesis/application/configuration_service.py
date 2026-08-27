"""Application service for Arnesis configuration CRUD.

The UI and future controller use this service instead of working directly with
SQLAlchemy sessions. Every public operation owns a short database transaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from arnesis.core.database import DatabaseManager
from arnesis.domain.entities import GroupStatus
from arnesis.infrastructure.repositories import (
    CameraRepository,
    GroupRepository,
    ModelRepository,
    ProcessingProfileRepository,
    RoiRepository,
)


@dataclass(frozen=True, slots=True)
class GroupSummary:
    id: int
    code: str
    name: str
    enabled: bool
    status: str
    preferred_gpu_index: int | None
    max_gpu_memory_mb: int
    max_concurrent_streams: int
    camera_count: int
    roi_count: int


class ConfigurationService:
    """Coordinates configuration use cases for groups, cameras, ROIs and models."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def create_group(
        self,
        *,
        code: str,
        name: str,
        description: str | None = None,
        preferred_gpu_index: int | None = None,
        max_gpu_memory_mb: int = 8192,
        max_concurrent_streams: int = 4,
    ) -> dict[str, Any]:
        with self.database.session_scope() as session:
            group = GroupRepository(session).create(
                code=code,
                name=name,
                description=description,
                preferred_gpu_index=preferred_gpu_index,
                max_gpu_memory_mb=max_gpu_memory_mb,
                max_concurrent_streams=max_concurrent_streams,
            )
            return self._group_to_dict(group)

    def update_group(self, group_id: int, **changes: Any) -> dict[str, Any]:
        with self.database.session_scope() as session:
            group = GroupRepository(session).update_fields(group_id, **changes)
            return self._group_to_dict(group)

    def delete_group(self, group_id: int) -> None:
        with self.database.session_scope() as session:
            GroupRepository(session).delete(group_id)

    def set_group_status(self, group_id: int, status: GroupStatus) -> dict[str, Any]:
        with self.database.session_scope() as session:
            group = GroupRepository(session).set_status(group_id, status)
            return self._group_to_dict(group)

    def list_groups(self) -> list[dict[str, Any]]:
        with self.database.session_scope() as session:
            groups = GroupRepository(session).list_with_configuration()
            summaries = []
            for group in groups:
                summaries.append(
                    GroupSummary(
                        id=group.id,
                        code=group.code,
                        name=group.name,
                        enabled=group.enabled,
                        status=group.status,
                        preferred_gpu_index=group.preferred_gpu_index,
                        max_gpu_memory_mb=group.max_gpu_memory_mb,
                        max_concurrent_streams=group.max_concurrent_streams,
                        camera_count=len(group.cameras),
                        roi_count=sum(len(camera.rois) for camera in group.cameras),
                    )
                )
            return [asdict(summary) for summary in summaries]

    def create_camera(
        self,
        *,
        group_id: int,
        name: str,
        source_type: str,
        connection_uri: str,
        target_fps: float = 15.0,
        display_order: int = 0,
    ) -> dict[str, Any]:
        with self.database.session_scope() as session:
            camera = CameraRepository(session).create(
                group_id=group_id,
                name=name,
                source_type=source_type,
                connection_uri=connection_uri,
                target_fps=target_fps,
                display_order=display_order,
            )
            return self._entity_to_dict(camera)

    def update_camera(self, camera_id: int, **changes: Any) -> dict[str, Any]:
        with self.database.session_scope() as session:
            camera = CameraRepository(session).update_fields(camera_id, **changes)
            return self._entity_to_dict(camera)

    def delete_camera(self, camera_id: int) -> None:
        with self.database.session_scope() as session:
            CameraRepository(session).delete(camera_id)

    def create_roi(
        self,
        *,
        camera_id: int,
        name: str,
        normalized_points: list[dict[str, float]],
        processing_profile_id: int | None = None,
        display_order: int = 0,
    ) -> dict[str, Any]:
        with self.database.session_scope() as session:
            roi = RoiRepository(session).create(
                camera_id=camera_id,
                name=name,
                normalized_points=normalized_points,
                processing_profile_id=processing_profile_id,
                display_order=display_order,
            )
            return self._entity_to_dict(roi)

    def update_roi(self, roi_id: int, **changes: Any) -> dict[str, Any]:
        if "normalized_points" in changes:
            RoiRepository._validate_points(changes["normalized_points"])
        with self.database.session_scope() as session:
            roi = RoiRepository(session).update_fields(roi_id, **changes)
            return self._entity_to_dict(roi)

    def delete_roi(self, roi_id: int) -> None:
        with self.database.session_scope() as session:
            RoiRepository(session).delete(roi_id)

    def register_model(self, **model_data: Any) -> dict[str, Any]:
        with self.database.session_scope() as session:
            model = ModelRepository(session).create(**model_data)
            return self._entity_to_dict(model)

    def create_processing_profile(self, **profile_data: Any) -> dict[str, Any]:
        with self.database.session_scope() as session:
            profile = ProcessingProfileRepository(session).create(**profile_data)
            return self._entity_to_dict(profile)

    @staticmethod
    def _group_to_dict(group: Any) -> dict[str, Any]:
        return {
            "id": group.id,
            "code": group.code,
            "name": group.name,
            "description": group.description,
            "enabled": group.enabled,
            "status": group.status,
            "preferred_gpu_index": group.preferred_gpu_index,
            "max_gpu_memory_mb": group.max_gpu_memory_mb,
            "max_concurrent_streams": group.max_concurrent_streams,
            "assigned_worker": group.assigned_worker,
        }

    @staticmethod
    def _entity_to_dict(entity: Any) -> dict[str, Any]:
        return {
            column.name: getattr(entity, column.name)
            for column in entity.__table__.columns
        }
