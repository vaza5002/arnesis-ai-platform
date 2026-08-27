"""Repository layer for Arnesis real-time configuration CRUD."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from arnesis.domain.entities import (
    Camera,
    Group,
    GroupStatus,
    ModelDefinition,
    ProcessingProfile,
    Roi,
)

EntityT = TypeVar("EntityT")


class RepositoryError(RuntimeError):
    """Raised when persistence cannot complete a requested operation."""


class EntityNotFoundError(RepositoryError):
    """Raised when a requested persistent entity does not exist."""


class ValidationError(RepositoryError):
    """Raised when input violates an application rule."""


class BaseRepository(Generic[EntityT]):
    """Small reusable repository with predictable transaction behavior."""

    model: type[EntityT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity_id: int) -> EntityT | None:
        return self.session.get(self.model, entity_id)

    def require(self, entity_id: int) -> EntityT:
        entity = self.get(entity_id)
        if entity is None:
            raise EntityNotFoundError(
                f"{self.model.__name__} with id {entity_id} was not found."
            )
        return entity

    def list_all(self) -> Sequence[EntityT]:
        return self.session.scalars(select(self.model).order_by(self.model.id)).all()

    def add(self, entity: EntityT) -> EntityT:
        self.session.add(entity)
        self._flush()
        return entity

    def update_fields(self, entity_id: int, **changes: Any) -> EntityT:
        entity = self.require(entity_id)
        protected_fields = {"id", "created_at", "updated_at"}

        for field_name, value in changes.items():
            if field_name in protected_fields or not hasattr(entity, field_name):
                raise ValidationError(f"Field '{field_name}' cannot be updated.")
            setattr(entity, field_name, value)

        self._flush()
        return entity

    def delete(self, entity_id: int) -> None:
        entity = self.require(entity_id)
        self.session.delete(entity)
        self._flush()

    def _flush(self) -> None:
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise RepositoryError(
                "The operation violates a database uniqueness or relationship rule."
            ) from exc


class GroupRepository(BaseRepository[Group]):
    model = Group

    def list_with_configuration(self) -> Sequence[Group]:
        statement = (
            select(Group)
            .options(selectinload(Group.cameras).selectinload(Camera.rois))
            .order_by(Group.code)
        )
        return self.session.scalars(statement).unique().all()

    def get_by_code(self, code: str) -> Group | None:
        normalized_code = code.strip().upper()
        return self.session.scalar(select(Group).where(Group.code == normalized_code))

    def create(
        self,
        *,
        code: str,
        name: str,
        description: str | None = None,
        preferred_gpu_index: int | None = None,
        max_gpu_memory_mb: int = 8192,
        max_concurrent_streams: int = 4,
    ) -> Group:
        normalized_code = code.strip().upper()
        if not normalized_code:
            raise ValidationError("Group code is required.")
        if not name.strip():
            raise ValidationError("Group name is required.")
        if preferred_gpu_index is not None and preferred_gpu_index < 0:
            raise ValidationError("Preferred GPU index cannot be negative.")

        return self.add(
            Group(
                code=normalized_code,
                name=name.strip(),
                description=description.strip() if description else None,
                preferred_gpu_index=preferred_gpu_index,
                max_gpu_memory_mb=max_gpu_memory_mb,
                max_concurrent_streams=max_concurrent_streams,
            )
        )

    def set_status(self, group_id: int, status: GroupStatus) -> Group:
        group = self.require(group_id)
        group.status = status.value
        self._flush()
        return group

    def delete(self, entity_id: int) -> None:
        group = self.require(entity_id)
        if group.status not in {GroupStatus.STOPPED.value, GroupStatus.ERROR.value}:
            raise ValidationError("Only stopped or failed groups can be deleted.")
        super().delete(entity_id)


class CameraRepository(BaseRepository[Camera]):
    model = Camera

    def list_by_group(self, group_id: int) -> Sequence[Camera]:
        return self.session.scalars(
            select(Camera)
            .where(Camera.group_id == group_id)
            .order_by(Camera.display_order, Camera.id)
        ).all()

    def create(
        self,
        *,
        group_id: int,
        name: str,
        source_type: str,
        connection_uri: str,
        target_fps: float = 15.0,
        display_order: int = 0,
    ) -> Camera:
        if not name.strip() or not connection_uri.strip():
            raise ValidationError("Camera name and connection URI are required.")
        if target_fps <= 0:
            raise ValidationError("Target FPS must be greater than zero.")

        return self.add(
            Camera(
                group_id=group_id,
                name=name.strip(),
                source_type=source_type.strip().upper(),
                connection_uri=connection_uri.strip(),
                target_fps=target_fps,
                display_order=display_order,
            )
        )


class RoiRepository(BaseRepository[Roi]):
    model = Roi

    def list_by_camera(self, camera_id: int) -> Sequence[Roi]:
        return self.session.scalars(
            select(Roi)
            .where(Roi.camera_id == camera_id)
            .order_by(Roi.display_order, Roi.id)
        ).all()

    def create(
        self,
        *,
        camera_id: int,
        name: str,
        normalized_points: list[dict[str, float]],
        processing_profile_id: int | None = None,
        display_order: int = 0,
    ) -> Roi:
        self._validate_points(normalized_points)
        if not name.strip():
            raise ValidationError("ROI name is required.")

        return self.add(
            Roi(
                camera_id=camera_id,
                processing_profile_id=processing_profile_id,
                name=name.strip(),
                normalized_points=normalized_points,
                display_order=display_order,
            )
        )

    @staticmethod
    def _validate_points(points: list[dict[str, float]]) -> None:
        if len(points) < 3:
            raise ValidationError("A polygon ROI requires at least three points.")

        for point in points:
            if set(point) != {"x", "y"}:
                raise ValidationError("Each ROI point must contain only 'x' and 'y'.")
            if not 0.0 <= float(point["x"]) <= 1.0:
                raise ValidationError("ROI x coordinates must be between 0 and 1.")
            if not 0.0 <= float(point["y"]) <= 1.0:
                raise ValidationError("ROI y coordinates must be between 0 and 1.")


class ModelRepository(BaseRepository[ModelDefinition]):
    model = ModelDefinition

    def create(
        self,
        *,
        name: str,
        version: str,
        model_type: str,
        model_path: str,
        file_sha256: str | None = None,
        framework: str = "ultralytics",
    ) -> ModelDefinition:
        normalized_path = str(Path(model_path).expanduser().resolve())
        if not name.strip() or not version.strip():
            raise ValidationError("Model name and version are required.")
        if file_sha256 and len(file_sha256) != 64:
            raise ValidationError("Model SHA-256 must contain 64 hexadecimal characters.")

        return self.add(
            ModelDefinition(
                name=name.strip(),
                version=version.strip(),
                model_type=model_type.strip().upper(),
                model_path=normalized_path,
                file_sha256=file_sha256.lower() if file_sha256 else None,
                framework=framework.strip(),
            )
        )


class ProcessingProfileRepository(BaseRepository[ProcessingProfile]):
    model = ProcessingProfile

    def create(
        self,
        *,
        name: str,
        detector_model_id: int | None = None,
        classifier_model_id: int | None = None,
        pose_model_id: int | None = None,
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        frame_skip: int = 0,
        target_classes: list[str] | None = None,
        custom_parameters: dict[str, Any] | None = None,
    ) -> ProcessingProfile:
        if not name.strip():
            raise ValidationError("Processing profile name is required.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValidationError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValidationError("IoU threshold must be between 0 and 1.")
        if frame_skip < 0:
            raise ValidationError("Frame skip cannot be negative.")

        return self.add(
            ProcessingProfile(
                name=name.strip(),
                detector_model_id=detector_model_id,
                classifier_model_id=classifier_model_id,
                pose_model_id=pose_model_id,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                frame_skip=frame_skip,
                target_classes=target_classes or [],
                custom_parameters=custom_parameters or {},
            )
        )
