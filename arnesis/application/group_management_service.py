"""Group CRUD and runtime-control use cases for the Arnesis Controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import func, select

from arnesis.application.processing_service import ProcessingService
from arnesis.core.database import DatabaseManager
from arnesis.domain.entities import Camera, GpuCapacity, Group, GroupStatus, Roi
from arnesis.infrastructure.repositories import ValidationError


@dataclass(frozen=True, slots=True)
class GpuOption:
    device_index: int
    device_name: str
    enabled: bool
    maximum_memory_percent: float
    reserved_memory_mb: int
    maximum_groups: int
    maximum_streams: int

    @property
    def label(self) -> str:
        status = "Enabled" if self.enabled else "Disabled"
        return f"CUDA:{self.device_index} - {self.device_name} ({status})"


@dataclass(frozen=True, slots=True)
class GroupRecord:
    id: int
    code: str
    name: str
    description: str | None
    enabled: bool
    status: str
    preferred_gpu_index: int | None
    gpu_label: str
    max_gpu_memory_mb: int
    max_concurrent_streams: int
    assigned_worker: str | None
    camera_count: int
    roi_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GroupManagementService:
    """Validates group CRUD and delegates lifecycle commands to the runtime."""

    ACTIVE_STATUSES = {
        GroupStatus.STARTING.value,
        GroupStatus.RUNNING.value,
        GroupStatus.PAUSING.value,
        GroupStatus.PAUSED.value,
        GroupStatus.STOPPING.value,
    }

    def __init__(
        self,
        database: DatabaseManager,
        processing: ProcessingService,
    ) -> None:
        self.database = database
        self.processing = processing

    def list_gpu_options(self, include_disabled: bool = True) -> list[GpuOption]:
        with self.database.session_scope() as session:
            statement = select(GpuCapacity).order_by(
                GpuCapacity.priority, GpuCapacity.device_index
            )
            if not include_disabled:
                statement = statement.where(GpuCapacity.enabled.is_(True))
            devices = session.scalars(statement).all()
            return [
                GpuOption(
                    device_index=device.device_index,
                    device_name=device.device_name,
                    enabled=device.enabled,
                    maximum_memory_percent=device.maximum_memory_percent,
                    reserved_memory_mb=device.reserved_memory_mb,
                    maximum_groups=device.maximum_groups,
                    maximum_streams=device.maximum_streams,
                )
                for device in devices
            ]

    def list_groups(self) -> list[GroupRecord]:
        with self.database.session_scope() as session:
            camera_counts = dict(
                session.execute(
                    select(Camera.group_id, func.count(Camera.id)).group_by(Camera.group_id)
                ).all()
            )
            roi_counts = dict(
                session.execute(
                    select(Camera.group_id, func.count(Roi.id))
                    .join(Roi, Roi.camera_id == Camera.id)
                    .group_by(Camera.group_id)
                ).all()
            )
            gpu_names = {
                gpu.device_index: gpu.device_name
                for gpu in session.scalars(select(GpuCapacity)).all()
            }
            groups = session.scalars(select(Group).order_by(Group.code)).all()
            return [
                self._to_record(
                    group,
                    camera_counts.get(group.id, 0),
                    roi_counts.get(group.id, 0),
                    gpu_names,
                )
                for group in groups
            ]

    def get_group(self, group_id: int) -> GroupRecord:
        records = {record.id: record for record in self.list_groups()}
        if group_id not in records:
            raise ValidationError(f"Group id {group_id} was not found.")
        return records[group_id]

    def create_group(
        self,
        *,
        code: str,
        name: str,
        description: str | None,
        enabled: bool,
        preferred_gpu_index: int | None,
        max_gpu_memory_mb: int,
        max_concurrent_streams: int,
    ) -> GroupRecord:
        values = self._validate_values(
            code=code,
            name=name,
            description=description,
            enabled=enabled,
            preferred_gpu_index=preferred_gpu_index,
            max_gpu_memory_mb=max_gpu_memory_mb,
            max_concurrent_streams=max_concurrent_streams,
        )
        with self.database.session_scope() as session:
            self._ensure_unique_code(session, values["code"])
            self._validate_gpu(session, values["preferred_gpu_index"])
            group = Group(**values, status=GroupStatus.STOPPED.value)
            session.add(group)
            session.flush()
            group_id = group.id
        return self.get_group(group_id)

    def update_group(
        self,
        group_id: int,
        *,
        code: str,
        name: str,
        description: str | None,
        enabled: bool,
        preferred_gpu_index: int | None,
        max_gpu_memory_mb: int,
        max_concurrent_streams: int,
    ) -> GroupRecord:
        values = self._validate_values(
            code=code,
            name=name,
            description=description,
            enabled=enabled,
            preferred_gpu_index=preferred_gpu_index,
            max_gpu_memory_mb=max_gpu_memory_mb,
            max_concurrent_streams=max_concurrent_streams,
        )
        with self.database.session_scope() as session:
            group = session.get(Group, group_id)
            if group is None:
                raise ValidationError(f"Group id {group_id} was not found.")
            if group.status in self.ACTIVE_STATUSES:
                protected_changes = (
                    group.code != values["code"]
                    or group.preferred_gpu_index != values["preferred_gpu_index"]
                    or group.max_gpu_memory_mb != values["max_gpu_memory_mb"]
                    or group.max_concurrent_streams != values["max_concurrent_streams"]
                    or not values["enabled"]
                )
                if protected_changes:
                    raise ValidationError(
                        "Stop the group before changing its code, CUDA assignment, "
                        "capacity, stream limit, or enabled status."
                    )
            self._ensure_unique_code(session, values["code"], group_id)
            self._validate_gpu(session, values["preferred_gpu_index"])
            for field_name, value in values.items():
                setattr(group, field_name, value)
        return self.get_group(group_id)

    def duplicate_group(self, group_id: int, new_code: str) -> GroupRecord:
        source = self.get_group(group_id)
        return self.create_group(
            code=new_code,
            name=f"{source.name} Copy",
            description=source.description,
            enabled=False,
            preferred_gpu_index=source.preferred_gpu_index,
            max_gpu_memory_mb=source.max_gpu_memory_mb,
            max_concurrent_streams=source.max_concurrent_streams,
        )

    def delete_group(self, group_id: int) -> None:
        with self.database.session_scope() as session:
            group = session.get(Group, group_id)
            if group is None:
                raise ValidationError(f"Group id {group_id} was not found.")
            if group.status not in {GroupStatus.STOPPED.value, GroupStatus.ERROR.value}:
                raise ValidationError("Only stopped or failed groups can be deleted.")
            session.delete(group)

    def start_group(self, group_id: int) -> dict[str, object]:
        return self.processing.start_group(group_id)

    def pause_group(self, group_id: int) -> dict[str, object]:
        return self.processing.pause_group(group_id)

    def resume_group(self, group_id: int) -> dict[str, object]:
        return self.processing.resume_group(group_id)

    def stop_group(self, group_id: int) -> dict[str, object]:
        return self.processing.stop_group(group_id)

    def runtime_status(self) -> list[dict[str, object]]:
        return self.processing.get_runtime_status()

    @staticmethod
    def _validate_values(
        *,
        code: str,
        name: str,
        description: str | None,
        enabled: bool,
        preferred_gpu_index: int | None,
        max_gpu_memory_mb: int,
        max_concurrent_streams: int,
    ) -> dict[str, Any]:
        normalized_code = code.strip().upper()
        if not normalized_code:
            raise ValidationError("Group code is required.")
        if len(normalized_code) > 50:
            raise ValidationError("Group code cannot exceed 50 characters.")
        if not name.strip():
            raise ValidationError("Group name is required.")
        if len(name.strip()) > 150:
            raise ValidationError("Group name cannot exceed 150 characters.")
        if max_gpu_memory_mb < 256:
            raise ValidationError("Maximum GPU memory must be at least 256 MiB.")
        if max_concurrent_streams < 1 or max_concurrent_streams > 256:
            raise ValidationError("Maximum concurrent streams must be between 1 and 256.")
        return {
            "code": normalized_code,
            "name": name.strip(),
            "description": description.strip() if description and description.strip() else None,
            "enabled": bool(enabled),
            "preferred_gpu_index": preferred_gpu_index,
            "max_gpu_memory_mb": max_gpu_memory_mb,
            "max_concurrent_streams": max_concurrent_streams,
        }

    @staticmethod
    def _ensure_unique_code(
        session: Any,
        code: str,
        excluded_group_id: int | None = None,
    ) -> None:
        statement = select(Group).where(Group.code == code)
        existing = session.scalar(statement)
        if existing is not None and existing.id != excluded_group_id:
            raise ValidationError(f"A group with code '{code}' already exists.")

    @staticmethod
    def _validate_gpu(session: Any, device_index: int | None) -> None:
        if device_index is None:
            return
        gpu = session.scalar(
            select(GpuCapacity).where(GpuCapacity.device_index == device_index)
        )
        if gpu is None:
            raise ValidationError(f"CUDA:{device_index} is not registered.")
        if not gpu.enabled:
            raise ValidationError(f"CUDA:{device_index} is currently disabled.")

    @staticmethod
    def _to_record(
        group: Group,
        camera_count: int,
        roi_count: int,
        gpu_names: dict[int, str],
    ) -> GroupRecord:
        if group.preferred_gpu_index is None:
            gpu_label = "Automatic"
        else:
            gpu_name = gpu_names.get(group.preferred_gpu_index, "Unavailable")
            gpu_label = f"CUDA:{group.preferred_gpu_index} - {gpu_name}"
        return GroupRecord(
            id=group.id,
            code=group.code,
            name=group.name,
            description=group.description,
            enabled=group.enabled,
            status=group.status,
            preferred_gpu_index=group.preferred_gpu_index,
            gpu_label=gpu_label,
            max_gpu_memory_mb=group.max_gpu_memory_mb,
            max_concurrent_streams=group.max_concurrent_streams,
            assigned_worker=group.assigned_worker,
            camera_count=camera_count,
            roi_count=roi_count,
        )
