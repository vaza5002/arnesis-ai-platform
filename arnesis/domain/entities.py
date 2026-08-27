"""Persistent domain entities for real-time Arnesis configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from arnesis.core.database import Base


class JsonText(TypeDecorator):
    """Store JSON as text for SQLite and Oracle 11g portability."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: object) -> str:
        import json

        return json.dumps(value if value is not None else {}, separators=(",", ":"))

    def process_result_value(self, value: str | None, dialect: object) -> Any:
        import json

        return json.loads(value) if value else None


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class GroupStatus(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class SourceType(str, Enum):
    RTSP = "RTSP"
    USB = "USB"
    FILE = "FILE"


class RoiType(str, Enum):
    POLYGON = "POLYGON"
    RECTANGLE = "RECTANGLE"


class ModelType(str, Enum):
    DETECTION = "DETECTION"
    DETECTOR = "DETECTION"
    CLASSIFICATION = "CLASSIFICATION"
    CLASSIFIER = "CLASSIFICATION"
    POSE = "POSE"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class Group(Base, TimestampMixin):
    __tablename__ = "arn_group"
    __table_args__ = (
        CheckConstraint("max_concurrent_streams > 0", name="ck_group_streams_positive"),
        CheckConstraint("max_gpu_memory_mb > 0", name="ck_group_memory_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=GroupStatus.STOPPED.value, nullable=False
    )
    preferred_gpu_index: Mapped[int | None] = mapped_column(Integer)
    max_gpu_memory_mb: Mapped[int] = mapped_column(Integer, default=8192, nullable=False)
    max_concurrent_streams: Mapped[int] = mapped_column(Integer, default=4, nullable=False)
    assigned_worker: Mapped[str | None] = mapped_column(String(150))

    cameras: Mapped[list[Camera]] = relationship(
        back_populates="group", cascade="all, delete-orphan", order_by="Camera.display_order"
    )


class Camera(Base, TimestampMixin):
    __tablename__ = "arn_camera"
    __table_args__ = (
        UniqueConstraint("group_id", "name", name="uq_camera_group_name"),
        CheckConstraint("target_fps > 0", name="ck_camera_fps_positive"),
        CheckConstraint("reconnect_seconds >= 0", name="ck_camera_reconnect_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("arn_group.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    connection_uri: Mapped[str] = mapped_column(String(1000), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    target_fps: Mapped[float] = mapped_column(Float, default=15.0, nullable=False)
    reconnect_seconds: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    group: Mapped[Group] = relationship(back_populates="cameras")
    rois: Mapped[list[Roi]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", order_by="Roi.display_order"
    )


class ModelDefinition(Base, TimestampMixin):
    __tablename__ = "arn_model_definition"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), default="ultralytics", nullable=False)
    model_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_sha256: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    input_size: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(String(1000))


class ProcessingProfile(Base, TimestampMixin):
    __tablename__ = "arn_processing_profile"
    __table_args__ = (
        CheckConstraint(
            "confidence_threshold >= 0 AND confidence_threshold <= 1",
            name="ck_profile_confidence_range",
        ),
        CheckConstraint(
            "iou_threshold >= 0 AND iou_threshold <= 1",
            name="ck_profile_iou_range",
        ),
        CheckConstraint("frame_skip >= 0", name="ck_profile_frame_skip_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    detector_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("arn_model_definition.id")
    )
    classifier_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("arn_model_definition.id")
    )
    pose_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("arn_model_definition.id")
    )
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.50, nullable=False)
    iou_threshold: Mapped[float] = mapped_column(Float, default=0.45, nullable=False)
    frame_skip: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    debounce_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    minimum_event_duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    target_classes: Mapped[list[str]] = mapped_column(JsonText(), default=list, nullable=False)
    custom_parameters: Mapped[dict[str, Any]] = mapped_column(JsonText(), default=dict, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    detector_model: Mapped[ModelDefinition | None] = relationship(
        foreign_keys=[detector_model_id]
    )
    classifier_model: Mapped[ModelDefinition | None] = relationship(
        foreign_keys=[classifier_model_id]
    )
    pose_model: Mapped[ModelDefinition | None] = relationship(
        foreign_keys=[pose_model_id]
    )


class Roi(Base, TimestampMixin):
    __tablename__ = "arn_roi"
    __table_args__ = (
        UniqueConstraint("camera_id", "name", name="uq_roi_camera_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("arn_camera.id", ondelete="CASCADE"), nullable=False, index=True
    )
    processing_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("arn_processing_profile.id")
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    roi_type: Mapped[str] = mapped_column(
        String(20), default=RoiType.POLYGON.value, nullable=False
    )
    normalized_points: Mapped[list[dict[str, float]]] = mapped_column(
        JsonText(), default=list, nullable=False
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    color_hex: Mapped[str] = mapped_column(String(7), default="#29E6FF", nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    camera: Mapped[Camera] = relationship(back_populates="rois")
    processing_profile: Mapped[ProcessingProfile | None] = relationship()


class GpuCapacity(Base, TimestampMixin):
    __tablename__ = "arn_gpu_capacity"
    __table_args__ = (
        CheckConstraint(
            "maximum_memory_percent > 0 AND maximum_memory_percent <= 100",
            name="ck_gpu_memory_percent_range",
        ),
        CheckConstraint("maximum_groups > 0", name="ck_gpu_groups_positive"),
        CheckConstraint("maximum_streams > 0", name="ck_gpu_streams_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_uuid: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    device_index: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    device_name: Mapped[str] = mapped_column(String(150), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    maximum_memory_percent: Mapped[float] = mapped_column(Float, default=90.0, nullable=False)
    reserved_memory_mb: Mapped[int] = mapped_column(Integer, default=2048, nullable=False)
    maximum_groups: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    maximum_streams: Mapped[int] = mapped_column(Integer, default=32, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)


class SystemSetting(Base, TimestampMixin):
    """Persistent application setting compatible with SQLite and Oracle 11g."""

    __tablename__ = "arn_system_setting"

    setting_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    setting_value: Mapped[str] = mapped_column(String(1000), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))


class StationMetricSample(Base, TimestampMixin):
    """Anonymous per-ROI occupancy and performance observation."""

    __tablename__ = "arn_station_metric"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "camera_id", "roi_id", "observed_at",
            name="uq_station_metric_observation",
        ),
        CheckConstraint("head_count >= 0", name="ck_metric_head_nonnegative"),
        CheckConstraint("person_count >= 0", name="ck_metric_person_nonnegative"),
        CheckConstraint("va_count >= 0", name="ck_metric_va_nonnegative"),
        CheckConstraint("nva_count >= 0", name="ck_metric_nva_nonnegative"),
        CheckConstraint("neutral_count >= 0", name="ck_metric_neutral_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    group_code: Mapped[str] = mapped_column(String(50), nullable=False)
    camera_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    camera_name: Mapped[str] = mapped_column(String(150), nullable=False)
    roi_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    roi_name: Mapped[str] = mapped_column(String(150), nullable=False)
    head_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    person_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    occupied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    va_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    nva_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    neutral_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cuda_device: Mapped[str | None] = mapped_column(String(100))
    source_sequence: Mapped[int | None] = mapped_column(Integer)
