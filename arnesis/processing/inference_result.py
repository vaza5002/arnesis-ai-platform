"""Immutable contracts for ROI-constrained multi-stage CUDA inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class DetectionResult:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    roi_id: int
    station: str


@dataclass(frozen=True, slots=True)
class HeadDetectionResult:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class PerformanceClassificationResult:
    track_id: int
    roi_id: int
    station: str
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class RuntimeRoiResult:
    roi_id: int
    station: str
    color_hex: str
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class StationResult:
    roi_id: int
    station: str
    people_count: int
    head_count: int
    va_count: int
    nva_count: int
    neutral_count: int
    track_ids: tuple[int, ...]
    processing_profile_id: int | None


@dataclass(frozen=True, slots=True)
class CameraInferenceResult:
    group_id: int
    camera_id: int
    sequence: int
    timestamp_utc: str
    cuda_device: str
    person_model_path: str
    head_model_path: str
    capture_to_result_ms: float
    person_inference_ms: float
    head_inference_ms: float
    classification_inference_ms: float
    processing_fps: float
    detections: tuple[DetectionResult, ...]
    privacy_heads: tuple[HeadDetectionResult, ...]
    classifications: tuple[PerformanceClassificationResult, ...]
    rois: tuple[RuntimeRoiResult, ...]
    stations: tuple[StationResult, ...]
    privacy_blur_enabled: bool = True
    privacy_blur_kernel: int = 51
    privacy_box_expansion: float = 0.15
    privacy_minimum_region: int = 12
    error: str | None = None

    @property
    def inference_ms(self) -> float:
        """Compatibility alias used by older RT views."""
        return self.person_inference_ms

    @property
    def head_detections(self) -> tuple[HeadDetectionResult, ...]:
        """Compatibility alias used by the privacy renderer."""
        return self.privacy_heads

    @property
    def model_path(self) -> str:
        """Compatibility alias used by older diagnostics."""
        return self.person_model_path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
