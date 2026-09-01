"""ROI-constrained person, head, and batched performance inference worker."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from arnesis.processing.cuda_model_manager import CudaModelManager
from arnesis.processing.inference_result import (
    CameraInferenceResult,
    DetectionResult,
    HeadDetectionResult,
    PerformanceClassificationResult,
    RuntimeRoiResult,
    StationResult,
)
from arnesis.processing.performance_classifier import (
    ClassifierConfiguration,
    PerformanceClassifierPool,
)
from arnesis.processing.result_buffer import LatestResultBuffer
from arnesis.processing.roi_assignment import (
    BoundingBox,
    PreparedRoiGeometryCache,
    RoiAssignmentEngine,
)
from arnesis.processing.tracking_service import CandidateDetection, IouTrackingService


@dataclass(frozen=True, slots=True)
class DetectionWorkerConfiguration:
    group_id: int
    camera_id: int
    device_index: int
    person_model_path: str
    person_model_sha256: str | None
    person_confidence: float
    person_iou: float
    frame_skip: int
    person_target_classes: tuple[str, ...]
    person_input_size: int | None
    head_model_path: str
    head_model_sha256: str | None
    head_input_size: int | None
    head_confidence: float = 0.50
    head_iou: float = 0.45
    head_frame_skip: int = 4
    minimum_process_interval_seconds: float = 0.20
    classification_interval_seconds: float = 0.75
    classification_cache_ttl_seconds: float = 3.0
    use_half_precision: bool = True

    def __post_init__(self) -> None:
        if self.frame_skip < 0 or self.head_frame_skip < 0:
            raise ValueError("Frame skip values cannot be negative.")
        if self.minimum_process_interval_seconds < 0:
            raise ValueError("Minimum process interval cannot be negative.")
        if self.classification_interval_seconds <= 0:
            raise ValueError("Classification interval must be greater than zero.")


@dataclass(slots=True)
class _CachedClassification:
    profile_id: int
    class_id: int
    class_name: str
    confidence: float
    classified_at: float
    last_seen_at: float


@dataclass(slots=True)
class _PendingClassification:
    track_id: int
    roi_id: int
    station: str
    profile_id: int
    crop: Any


class DetectionWorker:
    def __init__(
        self,
        config: DetectionWorkerConfiguration,
        frame_source: Any,
        roi_records: list[dict[str, Any]],
        classifiers: dict[int, ClassifierConfiguration],
        manager: CudaModelManager,
        results: LatestResultBuffer | None = None,
    ) -> None:
        self.config = config
        self.frame_source = frame_source
        self.roi_records = list(roi_records)
        self.manager = manager
        self.results = results or LatestResultBuffer()
        self.classifier_pool = PerformanceClassifierPool(
            manager,
            config.device_index,
            classifiers,
        )
        self.tracker = IouTrackingService()
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._started = threading.Event()
        self._person_lease = None
        self._head_lease = None
        self._last_heads: tuple[HeadDetectionResult, ...] = ()
        self._classification_cache: dict[int, _CachedClassification] = {}
        self._roi_geometry_cache = PreparedRoiGeometryCache(self.roi_records)
        self._runtime_rois_cache: dict[
            tuple[int, int], tuple[RuntimeRoiResult, ...]
        ] = {}

    def start(self, timeout_seconds: float = 30.0) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self._stop.clear()
        self._pause.clear()
        self._started.clear()
        self.error = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"arnesis-multistage-{self.config.camera_id}",
            daemon=False,
        )
        self.thread.start()
        if not self._started.wait(timeout_seconds):
            self.stop(timeout_seconds)
            raise TimeoutError("Multi-stage inference worker startup timed out.")
        if self.error:
            raise RuntimeError(self.error)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self, timeout_seconds: float = 15.0) -> None:
        self._stop.set()
        self._pause.clear()
        if self.thread is not None:
            self.thread.join(timeout_seconds)
            if self.thread.is_alive():
                raise TimeoutError("Multi-stage inference worker did not stop safely.")
        self.thread = None

    def _run(self) -> None:
        received_counter = 0
        processed_counter = 0
        last_sequence: int | None = None
        last_processed_at = 0.0

        try:
            self._person_lease = self.manager.acquire(
                self.config.person_model_path,
                self.config.device_index,
                self.config.person_model_sha256,
            )
            self._head_lease = self.manager.acquire(
                self.config.head_model_path,
                self.config.device_index,
                self.config.head_model_sha256,
            )
            self.classifier_pool.start()
            self._started.set()

            while not self._stop.is_set():
                if self._pause.is_set():
                    self._stop.wait(0.05)
                    continue

                packet = self.frame_source.next_frame(timeout_seconds=0.50)
                if packet is None:
                    continue

                if last_sequence is not None and packet.sequence <= last_sequence:
                    self._stop.wait(0.002)
                    continue
                last_sequence = packet.sequence

                received_counter += 1
                if (received_counter - 1) % (self.config.frame_skip + 1) != 0:
                    continue

                now = time.monotonic()
                remaining = (
                    self.config.minimum_process_interval_seconds
                    - (now - last_processed_at)
                )
                if remaining > 0:
                    self._stop.wait(min(remaining, 0.05))
                    continue

                last_processed_at = now
                processed_counter += 1
                self.results.publish(self._process(packet, processed_counter))

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._started.set()
        finally:
            self.classifier_pool.close()
            if self._head_lease is not None:
                self._head_lease.release()
                self._head_lease = None
            if self._person_lease is not None:
                self._person_lease.release()
                self._person_lease = None
            self._classification_cache.clear()

    def _process(self, packet: Any, processed_counter: int) -> CameraInferenceResult:
        begun = time.perf_counter()
        now = time.monotonic()
        height, width = packet.frame.shape[:2]
        rois = self._roi_geometry_cache.get(width, height)

        person_started = time.perf_counter()
        raw_people = self._extract_boxes(
            self._predict(
                self._person_lease,
                packet.frame,
                self.config.person_confidence,
                self.config.person_iou,
                self.config.person_input_size,
            ),
            self.config.person_target_classes,
        )
        person_ms = (time.perf_counter() - person_started) * 1000.0

        accepted: list[CandidateDetection] = []
        for person in raw_people:
            assignment = RoiAssignmentEngine.assign_track(
                0,
                BoundingBox(person.x1, person.y1, person.x2, person.y2),
                rois,
            )
            if assignment is not None:
                accepted.append(person)
        tracked = self.tracker.update(accepted)

        if (processed_counter - 1) % (self.config.head_frame_skip + 1) == 0:
            head_started = time.perf_counter()
            raw_heads = self._extract_boxes(
                self._predict(
                    self._head_lease,
                    packet.frame,
                    self.config.head_confidence,
                    self.config.head_iou,
                    self.config.head_input_size,
                ),
                (),
            )
            head_ms = (time.perf_counter() - head_started) * 1000.0
            self._last_heads = tuple(
                HeadDetectionResult(
                    item.class_id,
                    item.class_name,
                    item.confidence,
                    item.x1,
                    item.y1,
                    item.x2,
                    item.y2,
                )
                for item in raw_heads
            )
        else:
            head_ms = 0.0

        head_counts = {roi.roi_id: 0 for roi in rois}
        for head in self._last_heads:
            assignment = RoiAssignmentEngine.assign_track(
                0,
                BoundingBox(head.x1, head.y1, head.x2, head.y2),
                rois,
            )
            if assignment is not None:
                head_counts[assignment.roi_id] += 1

        tracked_items = list(tracked)
        tracked_assignments = RoiAssignmentEngine.assign_tracks(
            (
                (
                    track_id,
                    BoundingBox(person.x1, person.y1, person.x2, person.y2),
                )
                for track_id, person in tracked_items
            ),
            rois,
        )
        detections: list[DetectionResult] = []
        detections_by_roi: dict[int, list[DetectionResult]] = {
            roi.roi_id: [] for roi in rois
        }
        classifications: list[PerformanceClassificationResult] = []
        classifications_by_roi: dict[
            int, list[PerformanceClassificationResult]
        ] = {roi.roi_id: [] for roi in rois}
        pending_by_profile: dict[int, list[_PendingClassification]] = {}
        active_track_ids: set[int] = set()
        for track_id, person in tracked_items:
            assignment = tracked_assignments.get(track_id)
            if assignment is None:
                continue
            active_track_ids.add(track_id)
            detection = DetectionResult(
                track_id,
                person.class_id,
                person.class_name,
                person.confidence,
                person.x1,
                person.y1,
                person.x2,
                person.y2,
                assignment.roi_id,
                assignment.station,
            )
            detections.append(detection)
            detections_by_roi[assignment.roi_id].append(detection)
            profile_id = assignment.processing_profile_id
            if profile_id is None:
                continue
            cached = self._classification_cache.get(track_id)
            should_classify = (
                cached is None
                or cached.profile_id != profile_id
                or now - cached.classified_at
                >= self.config.classification_interval_seconds
            )
            if not should_classify and cached is not None:
                cached.last_seen_at = now
                classification = PerformanceClassificationResult(
                    track_id,
                    assignment.roi_id,
                    assignment.station,
                    cached.class_id,
                    cached.class_name,
                    cached.confidence,
                )
                classifications.append(classification)
                classifications_by_roi[assignment.roi_id].append(classification)
                continue
            x1 = max(0, int(person.x1))
            y1 = max(0, int(person.y1))
            x2 = min(width, int(person.x2))
            y2 = min(height, int(person.y2))
            pending_by_profile.setdefault(int(profile_id), []).append(
                _PendingClassification(
                    track_id,
                    assignment.roi_id,
                    assignment.station,
                    int(profile_id),
                    packet.frame[y1:y2, x1:x2],
                )
            )
        classification_started = time.perf_counter()
        for profile_id, pending in pending_by_profile.items():
            values = self.classifier_pool.classify_batch(
                profile_id,
                [item.crop for item in pending],
            )
            for item, value in zip(pending, values):
                if value is None:
                    continue
                class_id, class_name, confidence = value
                self._classification_cache[item.track_id] = _CachedClassification(
                    profile_id=item.profile_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    classified_at=now,
                    last_seen_at=now,
                )
                classification = PerformanceClassificationResult(
                    item.track_id,
                    item.roi_id,
                    item.station,
                    class_id,
                    class_name,
                    confidence,
                )
                classifications.append(classification)
                classifications_by_roi[item.roi_id].append(classification)
        classification_ms = (
            time.perf_counter() - classification_started
        ) * 1000.0

        self._expire_classification_cache(now, active_track_ids)

        stations: list[StationResult] = []
        for roi in rois:
            roi_detections = detections_by_roi[roi.roi_id]
            track_ids = tuple(item.track_id for item in roi_detections)
            label_counts = {"VA": 0, "NVA": 0, "NEUTRAL": 0}
            for item in classifications_by_roi[roi.roi_id]:
                label = self._normalize_label(item.class_name)
                if label in label_counts:
                    label_counts[label] += 1
            stations.append(
                StationResult(
                    roi.roi_id,
                    roi.name,
                    len(track_ids),
                    head_counts[roi.roi_id],
                    label_counts["VA"],
                    label_counts["NVA"],
                    label_counts["NEUTRAL"],
                    track_ids,
                    roi.processing_profile_id,
                )
            )
        resolution_key = (width, height)
        runtime_rois = self._runtime_rois_cache.get(resolution_key)
        if runtime_rois is None:
            runtime_rois = tuple(
                RuntimeRoiResult(
                    roi.roi_id,
                    roi.name,
                    roi.color_hex,
                    tuple(
                        (int(point[0][0]), int(point[0][1]))
                        for point in roi.points
                    ),
                )
                for roi in rois
            )
            self._runtime_rois_cache[resolution_key] = runtime_rois
        elapsed_ms = (time.perf_counter() - begun) * 1000.0
        return CameraInferenceResult(
            self.config.group_id,
            self.config.camera_id,
            packet.sequence,
            CameraInferenceResult.utc_timestamp(),
            f"cuda:{self.config.device_index}",
            self.config.person_model_path,
            self.config.head_model_path,
            elapsed_ms,
            person_ms,
            head_ms,
            classification_ms,
            1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0,
            tuple(detections),
            self._last_heads,
            tuple(classifications),
            runtime_rois,
            tuple(stations),
        )

    def _predict(
        self,
        lease: Any,
        frame: Any,
        confidence: float,
        iou: float,
        input_size: int | None,
    ):
        arguments: dict[str, object] = {
            "source": frame,
            "device": self.config.device_index,
            "conf": confidence,
            "iou": iou,
            "verbose": False,
            "quantize": 16 if self.config.use_half_precision else 32,
        }
        if input_size:
            arguments["imgsz"] = input_size
        return lease.predict(**arguments)

    def _expire_classification_cache(
        self,
        now: float,
        active_track_ids: set[int],
    ) -> None:
        expired = [
            track_id
            for track_id, item in self._classification_cache.items()
            if track_id not in active_track_ids
            and now - item.last_seen_at
            >= self.config.classification_cache_ttl_seconds
        ]
        for track_id in expired:
            self._classification_cache.pop(track_id, None)

    @staticmethod
    def _extract_boxes(
        outputs: Any,
        target_classes: tuple[str, ...],
    ) -> list[CandidateDetection]:
        import torch

        accepted = {
            item.strip().casefold()
            for item in target_classes
            if item.strip()
        }
        detections: list[CandidateDetection] = []

        for output in outputs:
            boxes = output.boxes
            if boxes is None or len(boxes) == 0:
                continue

            combined = torch.cat(
                (
                    boxes.xyxy,
                    boxes.conf.unsqueeze(1),
                    boxes.cls.unsqueeze(1),
                ),
                dim=1,
            ).detach().float().cpu().numpy()

            for x1, y1, x2, y2, confidence, class_value in combined:
                class_id = int(class_value)
                names = output.names
                class_name = str(
                    names[class_id]
                    if isinstance(names, dict)
                    else names[class_id]
                )
                if accepted and class_name.casefold() not in accepted:
                    continue
                detections.append(
                    CandidateDetection(
                        class_id,
                        class_name,
                        float(confidence),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    )
                )

        return detections

    @staticmethod
    def _normalize_label(value: str) -> str:
        compact = value.strip().casefold().replace("-", "_").replace(" ", "_")
        mapping = {
            "va": "VA",
            "value_added": "VA",
            "nva": "NVA",
            "non_value_added": "NVA",
            "neutral": "NEUTRAL",
            "neutro": "NEUTRAL",
        }
        return mapping.get(compact, compact.upper())
