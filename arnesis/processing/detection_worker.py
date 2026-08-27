"""ROI-constrained person, head, and performance inference worker."""
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
from arnesis.processing.performance_classifier import ClassifierConfiguration, PerformanceClassifierPool
from arnesis.processing.result_buffer import LatestResultBuffer
from arnesis.processing.roi_assignment import BoundingBox, RoiAssignmentEngine
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
    head_frame_skip: int = 0


class DetectionWorker:
    def __init__(self, config: DetectionWorkerConfiguration, frame_source: Any,
                 roi_records: list[dict[str, Any]], classifiers: dict[int, ClassifierConfiguration],
                 manager: CudaModelManager, results: LatestResultBuffer | None = None) -> None:
        self.config = config
        self.frame_source = frame_source
        self.roi_records = list(roi_records)
        self.manager = manager
        self.results = results or LatestResultBuffer()
        self.classifier_pool = PerformanceClassifierPool(manager, config.device_index, classifiers)
        self.tracker = IouTrackingService()
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._started = threading.Event()
        self._person_lease = None
        self._head_lease = None
        self._last_heads: tuple[HeadDetectionResult, ...] = ()

    def start(self, timeout_seconds: float = 30.0) -> None:
        self.thread = threading.Thread(target=self._run,
            name=f"arnesis-multistage-{self.config.camera_id}", daemon=False)
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
        frame_counter = 0
        try:
            self._person_lease = self.manager.acquire(
                self.config.person_model_path, self.config.device_index,
                self.config.person_model_sha256)
            self._head_lease = self.manager.acquire(
                self.config.head_model_path, self.config.device_index,
                self.config.head_model_sha256)
            self.classifier_pool.start()
            self._started.set()
            while not self._stop.is_set():
                if self._pause.is_set():
                    self._stop.wait(0.05)
                    continue
                packet = self.frame_source.next_frame(timeout_seconds=0.50)
                if packet is None:
                    continue
                frame_counter += 1
                if (frame_counter - 1) % (self.config.frame_skip + 1) != 0:
                    continue
                self.results.publish(self._process(packet, frame_counter))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._started.set()
        finally:
            self.classifier_pool.close()
            if self._head_lease is not None:
                self._head_lease.release()
            if self._person_lease is not None:
                self._person_lease.release()

    def _process(self, packet: Any, frame_counter: int) -> CameraInferenceResult:
        begun = time.perf_counter()
        height, width = packet.frame.shape[:2]
        rois = RoiAssignmentEngine.prepare_rois(self.roi_records, width, height)

        person_started = time.perf_counter()
        raw_people = self._extract_boxes(self._predict(
            self._person_lease.model, packet.frame, self.config.person_confidence,
            self.config.person_iou, self.config.person_input_size),
            self.config.person_target_classes)
        person_ms = (time.perf_counter() - person_started) * 1000.0

        accepted: list[CandidateDetection] = []
        for person in raw_people:
            assignment = RoiAssignmentEngine.assign_track(
                0, BoundingBox(person.x1, person.y1, person.x2, person.y2), rois)
            if assignment is not None:
                accepted.append(person)
        tracked = self.tracker.update(accepted)

        if (frame_counter - 1) % (self.config.head_frame_skip + 1) == 0:
            head_started = time.perf_counter()
            raw_heads = self._extract_boxes(self._predict(
                self._head_lease.model, packet.frame, self.config.head_confidence,
                self.config.head_iou, self.config.head_input_size), ())
            head_ms = (time.perf_counter() - head_started) * 1000.0
            self._last_heads = tuple(HeadDetectionResult(
                item.class_id, item.class_name, item.confidence,
                item.x1, item.y1, item.x2, item.y2) for item in raw_heads)
        else:
            head_ms = 0.0

        head_counts = {roi.roi_id: 0 for roi in rois}
        for head in self._last_heads:
            assignment = RoiAssignmentEngine.assign_track(
                0, BoundingBox(head.x1, head.y1, head.x2, head.y2), rois)
            if assignment is not None:
                head_counts[assignment.roi_id] += 1

        detections: list[DetectionResult] = []
        classifications: list[PerformanceClassificationResult] = []
        classification_ms = 0.0
        for track_id, person in tracked:
            assignment = RoiAssignmentEngine.assign_track(
                track_id, BoundingBox(person.x1, person.y1, person.x2, person.y2), rois)
            if assignment is None:
                continue
            detections.append(DetectionResult(
                track_id, person.class_id, person.class_name, person.confidence,
                person.x1, person.y1, person.x2, person.y2,
                assignment.roi_id, assignment.station))
            x1 = max(0, int(person.x1)); y1 = max(0, int(person.y1))
            x2 = min(width, int(person.x2)); y2 = min(height, int(person.y2))
            classification_started = time.perf_counter()
            value = self.classifier_pool.classify(
                assignment.processing_profile_id, packet.frame[y1:y2, x1:x2])
            classification_ms += (time.perf_counter() - classification_started) * 1000.0
            if value is not None:
                class_id, class_name, confidence = value
                classifications.append(PerformanceClassificationResult(
                    track_id, assignment.roi_id, assignment.station,
                    class_id, class_name, confidence))

        stations: list[StationResult] = []
        for roi in rois:
            track_ids = tuple(item.track_id for item in detections if item.roi_id == roi.roi_id)
            labels = [self._normalize_label(item.class_name)
                for item in classifications if item.roi_id == roi.roi_id]
            stations.append(StationResult(
                roi.roi_id, roi.name, len(track_ids), head_counts[roi.roi_id],
                labels.count("VA"), labels.count("NVA"), labels.count("NEUTRAL"),
                track_ids, roi.processing_profile_id))

        runtime_rois = tuple(RuntimeRoiResult(
            roi.roi_id, roi.name, roi.color_hex,
            tuple((int(point[0][0]), int(point[0][1])) for point in roi.points))
            for roi in rois)
        elapsed_ms = (time.perf_counter() - begun) * 1000.0
        return CameraInferenceResult(
            self.config.group_id, self.config.camera_id, packet.sequence,
            CameraInferenceResult.utc_timestamp(), f"cuda:{self.config.device_index}",
            self.config.person_model_path, self.config.head_model_path,
            elapsed_ms, person_ms, head_ms, classification_ms,
            1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0,
            tuple(detections), self._last_heads, tuple(classifications),
            runtime_rois, tuple(stations))

    def _predict(self, model: Any, frame: Any, confidence: float,
                 iou: float, input_size: int | None):
        arguments: dict[str, object] = {
            "source": frame, "device": self.config.device_index,
            "conf": confidence, "iou": iou, "verbose": False,
        }
        if input_size:
            arguments["imgsz"] = input_size
        return model.predict(**arguments)

    @staticmethod
    def _extract_boxes(outputs: Any, target_classes: tuple[str, ...]) -> list[CandidateDetection]:
        accepted = {item.strip().casefold() for item in target_classes if item.strip()}
        detections: list[CandidateDetection] = []
        for output in outputs:
            if output.boxes is None:
                continue
            for coordinates, confidence, class_value in zip(
                output.boxes.xyxy.detach().cpu().tolist(),
                output.boxes.conf.detach().cpu().tolist(),
                output.boxes.cls.detach().cpu().tolist()):
                class_id = int(class_value)
                names = output.names
                class_name = str(names[class_id] if isinstance(names, dict) else names[class_id])
                if accepted and class_name.casefold() not in accepted:
                    continue
                detections.append(CandidateDetection(
                    class_id, class_name, float(confidence),
                    float(coordinates[0]), float(coordinates[1]),
                    float(coordinates[2]), float(coordinates[3])))
        return detections

    @staticmethod
    def _normalize_label(value: str) -> str:
        compact = value.strip().casefold().replace("-", "_").replace(" ", "_")
        mapping = {
            "va": "VA", "value_added": "VA",
            "nva": "NVA", "non_value_added": "NVA",
            "neutral": "NEUTRAL", "neutro": "NEUTRAL",
        }
        return mapping.get(compact, compact.upper())
