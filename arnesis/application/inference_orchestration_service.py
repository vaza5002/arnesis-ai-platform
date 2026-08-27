"""Database assembly and lifecycle for ROI-constrained multi-stage workers."""
from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from arnesis.domain.entities import Camera, Group, ModelDefinition, ProcessingProfile
from arnesis.processing.cuda_model_manager import CudaModelManager
from arnesis.processing.detection_worker import DetectionWorker, DetectionWorkerConfiguration
from arnesis.processing.performance_classifier import ClassifierConfiguration
from arnesis.processing.result_buffer import LatestResultBuffer


class _FrameSource:
    def __init__(self, realtime: Any, group_id: int, camera_id: int) -> None:
        self.realtime = realtime
        self.group_id = group_id
        self.camera_id = camera_id

    def next_frame(self, timeout_seconds: float | None = None):
        return self.realtime.next_inference_frame(
            self.group_id, self.camera_id, timeout_seconds)


class InferenceOrchestrationService:
    HEAD_COUNTER_NAME = "head counter"

    def __init__(self, database: Any, realtime: Any) -> None:
        self.database = database
        self.realtime = realtime
        self.manager = CudaModelManager()
        self.groups: dict[int, dict[int, tuple[DetectionWorker, LatestResultBuffer]]] = {}
        self.lock = threading.RLock()

    def start_group(self, group_id: int, device_index: int) -> list[dict[str, object]]:
        created: dict[int, tuple[DetectionWorker, LatestResultBuffer]] = {}
        try:
            for camera_id, config, rois, classifiers in self._load(group_id, device_index):
                results = LatestResultBuffer()
                worker = DetectionWorker(config, _FrameSource(
                    self.realtime, group_id, camera_id), rois, classifiers,
                    self.manager, results)
                worker.start()
                created[camera_id] = (worker, results)
        except Exception:
            for worker, results in created.values():
                try:
                    worker.stop()
                except Exception:
                    pass
                results.close()
            raise
        with self.lock:
            self.groups[group_id] = created
        return self.snapshot(group_id)

    def pause_group(self, group_id: int) -> None:
        for worker, _ in self.groups[group_id].values():
            worker.pause()

    def resume_group(self, group_id: int) -> None:
        for worker, _ in self.groups[group_id].values():
            worker.resume()

    def stop_group(self, group_id: int) -> None:
        with self.lock:
            items = self.groups.pop(group_id, {})
        for worker, results in items.values():
            worker.stop()
            results.close()

    def latest_result(self, group_id: int, camera_id: int,
                      after_sequence: int | None = None):
        item = self.groups.get(group_id, {}).get(camera_id)
        return None if item is None else item[1].get_latest(after_sequence)

    def snapshot(self, group_id: int) -> list[dict[str, object]]:
        return [{
            "camera_id": camera_id,
            "alive": worker.thread is not None and worker.thread.is_alive(),
            "error": worker.error,
        } for camera_id, (worker, _) in self.groups.get(group_id, {}).items()]

    def _load(self, group_id: int, device_index: int):
        with self.database.session_scope() as session:
            group = session.scalar(select(Group).options(
                selectinload(Group.cameras).selectinload(Camera.rois)
            ).where(Group.id == group_id))
            if group is None:
                raise ValueError(f"Group id {group_id} was not found.")

            head_models = session.scalars(select(ModelDefinition).where(
                func.lower(func.trim(ModelDefinition.name)) == self.HEAD_COUNTER_NAME,
                ModelDefinition.enabled.is_(True))).all()
            if len(head_models) != 1:
                raise ValueError("Enable exactly one DETECTION model named 'Head Counter'.")
            head_model = head_models[0]
            prepared = []

            for camera in (item for item in group.cameras if item.enabled):
                rois = [roi for roi in camera.rois
                        if roi.enabled and roi.processing_profile_id is not None]
                if not rois:
                    continue
                profiles = {roi.processing_profile_id:
                            session.get(ProcessingProfile, roi.processing_profile_id)
                            for roi in rois}
                if any(profile is None or not profile.enabled
                       or profile.detector_model_id is None
                       or profile.classifier_model_id is None
                       for profile in profiles.values()):
                    raise ValueError(f"Camera {camera.name} requires an enabled detector and classifier on every ROI profile.")
                detector_ids = {profile.detector_model_id for profile in profiles.values()}
                if len(detector_ids) != 1:
                    raise ValueError(f"Camera {camera.name} requires one shared person detector.")
                person_model = session.get(ModelDefinition, next(iter(detector_ids)))
                if person_model is None or not person_model.enabled:
                    raise ValueError(f"Camera {camera.name} person detector is missing or disabled.")

                classifiers: dict[int, ClassifierConfiguration] = {}
                for profile_id, profile in profiles.items():
                    classifier = session.get(ModelDefinition, profile.classifier_model_id)
                    if classifier is None or not classifier.enabled:
                        raise ValueError(f"Profile {profile.name} classifier is missing or disabled.")
                    custom = dict(profile.custom_parameters or {})
                    classifiers[int(profile_id)] = ClassifierConfiguration(
                        int(profile_id), classifier.model_path, classifier.file_sha256,
                        classifier.input_size,
                        float(custom.get("classification_confidence", 0.0)))

                primary = next(iter(profiles.values()))
                records = [{
                    "id": roi.id, "camera_id": roi.camera_id, "name": roi.name,
                    "roi_type": roi.roi_type, "points": roi.normalized_points,
                    "enabled": roi.enabled, "color_hex": roi.color_hex,
                    "display_order": roi.display_order,
                    "processing_profile_id": roi.processing_profile_id,
                } for roi in rois]
                config = DetectionWorkerConfiguration(
                    group.id, camera.id, device_index,
                    person_model.model_path, person_model.file_sha256,
                    primary.confidence_threshold, primary.iou_threshold,
                    primary.frame_skip, tuple(primary.target_classes or []),
                    person_model.input_size,
                    head_model.model_path, head_model.file_sha256,
                    head_model.input_size)
                prepared.append((camera.id, config, records, classifiers))

            if not prepared:
                raise ValueError(f"Group {group.code} has no enabled ROI profiles.")
            return prepared
