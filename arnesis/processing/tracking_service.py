"""Lightweight class-aware IoU tracker for ROI-accepted detections."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class _Track:
    detection: CandidateDetection
    missed_frames: int = 0


class IouTrackingService:
    def __init__(self, threshold: float = 0.30, maximum_missed_frames: int = 15) -> None:
        self.threshold = threshold
        self.maximum_missed_frames = maximum_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[CandidateDetection]) -> list[tuple[int, CandidateDetection]]:
        unused = set(self._tracks)
        output: list[tuple[int, CandidateDetection]] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            choices = [
                (track_id, self._iou(track.detection, detection))
                for track_id, track in self._tracks.items()
                if track_id in unused and track.detection.class_id == detection.class_id
            ]
            track_id: int | None = None
            if choices:
                candidate_id, score = max(choices, key=lambda item: item[1])
                if score >= self.threshold:
                    track_id = candidate_id
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = _Track(detection)
            unused.discard(track_id)
            output.append((track_id, detection))
        for track_id in unused:
            self._tracks[track_id].missed_frames += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed_frames <= self.maximum_missed_frames
        }
        return output

    @staticmethod
    def _iou(left: CandidateDetection, right: CandidateDetection) -> float:
        x1 = max(left.x1, right.x1)
        y1 = max(left.y1, right.y1)
        x2 = min(left.x2, right.x2)
        y2 = min(left.y2, right.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
        right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
        union = left_area + right_area - intersection
        return 0.0 if union <= 0.0 else intersection / union
