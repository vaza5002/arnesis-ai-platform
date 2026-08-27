"""Polygon ROI conversion, assignment, and station counting.

The implementation preserves the original Arnesis rule: the center of a
tracked person's bounding box determines ROI membership, and the first matching
ROI in display order wins when polygons overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PixelRoi:
    """Immutable runtime polygon converted to the current frame resolution."""

    roi_id: int
    camera_id: int
    name: str
    points: np.ndarray
    color_hex: str
    processing_profile_id: int | None
    display_order: int

    def contains(self, x: float, y: float) -> bool:
        """Return True when the point is inside or on the polygon boundary."""
        return cv2.pointPolygonTest(self.points, (float(x), float(y)), False) >= 0


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel bounding box for one detected or tracked person."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True, slots=True)
class RoiAssignment:
    """Association between a tracked person and one station ROI."""

    track_id: int
    roi_id: int
    station: str
    center_x: int
    center_y: int
    processing_profile_id: int | None


class RoiAssignmentEngine:
    """Convert normalized ROIs and assign tracked people deterministically."""

    @staticmethod
    def prepare_rois(
        roi_records: Iterable[Mapping[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> tuple[PixelRoi, ...]:
        """Convert enabled normalized polygons into pixel polygons."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame width and height must be greater than zero.")

        prepared: list[PixelRoi] = []
        ordered = sorted(
            roi_records,
            key=lambda item: (
                int(item.get("display_order", 0)),
                int(item.get("id", 0)),
            ),
        )
        for record in ordered:
            if not bool(record.get("enabled", True)):
                continue
            if str(record.get("roi_type", "POLYGON")).upper() != "POLYGON":
                continue

            points = record.get("points") or record.get("normalized_points") or []
            pixel_points = RoiAssignmentEngine._to_pixel_polygon(
                points,
                frame_width,
                frame_height,
            )
            prepared.append(
                PixelRoi(
                    roi_id=int(record["id"]),
                    camera_id=int(record["camera_id"]),
                    name=str(record["name"]).strip(),
                    points=pixel_points,
                    color_hex=str(
                        record.get("color_hex", record.get("color", "#29E6FF"))
                    ),
                    processing_profile_id=RoiAssignmentEngine._optional_int(
                        record.get(
                            "processing_profile_id",
                            record.get("profile_id"),
                        )
                    ),
                    display_order=int(record.get("display_order", 0)),
                )
            )
        return tuple(prepared)

    @staticmethod
    def assign_track(
        track_id: int,
        bounding_box: BoundingBox | Sequence[float],
        rois: Sequence[PixelRoi],
    ) -> RoiAssignment | None:
        """Assign one track to the first matching ROI."""
        box = RoiAssignmentEngine._coerce_box(bounding_box)
        center_x, center_y = box.center
        for roi in rois:
            if roi.contains(center_x, center_y):
                return RoiAssignment(
                    track_id=int(track_id),
                    roi_id=roi.roi_id,
                    station=roi.name,
                    center_x=int(round(center_x)),
                    center_y=int(round(center_y)),
                    processing_profile_id=roi.processing_profile_id,
                )
        return None

    @staticmethod
    def assign_tracks(
        tracked_boxes: Iterable[tuple[int, BoundingBox | Sequence[float]]],
        rois: Sequence[PixelRoi],
    ) -> dict[int, RoiAssignment]:
        """Assign every track to at most one station ROI."""
        assignments: dict[int, RoiAssignment] = {}
        for track_id, bounding_box in tracked_boxes:
            assignment = RoiAssignmentEngine.assign_track(
                track_id,
                bounding_box,
                rois,
            )
            if assignment is not None:
                assignments[int(track_id)] = assignment
        return assignments

    @staticmethod
    def count_people_by_station(
        assignments: Mapping[int, RoiAssignment],
        rois: Sequence[PixelRoi],
    ) -> dict[str, int]:
        """Count unique tracks per station for the current frame."""
        counts = {roi.name: 0 for roi in rois}
        seen_tracks: set[int] = set()
        for track_id, assignment in assignments.items():
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            counts.setdefault(assignment.station, 0)
            counts[assignment.station] += 1
        return counts

    @staticmethod
    def draw_rois(frame: np.ndarray, rois: Sequence[PixelRoi]) -> np.ndarray:
        """Draw ROI polygons and station names on a copy of the frame."""
        output = frame.copy()
        for roi in rois:
            color = RoiAssignmentEngine._hex_to_bgr(roi.color_hex)
            cv2.polylines(output, [roi.points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(int(value) for value in roi.points[0][0])
            cv2.putText(
                output,
                roi.name,
                (anchor[0], max(18, anchor[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def _to_pixel_polygon(
        points: Iterable[Mapping[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        normalized = list(points)
        if len(normalized) < 3:
            raise ValueError("A runtime polygon requires at least three points.")

        converted: list[list[int]] = []
        for point in normalized:
            x = float(point["x"])
            y = float(point["y"])
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError("Runtime ROI points must be normalized from 0 to 1.")
            pixel_x = min(frame_width - 1, max(0, int(round(x * (frame_width - 1)))))
            pixel_y = min(frame_height - 1, max(0, int(round(y * (frame_height - 1)))))
            converted.append([pixel_x, pixel_y])

        polygon = np.asarray(converted, dtype=np.int32).reshape((-1, 1, 2))
        if abs(cv2.contourArea(polygon)) < 1.0:
            raise ValueError("Runtime ROI polygon area must be greater than zero.")
        return polygon

    @staticmethod
    def _coerce_box(value: BoundingBox | Sequence[float]) -> BoundingBox:
        if isinstance(value, BoundingBox):
            return value
        if len(value) != 4:
            raise ValueError("A bounding box requires x1, y1, x2, and y2 values.")
        x1, y1, x2, y2 = (float(item) for item in value)
        if x2 < x1 or y2 < y1:
            raise ValueError("Bounding box maximum coordinates cannot be smaller.")
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _hex_to_bgr(value: str) -> tuple[int, int, int]:
        color = value.strip().lstrip("#")
        if len(color) != 6:
            return (255, 230, 41)
        red, green, blue = (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )
        return (blue, green, red)
