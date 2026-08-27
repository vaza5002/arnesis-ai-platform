"""ROI persistence and runtime preparation service.

The service stores normalized polygon vertices for database portability and
converts ORM entities into immutable application-friendly dictionaries.
Pixel conversion remains a processing-layer responsibility.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from arnesis.domain.entities import Roi, RoiType


class RoiService:
    """Create, update, query, order, enable, and delete camera ROIs."""

    DEFAULT_COLOR = "#29E6FF"

    def __init__(self, database: Any) -> None:
        self.database = database

    def list_by_camera(
        self,
        camera_id: int,
        *,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return ROIs for one camera in deterministic assignment order."""
        self._validate_positive_id(camera_id, "Camera id")

        statement = (
            select(Roi)
            .where(Roi.camera_id == camera_id)
            .order_by(Roi.display_order, Roi.id)
        )
        if enabled_only:
            statement = statement.where(Roi.enabled.is_(True))

        with self.database.session_scope() as session:
            rows = session.scalars(statement).all()
            return [self._dto(row) for row in rows]

    def list_runtime_polygons(self, camera_id: int) -> list[dict[str, Any]]:
        """Return enabled polygon ROIs prepared for inference configuration."""
        rois = self.list_by_camera(camera_id, enabled_only=True)
        return [roi for roi in rois if roi["roi_type"] == RoiType.POLYGON.value]

    def get(self, roi_id: int) -> dict[str, Any]:
        """Return one ROI or raise when the record does not exist."""
        self._validate_positive_id(roi_id, "ROI id")
        with self.database.session_scope() as session:
            roi = session.get(Roi, roi_id)
            if roi is None:
                raise ValueError(f"ROI id {roi_id} was not found.")
            return self._dto(roi)

    def save_polygon(
        self,
        camera_id: int,
        name: str,
        points: list[dict[str, float]],
        roi_id: int | None = None,
        profile_id: int | None = None,
        color_hex: str = DEFAULT_COLOR,
        display_order: int | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create or update a normalized polygon ROI."""
        self._validate_positive_id(camera_id, "Camera id")
        if roi_id is not None:
            self._validate_positive_id(roi_id, "ROI id")
        if profile_id is not None:
            self._validate_positive_id(profile_id, "Processing profile id")

        normalized_name = name.strip()
        normalized_points = self._normalize_points(points)
        normalized_color = self._normalize_color(color_hex)
        if not normalized_name:
            raise ValueError("ROI name is required.")

        with self.database.session_scope() as session:
            roi = session.get(Roi, roi_id) if roi_id is not None else None
            if roi_id is not None and roi is None:
                raise ValueError(f"ROI id {roi_id} was not found.")

            if roi is None:
                roi = Roi(
                    camera_id=camera_id,
                    name=normalized_name,
                    roi_type=RoiType.POLYGON.value,
                    normalized_points=normalized_points,
                )

            roi.camera_id = camera_id
            roi.name = normalized_name
            roi.roi_type = RoiType.POLYGON.value
            roi.normalized_points = normalized_points
            roi.processing_profile_id = profile_id
            roi.color_hex = normalized_color
            roi.enabled = bool(enabled)
            if display_order is not None:
                if display_order < 0:
                    raise ValueError("ROI display order cannot be negative.")
                roi.display_order = int(display_order)

            session.add(roi)
            session.commit()
            session.refresh(roi)
            return self._dto(roi)

    def set_enabled(self, roi_id: int, enabled: bool) -> dict[str, Any]:
        """Enable or disable an ROI without deleting its configuration."""
        self._validate_positive_id(roi_id, "ROI id")
        with self.database.session_scope() as session:
            roi = session.get(Roi, roi_id)
            if roi is None:
                raise ValueError(f"ROI id {roi_id} was not found.")
            roi.enabled = bool(enabled)
            session.commit()
            session.refresh(roi)
            return self._dto(roi)

    def reorder(self, camera_id: int, ordered_roi_ids: list[int]) -> list[dict[str, Any]]:
        """Persist first-match priority for overlapping camera ROIs."""
        self._validate_positive_id(camera_id, "Camera id")
        if len(set(ordered_roi_ids)) != len(ordered_roi_ids):
            raise ValueError("ROI order cannot contain duplicate ids.")

        with self.database.session_scope() as session:
            rows = session.scalars(
                select(Roi).where(Roi.camera_id == camera_id)
            ).all()
            by_id = {row.id: row for row in rows}
            unknown = [roi_id for roi_id in ordered_roi_ids if roi_id not in by_id]
            if unknown:
                raise ValueError(
                    f"ROI ids do not belong to camera id {camera_id}: {unknown}"
                )

            for order, roi_id in enumerate(ordered_roi_ids):
                by_id[roi_id].display_order = order
            session.commit()

        return self.list_by_camera(camera_id)

    def delete(self, roi_id: int) -> None:
        """Delete an ROI and preserve all unrelated camera configuration."""
        self._validate_positive_id(roi_id, "ROI id")
        with self.database.session_scope() as session:
            roi = session.get(Roi, roi_id)
            if roi is None:
                raise ValueError(f"ROI id {roi_id} was not found.")
            session.delete(roi)
            session.commit()

    @staticmethod
    def _normalize_points(
        points: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        if len(points) < 3:
            raise ValueError("A polygon ROI requires at least three points.")

        normalized: list[dict[str, float]] = []
        for index, point in enumerate(points):
            if "x" not in point or "y" not in point:
                raise ValueError(f"ROI point {index + 1} requires x and y values.")
            x = float(point["x"])
            y = float(point["y"])
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(
                    "ROI coordinates must be normalized between 0 and 1."
                )
            normalized.append({"x": x, "y": y})

        if len({(point["x"], point["y"]) for point in normalized}) < 3:
            raise ValueError("A polygon ROI requires three distinct points.")
        return normalized

    @staticmethod
    def _normalize_color(color_hex: str) -> str:
        value = color_hex.strip().upper()
        if len(value) != 7 or not value.startswith("#"):
            raise ValueError("ROI color must use #RRGGBB format.")
        try:
            int(value[1:], 16)
        except ValueError as exc:
            raise ValueError("ROI color must use hexadecimal #RRGGBB format.") from exc
        return value

    @staticmethod
    def _validate_positive_id(value: int, label: str) -> None:
        if int(value) <= 0:
            raise ValueError(f"{label} must be greater than zero.")

    @staticmethod
    def _dto(roi: Roi) -> dict[str, Any]:
        return {
            "id": roi.id,
            "camera_id": roi.camera_id,
            "name": roi.name,
            "roi_type": roi.roi_type,
            "points": [dict(point) for point in (roi.normalized_points or [])],
            "color": roi.color_hex,
            "enabled": roi.enabled,
            "profile_id": roi.processing_profile_id,
            "display_order": roi.display_order,
        }
