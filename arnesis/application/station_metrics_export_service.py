"""Database persistence and buffered CSV export for anonymous ROI metrics."""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from arnesis.application.export_settings_service import ExportSettingsService
from arnesis.domain.entities import Camera, Group, StationMetricSample


CSV_COLUMNS = (
    "Timestamp", "GroupId", "GroupCode", "CameraId", "CameraName",
    "RoiId", "RoiName", "HeadCount", "PersonCount", "Occupied",
    "VACount", "NVACount", "NeutralCount", "CUDADevice", "Sequence",
)


@dataclass(frozen=True, slots=True)
class ExportStatus:
    state: str
    output_root: str
    pending_rows: int
    last_write_utc: str | None
    last_error: str | None


class StationMetricsExportService:
    def __init__(self, database) -> None:
        self.database = database
        self.settings_service = ExportSettingsService(database)
        self.settings_service.ensure_defaults()
        self._pending: list[dict[str, object]] = []
        self._seen_sequences: dict[tuple[int, int], int] = {}
        self._lock = threading.RLock()
        self._last_write_utc: str | None = None
        self._last_error: str | None = None

    def record_result(self, group_id: int, camera_id: int, result: Any) -> int:
        key = (int(group_id), int(camera_id))
        sequence = int(result.sequence)
        with self._lock:
            if self._seen_sequences.get(key, -1) >= sequence:
                return 0
            self._seen_sequences[key] = sequence

        observed_at = self._parse_timestamp(result.timestamp_utc)
        with self.database.session_scope() as session:
            group = session.get(Group, int(group_id))
            camera = session.get(Camera, int(camera_id))
            if group is None or camera is None:
                raise ValueError("Group or camera was not found while persisting station metrics.")
            group_code = group.code
            camera_name = camera.name
            rows: list[dict[str, object]] = []
            for station in result.stations:
                row = {
                    "Timestamp": observed_at.isoformat(timespec="seconds"),
                    "GroupId": int(group_id),
                    "GroupCode": group_code,
                    "CameraId": int(camera_id),
                    "CameraName": camera_name,
                    "RoiId": int(station.roi_id),
                    "RoiName": station.station,
                    "HeadCount": int(station.head_count),
                    "PersonCount": int(station.people_count),
                    "Occupied": bool(station.head_count > 0 or station.people_count > 0),
                    "VACount": int(station.va_count),
                    "NVACount": int(station.nva_count),
                    "NeutralCount": int(station.neutral_count),
                    "CUDADevice": result.cuda_device,
                    "Sequence": sequence,
                }
                rows.append(row)
                session.add(StationMetricSample(
                    observed_at=observed_at,
                    group_id=int(group_id), group_code=group_code,
                    camera_id=int(camera_id), camera_name=camera_name,
                    roi_id=int(station.roi_id), roi_name=station.station,
                    head_count=int(station.head_count),
                    person_count=int(station.people_count),
                    occupied=bool(row["Occupied"]),
                    va_count=int(station.va_count),
                    nva_count=int(station.nva_count),
                    neutral_count=int(station.neutral_count),
                    cuda_device=result.cuda_device,
                    source_sequence=sequence,
                ))
        with self._lock:
            self._pending.extend(rows)
        return len(rows)

    def flush(self) -> int:
        settings = self.settings_service.get()
        if not settings.enabled:
            return 0
        with self._lock:
            rows = list(self._pending)
        if not rows:
            return 0
        try:
            root = self.settings_service.validate_output_root(settings.output_root)
            grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
            for row in rows:
                day = str(row["Timestamp"])[:10]
                group_code = self._safe_name(str(row["GroupCode"]))
                grouped.setdefault((group_code, day), []).append(row)
            for (group_code, day), records in grouped.items():
                self._append(root / "Combined" / f"station_metrics_{group_code}_{day}.csv",
                             records, settings.encoding, settings.delimiter)
                self._append(root / "Occupancy" / f"occupancy_{group_code}_{day}.csv",
                             records, settings.encoding, settings.delimiter)
                self._append(root / "Performance" / f"performance_{group_code}_{day}.csv",
                             records, settings.encoding, settings.delimiter)
            with self._lock:
                del self._pending[:len(rows)]
                self._last_write_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._last_error = None
            return len(rows)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            return 0

    def status(self) -> ExportStatus:
        settings = self.settings_service.get()
        with self._lock:
            return ExportStatus(
                state="ERROR" if self._last_error else ("ENABLED" if settings.enabled else "DISABLED"),
                output_root=settings.output_root,
                pending_rows=len(self._pending),
                last_write_utc=self._last_write_utc,
                last_error=self._last_error,
            )

    @staticmethod
    def _append(path: Path, rows: list[dict[str, object]],
                encoding: str, delimiter: str) -> None:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding=encoding) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS,
                                    delimiter=delimiter, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
