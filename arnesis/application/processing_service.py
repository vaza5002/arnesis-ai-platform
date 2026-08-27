"""Integrated group, camera, CUDA inference, persistence, and CSV export lifecycle."""
from __future__ import annotations

import threading
import time
from typing import Any

from arnesis.application.inference_orchestration_service import InferenceOrchestrationService
from arnesis.application.realtime_processing_service import RealtimeProcessingService
from arnesis.application.station_metrics_export_service import StationMetricsExportService
from arnesis.domain.entities import Group, GroupStatus
from arnesis.processing.group_session import GroupSessionConfiguration, SessionState
from arnesis.processing.processing_runtime import ProcessingRuntime


class ProcessingServiceError(RuntimeError):
    pass


class ProcessingService:
    def __init__(self, database, gpu_capacity, runtime=None, realtime=None,
                 inference=None, metrics_export=None) -> None:
        self.database = database
        self.gpu_capacity = gpu_capacity
        self.lock = threading.RLock()
        self.runtime = runtime or ProcessingRuntime(self._persist)
        self.realtime = realtime or RealtimeProcessingService(database)
        self.inference = inference or InferenceOrchestrationService(database, self.realtime)
        self.metrics_export = metrics_export or StationMetricsExportService(database)
        self._collector_stop = threading.Event()
        self._collector_thread: threading.Thread | None = None
        self._collector_sequences: dict[tuple[int, int], int] = {}

    def start_group(self, group_id: int) -> dict[str, object]:
        if self.runtime.contains(group_id):
            return self._snapshot(group_id)
        prepared = self.realtime.prepare_group(group_id)
        config = self._prepare(group_id, len(prepared.camera_configs))
        try:
            result = self.runtime.start_group(config).to_dict()
            result["cameras"] = self.realtime.start_prepared_group(prepared)
            result["inference"] = self.inference.start_group(group_id, config.gpu_index)
            self._ensure_collector()
            return result
        except Exception as exc:
            try:
                self.inference.stop_group(group_id)
            except Exception:
                pass
            try:
                self.realtime.stop_group(group_id)
            except Exception:
                pass
            if self.runtime.contains(group_id):
                try:
                    self.runtime.stop_group(group_id)
                except Exception:
                    pass
            self._write(group_id, GroupStatus.ERROR.value)
            raise ProcessingServiceError(
                f"Unable to start group id {group_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def pause_group(self, group_id: int) -> dict[str, object]:
        self.inference.pause_group(group_id)
        cameras = self.realtime.pause_group(group_id)
        result = self.runtime.pause_group(group_id).to_dict()
        result.update(cameras=cameras, inference=self.inference.snapshot(group_id))
        return result

    def resume_group(self, group_id: int) -> dict[str, object]:
        cameras = self.realtime.resume_group(group_id)
        self.inference.resume_group(group_id)
        result = self.runtime.resume_group(group_id).to_dict()
        result.update(cameras=cameras, inference=self.inference.snapshot(group_id))
        return result

    def stop_group(self, group_id: int) -> dict[str, object]:
        errors: list[str] = []
        try:
            self.inference.stop_group(group_id)
        except Exception as exc:
            errors.append(str(exc))
        try:
            cameras = self.realtime.stop_group(group_id)
        except Exception as exc:
            cameras = []
            errors.append(str(exc))
        result = (self.runtime.stop_group(group_id).to_dict()
                  if self.runtime.contains(group_id)
                  else {"group_id": group_id, "state": "STOPPED"})
        result.update(cameras=cameras, inference=[])
        self.metrics_export.flush()
        self._write(group_id, "ERROR" if errors else "STOPPED")
        if errors:
            raise ProcessingServiceError("; ".join(errors))
        return result

    def stop_all(self) -> list[dict[str, object]]:
        results = [self.stop_group(int(item["group_id"]))
                   for item in self.get_runtime_status()]
        self._collector_stop.set()
        thread = self._collector_thread
        if thread is not None:
            thread.join(5.0)
        self.metrics_export.flush()
        return results

    def get_runtime_status(self) -> list[dict[str, object]]:
        return [self._snapshot(item.group_id) for item in self.runtime.list_groups()]

    def subscribe_preview(self, group_id: int, camera_id: int):
        return self.realtime.subscribe_preview(group_id, camera_id)

    def unsubscribe_preview(self, group_id: int, camera_id: int):
        return self.realtime.unsubscribe_preview(group_id, camera_id)

    def preview_frame(self, group_id: int, camera_id: int):
        return self.realtime.preview_frame(group_id, camera_id)

    def latest_inference_result(self, group_id: int, camera_id: int,
                                after_sequence: int | None = None):
        return self.inference.latest_result(group_id, camera_id, after_sequence)

    def export_status(self):
        return self.metrics_export.status()

    def _ensure_collector(self) -> None:
        if self._collector_thread is not None and self._collector_thread.is_alive():
            return
        self._collector_stop.clear()
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            name="arnesis-station-metrics-export",
            daemon=True,
        )
        self._collector_thread.start()

    def _collector_loop(self) -> None:
        last_flush = time.monotonic()
        while not self._collector_stop.wait(1.0):
            try:
                for runtime_group in self.runtime.list_groups():
                    group_id = int(runtime_group.group_id)
                    for camera in self.realtime.group_snapshot(group_id):
                        camera_id = int(camera["camera_id"])
                        key = (group_id, camera_id)
                        result = self.inference.latest_result(
                            group_id, camera_id, self._collector_sequences.get(key))
                        if result is not None:
                            self.metrics_export.record_result(group_id, camera_id, result)
                            self._collector_sequences[key] = int(result.sequence)
                interval = self.metrics_export.settings_service.get().flush_interval_seconds
                if time.monotonic() - last_flush >= interval:
                    self.metrics_export.flush()
                    last_flush = time.monotonic()
            except Exception:
                # Export failure must never stop camera or CUDA inference.
                continue

    def _snapshot(self, group_id: int) -> dict[str, object]:
        result = self.runtime.get_group(group_id).to_dict()
        result["cameras"] = self.realtime.group_snapshot(group_id)
        result["inference"] = self.inference.snapshot(group_id)
        result["csv_export"] = {
            "state": self.metrics_export.status().state,
            "output_root": self.metrics_export.status().output_root,
            "pending_rows": self.metrics_export.status().pending_rows,
            "last_write_utc": self.metrics_export.status().last_write_utc,
            "last_error": self.metrics_export.status().last_error,
        }
        return result

    def _prepare(self, group_id: int, streams: int) -> GroupSessionConfiguration:
        with self.database.session_scope() as session:
            group = session.get(Group, group_id)
            if group is None:
                raise ProcessingServiceError(f"Group id {group_id} was not found.")
            allocation = self.gpu_capacity.select_device(
                session,
                requested_memory_mb=group.max_gpu_memory_mb,
                requested_streams=max(1, streams),
                preferred_gpu_index=group.preferred_gpu_index,
            )
            group.preferred_gpu_index = allocation.device_index
            group.status = "STARTING"
            return GroupSessionConfiguration(
                group.id, group.code, group.name,
                allocation.device_index, allocation.device_name,
                group.max_gpu_memory_mb, max(1, streams),
            )

    def _persist(self, group_id: int, state: SessionState, error: str | None) -> None:
        del error
        self._write(group_id, state.value)

    def _write(self, group_id: int, status: str) -> None:
        with self.lock:
            with self.database.session_scope() as session:
                group = session.get(Group, group_id)
                if group is not None:
                    group.status = status
