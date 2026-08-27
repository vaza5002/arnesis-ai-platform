"""Lifecycle orchestration for camera pipelines and CUDA inference workers."""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Iterable

from arnesis.processing.cuda_model_manager import CudaModelManager
from arnesis.processing.detection_worker import DetectionWorker


class InferenceGroupEngine:
    """Start, pause, resume, and stop one group as an atomic unit."""

    def __init__(
        self,
        camera_pipelines: Iterable[object],
        workers: Iterable[DetectionWorker],
        model_manager: CudaModelManager,
    ) -> None:
        self.camera_pipelines = tuple(camera_pipelines)
        self.workers = tuple(workers)
        self.model_manager = model_manager
        self._lock = threading.RLock()
        self._running = False
        self._paused = False

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._running:
                return self.snapshot()
            started_pipelines: list[object] = []
            started_workers: list[DetectionWorker] = []
            try:
                for pipeline in self.camera_pipelines:
                    pipeline.start()
                    started_pipelines.append(pipeline)
                for worker in self.workers:
                    worker.start()
                    started_workers.append(worker)
                self._running = True
                self._paused = False
                return self.snapshot()
            except Exception:
                for worker in reversed(started_workers):
                    try: worker.stop()
                    except Exception: pass
                for pipeline in reversed(started_pipelines):
                    try: pipeline.stop()
                    except Exception: pass
                raise

    def pause(self) -> dict[str, object]:
        with self._lock:
            for worker in self.workers: worker.pause()
            for pipeline in self.camera_pipelines: pipeline.pause()
            self._paused = True
            return self.snapshot()

    def resume(self) -> dict[str, object]:
        with self._lock:
            for pipeline in self.camera_pipelines: pipeline.resume()
            for worker in self.workers: worker.resume()
            self._paused = False
            return self.snapshot()

    def stop(self) -> dict[str, object]:
        errors: list[str] = []
        with self._lock:
            for worker in reversed(self.workers):
                try: worker.stop()
                except Exception as exc: errors.append(f"worker: {type(exc).__name__}: {exc}")
            for pipeline in reversed(self.camera_pipelines):
                try: pipeline.stop()
                except Exception as exc: errors.append(f"pipeline: {type(exc).__name__}: {exc}")
            self._running = False
            self._paused = False
        result = self.snapshot()
        if errors:
            raise RuntimeError("Inference group shutdown incomplete: " + "; ".join(errors))
        return result

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self._running,
            "paused": self._paused,
            "cameras": len(self.camera_pipelines),
            "workers": [worker.snapshot() for worker in self.workers],
            "models": [asdict(snapshot) for snapshot in self.model_manager.snapshots()],
        }
