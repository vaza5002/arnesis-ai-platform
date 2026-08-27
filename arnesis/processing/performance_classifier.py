"""CUDA-only performance classifier pool selected by ROI profile."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arnesis.processing.cuda_model_manager import CudaModelLease, CudaModelManager


@dataclass(frozen=True, slots=True)
class ClassifierConfiguration:
    profile_id: int
    model_path: str
    model_sha256: str | None
    input_size: int | None
    minimum_confidence: float = 0.0


class PerformanceClassifierPool:
    def __init__(self, manager: CudaModelManager, device_index: int,
                 configurations: dict[int, ClassifierConfiguration]) -> None:
        self.manager = manager
        self.device_index = device_index
        self.configurations = configurations
        self.leases: dict[int, CudaModelLease] = {}

    def start(self) -> None:
        for profile_id, configuration in self.configurations.items():
            self.leases[profile_id] = self.manager.acquire(
                configuration.model_path,
                self.device_index,
                configuration.model_sha256,
            )

    def classify(self, profile_id: int | None, crop: Any):
        if profile_id is None or crop is None or crop.size == 0:
            return None
        configuration = self.configurations.get(profile_id)
        lease = self.leases.get(profile_id)
        if configuration is None or lease is None:
            return None
        arguments: dict[str, object] = {
            "source": crop,
            "device": self.device_index,
            "verbose": False,
        }
        if configuration.input_size:
            arguments["imgsz"] = configuration.input_size
        output = lease.model.predict(**arguments)[0]
        probabilities = getattr(output, "probs", None)
        if probabilities is None:
            raise RuntimeError("Performance Classification returned no probabilities.")
        class_id = int(probabilities.top1)
        confidence = float(probabilities.top1conf.detach().cpu().item())
        if confidence < configuration.minimum_confidence:
            return None
        names = output.names
        class_name = str(names[class_id] if isinstance(names, dict) else names[class_id])
        return class_id, class_name, confidence

    def close(self) -> None:
        for lease in self.leases.values():
            lease.release()
        self.leases.clear()
