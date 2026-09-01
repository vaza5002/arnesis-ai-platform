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
    def __init__(
        self,
        manager: CudaModelManager,
        device_index: int,
        configurations: dict[int, ClassifierConfiguration],
    ) -> None:
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
        values = self.classify_batch(profile_id, [crop])
        return values[0] if values else None

    def classify_batch(
        self,
        profile_id: int | None,
        crops: list[Any],
    ) -> list[tuple[int, str, float] | None]:
        if profile_id is None:
            return [None] * len(crops)

        configuration = self.configurations.get(profile_id)
        lease = self.leases.get(profile_id)
        if configuration is None or lease is None:
            return [None] * len(crops)

        valid_indices: list[int] = []
        valid_crops: list[Any] = []
        for index, crop in enumerate(crops):
            if crop is not None and getattr(crop, "size", 0) > 0:
                valid_indices.append(index)
                valid_crops.append(crop)

        values: list[tuple[int, str, float] | None] = [None] * len(crops)
        if not valid_crops:
            return values

        arguments: dict[str, object] = {
            "source": valid_crops,
            "device": self.device_index,
            "verbose": False,
            "quantize": 16,
        }
        if configuration.input_size:
            arguments["imgsz"] = configuration.input_size

        outputs = list(lease.predict(**arguments))
        if len(outputs) != len(valid_indices):
            raise RuntimeError(
                "Performance Classification returned an unexpected result count."
            )

        probability_tensors = []
        for output in outputs:
            probabilities = getattr(output, "probs", None)
            probability_data = getattr(probabilities, "data", None)
            if probability_data is None:
                raise RuntimeError(
                    "Performance Classification returned no probabilities."
                )
            probability_tensors.append(probability_data)

        import torch

        probability_batch = torch.stack(probability_tensors, dim=0)
        confidence_batch, class_batch = probability_batch.max(dim=1)
        host_results = torch.stack(
            (class_batch.to(dtype=torch.float32), confidence_batch.float()),
            dim=1,
        ).detach().cpu().numpy()

        for target_index, output, host_result in zip(
            valid_indices,
            outputs,
            host_results,
        ):
            class_id = int(host_result[0])
            confidence = float(host_result[1])
            if confidence < configuration.minimum_confidence:
                continue

            names = output.names
            class_name = str(
                names[class_id] if isinstance(names, dict) else names[class_id]
            )
            values[target_index] = class_id, class_name, confidence

        return values

    def close(self) -> None:
        for lease in self.leases.values():
            lease.release()
        self.leases.clear()
