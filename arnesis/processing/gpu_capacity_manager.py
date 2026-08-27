"""CUDA-only GPU capacity synchronization and allocation for Arnesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from arnesis.domain.entities import GpuCapacity, Group, GroupStatus
from arnesis.processing.cuda_device import CudaDeviceService


class CudaCapacityError(RuntimeError):
    """Raised when CUDA capacity is invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class GpuAllocation:
    device_index: int
    device_uuid: str
    device_name: str
    total_memory_mb: int
    free_memory_mb: int
    configured_limit_mb: int
    allocated_group_memory_mb: int
    requested_memory_mb: int
    active_groups: int
    active_streams: int

    @property
    def cuda_label(self) -> str:
        return f"CUDA:{self.device_index} - {self.device_name}"


class GpuCapacityManager:
    """Synchronizes detected devices and selects capacity without CPU fallback."""

    ACTIVE_STATUSES = {
        GroupStatus.STARTING.value,
        GroupStatus.RUNNING.value,
        GroupStatus.PAUSING.value,
        GroupStatus.PAUSED.value,
    }

    def require_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise CudaCapacityError(
                "CUDA is required by Arnesis, but PyTorch cannot access a CUDA device."
            )
        if torch.cuda.device_count() < 1:
            raise CudaCapacityError("Arnesis requires at least one CUDA device.")

    def synchronize_devices(self, session: Session) -> list[GpuCapacity]:
        self.require_cuda()
        detected = list(CudaDeviceService.get_available_devices())
        if not detected:
            raise CudaCapacityError("CudaDeviceService returned no CUDA devices.")

        synchronized: list[GpuCapacity] = []
        detected_indices: set[int] = set()

        for device in detected:
            index = self._read_int(device, "index", "device_index", "cuda_index")
            detected_indices.add(index)
            name = self._read_text(device, "name", "device_name")
            uuid = self._resolve_uuid(index, device)

            record = session.scalar(
                select(GpuCapacity).where(GpuCapacity.device_index == index)
            )
            if record is None:
                record = GpuCapacity(
                    device_uuid=uuid,
                    device_index=index,
                    device_name=name,
                    enabled=True,
                    maximum_memory_percent=90.0,
                    reserved_memory_mb=2048,
                    maximum_groups=8,
                    maximum_streams=32,
                    priority=100,
                )
                session.add(record)
            else:
                record.device_uuid = uuid
                record.device_name = name

            synchronized.append(record)

        existing = session.scalars(select(GpuCapacity)).all()
        for record in existing:
            if record.device_index not in detected_indices:
                record.enabled = False

        session.flush()
        return sorted(synchronized, key=lambda item: item.device_index)

    def select_device(
        self,
        session: Session,
        *,
        requested_memory_mb: int,
        requested_streams: int,
        preferred_gpu_index: int | None = None,
    ) -> GpuAllocation:
        self.require_cuda()
        if requested_memory_mb <= 0:
            raise CudaCapacityError("Requested GPU memory must be greater than zero.")
        if requested_streams <= 0:
            raise CudaCapacityError("Requested stream count must be greater than zero.")

        records = session.scalars(
            select(GpuCapacity)
            .where(GpuCapacity.enabled.is_(True))
            .order_by(GpuCapacity.priority, GpuCapacity.device_index)
        ).all()
        if preferred_gpu_index is not None:
            records = sorted(
                records,
                key=lambda item: (item.device_index != preferred_gpu_index, item.priority),
            )

        candidates: list[GpuAllocation] = []
        for record in records:
            allocation = self._evaluate_device(
                session,
                record,
                requested_memory_mb=requested_memory_mb,
                requested_streams=requested_streams,
            )
            if allocation is not None:
                candidates.append(allocation)

        if not candidates:
            raise CudaCapacityError(
                "No enabled CUDA device has sufficient configured capacity for this group."
            )

        return max(
            candidates,
            key=lambda item: (
                item.free_memory_mb - item.requested_memory_mb,
                -item.active_groups,
                -item.device_index,
            ),
        )

    def _evaluate_device(
        self,
        session: Session,
        record: GpuCapacity,
        *,
        requested_memory_mb: int,
        requested_streams: int,
    ) -> GpuAllocation | None:
        if record.device_index >= torch.cuda.device_count():
            return None

        free_bytes, total_bytes = torch.cuda.mem_get_info(record.device_index)
        free_mb = int(free_bytes / 1024**2)
        total_mb = int(total_bytes / 1024**2)
        configured_limit_mb = int(total_mb * record.maximum_memory_percent / 100)
        usable_limit_mb = max(0, configured_limit_mb - record.reserved_memory_mb)

        active_groups = session.scalars(
            select(Group).where(
                Group.preferred_gpu_index == record.device_index,
                Group.status.in_(self.ACTIVE_STATUSES),
            )
        ).all()
        active_group_count = len(active_groups)
        active_streams = sum(group.max_concurrent_streams for group in active_groups)
        allocated_memory = sum(group.max_gpu_memory_mb for group in active_groups)

        has_group_capacity = active_group_count + 1 <= record.maximum_groups
        has_stream_capacity = active_streams + requested_streams <= record.maximum_streams
        has_configured_memory = allocated_memory + requested_memory_mb <= usable_limit_mb
        has_physical_memory = requested_memory_mb <= max(
            0, free_mb - record.reserved_memory_mb
        )

        if not all(
            (has_group_capacity, has_stream_capacity, has_configured_memory, has_physical_memory)
        ):
            return None

        return GpuAllocation(
            device_index=record.device_index,
            device_uuid=record.device_uuid,
            device_name=record.device_name,
            total_memory_mb=total_mb,
            free_memory_mb=free_mb,
            configured_limit_mb=usable_limit_mb,
            allocated_group_memory_mb=allocated_memory,
            requested_memory_mb=requested_memory_mb,
            active_groups=active_group_count,
            active_streams=active_streams,
        )

    @staticmethod
    def _resolve_uuid(index: int, device: Any) -> str:
        for attribute in ("uuid", "device_uuid"):
            value = getattr(device, attribute, None)
            if value:
                return str(value)

        properties = torch.cuda.get_device_properties(index)
        value = getattr(properties, "uuid", None)
        return str(value) if value else f"CUDA-{index}-{properties.name}"

    @staticmethod
    def _read_int(device: Any, *names: str) -> int:
        for name in names:
            value = getattr(device, name, None)
            if value is not None:
                return int(value)
        raise CudaCapacityError(f"CUDA device has no index attribute: {device!r}")

    @staticmethod
    def _read_text(device: Any, *names: str) -> str:
        for name in names:
            value = getattr(device, name, None)
            if value:
                return str(value)
        return "Unknown CUDA device"
