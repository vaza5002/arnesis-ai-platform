"""
CUDA device discovery and validation utilities for Arnesis.

This module enforces the GPU-only execution policy used by the real-time
processing system. CPU inference is intentionally unsupported.

The classes defined here provide CUDA device information to the processing
runtime, GPU resource manager, controller, and user interface.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "PyTorch is not installed. Arnesis requires a CUDA-enabled PyTorch "
        "installation and does not support CPU inference."
    ) from exc


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA is unavailable or no compatible GPU is detected."""


class InvalidCudaDeviceError(ValueError):
    """Raised when a requested CUDA device index does not exist."""


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    """
    Describes an NVIDIA CUDA device available to Arnesis.

    Attributes:
        device_index: Zero-based CUDA device index.
        device_name: GPU name reported by PyTorch.
        total_memory_bytes: Total physical GPU memory in bytes.
        compute_capability_major: CUDA compute capability major version.
        compute_capability_minor: CUDA compute capability minor version.
    """

    device_index: int
    device_name: str
    total_memory_bytes: int
    compute_capability_major: int
    compute_capability_minor: int

    @property
    def torch_device(self) -> str:
        """Return the PyTorch device identifier."""

        return f"cuda:{self.device_index}"

    @property
    def total_memory_mb(self) -> float:
        """Return total GPU memory in mebibytes."""

        return self.total_memory_bytes / (1024**2)

    @property
    def total_memory_gb(self) -> float:
        """Return total GPU memory in gibibytes."""

        return self.total_memory_bytes / (1024**3)

    @property
    def compute_capability(self) -> str:
        """Return the CUDA compute capability as a display value."""

        return (
            f"{self.compute_capability_major}."
            f"{self.compute_capability_minor}"
        )

    @property
    def display_name(self) -> str:
        """Return a user-friendly description for the UI."""

        return (
            f"CUDA:{self.device_index} - {self.device_name} "
            f"({self.total_memory_gb:.2f} GB)"
        )


@dataclass(frozen=True, slots=True)
class CudaMemorySnapshot:
    """
    Represents the current memory state of a CUDA device.

    Attributes:
        device_index: Zero-based CUDA device index.
        free_memory_bytes: Currently available GPU memory.
        total_memory_bytes: Total physical GPU memory.
        allocated_memory_bytes: Memory occupied by active PyTorch tensors.
        reserved_memory_bytes: Memory reserved by the PyTorch caching allocator.
    """

    device_index: int
    free_memory_bytes: int
    total_memory_bytes: int
    allocated_memory_bytes: int
    reserved_memory_bytes: int

    @property
    def free_memory_mb(self) -> float:
        """Return available GPU memory in mebibytes."""

        return self.free_memory_bytes / (1024**2)

    @property
    def total_memory_mb(self) -> float:
        """Return total GPU memory in mebibytes."""

        return self.total_memory_bytes / (1024**2)

    @property
    def allocated_memory_mb(self) -> float:
        """Return PyTorch-allocated memory in mebibytes."""

        return self.allocated_memory_bytes / (1024**2)

    @property
    def reserved_memory_mb(self) -> float:
        """Return PyTorch-reserved memory in mebibytes."""

        return self.reserved_memory_bytes / (1024**2)

    @property
    def utilization_percent(self) -> float:
        """Return the percentage of total memory currently unavailable."""

        if self.total_memory_bytes <= 0:
            return 0.0

        used_memory_bytes = self.total_memory_bytes - self.free_memory_bytes
        return (used_memory_bytes / self.total_memory_bytes) * 100.0


class CudaDeviceService:
    """
    Validates CUDA availability and provides GPU device information.

    Arnesis must call `require_cuda` before initializing models or starting
    any real-time processing session. This service never returns a CPU device.
    """

    @staticmethod
    def require_cuda() -> None:
        """
        Verify that CUDA is available and at least one GPU is detected.

        Raises:
            CudaUnavailableError: If the installed PyTorch build has no CUDA
                support, the NVIDIA runtime is unavailable, or no CUDA device
                is detected.
        """

        if torch.version.cuda is None:
            raise CudaUnavailableError(
                "The installed PyTorch build does not include CUDA support. "
                "Arnesis cannot run inference on CPU."
            )

        if not torch.cuda.is_available():
            raise CudaUnavailableError(
                "CUDA is not available. Verify the NVIDIA driver, CUDA-enabled "
                "PyTorch installation, and GPU access. Arnesis cannot run "
                "inference on CPU."
            )

        if torch.cuda.device_count() < 1:
            raise CudaUnavailableError(
                "No CUDA-compatible GPU was detected. Arnesis cannot start "
                "real-time processing without an NVIDIA CUDA device."
            )

    @classmethod
    def get_available_devices(cls) -> tuple[CudaDeviceInfo, ...]:
        """
        Return all CUDA devices currently available to Arnesis.

        Returns:
            An immutable tuple containing the detected CUDA devices.

        Raises:
            CudaUnavailableError: If CUDA is unavailable.
        """

        cls.require_cuda()

        devices: list[CudaDeviceInfo] = []

        for device_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device_index)

            devices.append(
                CudaDeviceInfo(
                    device_index=device_index,
                    device_name=properties.name,
                    total_memory_bytes=properties.total_memory,
                    compute_capability_major=properties.major,
                    compute_capability_minor=properties.minor,
                )
            )

        return tuple(devices)

    @classmethod
    def get_device(cls, device_index: int) -> CudaDeviceInfo:
        """
        Return information for a specific CUDA device.

        Args:
            device_index: Zero-based CUDA device index.

        Returns:
            Information for the requested CUDA device.

        Raises:
            CudaUnavailableError: If CUDA is unavailable.
            InvalidCudaDeviceError: If the device index does not exist.
        """

        cls.require_cuda()
        cls._validate_device_index(device_index)

        properties = torch.cuda.get_device_properties(device_index)

        return CudaDeviceInfo(
            device_index=device_index,
            device_name=properties.name,
            total_memory_bytes=properties.total_memory,
            compute_capability_major=properties.major,
            compute_capability_minor=properties.minor,
        )

    @classmethod
    def get_memory_snapshot(
        cls,
        device_index: int,
    ) -> CudaMemorySnapshot:
        """
        Return the current memory state for a CUDA device.

        Args:
            device_index: Zero-based CUDA device index.

        Returns:
            Current CUDA and PyTorch memory measurements.

        Raises:
            CudaUnavailableError: If CUDA is unavailable.
            InvalidCudaDeviceError: If the device index does not exist.
        """

        cls.require_cuda()
        cls._validate_device_index(device_index)

        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(
            device_index
        )

        return CudaMemorySnapshot(
            device_index=device_index,
            free_memory_bytes=free_memory_bytes,
            total_memory_bytes=total_memory_bytes,
            allocated_memory_bytes=torch.cuda.memory_allocated(device_index),
            reserved_memory_bytes=torch.cuda.memory_reserved(device_index),
        )

    @classmethod
    def set_active_device(cls, device_index: int) -> CudaDeviceInfo:
        """
        Set and return the active CUDA device for the current process.

        Args:
            device_index: Zero-based CUDA device index.

        Returns:
            Information for the selected CUDA device.

        Raises:
            CudaUnavailableError: If CUDA is unavailable.
            InvalidCudaDeviceError: If the device index does not exist.
        """

        device = cls.get_device(device_index)
        torch.cuda.set_device(device_index)

        return device

    @staticmethod
    def get_cuda_runtime_version() -> str:
        """Return the CUDA runtime version reported by PyTorch."""

        return str(torch.version.cuda or "Unavailable")

    @staticmethod
    def get_pytorch_version() -> str:
        """Return the installed PyTorch version."""

        return str(torch.__version__)

    @staticmethod
    def _validate_device_index(device_index: int) -> None:
        """
        Validate a CUDA device index.

        Args:
            device_index: Device index to validate.

        Raises:
            InvalidCudaDeviceError: If the value is invalid or unavailable.
        """

        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise InvalidCudaDeviceError(
                "The CUDA device index must be an integer."
            )

        device_count = torch.cuda.device_count()

        if device_index < 0 or device_index >= device_count:
            raise InvalidCudaDeviceError(
                f"CUDA device index {device_index} is invalid. "
                f"Available device indexes: 0 through {device_count - 1}."
            )