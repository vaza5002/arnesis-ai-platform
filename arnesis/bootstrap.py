"""Application bootstrap for the Arnesis real-time platform."""

from __future__ import annotations

from dataclasses import dataclass

from arnesis.application.configuration_service import ConfigurationService
from arnesis.core.database import DatabaseManager
from arnesis.processing.gpu_capacity_manager import GpuCapacityManager


@dataclass(slots=True)
class ApplicationContext:
    database: DatabaseManager
    configuration: ConfigurationService
    gpu_capacity: GpuCapacityManager

    def close(self) -> None:
        self.database.dispose()


def bootstrap_application(database_url: str | None = None) -> ApplicationContext:
    """Create the schema, require CUDA, and synchronize configured GPU devices."""
    database = DatabaseManager(database_url)
    database.create_schema()

    gpu_capacity = GpuCapacityManager()
    gpu_capacity.require_cuda()

    with database.session_scope() as session:
        gpu_capacity.synchronize_devices(session)

    return ApplicationContext(
        database=database,
        configuration=ConfigurationService(database),
        gpu_capacity=gpu_capacity,
    )


def main() -> int:
    context = bootstrap_application()
    try:
        with context.database.session_scope() as session:
            devices = context.gpu_capacity.synchronize_devices(session)
            print("[OK] Arnesis bootstrap completed.")
            print(f"[OK] Database: {context.database.database_url}")
            for device in devices:
                print(
                    f"[OK] CUDA:{device.device_index} - {device.device_name} | "
                    f"memory limit={device.maximum_memory_percent:.1f}% | "
                    f"groups={device.maximum_groups} | streams={device.maximum_streams}"
                )
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
