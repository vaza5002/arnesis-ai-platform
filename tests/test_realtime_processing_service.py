"""Offline orchestration test for real-time group and camera lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from arnesis.application.processing_service import ProcessingService
from arnesis.application.realtime_processing_service import PreparedCameraGroup
from arnesis.processing.group_session import GroupSessionConfiguration


@dataclass
class FakeSnapshot:
    group_id: int
    state: str

    def to_dict(self):
        return {"group_id": self.group_id, "state": self.state, "cuda_device": "CUDA:0 - TEST"}


class FakeGroupRuntime:
    def __init__(self):
        self.states = {}

    def contains(self, group_id):
        return group_id in self.states

    def start_group(self, config):
        self.states[config.group_id] = "RUNNING"
        return FakeSnapshot(config.group_id, "RUNNING")

    def get_group(self, group_id):
        return FakeSnapshot(group_id, self.states[group_id])

    def pause_group(self, group_id):
        self.states[group_id] = "PAUSED"
        return FakeSnapshot(group_id, "PAUSED")

    def resume_group(self, group_id):
        self.states[group_id] = "RUNNING"
        return FakeSnapshot(group_id, "RUNNING")

    def stop_group(self, group_id):
        self.states.pop(group_id, None)
        return FakeSnapshot(group_id, "STOPPED")

    def list_groups(self):
        return [FakeSnapshot(group_id, state) for group_id, state in self.states.items()]


class FakeRealtime:
    def __init__(self):
        self.state = "STOPPED"

    def prepare_group(self, group_id):
        return PreparedCameraGroup(group_id, "TEST", (object(), object()))

    def start_prepared_group(self, prepared):
        self.state = "RUNNING"
        return self.group_snapshot(prepared.group_id)

    def pause_group(self, group_id):
        self.state = "PAUSED"
        return self.group_snapshot(group_id)

    def resume_group(self, group_id):
        self.state = "RUNNING"
        return self.group_snapshot(group_id)

    def stop_group(self, group_id):
        self.state = "STOPPED"
        return self.group_snapshot(group_id)

    def stop_all(self):
        self.state = "STOPPED"
        return []

    def group_snapshot(self, group_id):
        return [{"camera_id": 1, "group_id": group_id, "state": self.state}]

    def latest_frame(self, group_id, camera_id, **kwargs):
        return None


class TestProcessingService(ProcessingService):
    def _prepare_group_runtime(self, prepared):
        return GroupSessionConfiguration(
            group_id=prepared.group_id,
            group_code=prepared.group_code,
            group_name="Test Group",
            gpu_index=0,
            gpu_name="TEST GPU",
            requested_memory_mb=1024,
            maximum_streams=len(prepared.camera_configs),
        )

    def _write_group_status(self, group_id, status):
        pass


def main():
    service = TestProcessingService(
        database=None,
        gpu_capacity=None,
        runtime=FakeGroupRuntime(),
        realtime=FakeRealtime(),
    )

    started = service.start_group(1)
    assert started["state"] == "RUNNING"
    assert started["cameras"][0]["state"] == "RUNNING"

    paused = service.pause_group(1)
    assert paused["state"] == "PAUSED"
    assert paused["cameras"][0]["state"] == "PAUSED"

    resumed = service.resume_group(1)
    assert resumed["state"] == "RUNNING"
    assert resumed["cameras"][0]["state"] == "RUNNING"

    stopped = service.stop_group(1)
    assert stopped["state"] == "STOPPED"
    assert stopped["cameras"][0]["state"] == "STOPPED"
    assert service.get_runtime_status() == []

    print("[OK] Real-time ProcessingService lifecycle passed.")


if __name__ == "__main__":
    main()
