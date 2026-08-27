"""Validate non-destructive preview-frame support without opening a camera."""

from __future__ import annotations

import inspect

import numpy as np

from arnesis.processing.camera_runtime import CameraRuntime
from arnesis.processing.camera_session import CameraSession, CameraSessionConfig


class FakeQueue:
    """Record publications while proving preview reads do not consume items."""

    def __init__(self) -> None:
        self.publish_count = 0
        self.get_count = 0

    def publish(self, frame, captured_at=None) -> None:
        self.publish_count += 1

    def get_latest(self, timeout_seconds=0.0, after_sequence=None):
        self.get_count += 1
        return None


def main() -> None:
    queue = FakeQueue()
    config = CameraSessionConfig(
        camera_id=901,
        group_id=902,
        camera_name="Preview Hotfix Test",
        source_uri="rtsp://test:*****@127.0.0.1/test",
    )
    session = CameraSession(config, queue=queue)
    frame = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

    session._publish_frame(frame, captured_at=123.45)
    preview = session.get_preview_frame(copy_frame=True)

    assert preview is not None
    assert preview.camera_id == 901
    assert preview.captured_at == 123.45
    assert np.array_equal(preview.frame, frame)
    assert preview.frame is not frame
    assert queue.publish_count == 1
    assert queue.get_count == 0
    assert "preview_frame" in dir(CameraRuntime)
    assert tuple(inspect.signature(CameraRuntime.preview_frame).parameters) == (
        "self",
        "camera_id",
        "copy_frame",
    )

    print("[OK] Preview frame is cached independently from the inference queue.")
    print("[OK] Preview retrieval returns a defensive frame copy.")
    print("[OK] Inference queue consumption count remained zero.")
    print("[OK] CameraRuntime preview API contract passed.")


if __name__ == "__main__":
    main()
