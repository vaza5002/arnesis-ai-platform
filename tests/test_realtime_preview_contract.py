"""Regression test for the real-time camera preview packet contract."""

from __future__ import annotations

import numpy as np

from arnesis.processing.camera_session import CameraSession, CameraSessionConfig
from arnesis.processing.latest_frame_queue import FramePacket, LatestFrameQueue


def main() -> None:
    queue = LatestFrameQueue(capacity=2)
    config = CameraSessionConfig(
        camera_id=101,
        group_id=501,
        camera_name="Preview Contract Test",
        source_uri="rtsp://USER:*****@127.0.0.1:554/Streaming/Channels/101",
    )
    session = CameraSession(config, queue=queue)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    session._publish_frame(frame, captured_at=123.456)

    queued = queue.peek_latest(copy_frame=False)
    assert isinstance(queued, FramePacket), type(queued)
    assert queued.camera_id == 101
    assert queued.sequence == 1
    assert queued.frame is frame

    preview = session.get_preview_frame(copy_frame=True)
    assert preview is not None
    assert preview.camera_id == 101
    assert preview.frame.shape == (48, 64, 3)
    assert preview.frame is not frame

    latest = session.latest_frame()
    assert isinstance(latest, FramePacket)
    assert latest.frame.shape == (48, 64, 3)

    consumed = session.next_frame(timeout_seconds=0.0)
    assert isinstance(consumed, FramePacket)
    assert consumed.sequence == 1

    retained_preview = session.get_preview_frame(copy_frame=False)
    assert retained_preview is not None
    assert retained_preview.frame.shape == (48, 64, 3)

    print("[OK] Typed FramePacket queue contract passed.")
    print("[OK] Non-destructive preview cache passed.")
    print("[OK] Inference queue consumption remains independent.")
    print("[OK] Real-time preview regression test passed.")


if __name__ == "__main__":
    main()
