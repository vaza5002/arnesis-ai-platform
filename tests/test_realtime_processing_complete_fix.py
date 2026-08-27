"""Offline regression tests for the complete real-time preview fix."""

from __future__ import annotations

import numpy as np

from arnesis.processing.camera_session import CameraSession, CameraSessionConfig
from arnesis.processing.latest_frame_queue import FramePacket, LatestFrameQueue


def main() -> None:
    queue = LatestFrameQueue(capacity=2)

    try:
        queue.put(np.zeros((2, 2, 3), dtype=np.uint8))
    except TypeError:
        pass
    else:
        raise AssertionError("LatestFrameQueue accepted an untyped NumPy frame.")

    config = CameraSessionConfig(
        camera_id=7,
        group_id=307,
        camera_name="Offline Camera",
        source_uri="rtsp://USER:*****@127.0.0.1:554/Streaming/Channels/101",
    )
    session = CameraSession(config, queue=queue)
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    session._publish_frame(frame, captured_at=55.0)

    cached = queue.peek_latest(copy_frame=False)
    assert isinstance(cached, FramePacket)
    assert cached.camera_id == 7
    assert cached.sequence == 1
    assert cached.frame is frame

    preview = session.get_preview_frame(copy_frame=True)
    assert preview is not None
    assert preview.frame.shape == (72, 128, 3)
    assert preview.frame is not frame

    inference = session.next_frame(timeout_seconds=0.0)
    assert isinstance(inference, FramePacket)
    assert inference.sequence == 1

    retained = session.get_preview_frame(copy_frame=False)
    assert retained is not None
    assert retained.frame.shape == (72, 128, 3)

    print("[OK] Raw ndarray rejection passed.")
    print("[OK] Typed FramePacket publication passed.")
    print("[OK] Persistent non-destructive preview passed.")
    print("[OK] Independent inference consumption passed.")
    print("[OK] Complete Real-Time Processing fix validation passed.")


if __name__ == "__main__":
    main()
