"""Camera pipeline composed of capture and latest-frame buffering."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from arnesis.processing.camera_capture import (
    CameraCaptureConfiguration,
    CameraCaptureWorker,
)
from arnesis.processing.frame_buffer import FramePacket, LatestFrameBuffer


@dataclass(frozen=True, slots=True)
class CameraPipelineSnapshot:
    capture: dict[str, object]
    buffer: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CameraPipeline:
    """Public lifecycle and frame-consumption API for one camera."""

    def __init__(
        self,
        configuration: CameraCaptureConfiguration,
        buffer_capacity: int = 2,
        copy_frames: bool = False,
    ) -> None:
        self.configuration = configuration
        self.buffer = LatestFrameBuffer(buffer_capacity, copy_frames)
        self.capture = CameraCaptureWorker(configuration, self.buffer)

    def start(self) -> CameraPipelineSnapshot:
        self.capture.start()
        return self.snapshot()

    def pause(self) -> CameraPipelineSnapshot:
        self.capture.pause()
        return self.snapshot()

    def resume(self) -> CameraPipelineSnapshot:
        self.capture.resume()
        return self.snapshot()

    def stop(self) -> CameraPipelineSnapshot:
        self.capture.stop()
        self.buffer.close()
        return self.snapshot()

    def get_latest_frame(
        self,
        timeout_seconds: float | None = 1.0,
        after_sequence: int | None = None,
    ) -> FramePacket | None:
        return self.buffer.get_latest(timeout_seconds, after_sequence)

    def snapshot(self) -> CameraPipelineSnapshot:
        return CameraPipelineSnapshot(
            capture=self.capture.snapshot().to_dict(),
            buffer=asdict(self.buffer.snapshot()),
        )

    def __enter__(self) -> CameraPipeline:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
