from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arnesis.processing.latest_frame_queue import FramePacket, LatestFrameQueue
q = LatestFrameQueue(2)
q.put(FramePacket(5, 10, 50.0, bytearray(b"frame")))
assert q.peek_latest() is not None
assert q.peek_latest() is not None
assert q.get_latest(0) is not None
assert q.peek_latest() is not None
print("[OK] Persistent preview cache test passed.")
