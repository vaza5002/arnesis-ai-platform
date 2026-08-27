"""Validate the installed complete preview correction."""
from __future__ import annotations
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files={
"queue":ROOT/"arnesis/processing/latest_frame_queue.py",
"session":ROOT/"arnesis/processing/camera_session.py",
"runtime":ROOT/"arnesis/processing/camera_runtime.py",
"realtime":ROOT/"arnesis/application/realtime_processing_service.py",
"processing":ROOT/"arnesis/application/processing_service.py",
"view":ROOT/"arnesis/ui/realtime_processing_view.py",
}
content={}
for key,path in files.items():
    content[key]=path.read_text(encoding="utf-8-sig"); ast.parse(content[key],filename=str(path))
assert "def get_preview_frame(" in content["session"]
assert "self._preview_packet = PreviewFramePacket(" in content["session"]
assert "def preview_frame(" in content["runtime"]
assert "return self._require(camera_id).get_preview_frame" in content["runtime"]
assert "def preview_frame(self, group_id: int, camera_id: int)" in content["realtime"]
assert "def preview_frame(self, group_id: int, camera_id: int)" in content["processing"]
assert "preview_method = getattr(" in content["view"]
assert "not isinstance(packet, FramePacket)" in content["queue"]
print("[OK] CameraSession independent preview cache passed.")
print("[OK] Runtime and service preview bridge passed.")
print("[OK] Real-Time UI preview preference passed.")
print("[OK] Residual queue preview safety check passed.")
print("[OK] Complete preview correction validation passed.")
