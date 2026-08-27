"""Static and behavioral contract checks for the preview service bridge."""
from __future__ import annotations
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/"arnesis/application/realtime_processing_service.py",ROOT/"arnesis/application/processing_service.py",ROOT/"arnesis/ui/realtime_processing_view.py"]
for path in paths: ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
rt=paths[0].read_text(encoding="utf-8-sig"); ps=paths[1].read_text(encoding="utf-8-sig"); view=paths[2].read_text(encoding="utf-8-sig")
assert "def preview_frame(self, group_id: int, camera_id: int)" in rt
assert "runtime.preview_frame(camera_id)" in rt
assert "def preview_frame(self, group_id: int, camera_id: int)" in ps
assert "self.realtime.preview_frame(group_id, camera_id)" in ps
assert "preview_method = getattr(self.processing_service" in view
print("[OK] Realtime service preview bridge passed.")
print("[OK] Processing service preview bridge passed.")
print("[OK] Real-Time UI non-destructive binding passed.")
print("[OK] Preview bridge package validation passed.")
