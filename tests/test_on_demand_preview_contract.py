"""Static cross-layer contract validation for on-demand preview."""
from __future__ import annotations
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=(
"arnesis/processing/camera_session.py",
"arnesis/processing/camera_runtime.py",
"arnesis/application/realtime_processing_service.py",
"arnesis/application/processing_service.py",
"arnesis/ui/realtime_processing_view.py",
)
content={}
for rel in files:
    text=(ROOT/rel).read_text(encoding="utf-8-sig"); ast.parse(text,filename=rel); content[rel]=text
assert "preview_subscriber_count" in content[files[0]]
assert "preview_enabled = self._preview_subscribers > 0" in content[files[0]]
assert "def subscribe_preview(" in content[files[1]]
assert "def unsubscribe_preview(" in content[files[2]]
assert "return self.realtime.subscribe_preview" in content[files[3]]
assert "_activate_preview_subscriptions" in content[files[4]]
assert "_release_preview_subscriptions" in content[files[4]]
assert "self._release_preview_subscriptions()" in content[files[4]]
print("[OK] CameraSession subscription gate is present.")
print("[OK] Runtime and service subscription bridge is present.")
print("[OK] Group detail activation and release hooks are present.")
print("[OK] On-demand preview cross-layer contract passed.")
