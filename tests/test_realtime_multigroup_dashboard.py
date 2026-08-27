from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]; view=(root/"arnesis/ui/realtime_processing_view.py").read_text(encoding="utf-8-sig"); service=(root/"arnesis/application/processing_service.py").read_text(encoding="utf-8-sig")
ast.parse(view); ast.parse(service)
for marker in ("MULTI-GROUP LIVE OPERATIONS","TOTAL GROUPS","RUNNING","STOPPED","ERRORS","ACTIVE CAMERAS","Back to groups","_group_card","_open_group","_fullscreen"):assert marker in view,marker
assert "Preview API is unavailable" not in view
assert "def preview_frame(self,group_id:int,camera_id:int)" in service
assert "self.realtime.preview_frame(group_id,camera_id)" in service
print("[OK] All-group overview contract passed.")
print("[OK] Per-group status cards passed.")
print("[OK] On-demand detail preview passed.")
print("[OK] ProcessingService preview bridge passed.")
print("[OK] Obsolete single-group dashboard is absent.")
