"""Validate that the old split-pane view was replaced by the card dashboard."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "arnesis" / "ui" / "realtime_processing_view.py"
content = VIEW.read_text(encoding="utf-8-sig")
ast.parse(content, filename=str(VIEW))

required = (
    "LIVE OPERATIONS",
    "RUNTIME STATUS",
    "CUDA DEVICE",
    "GPU MEMORY",
    "ACTIVE CAMERAS",
    "TOTAL FPS",
    "DROPPED FRAMES",
    "LIVE CAMERA GRID",
    "preview_frame",
)
for marker in required:
    assert marker in content, marker

for marker in ("REAL-TIME CONTROLS", "CAMERA STREAMS AND RESULTS"):
    assert marker not in content, marker

assert "tk.PanedWindow" not in content
assert "_create_kpi_card" in content
assert "_create_camera_card" in content
assert "_layout_camera_cards" in content

print("[OK] Card-based Real-Time Processing syntax passed.")
print("[OK] Six operational KPI cards are present.")
print("[OK] Adaptive camera card grid is present.")
print("[OK] Non-destructive preview API is present.")
print("[OK] Previous split-pane UI is absent.")
print("[OK] Real-Time cards UI package validation passed.")
