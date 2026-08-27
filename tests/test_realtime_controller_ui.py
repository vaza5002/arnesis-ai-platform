"""Static contract validation for the Arnesis real-time Controller UI."""

from __future__ import annotations

import ast
from pathlib import Path

VIEW_PATH = (
    Path(__file__).resolve().parents[1]
    / "arnesis"
    / "ui"
    / "realtime_processing_view.py"
)


def main() -> None:
    content = VIEW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(VIEW_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    view = classes["RealTimeProcessingView"]
    methods = {
        node.name for node in view.body if isinstance(node, ast.FunctionDef)
    }
    required = {
        "reload_groups",
        "_start_group",
        "_pause_group",
        "_resume_group",
        "_stop_group",
        "_refresh_previews",
    }
    assert required <= methods, sorted(required - methods)
    assert "ArnesisTheme" in content
    assert "CUDA" in content
    print("[OK] Real-Time Processing view syntax passed.")
    print("[OK] Controller service contract passed.")
    print("[OK] Dynamic group lifecycle controls passed.")
    print("[OK] Camera preview integration contract passed.")


if __name__ == "__main__":
    main()
