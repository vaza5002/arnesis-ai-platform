"""Static validation for the Arnesis Model Registry package."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for rel in ("arnesis/application/model_registry_service.py","arnesis/ui/model_registry_view.py","tools/install_model_registry.py"):
    ast.parse((root/rel).read_text(encoding="utf-8"),filename=rel); print("[OK]",rel)
print("[OK] Model Registry package validation passed.")
