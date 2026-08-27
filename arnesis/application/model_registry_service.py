"""Persistent model registry and CUDA-only model validation for Arnesis."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import text
from arnesis.processing.cuda_device import CudaDeviceService

ALLOWED_SUFFIXES={".pt",".pth",".onnx",".engine",".torchscript"}
ALLOWED_TYPES={"DETECTION","CLASSIFICATION","POSE"}

@dataclass(frozen=True,slots=True)
class ModelRecord:
    id:int; name:str; version:str; model_type:str; framework:str
    model_path:str; file_sha256:str|None; enabled:bool
    input_size:int|None; notes:str|None

class ModelRegistryService:
    """Manage model metadata while preserving verified absolute paths."""
    def __init__(self,database): self.database=database
    def list_models(self)->list[ModelRecord]:
        with self.database.session_scope() as s:
            rows=s.execute(text("SELECT id,name,version,model_type,framework,model_path,file_sha256,enabled,input_size,notes FROM arn_model_definition ORDER BY name,version")).mappings().all()
            return [ModelRecord(**dict(r)) for r in rows]
    def save(self,*,model_id:int|None,name:str,version:str,model_type:str,framework:str,model_path:str,input_size:int|None,notes:str|None,enabled:bool=True)->ModelRecord:
        path=self._validate_path(model_path); kind=model_type.strip().upper()
        if not name.strip() or not version.strip(): raise ValueError("Model name and version are required.")
        if kind not in ALLOWED_TYPES: raise ValueError("Model type must be DETECTION, CLASSIFICATION, or POSE.")
        digest=self._hash(path); now=datetime.now(timezone.utc)
        with self.database.session_scope() as s:
            if model_id is None:
                result=s.execute(text("INSERT INTO arn_model_definition (name,version,model_type,framework,model_path,file_sha256,enabled,input_size,notes,created_at,updated_at) VALUES (:name,:version,:type,:framework,:path,:sha,:enabled,:size,:notes,:now,:now)"),{"name":name.strip(),"version":version.strip(),"type":kind,"framework":framework.strip(),"path":str(path),"sha":digest,"enabled":enabled,"size":input_size,"notes":notes,"now":now})
                model_id=int(result.lastrowid)
            else:
                s.execute(text("UPDATE arn_model_definition SET name=:name,version=:version,model_type=:type,framework=:framework,model_path=:path,file_sha256=:sha,enabled=:enabled,input_size=:size,notes=:notes,updated_at=:now WHERE id=:id"),{"id":model_id,"name":name.strip(),"version":version.strip(),"type":kind,"framework":framework.strip(),"path":str(path),"sha":digest,"enabled":enabled,"size":input_size,"notes":notes,"now":now})
        return self.get(model_id)
    def get(self,model_id:int)->ModelRecord:
        with self.database.session_scope() as s:
            row=s.execute(text("SELECT id,name,version,model_type,framework,model_path,file_sha256,enabled,input_size,notes FROM arn_model_definition WHERE id=:id"),{"id":model_id}).mappings().one()
            return ModelRecord(**dict(row))
    def delete(self,model_id:int)->None:
        with self.database.session_scope() as s: s.execute(text("DELETE FROM arn_model_definition WHERE id=:id"),{"id":model_id})
    def verify_file(self,model_id:int)->dict[str,object]:
        record=self.get(model_id); path=self._validate_path(record.model_path); digest=self._hash(path)
        return {"path":str(path),"sha256":digest,"matches":digest==record.file_sha256,"size_bytes":path.stat().st_size}
    def validate_cuda_load(self,model_id:int,device_index:int)->dict[str,object]:
        CudaDeviceService.require_cuda(); device=CudaDeviceService.get_device(device_index); record=self.get(model_id); path=self._validate_path(record.model_path)
        if path.suffix.lower() not in {".pt",".pth"}: raise ValueError("CUDA load validation currently supports PyTorch/Ultralytics .pt or .pth models.")
        from ultralytics import YOLO
        model=YOLO(str(path)); model.to(device.torch_device)
        parameter=next(model.model.parameters()); actual=str(parameter.device)
        if not actual.startswith("cuda:"): raise RuntimeError("Model remained on CPU. Arnesis prohibits CPU inference.")
        return {"model_id":record.id,"cuda_device":device.display_name,"parameter_device":actual,"path":str(path)}
    @staticmethod
    def _validate_path(value:str)->Path:
        path=Path(value).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f"Model file not found: {path}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES: raise ValueError(f"Unsupported model extension: {path.suffix}")
        return path
    @staticmethod
    def _hash(path:Path)->str:
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        return h.hexdigest()
