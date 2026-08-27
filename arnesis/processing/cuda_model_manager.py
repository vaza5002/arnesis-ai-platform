"""SHA-verified reference-counted CUDA-only model cache."""
from __future__ import annotations
import hashlib,threading
from pathlib import Path
from arnesis.processing.cuda_device import CudaDeviceService
class CudaModelLease:
    def __init__(self,manager,key,model): self.manager=manager; self.key=key; self.model=model; self.released=False
    @property
    def cuda_device(self): return f"cuda:{self.key[1]}"
    def release(self):
        if not self.released: self.released=True; self.manager.release(self.key)
class CudaModelManager:
    def __init__(self): self._entries={}; self._lock=threading.RLock()
    def acquire(self,path_value,device_index,expected_sha256=None):
        device=CudaDeviceService.set_active_device(int(device_index)); path=Path(path_value).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f"Model file not found: {path}")
        digest=self._hash(path)
        if expected_sha256 and digest.lower()!=expected_sha256.lower(): raise RuntimeError(f"Model SHA-256 mismatch: {path}")
        key=(str(path),device.device_index,digest)
        with self._lock:
            if key in self._entries:
                self._entries[key][1]+=1; return CudaModelLease(self,key,self._entries[key][0])
            from ultralytics import YOLO
            model=YOLO(str(path)); model.to(device.torch_device)
            actual=str(next(model.model.parameters()).device)
            if actual!=device.torch_device: raise RuntimeError(f"Model remained on {actual}; CPU inference is prohibited.")
            self._entries[key]=[model,1]; return CudaModelLease(self,key,model)
    def release(self,key):
        with self._lock:
            if key not in self._entries:return
            self._entries[key][1]-=1
            if self._entries[key][1]>0:return
            del self._entries[key]
        import torch
        with torch.cuda.device(key[1]): torch.cuda.empty_cache()
    @staticmethod
    def _hash(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        return h.hexdigest()
