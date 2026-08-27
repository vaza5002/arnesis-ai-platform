"""Integrated Arnesis group, camera, and CUDA inference lifecycle."""
from __future__ import annotations
import threading
from arnesis.application.inference_orchestration_service import InferenceOrchestrationService
from arnesis.application.realtime_processing_service import RealtimeProcessingService
from arnesis.domain.entities import Group,GroupStatus
from arnesis.processing.group_session import GroupSessionConfiguration,SessionState
from arnesis.processing.processing_runtime import ProcessingRuntime
class ProcessingServiceError(RuntimeError):pass
class ProcessingService:
    def __init__(self,database,gpu_capacity,runtime=None,realtime=None,inference=None):
        self.database=database;self.gpu_capacity=gpu_capacity;self.lock=threading.RLock();self.runtime=runtime or ProcessingRuntime(self._persist);self.realtime=realtime or RealtimeProcessingService(database);self.inference=inference or InferenceOrchestrationService(database,self.realtime)
    def start_group(self,group_id):
        if self.runtime.contains(group_id):return self._snapshot(group_id)
        prepared=self.realtime.prepare_group(group_id);config=self._prepare(group_id,len(prepared.camera_configs))
        try:
            result=self.runtime.start_group(config).to_dict();result['cameras']=self.realtime.start_prepared_group(prepared);result['inference']=self.inference.start_group(group_id,config.gpu_index);return result
        except Exception as exc:
            try:self.inference.stop_group(group_id)
            except Exception:pass
            try:self.realtime.stop_group(group_id)
            except Exception:pass
            if self.runtime.contains(group_id):
                try:self.runtime.stop_group(group_id)
                except Exception:pass
            self._write(group_id,GroupStatus.ERROR.value);raise ProcessingServiceError(f"Unable to start group id {group_id}: {type(exc).__name__}: {exc}") from exc
    def pause_group(self,group_id):
        self.inference.pause_group(group_id);cameras=self.realtime.pause_group(group_id);result=self.runtime.pause_group(group_id).to_dict();result.update(cameras=cameras,inference=self.inference.snapshot(group_id));return result
    def resume_group(self,group_id):
        cameras=self.realtime.resume_group(group_id);self.inference.resume_group(group_id);result=self.runtime.resume_group(group_id).to_dict();result.update(cameras=cameras,inference=self.inference.snapshot(group_id));return result
    def stop_group(self,group_id):
        errors=[]
        try:self.inference.stop_group(group_id)
        except Exception as exc:errors.append(str(exc))
        try:cameras=self.realtime.stop_group(group_id)
        except Exception as exc:cameras=[];errors.append(str(exc))
        result=self.runtime.stop_group(group_id).to_dict() if self.runtime.contains(group_id) else {'group_id':group_id,'state':'STOPPED'};result['cameras']=cameras;result['inference']=[];self._write(group_id,'ERROR' if errors else 'STOPPED')
        if errors:raise ProcessingServiceError('; '.join(errors))
        return result
    def stop_all(self):return [self.stop_group(int(x['group_id'])) for x in self.get_runtime_status()]
    def get_runtime_status(self):return [self._snapshot(x.group_id) for x in self.runtime.list_groups()]
    def _snapshot(self,group_id):
        result=self.runtime.get_group(group_id).to_dict();result['cameras']=self.realtime.group_snapshot(group_id);result['inference']=self.inference.snapshot(group_id);return result
    def subscribe_preview(self,g,c):return self.realtime.subscribe_preview(g,c)
    def unsubscribe_preview(self,g,c):return self.realtime.unsubscribe_preview(g,c)
    def preview_frame(self,g,c):return self.realtime.preview_frame(g,c)
    def latest_inference_result(self,g,c,after_sequence=None):return self.inference.latest_result(g,c,after_sequence)
    def _prepare(self,group_id,streams):
        with self.database.session_scope() as session:
            group=session.get(Group,group_id)
            if group is None:raise ProcessingServiceError(f"Group id {group_id} was not found.")
            allocation=self.gpu_capacity.select_device(session,requested_memory_mb=group.max_gpu_memory_mb,requested_streams=max(1,streams),preferred_gpu_index=group.preferred_gpu_index);group.preferred_gpu_index=allocation.device_index;group.status='STARTING'
            return GroupSessionConfiguration(group.id,group.code,group.name,allocation.device_index,allocation.device_name,group.max_gpu_memory_mb,max(1,streams))
    def _persist(self,group_id,state:SessionState,error):self._write(group_id,state.value)
    def _write(self,group_id,status):
        with self.lock:
            with self.database.session_scope() as session:
                group=session.get(Group,group_id)
                if group is not None:group.status=status
