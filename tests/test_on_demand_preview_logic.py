"""Behavioral test for Arnesis on-demand preview memory control."""
from __future__ import annotations
import numpy as np
from arnesis.processing.camera_session import CameraSession, CameraSessionConfig

class FakeQueue:
    def __init__(self): self.items=[]
    def put(self, packet): self.items.append(packet)
    def get_latest(self, *args, **kwargs): return self.items[-1] if self.items else None

def main():
    queue=FakeQueue()
    config=CameraSessionConfig(camera_id=991,group_id=992,camera_name="On Demand Test",source_uri="rtsp://test:*****@127.0.0.1/test")
    session=CameraSession(config,queue=queue)
    first=np.zeros((16,16,3),dtype=np.uint8)
    session._publish_frame(first,1.0)
    assert len(queue.items)==1, "Processing packet must continue without preview."
    assert session.preview_subscriber_count==0
    assert session.get_preview_frame() is None
    assert getattr(session,"_preview_packet") is None

    assert session.subscribe_preview()==1
    second=np.ones((16,16,3),dtype=np.uint8)
    session._publish_frame(second,2.0)
    preview=session.get_preview_frame()
    assert len(queue.items)==2, "Processing packets must remain independent."
    assert preview is not None and np.array_equal(preview.frame,second)
    assert preview.frame is not second

    assert session.subscribe_preview()==2
    assert session.unsubscribe_preview()==1
    assert session.get_preview_frame() is not None
    assert session.unsubscribe_preview()==0
    assert session.get_preview_frame() is None
    assert getattr(session,"_preview_packet") is None

    third=np.full((16,16,3),2,dtype=np.uint8)
    session._publish_frame(third,3.0)
    assert len(queue.items)==3, "Background processing must continue after preview closes."
    assert getattr(session,"_preview_packet") is None

    print("[OK] Processing packets continue with zero preview subscribers.")
    print("[OK] No preview cache is created when nobody is watching.")
    print("[OK] Preview cache activates after subscription.")
    print("[OK] Multiple subscribers use reference counting.")
    print("[OK] Last unsubscribe releases the cached preview frame.")
    print("[OK] Background processing continues after preview closes.")

if __name__=="__main__": main()
