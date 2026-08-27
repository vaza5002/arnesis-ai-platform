from arnesis.processing.latest_frame_queue import FramePacket, LatestFrameQueue
q=LatestFrameQueue(2)
for i in range(3): q.put(FramePacket(1,i,0.0,i))
assert q.get_latest(0).frame==2
assert q.dropped_frames==2
print('[OK] Latest-frame queue validation passed.')
