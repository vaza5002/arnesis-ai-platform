from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arnesis.ui.camera_manager_dialog import CameraManagerDialog
assert 'Authentication failed' in CameraManagerDialog._friendly_connection_error('method DESCRIBE failed: 401 Unauthorized')
assert 'stream path' in CameraManagerDialog._friendly_connection_error('404 Not Found')
assert 'timeout' in CameraManagerDialog._friendly_connection_error('Connection timed out')
print('[OK] Friendly RTSP errors passed.')
