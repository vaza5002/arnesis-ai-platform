import sys
import argparse
import os
from pathlib import Path
from typing import Optional
import threading
import time
import numpy as np
import json
import csv
import re
import shutil
import random
import io
from collections import OrderedDict
from datetime import datetime, timedelta
from queue import Queue

import atexit
import signal

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

try:
	import torch
	TORCH_AVAILABLE = True
except Exception:
	TORCH_AVAILABLE = False
	torch = None

try:
	from ultralytics import YOLO
	YOLO_AVAILABLE = True
except Exception:
	YOLO_AVAILABLE = False
	YOLO = None

try:
	from PIL import Image, ImageTk  # Pillow for image handling
	from PIL import ImageOps
except Exception:  # Pillow is listed in project deps; fallback if unavailable
	Image = None
	ImageTk = None
	ImageOps = None

# Optional MediaPipe for ergonomics (pose estimation)
try:
	import mediapipe as mp
	MP_AVAILABLE = True
	mp_pose = mp.solutions.pose
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles
except Exception:
	MP_AVAILABLE = False
	mp_pose = None
	mp_drawing = None
	mp_drawing_styles = None

# Optional drag-and-drop support (tkinterdnd2)
try:
	from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
	DND_AVAILABLE = True
except Exception:
	DND_AVAILABLE = False

                                                                                                  
APP_TITLE = "Arnesis — Consolidation Tool"
# Small, square window
WINDOW_WIDTH = 520
WINDOW_HEIGHT = 520
LOGO_MAX_WIDTH = 750  # scale for main logo

# Color palette - New design
BG_COLOR = "#02234e"       # main background
FG_COLOR = "#E6EEF9"       # light text
BUTTON_BG = "#021e44"      # button normal state
BUTTON_HOVER = "#043c86"   # button hover state
BUTTON_ACTIVE = "#065ed4"  # button active/pressed state
# Legacy colors kept for training/processing windows
GRAY_BG = "#1A3A7A"        
NAVY_HOVER = "#0B2D7A"     
GRAY_HOVER = "#224C9A"

# Global hook to invoke processing starter without early binding issues
BEGIN_PROCESSING_FN = None

# Global context for begin_processing function
# This dictionary holds references to functions and variables needed by begin_processing
PROCESSING_CONTEXT = {
	"processing_type": None,
	"begin_rt_processing": None,
	"launch_real_pipeline": None,
	"progress_var": None,
	"update_progress_bar": None,
	"_update_processing_nav_state": None,
	"_append_log_line": None
}


# ============================================================================
# Distributed Worker/Controller Control Plane (MVP)
# ============================================================================
WORKER_CONTROL_STATE = {
	"node_id": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "worker-node",
	"role": "worker",
	"processing": {
		"rt_running": False,
		"mode": "normal",
		"camera_count": 0,
		"groups": [],
	},
	"callbacks": {
		"set_light_mode": None,
		"stop_processing": None,
	},
	"last_update": "",
}

KNOWN_CONTROL_PANEL_GROUPS = {
	"501", "502", "504", "404", "406", "204", "508", "311", "407", "507", "312", "FOAMV2",
}

WORKER_CONTROL_SERVER = {
	"server": None,
	"thread": None,
	"port": None,
}

LIVE_RT_FRAMES = {}
LIVE_RT_FRAMES_LOCK = threading.Lock()

# Dashboard child processes tracked here so we can clean them on exit
DASHBOARD_CHILD_PROCS = []

# Background reporter processes (entries/exits) tracked here
REPORTER_CHILD_PROCS = []


def _cleanup_dashboard_children():
	"""Terminate any dashboard processes we spawned. Safe to call multiple times."""
	try:
		# Try to terminate tracked processes
		for p in list(DASHBOARD_CHILD_PROCS):
			try:
				if p is None:
					continue
				# if process still running
				alive = getattr(p, 'poll', lambda: 1)()
				if alive is None:
					try:
						p.terminate()
					except Exception:
						try:
							p.kill()
						except Exception:
							pass
			except Exception:
				pass
		# Try to terminate any reporter processes we spawned
		for p in list(REPORTER_CHILD_PROCS):
			try:
				if p is None:
					continue
				alive = getattr(p, 'poll', lambda: 1)()
				if alive is None:
					try:
						p.terminate()
					except Exception:
						try:
							p.kill()
						except Exception:
							pass
			except Exception:
				pass
		# also try to terminate any streamlit_process global if present
		try:
			sp = globals().get('streamlit_process')
			if isinstance(sp, dict):
				p = sp.get('p')
				if p is not None and getattr(p, 'poll', lambda: 1)() is None:
					try:
						p.terminate()
					except Exception:
						try:
							p.kill()
						except Exception:
							pass
		except Exception:
			pass
		# also try to terminate any reporter_process global if present
		try:
			rp = globals().get('reporter_process')
			if isinstance(rp, dict):
				p = rp.get('p')
				if p is not None and getattr(p, 'poll', lambda: 1)() is None:
					try:
						p.terminate()
					except Exception:
						try:
							p.kill()
						except Exception:
							pass
		except Exception:
			pass
	except Exception:
		pass


# Register cleanup at interpreter exit
atexit.register(_cleanup_dashboard_children)

# Also try to handle signals gracefully
def _signal_cleanup_handler(signum, frame):
	try:
		_cleanup_dashboard_children()
	finally:
		try:
			sys.exit(0)
		except Exception:
			os._exit(0)

for sig in ('SIGINT', 'SIGTERM'):
	if hasattr(signal, sig):
		try:
			signal.signal(getattr(signal, sig), _signal_cleanup_handler)
		except Exception:
			pass


def _set_worker_processing_state(rt_running: Optional[bool] = None,
								mode: Optional[str] = None,
								camera_count: Optional[int] = None,
								groups: Optional[list] = None) -> None:
	"""Update runtime worker processing state for remote controller visibility."""
	try:
		if rt_running is not None:
			WORKER_CONTROL_STATE["processing"]["rt_running"] = bool(rt_running)
		if mode is not None:
			WORKER_CONTROL_STATE["processing"]["mode"] = "light" if str(mode).lower() == "light" else "normal"
		if camera_count is not None:
			WORKER_CONTROL_STATE["processing"]["camera_count"] = int(camera_count)
		if groups is not None:
			normalized = []
			seen = set()
			for g in groups:
				g_norm = str(g).strip().upper()
				if not g_norm or g_norm in seen:
					continue
				seen.add(g_norm)
				normalized.append(g_norm)
			WORKER_CONTROL_STATE["processing"]["groups"] = normalized
		WORKER_CONTROL_STATE["last_update"] = datetime.now().isoformat(timespec="seconds")
	except Exception:
		pass


def _get_control_auth_token() -> str:
	"""Get shared token used to secure controller/worker requests."""
	try:
		return str(os.environ.get("ARNESIS_CONTROL_TOKEN", "")).strip()
	except Exception:
		return ""


def _update_live_rt_frame(stream_id: str, camera_name: str, frame_bgr, camera_group: str = "") -> None:
	"""Store latest processed RT frame in memory for controller pull."""
	try:
		ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
		if not ok:
			print(f"[WORKER_FRAMES] encode_failed stream_id={stream_id}")
			return

		payload = encoded.tobytes()
		h, w = frame_bgr.shape[:2]
		with LIVE_RT_FRAMES_LOCK:
			LIVE_RT_FRAMES[str(stream_id)] = {
				"jpeg": payload,
				"camera": camera_name,
				"group": str(camera_group or "").strip(),
				"updated_at": datetime.now().isoformat(timespec="seconds"),
				"width": int(w),
				"height": int(h),
			}
	except Exception as e:
		print(f"[WORKER_FRAMES] update_failed stream_id={stream_id}: {e}")


def _collect_worker_frames_info() -> list:
	"""Collect metadata for processed RT frames (no image payload)."""
	frames = []
	try:
		with LIVE_RT_FRAMES_LOCK:
			for stream_id, payload in LIVE_RT_FRAMES.items():
				if not isinstance(payload, dict):
					continue
				frame_bytes = payload.get("jpeg") or b""
				frames.append({
					"stream_id": str(stream_id),
					"frame_url": f"/frame/{stream_id}",
					"updated_at": payload.get("updated_at", ""),
					"camera": payload.get("camera", ""),
					"group": payload.get("group", ""),
					"frame_bytes": len(frame_bytes),
					"width": payload.get("width", 0),
					"height": payload.get("height", 0),
				})

		print(f"[WORKER_FRAMES] discovered={len(frames)} in-memory streams")
	except Exception:
		print("[WORKER_FRAMES] exception while collecting frames")
		pass
	return frames


def _http_bytes_request(url: str, timeout: float = 2.0, auth_token: Optional[str] = None) -> bytes:
	"""Small bytes HTTP helper for controller-to-worker frame download."""
	import urllib.request

	headers = {}
	if auth_token:
		headers["Authorization"] = f"Bearer {auth_token}"

	req = urllib.request.Request(url=url, headers=headers, method="GET")
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return resp.read()


def _http_json_request(url: str, method: str = "GET", payload: Optional[dict] = None,
					   timeout: float = 1.5, auth_token: Optional[str] = None) -> dict:
	"""Small JSON HTTP helper for controller-to-worker communication."""
	import urllib.request
	import urllib.error

	headers = {"Content-Type": "application/json"}
	if auth_token:
		headers["Authorization"] = f"Bearer {auth_token}"
	data = None
	if payload is not None:
		data = json.dumps(payload).encode("utf-8")

	req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		raw = resp.read().decode("utf-8", errors="replace")
		if not raw.strip():
			return {}
		return json.loads(raw)


def _start_worker_control_server(port: int = 8765) -> None:
	"""Start local worker control HTTP server (status + commands)."""
	if WORKER_CONTROL_SERVER["server"] is not None:
		return

	from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

	class _WorkerControlHandler(BaseHTTPRequestHandler):
		def _send_json(self, code: int, obj: dict):
			payload = json.dumps(obj).encode("utf-8")
			self.send_response(code)
			self.send_header("Content-Type", "application/json; charset=utf-8")
			self.send_header("Content-Length", str(len(payload)))
			self.end_headers()
			self.wfile.write(payload)

		def _send_bytes(self, code: int, payload: bytes, content_type: str = "application/octet-stream"):
			self.send_response(code)
			self.send_header("Content-Type", content_type)
			self.send_header("Content-Length", str(len(payload)))
			self.end_headers()
			self.wfile.write(payload)

		def _is_auth_ok(self) -> bool:
			expected = _get_control_auth_token()
			if not expected:
				return True

			provided = str(self.headers.get("Authorization", "")).strip()
			if provided.lower().startswith("bearer "):
				provided = provided[7:].strip()
			else:
				provided = str(self.headers.get("X-Arnesis-Token", "")).strip()
			return provided == expected

		def do_GET(self):
			if not self._is_auth_ok():
				print(f"[WORKER_HTTP] unauthorized GET path={self.path}")
				self._send_json(401, {"ok": False, "error": "unauthorized"})
				return

			clean_path = self.path.split("?", 1)[0]
			if clean_path.startswith("/frame/"):
				stream_id = clean_path[len("/frame/"):].strip().strip("/")
				if not stream_id:
					print("[WORKER_HTTP] /frame requested without stream_id")
					self._send_json(400, {"ok": False, "error": "missing_stream_id"})
					return

				with LIVE_RT_FRAMES_LOCK:
					entry = LIVE_RT_FRAMES.get(str(stream_id))

				if not isinstance(entry, dict) or not entry.get("jpeg"):
					print(f"[WORKER_HTTP] frame_not_found stream_id={stream_id} in-memory")
					self._send_json(404, {"ok": False, "error": "frame_not_found"})
					return
				try:
					payload = entry.get("jpeg") or b""
					print(f"[WORKER_HTTP] serving frame stream_id={stream_id} bytes={len(payload)} from-memory")
					self._send_bytes(200, payload, content_type="image/jpeg")
				except Exception as e:
					print(f"[WORKER_HTTP] error serving frame stream_id={stream_id}: {e}")
					self._send_json(500, {"ok": False, "error": str(e)})
				return

			if clean_path.rstrip("/") == "/status":
				self._send_json(200, {
					"ok": True,
					"node_id": WORKER_CONTROL_STATE.get("node_id"),
					"role": WORKER_CONTROL_STATE.get("role"),
					"processing": WORKER_CONTROL_STATE.get("processing", {}),
					"last_update": WORKER_CONTROL_STATE.get("last_update", ""),
				})
				return
			if clean_path.rstrip("/") == "/frames":
				frames_info = _collect_worker_frames_info()
				print(f"[WORKER_HTTP] /frames requested returning={len(frames_info)}")
				self._send_json(200, {
					"ok": True,
					"frames": frames_info,
				})
				return
			self._send_json(404, {"ok": False, "error": "not_found"})

		def do_POST(self):
			if not self._is_auth_ok():
				self._send_json(401, {"ok": False, "error": "unauthorized"})
				return

			if self.path.rstrip("/") == "/set_mode":
				try:
					length = int(self.headers.get("Content-Length", "0"))
					raw = self.rfile.read(length) if length > 0 else b"{}"
					body = json.loads(raw.decode("utf-8", errors="replace") or "{}")
					mode = str(body.get("mode", "normal")).strip().lower()
					if mode not in {"light", "normal"}:
						self._send_json(400, {"ok": False, "error": "invalid_mode"})
						return

					cb = WORKER_CONTROL_STATE.get("callbacks", {}).get("set_light_mode")
					if callable(cb):
						cb(mode == "light")
					_set_worker_processing_state(mode=mode)
					self._send_json(200, {"ok": True, "mode": mode})
				except Exception as e:
					self._send_json(500, {"ok": False, "error": str(e)})
				return

			if self.path.rstrip("/") == "/stop":
				try:
					cb = WORKER_CONTROL_STATE.get("callbacks", {}).get("stop_processing")
					if callable(cb):
						cb()
					_set_worker_processing_state(rt_running=False)
					self._send_json(200, {"ok": True})
				except Exception as e:
					self._send_json(500, {"ok": False, "error": str(e)})
				return

			if self.path.rstrip("/") == "/start_group":
				try:
					length = int(self.headers.get("Content-Length", "0"))
					raw = self.rfile.read(length) if length > 0 else b"{}"
					body = json.loads(raw.decode("utf-8", errors="replace") or "{}")
					group = str(body.get("group", "")).strip().upper()
					if not group:
						self._send_json(400, {"ok": False, "error": "missing_group"})
						return
					cb = WORKER_CONTROL_STATE.get("callbacks", {}).get("start_group")
					if callable(cb):
						cb(group)
						self._send_json(200, {"ok": True, "group": group})
					else:
						self._send_json(503, {"ok": False, "error": "no_start_group_callback"})
				except Exception as e:
					self._send_json(500, {"ok": False, "error": str(e)})
				return

			self._send_json(404, {"ok": False, "error": "not_found"})

		def log_message(self, format, *args):
			return

	try:
		server = ThreadingHTTPServer(("0.0.0.0", int(port)), _WorkerControlHandler)
		thread = threading.Thread(target=server.serve_forever, daemon=True)
		thread.start()
		WORKER_CONTROL_SERVER["server"] = server
		WORKER_CONTROL_SERVER["thread"] = thread
		WORKER_CONTROL_SERVER["port"] = int(port)
	except Exception:
		pass


# ============================================================================
# PandasAI Analysis Functions (using subprocess to isolated environment)
# ============================================================================

def call_pandasai_script(csv_path, question):
	"""Call external PandasAI script in isolated conda environment."""
	import subprocess
	import json
	
	# Get the directory where this script is located
	script_dir = os.path.dirname(os.path.abspath(__file__))
	pandasai_script = os.path.join(script_dir, "llm_query.py")
	
	if not os.path.exists(pandasai_script):
		return {
			"success": False,
			"error": f"Script llm_query.py no encontrado en {script_dir}"
		}
	
	try:
		# Use conda run for reliable environment execution with correct argparse format
		cmd = [
			"conda", "run", "-n", "pandasai_env",
			"python", pandasai_script,
			"--query", question,
			"--archivo", csv_path
		]
		
		# Execute command
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=120  # 2 minutes timeout per query
		)
		
		if result.returncode != 0:
			return {
				"success": False,
				"error": f"Error ejecutando script (código {result.returncode}): {result.stderr}"
			}
		
		# Parse JSON output
		output = result.stdout.strip()
		if not output:
			return {
				"success": False,
				"error": "Sin respuesta del script PandasAI"
			}
		
		try:
			# Ensure UTF-8 encoding is preserved
			response = json.loads(output, strict=False)
			return response
		except json.JSONDecodeError as je:
			return {
				"success": False,
				"error": f"Error parseando respuesta JSON: {je}\nOutput: {output}"
			}
		
	except subprocess.TimeoutExpired:
		return {
			"success": False,
			"error": "Timeout: La consulta tardó más de 2 minutos"
		}
	except Exception as e:
		import traceback
		return {
			"success": False,
			"error": f"Error ejecutando subprocess: {e}\n{traceback.format_exc()}"
		}

def call_llm_query(question, csv_path=None, system_prompt=None, temperature=0.8, conversation_history=None):
	"""Call LLM query script with optional CSV file and conversation history."""
	import subprocess
	import json
	
	print("\n[DEBUG - call_llm_query] ===================")
	print(f"[DEBUG - call_llm_query] Question: {question[:100]}..." if len(question) > 100 else f"[DEBUG - call_llm_query] Question: {question}")
	print(f"[DEBUG - call_llm_query] CSV Path: {csv_path}")
	print(f"[DEBUG - call_llm_query] System Prompt Length: {len(system_prompt) if system_prompt else 0}")
	print(f"[DEBUG - call_llm_query] Conversation History: {len(conversation_history) if conversation_history else 0} messages")
	print(f"[DEBUG - call_llm_query] Temperature: {temperature}")
	
	script_dir = os.path.dirname(os.path.abspath(__file__))
	llm_script = os.path.join(script_dir, "llm_query.py")
	
	if not os.path.exists(llm_script):
		print(f"[DEBUG - call_llm_query] ERROR: Script no encontrado en {script_dir}")
		return {
			"success": False,
			"error": f"Script llm_query.py no encontrado en {script_dir}",
			"error_type": "script_not_found"
		}
	
	try:
		cmd = [
			"conda", "run", "-n", "pandasai_env",
			"python", llm_script,
			"--query", question
		]
		
		if csv_path:
			cmd.extend(["--archivo", csv_path])
		if system_prompt:
			cmd.extend(["--system_prompt", system_prompt])
		if conversation_history:
			# Send conversation history as JSON
			cmd.extend(["--conversation_history", json.dumps(conversation_history, ensure_ascii=False)])
		if temperature != 0.8:
			cmd.extend(["--temperature", str(temperature)])
		
		print(f"[DEBUG - call_llm_query] Command: {' '.join(cmd[:6])}... (truncated)")
		
		print("[DEBUG - call_llm_query] Ejecutando subprocess...")
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=60  # 1 minute timeout
		)
		
		print(f"[DEBUG - call_llm_query] Return code: {result.returncode}")
		print(f"[DEBUG - call_llm_query] Stdout length: {len(result.stdout)}")
		print(f"[DEBUG - call_llm_query] Stderr length: {len(result.stderr)}")
		
		if result.returncode != 0:
			print(f"[DEBUG - call_llm_query] ERROR: {result.stderr[:200]}")
			# Check if it's a connection error
			if "ConnectionError" in result.stderr or "connection" in result.stderr.lower() or "refused" in result.stderr.lower():
				return {
					"success": False,
					"error": "CONNECTION_ERROR",
					"error_type": "connection"
				}
			return {
				"success": False,
				"error": f"Error ejecutando script (código {result.returncode}): {result.stderr}",
				"error_type": "script_error"
			}
		
		output = result.stdout.strip()
		if not output:
			print("[DEBUG - call_llm_query] ERROR: Sin salida del script")
			return {
				"success": False,
				"error": "Sin respuesta del script LLM",
				"error_type": "empty_response"
			}
		
		try:
			parsed = json.loads(output)
			print(f"[DEBUG - call_llm_query] Success: {parsed.get('success')}")
			if parsed.get('success'):
				result_preview = parsed.get('result', '')[:100]
				print(f"[DEBUG - call_llm_query] Result preview: {result_preview}...")
			else:
				print(f"[DEBUG - call_llm_query] Error in response: {parsed.get('error', 'Unknown')}")
			print("[DEBUG - call_llm_query] ===================")
			return parsed
		except json.JSONDecodeError as je:
			print(f"[DEBUG - call_llm_query] JSON ERROR: {je}")
			print(f"[DEBUG - call_llm_query] Output: {output[:200]}...")
			return {
				"success": False,
				"error": f"Error parseando respuesta JSON: {je}\nOutput: {output}",
				"error_type": "json_error"
			}
	
	except subprocess.TimeoutExpired:
		print("[DEBUG - call_llm_query] TIMEOUT: Consulta tardó más de 1 minuto")
		return {
			"success": False,
			"error": "TIMEOUT",
			"error_type": "timeout"
		}
	except Exception as e:
		import traceback
		print(f"[DEBUG - call_llm_query] EXCEPTION: {e}")
		print(f"[DEBUG - call_llm_query] Traceback: {traceback.format_exc()[:300]}")
		# Check if it's a connection-related error
		if "ConnectionError" in str(e) or "connection" in str(e).lower():
			return {
				"success": False,
				"error": "CONNECTION_ERROR",
				"error_type": "connection"
			}
		return {
			"success": False,
			"error": f"Error ejecutando subprocess: {e}\n{traceback.format_exc()}",
			"error_type": "general_error"
		}

def analyze_productivity_by_roi(csv_path):
	"""Análisis 1: Productividad por Estacion."""
	question = "¿Cuál Station tiene mayor indice de VA?"
	response = call_pandasai_script(csv_path, question)
	
	if response.get("success"):
		return response.get("result", "No se obtuvo respuesta")
	else:
		return f"Error: {response.get('error', 'Error desconocido')}"

def analyze_temporal_patterns(csv_path):
	"""Análisis 2: Patrones temporales."""
	question = "¿A qué hora del día hay más actividades NVA?"
	response = call_pandasai_script(csv_path, question)
	
	if response.get("success"):
		return response.get("result", "No se obtuvo respuesta")
	else:
		return f"Error: {response.get('error', 'Error desconocido')}"

def analyze_ergonomics_correlation(csv_path):
	"""Análisis 3: Correlación de ergonomía."""
	question = "¿Hay correlación entre Station y postura incorrecta (NG)?"
	response = call_pandasai_script(csv_path, question)
	
	if response.get("success"):
		return response.get("result", "No se obtuvo respuesta")
	else:
		return f"Error: {response.get('error', 'Error desconocido')}"

def generate_recommendations(csv_path):
	"""Análisis 4: Generar recomendaciones."""
	question = "Basándote en los datos, genera 5 recomendaciones priorizadas para mejorar la productividad (VA) y ergonomía (OK)."
	response = call_pandasai_script(csv_path, question)
	
	if response.get("success"):
		return response.get("result", "No se obtuvo respuesta")
	else:
		return f"Error: {response.get('error', 'Error desconocido')}"

def run_complete_analysis(csv_path):
	"""Run complete analysis pipeline on CSV data."""
	results = {
		"success": False,
		"error": None,
		"analyses": {}
	}
	
	# Verify CSV exists
	if not os.path.exists(csv_path):
		results["error"] = f"Archivo CSV no encontrado: {csv_path}"
		return results
	
	# Run all analyses
	print("\n[PandasAI] 📊 Iniciando análisis completo...")
	
	print("[PandasAI] 📊 Análisis 1: Productividad por ROI")
	results["analyses"]["productivity"] = analyze_productivity_by_roi(csv_path)
	
	print("[PandasAI] ⏰ Análisis 2: Patrones temporales")
	results["analyses"]["temporal"] = analyze_temporal_patterns(csv_path)
	
	print("[PandasAI] 🧍 Análisis 3: Ergonomía")
	results["analyses"]["ergonomics"] = analyze_ergonomics_correlation(csv_path)
	
	print("[PandasAI] 💡 Análisis 4: Recomendaciones")
	results["analyses"]["recommendations"] = generate_recommendations(csv_path)
	
	results["success"] = True
	print("[PandasAI] ✅ Análisis completo finalizado")
	
	return results


def begin_processing(config_data):
	"""Module-level function to start processing (RT or batch mode).
	
	This function is defined at module level to avoid scope issues with globals().
	It uses PROCESSING_CONTEXT to access the necessary functions and variables.
	"""
	try:
		if not config_data:
			raise ValueError("No se proporcionó configuración para el procesamiento")
		
		# Get context references
		processing_type = PROCESSING_CONTEXT.get("processing_type")
		begin_rt_processing = PROCESSING_CONTEXT.get("begin_rt_processing")
		launch_real_pipeline = PROCESSING_CONTEXT.get("launch_real_pipeline")
		progress_var = PROCESSING_CONTEXT.get("progress_var")
		update_progress_bar = PROCESSING_CONTEXT.get("update_progress_bar")
		_update_processing_nav_state = PROCESSING_CONTEXT.get("_update_processing_nav_state")
		_append_log_line = PROCESSING_CONTEXT.get("_append_log_line")
		
		if not all([processing_type, begin_rt_processing, launch_real_pipeline]):
			raise RuntimeError("PROCESSING_CONTEXT no está completamente inicializado")
		
		if processing_type["value"] == "rt":
			# RT mode: start RT processing threads
			begin_rt_processing(config_data)
		else:
			# Batch mode: start subprocess pipeline
			if progress_var:
				progress_var.set(0)
			if update_progress_bar:
				update_progress_bar(0)
			if _update_processing_nav_state:
				_update_processing_nav_state()
			launch_real_pipeline(config_data)
			
	except Exception as e:
		import traceback
		error_msg = f"Error en begin_processing:\n\n{str(e)}\n\nDetalle técnico:\n{traceback.format_exc()}"
		from tkinter import messagebox
		messagebox.showerror(
			"Error Iniciando Procesamiento",
			error_msg
		)
		print(f"[ERROR] begin_processing failed: {error_msg}")
		_append_log_line_fn = PROCESSING_CONTEXT.get("_append_log_line")
		if _append_log_line_fn:
			_append_log_line_fn(f"[ERROR] {error_msg}")


# Register the begin_processing function in globals immediately after definition
BEGIN_PROCESSING_FN = begin_processing


def _center_window(window: tk.Tk, width: int, height: int) -> None:
	window.update_idletasks()
	screen_width = window.winfo_screenwidth()
	screen_height = window.winfo_screenheight()
	x = int((screen_width - width) / 2)
	y = int((screen_height - height) / 3)
	window.geometry(f"{width}x{height}+{x}+{y}")


def _base_root() -> Path:
	"""Determine base root considering PyInstaller frozen mode.

	When bundled with PyInstaller (.exe), returns the directory containing the .exe
	so that assets/ folder is searched next to the executable.
	In development mode, returns the project root.
	"""
	if getattr(sys, "frozen", False):
		# When frozen, use the directory containing the .exe
		return Path(os.path.dirname(sys.executable))
	return Path(__file__).resolve().parent.parent


def _resolve_node_role(default_role: str = "worker") -> str:
	"""Resolve runtime node role for distributed deployment.

	Priority:
	1) ENV ARNESIS_NODE_ROLE
	2) gw_config.json key: node_role
	3) default_role
	"""
	role = os.environ.get("ARNESIS_NODE_ROLE", "").strip().lower()
	if role in {"worker", "controller"}:
		return role

	try:
		cfg_path = _base_root() / "gw_config.json"
		if cfg_path.exists():
			with open(cfg_path, "r", encoding="utf-8") as f:
				cfg = json.load(f)
			cfg_role = str(cfg.get("node_role", "")).strip().lower()
			if cfg_role in {"worker", "controller"}:
				return cfg_role
	except Exception:
		pass

	return default_role


def _find_logo_path() -> Optional[Path]:
	"""Locate the arnesis logo within the bundle or repo.

	Priority:
	1) assets/arnesis_logo.png under base root (repo or _MEIPASS)
	2) same folder as this file
	3) base root directly (rare)
	"""
	base = _base_root()
	here = Path(__file__).resolve()
	candidates = [
		base / "assets" / "arnesis_logo.png",
		here.parent / "arnesis_logo.png",
		base / "arnesis_logo.png",
	]
	for p in candidates:
		if p.exists():
			return p
	return None


def _load_logo_image(logo_path: Path) -> Optional[object]:
	if Image is None or ImageTk is None:
		return None
	try:
		img = Image.open(logo_path)
		# Scale if too wide for the window
		if img.width > LOGO_MAX_WIDTH:
			ratio = LOGO_MAX_WIDTH / float(img.width)
			new_size = (int(img.width * ratio), int(img.height * ratio))
			img = img.resize(new_size, Image.LANCZOS)
		return ImageTk.PhotoImage(img)
	except Exception:
		return None


def _dominant_color(img_path: Path) -> str:
	if Image is None:
		return FG_COLOR
	try:
		img = Image.open(img_path).convert("RGBA")
		# Downscale to speed up, then get color histogram
		small = img.resize((32, 32), Image.LANCZOS)
		data = list(small.getdata())
		# Filter near-transparent
		data = [p for p in data if p[3] > 10]
		# Count colors excluding extremes (near white/near black)
		counts = {}
		for r, g, b, a in data:
			if r > 240 and g > 240 and b > 240:
				continue
			if r < 10 and g < 10 and b < 10:
				continue
			key = (r, g, b)
			counts[key] = counts.get(key, 0) + 1
		if not counts:
			return FG_COLOR
		(r, g, b), _ = max(counts.items(), key=lambda kv: kv[1])
		return f"#{r:02X}{g:02X}{b:02X}"
	except Exception:
		return FG_COLOR


def main(node_role: Optional[str] = None) -> None:
	root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
	resolved_role = (node_role or _resolve_node_role()).strip().lower()
	if resolved_role not in {"worker", "controller"}:
		resolved_role = "worker"
	WORKER_CONTROL_STATE["role"] = resolved_role
	WORKER_CONTROL_STATE["node_id"] = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "worker-node"
	_set_worker_processing_state(rt_running=False, mode="normal", camera_count=0, groups=[])
	if resolved_role == "worker":
		ctrl_port = int(os.environ.get("ARNESIS_WORKER_CONTROL_PORT", "8765"))
		_start_worker_control_server(port=ctrl_port)
	role_label = "Worker" if resolved_role == "worker" else "Controller"
	root.title(f"{APP_TITLE} [{role_label}]")
	# Start maximized
	try:
		root.state("zoomed")
	except Exception:
		try:
			root.attributes("-zoomed", True)
		except Exception:
			pass
	# Keep a fallback geometry if zoom not supported
	_center_window(root, WINDOW_WIDTH, WINDOW_HEIGHT)
	# Set base window background
	root.configure(bg=BG_COLOR)

	# Use ttk styling for a cleaner look
	style = ttk.Style()
	# Use 'clam' to honor custom colors on Windows
	try:
		style.theme_use("clam")
	except Exception:
		pass

	# Base theming (only minimal ttk usage here)
	style.configure("Root.TFrame", background=BG_COLOR)
	style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)

	# Container uses vertical layout: logo on top, buttons below
	container = tk.Frame(root, bg=BG_COLOR)
	container.pack(fill=tk.BOTH, expand=True)

	# Helpers
	def _noop():
		return None
	
	def create_rounded_button(parent, text, bg_color, fg_color="white", active_bg=None, 
	                          font=("Arial", 11, "bold"), padx=20, pady=10, 
	                          corner_radius=8, command=None, width=None):
		"""Create a button with rounded corners using Canvas"""
		if active_bg is None:
			# Darken the bg color for hover
			active_bg = bg_color
		
		# State storage
		state = {
			"text": text,
			"bg_color": bg_color,
			"fg_color": fg_color,
			"active_bg": active_bg,
			"enabled": True
		}
		
		# Calculate button dimensions
		temp_label = tk.Label(parent, text=text, font=font)
		temp_label.update_idletasks()
		text_width = temp_label.winfo_reqwidth()
		text_height = temp_label.winfo_reqheight()
		temp_label.destroy()
		
		# Use provided width or calculate from text
		if width is not None:
			btn_width = width
		else:
			btn_width = text_width + padx * 2
		btn_height = text_height + pady * 2
		
		canvas = tk.Canvas(parent, width=btn_width, height=btn_height, 
		                   bg=parent.cget("bg"), highlightthickness=0, cursor="hand2")
		
		# Store btn_width for draw function
		state["btn_width"] = btn_width
		
		# Draw rounded rectangle
		def draw_rounded_rect(fill_color):
			canvas.delete("all")
			r = corner_radius
			w = state["btn_width"]
			canvas.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=fill_color, outline=fill_color)
			canvas.create_arc(w-r*2, 0, w, r*2, start=0, extent=90, fill=fill_color, outline=fill_color)
			canvas.create_arc(0, btn_height-r*2, r*2, btn_height, start=180, extent=90, fill=fill_color, outline=fill_color)
			canvas.create_arc(w-r*2, btn_height-r*2, w, btn_height, start=270, extent=90, fill=fill_color, outline=fill_color)
			canvas.create_rectangle(r, 0, w-r, btn_height, fill=fill_color, outline=fill_color)
			canvas.create_rectangle(0, r, w, btn_height-r, fill=fill_color, outline=fill_color)
			canvas.create_text(w//2, btn_height//2, text=state["text"], fill=state["fg_color"], font=font)
		
		draw_rounded_rect(state["bg_color"])
		
		# Hover effects
		def on_enter(e):
			if state["enabled"]:
				draw_rounded_rect(state["active_bg"])
		
		def on_leave(e):
			draw_rounded_rect(state["bg_color"])
		
		def on_click(e):
			if state["enabled"] and command:
				command()
		
		canvas.bind("<Enter>", on_enter)
		canvas.bind("<Leave>", on_leave)
		canvas.bind("<Button-1>", on_click)
		
		# Add configure method to mimic tk.Button API
		def configure(**kwargs):
			if "text" in kwargs:
				state["text"] = kwargs["text"]
			if "bg" in kwargs:
				state["bg_color"] = kwargs["bg"]
			if "canvas_bg" in kwargs:
				canvas.config(bg=kwargs["canvas_bg"])
			if "state" in kwargs:
				state["enabled"] = (kwargs["state"] != "disabled")
				if not state["enabled"]:
					canvas.config(cursor="arrow")
				else:
					canvas.config(cursor="hand2")
			draw_rounded_rect(state["bg_color"])
		
		canvas.configure = configure
		
		return canvas
	
	def open_entrenamiento_window():
		# New color scheme for training window (same as processing)
		PROC_BG = "#01326a"  # Main background
		PROC_CONTENT_BG = "#02234e"  # Content area background
		PROC_TAB_ACTIVE = "#5BA8C9"  # Active tab text color (blue)
		PROC_TAB_PREVIOUS = "#ffc735"  # Previous tab text color (yellow)
		PROC_TAB_FUTURE = "#E6EEF9"  # Future tab text color (white)
		PROC_BTN_NORMAL = "#015aca"  # Normal button color
		PROC_BTN_CONFIRM = "#ffc735"  # Confirm/Continue button color
		
		# Load assets root early for header logo
		assets_root = _base_root() / "assets"
		
		train_win = tk.Toplevel(root)
		train_win.title("Entrenamiento")
		# Hide main window while training window is open
		root.withdraw()
		
		def _on_close_training():
			train_win.destroy()
			root.deiconify()
			root.state("zoomed")
		
		train_win.protocol("WM_DELETE_WINDOW", _on_close_training)
		
		# Start maximized
		try:
			train_win.state("zoomed")
		except Exception:
			try:
				train_win.attributes("-zoomed", True)
			except Exception:
				pass
		
		# Keep a fallback geometry if zoom not supported
		_center_window(train_win, 720, 520)
		train_win.configure(bg=PROC_BG)
		
		wrapper = tk.Frame(train_win, bg=PROC_BG)
		wrapper.pack(fill=tk.BOTH, expand=True)
		
		# ========== TOP HEADER: Logo + Breadcrumb Tabs ==========
		header_frame = tk.Frame(wrapper, bg=PROC_BG, height=80)
		header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
		header_frame.pack_propagate(False)
		
		# Mini logo on the left
		mini_logo_path = assets_root / "NGUI" / "ArnesisMiniLogo.png"
		menorque_path = assets_root / "NGUI" / "menorque.png"
		mini_logo_img = load_icon(mini_logo_path, 200, 100, invert=False)
		if mini_logo_img:
			logo_label = tk.Label(header_frame, image=mini_logo_img, bg=PROC_BG)
			logo_label.image = mini_logo_img
			logo_label.pack(side=tk.LEFT, padx=(0, 20))
		
		# Breadcrumb tabs container
		breadcrumb_frame = tk.Frame(header_frame, bg=PROC_BG)
		breadcrumb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		# Tab titles for training workflow
		tab_titles = [
			"Inicio",
			"Tipo de\nmodelo",
			"Cargar contenido\npara dataset",
			"Recorte",
			"Clasificacion",
			"Deteccion",
			"Data\nAugmentation",
			"Configuracion",
			"Entrenamiento",
			"Estadisticas",
			"Pruebas",
		]
		
		# Create tab labels with separators
		breadcrumb_labels = []
		separator_imgs = []
		menorque_img = load_icon(menorque_path, 12, 12, invert=False) if menorque_path.exists() else None
		
		for i, title in enumerate(tab_titles):
			if i > 0:
				# Add separator
				if menorque_img:
					sep_label = tk.Label(breadcrumb_frame, image=menorque_img, bg=PROC_BG)
					sep_label.image = menorque_img
					separator_imgs.append(sep_label)
				else:
					sep_label = tk.Label(breadcrumb_frame, text=">", font=("Arial", 14), 
					                     fg=FG_COLOR, bg=PROC_BG)
					separator_imgs.append(sep_label)
				sep_label.pack(side=tk.LEFT, padx=8)
			
			# Tab label
			tab_label = tk.Label(breadcrumb_frame, text=title, font=("Arial", 10, "bold"), 
			                     fg=PROC_TAB_FUTURE, bg=PROC_BG, justify=tk.CENTER, cursor="hand2")
			tab_label.pack(side=tk.LEFT, padx=8)
			breadcrumb_labels.append(tab_label)
			
			# Bind click event to navigate to this tab
			def make_tab_click_handler(tab_index):
				def handler(event):
					nonlocal current_step
					# Only navigate if the tab is visible
					if tab_index in visible_tabs:
						current_step = tab_index
						update_tabs_state()
				return handler
			
			tab_label.bind("<Button-1>", make_tab_click_handler(i))
		
		# ========== CONTENT AREA ==========
		content_frame = tk.Frame(wrapper, bg=PROC_CONTENT_BG)
		content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
		
		# Create tabs (frames for content management)
		tabs = []
		for _ in tab_titles:
			frm = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			tabs.append(frm)
		
		# Training workflow state
		current_step = 0
		can_advance = [False] * len(tab_titles)
		sync_rt_controls_state = {"func": None}
		
		# Track which tabs are visible based on model type
		# Initially only show "Inicio" and "Tipo de modelo"
		visible_tabs = [0, 1]  # Indices of visible tabs
		
		# Navigation buttons (will be created later in make_rounded_button style)
		nav = tk.Frame(wrapper, bg=PROC_BG)
		nav.pack(fill=tk.X, padx=20, pady=(0, 10))
		
		def update_breadcrumb():
			"""Update breadcrumb colors based on current step and visibility"""
			# First, unpack all labels and separators
			for label in breadcrumb_labels:
				label.pack_forget()
			for sep in separator_imgs:
				sep.pack_forget()
			
			# Now repack only visible tabs with their separators
			first_visible = True
			for i, label in enumerate(breadcrumb_labels):
				if i in visible_tabs:
					# Add separator before this tab (except for the first visible tab)
					if not first_visible and i > 0 and i-1 < len(separator_imgs):
						separator_imgs[i-1].pack(side=tk.LEFT, padx=8)
					first_visible = False
					
					# Pack the tab label
					label.pack(side=tk.LEFT, padx=8)
					
					# Update color based on current step
					if i < current_step:
						label.config(fg=PROC_TAB_PREVIOUS)  # Yellow for previous
					elif i == current_step:
						label.config(fg=PROC_TAB_ACTIVE)  # Blue for current
					else:
						label.config(fg=PROC_TAB_FUTURE)  # White for future
		
		def show_current_tab():
			"""Show only the current tab content"""
			for i, tab in enumerate(tabs):
				if i == current_step:
					tab.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
				else:
					tab.pack_forget()
		
		def update_tabs_state():
			update_breadcrumb()
			show_current_tab()
			update_nav_state()
		
		def update_nav_state():
			# Update button states based on current step and visible tabs
			# Check if there's a previous visible tab
			has_prev = any(i in visible_tabs for i in range(current_step))
			if has_prev:
				prev_btn_canvas.configure(state="normal")
			else:
				prev_btn_canvas.configure(state="disabled")
			
			# Check if there's a next visible tab and we can advance
			has_next = any(i in visible_tabs for i in range(current_step + 1, len(tabs)))
			if has_next and can_advance[current_step]:
				next_btn_canvas.configure(state="normal")
			else:
				next_btn_canvas.configure(state="disabled")
		
		def go_prev():
			nonlocal current_step
			if current_step > 0:
				# Find the previous visible tab
				for i in range(current_step - 1, -1, -1):
					if i in visible_tabs:
						current_step = i
						update_tabs_state()
						break
		
		def go_next():
			nonlocal current_step
			if current_step < len(tabs) - 1 and can_advance[current_step]:
				# Find the next visible tab
				for i in range(current_step + 1, len(tabs)):
					if i in visible_tabs:
						current_step = i
						update_tabs_state()
						break
		
		# Helper function for rounded buttons
		def make_rounded_button(parent, text, command, bg_color, width=120, height=40, fg=None):
			"""Create a button with rounded corners"""
			container = tk.Frame(parent, bg=PROC_BG)
			canvas = tk.Canvas(container, width=width, height=height, bg=PROC_BG, 
			                   highlightthickness=0, cursor="hand2")
			canvas.pack()
			
			# Store command reference so it can be updated
			command_ref = {"func": command}
			
			def draw_button(color, state="normal"):
				canvas.delete("all")
				# Adjust color for disabled state
				if state == "disabled":
					color = "#666666"
				# Draw rounded rectangle
				radius = 8
				canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(width-radius*2, 0, width, radius*2, start=0, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(0, height-radius*2, radius*2, height, start=180, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(width-radius*2, height-radius*2, width, height, start=270, extent=90, 
				                  fill=color, outline=color)
				canvas.create_rectangle(radius, 0, width-radius, height, fill=color, outline=color)
				canvas.create_rectangle(0, radius, width, height-radius, fill=color, outline=color)
				# Draw text
				if fg:
					text_color = fg
				else:
					text_color = "#000000" if bg_color == PROC_BTN_CONFIRM else "#FFFFFF"
				if state == "disabled":
					text_color = "#999999"
				canvas.create_text(width/2, height/2, text=text, font=("Arial", 11, "bold"), 
				                   fill=text_color)
			
			# Store state
			button_state = {"state": "normal"}
			
			def redraw():
				draw_button(bg_color, button_state["state"])
			
			redraw()
			
			def on_enter(e):
				if button_state["state"] == "normal":
					# Lighter on hover
					hover_color = bg_color if bg_color == PROC_BTN_CONFIRM else "#0374e6"
					draw_button(hover_color, button_state["state"])
			
			def on_leave(e):
				redraw()
			
			def on_click(e):
				if button_state["state"] == "normal" and command_ref["func"]:
					command_ref["func"]()
			
			canvas.bind("<Enter>", on_enter)
			canvas.bind("<Leave>", on_leave)
			canvas.bind("<Button-1>", on_click)
			
			# Add configure method to mimic button API
			def configure(state=None, command=None):
				if state is not None:
					button_state["state"] = state
					redraw()
				if command is not None:
					command_ref["func"] = command
			
			canvas.configure = configure
			return container
		
		# Create navigation buttons
		prev_btn_container = make_rounded_button(nav, "Anterior", go_prev, PROC_BTN_NORMAL, width=120, height=40)
		prev_btn_canvas = prev_btn_container.winfo_children()[0]
		prev_btn_container.pack(side=tk.LEFT, padx=10)
		
		next_btn_container = make_rounded_button(nav, "Siguiente", go_next, PROC_BTN_CONFIRM, width=120, height=40)
		next_btn_canvas = next_btn_container.winfo_children()[0]
		next_btn_container.pack(side=tk.RIGHT, padx=10)
		
		# Initialize the first tab
		update_tabs_state()
		
		# ============================================================
		# TAB 0: Inicio
		# ============================================================
		inicio_tab = tabs[0]
		inicio_tab.configure(bg=PROC_CONTENT_BG)
		inicio_tab.grid_rowconfigure(0, weight=1)
		inicio_tab.grid_columnconfigure(0, weight=1)
		inicio_tab.grid_columnconfigure(1, weight=1)
		
		# Helper function to create large square button boxes with icon and description
		def make_inicio_button(parent, title, description, icon_path, command, row, col, bg_color="#021e44", hover_color="#043c86", padx=25, pady=30):
			btn_frame = tk.Frame(parent, bg=bg_color, cursor="hand2", bd=3)
			# TAMAÑOS DE BOTÓN: padx y pady controlan el espaciado alrededor del botón
			btn_frame.grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
			
			# Make it square by setting aspect ratio
			btn_frame.grid_propagate(False)
			
			# Inner container for centering content
			inner_container = tk.Frame(btn_frame, bg=bg_color)
			inner_container.place(relx=0.5, rely=0.5, anchor="center")
			
			# Load and display icon
			# TAMAÑO DE ICONO: Primer 90 = ancho, segundo 90 = alto
			icon_img = load_icon(icon_path, 200, 200, invert=False)
			if icon_img:
				icon_label = tk.Label(inner_container, image=icon_img, bg=bg_color)
				icon_label.image = icon_img  # Keep reference
				icon_label.pack(pady=(0, 10))
			
			# Main title label (white, bold)
			# TAMAÑO DE FUENTE TÍTULO: "Arial", 18 (antes era 24)
			title_label = tk.Label(inner_container, text=title, font=("Arial", 28, "bold"), 
			                       fg=FG_COLOR, bg=bg_color, wraplength=300)
			title_label.pack(pady=(5, 10))
			
			# Description label (yellow, smaller)
			# TAMAÑO DE FUENTE DESCRIPCIÓN: "Arial", 10 (antes era 11)
			desc_label = tk.Label(inner_container, text=description, font=("Arial", 19), 
			                      fg="#ffc000", bg=bg_color, wraplength=500, justify=tk.CENTER)
			desc_label.pack(pady=(50, 0))
			
			# Store all widgets for hover effects
			all_widgets = [btn_frame, inner_container, title_label, desc_label]
			if icon_img:
				all_widgets.append(icon_label)
			
			# Hover effects
			def on_enter(e):
				for widget in all_widgets:
					try:
						widget.configure(bg=hover_color)
					except:
						pass
			
			def on_leave(e):
				for widget in all_widgets:
					try:
						widget.configure(bg=bg_color)
					except:
						pass
			
			def on_click(e):
				command()
			
			# Bind events to all widgets
			for widget in all_widgets:
				widget.bind("<Enter>", on_enter)
				widget.bind("<Leave>", on_leave)
				widget.bind("<Button-1>", on_click)
			
			return btn_frame
		
		# Function to open manual PDF (placeholder)
		def open_manual():
			# TODO: Agregar ruta del archivo PDF real
			pdf_path = "assets\\NGUI\\Entrenamiento de Modelos AI.pdf"
			try:
				import os
				import subprocess
				if os.path.exists(pdf_path):
					subprocess.Popen([pdf_path], shell=True)
				else:
					messagebox.showwarning("Manual no encontrado", 
					                       f"El archivo {pdf_path} no existe.\nPor favor, configure la ruta correcta.")
			except Exception as e:
				messagebox.showerror("Error", f"No se pudo abrir el manual:\n{str(e)}")
		
		# Function to start (advance to next tab)
		def start_training_workflow():
			can_advance[0] = True
			enhanced_go_next()
		
		# Load icon paths
		manual_icon = assets_root / "NGUI" / "manual.png"
		iniciar_icon = assets_root / "NGUI" / "iniciar.png"

		# Create two large square buttons
		make_inicio_button(
			inicio_tab, 
			"ABRIR MANUAL", 
			"Leer el manual de usuario si se requiere ayuda con alguna sección para el entrenamiento del modelo",
			manual_icon,
			open_manual, 
			0, 0,
			padx=(300, 20), pady=(118, 170)
		)
		
		make_inicio_button(
			inicio_tab, 
			"INICIAR", 
			"Comenzar con el proceso para entrenar un modelo de AI de Visión por Computadora",
			iniciar_icon,
			start_training_workflow, 
			0, 1,
			padx=(0, 300), pady=(118, 170)
		)
		
		# ============================================================
		# TAB 1: Tipo de modelo
		# ============================================================
		tipo_tab = tabs[1]
		tipo_tab.configure(bg=PROC_CONTENT_BG)
		tipo_tab.grid_rowconfigure(0, weight=0)  # Label row
		tipo_tab.grid_rowconfigure(1, weight=1)  # Buttons row
		tipo_tab.grid_columnconfigure(0, weight=1)
		tipo_tab.grid_columnconfigure(1, weight=1)
		
		# Add white label at the top
		title_label_ref = {"widget": None}
		title_label_ref["widget"] = tk.Label(tipo_tab, text="Elige el tipo de Modelo", 
		                                      font=("Arial", 20, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		title_label_ref["widget"].grid(row=0, column=0, columnspan=2, pady=(30, 30))
		
		model_type = {"value": None}
		
		# Store button references to destroy/recreate them
		button_refs = {"left": None, "right": None}
		
		# Function to select classification model
		def select_classification():
			nonlocal visible_tabs
			model_type["value"] = "classification"
			
			# Update visible tabs for classification
			# 0: Inicio, 1: Tipo de modelo, 2: Cargar contenido, 3: Recorte, 4: Clasificacion
			# 6: Data Augmentation, 7: Configuracion, 8: Entrenamiento, 9: Estadisticas, 10: Pruebas
			visible_tabs = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
			update_breadcrumb()
			
			# Destroy and recreate title label with new text
			if title_label_ref["widget"]:
				title_label_ref["widget"].destroy()
			title_label_ref["widget"] = tk.Label(tipo_tab, text="Clasificación de Acciones", 
			                                      font=("Arial", 20, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			title_label_ref["widget"].grid(row=0, column=0, columnspan=2, pady=(30, 30))
			
			# Destroy existing buttons
			if button_refs["left"]:
				button_refs["left"].destroy()
			if button_refs["right"]:
				button_refs["right"].destroy()
			
			# Load new icons
			nuevo_dataset_icon = assets_root / "NGUI" / "nuevo_dataset_icono.png"
			dataset_existente_icon = assets_root / "NGUI" / "dataset_existente_icono.png"
			
			# Function to advance to next tab after selecting nuevo dataset
			def on_nuevo_dataset():
				can_advance[1] = True
				update_tabs_state()
				update_nav_state()
				go_next()
			
			# Function to select an existing dataset
			def on_dataset_existente():
				nonlocal current_step
				folder = filedialog.askdirectory(title="Seleccionar carpeta del dataset existente")
				if folder:
					folder_path = Path(folder)
					
					# Look for a subfolder with "crop" in its name
					crop_folder = None
					if folder_path.exists() and folder_path.is_dir():
						for subfolder in folder_path.iterdir():
							if subfolder.is_dir() and "crop" in subfolder.name.lower():
								crop_folder = subfolder
								break
					
					# If crop folder exists, go to Classification tab
					if crop_folder:
						cropped_output_dir["path"] = str(crop_folder)
						can_advance[1] = True
						can_advance[3] = True  # Mark Recorte as complete
						current_step = 4  # Go to Clasificacion tab (index 4)
						update_tabs_state()
						update_nav_state()
						on_clasificacion_tab_focus()  # Load images and display
					else:
						# No crop folder found, go to Recorte tab
						# Load videos from selected folder
						video_files = []
						for ext in [".mp4", ".avi", ".mov", ".mkv"]:
							video_files.extend([str(f) for f in folder_path.glob(f"*{ext}")])
						
						if video_files:
							dataset_files.clear()
							dataset_files.extend(video_files)
						
						can_advance[1] = True
						current_step = 3  # Go to Recorte tab (index 3)
						update_tabs_state()
						update_nav_state()
			
			# Recreate buttons with new content
			button_refs["left"] = make_inicio_button(
				tipo_tab, 
				"NUEVO DATASET", 
				"Define un nombre único e iniciar la\nclasificación desde cero.​",
				nuevo_dataset_icon,
				on_nuevo_dataset,
				1, 0,
				padx=(300, 20), pady=(20, 170)
			)
			
			button_refs["right"] = make_inicio_button(
				tipo_tab, 
				"DATASET EXISTENTE", 
				"Selecciona un dataset ya creado para\nagregar más videos o seguir clasificando\nimágenes pendientes.​",
				dataset_existente_icon,
				on_dataset_existente,
				1, 1,
				padx=(0, 300), pady=(20, 170)
			)
		
		# Function to select detection model
		def select_detection():
			nonlocal visible_tabs
			model_type["value"] = "detection"
			
			# Update visible tabs for detection
			# 0: Inicio, 1: Tipo de modelo, 2: Cargar contenido, 5: Deteccion
			# 6: Data Augmentation, 7: Configuracion, 8: Entrenamiento, 9: Estadisticas, 10: Pruebas
			visible_tabs = [0, 1, 2, 5, 6, 7, 8, 9, 10]
			update_breadcrumb()
			
			# Destroy and recreate title label with new text
			if title_label_ref["widget"]:
				title_label_ref["widget"].destroy()
			title_label_ref["widget"] = tk.Label(tipo_tab, text="Detección", 
			                                      font=("Arial", 20, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			title_label_ref["widget"].grid(row=0, column=0, columnspan=2, pady=(30, 30))
			
			# Destroy existing buttons
			if button_refs["left"]:
				button_refs["left"].destroy()
			if button_refs["right"]:
				button_refs["right"].destroy()
			
			# Load new icons
			nuevo_dataset_icon = assets_root / "NGUI" / "nuevo_dataset_icono.png"
			dataset_existente_icon = assets_root / "NGUI" / "dataset_existente_icono.png"
			
			# Function to advance to next tab after selecting nuevo dataset
			def on_nuevo_dataset():
				can_advance[1] = True
				update_tabs_state()
				update_nav_state()
				go_next()
			
			# Function to select an existing dataset
			def on_dataset_existente():
				folder = filedialog.askdirectory(title="Seleccionar carpeta del dataset existente")
				if folder:
					cropped_output_dir["path"] = folder
					# Update dataset with images from selected folder
					folder_path = Path(folder)
					if folder_path.exists() and folder_path.is_dir():
						image_files = []
						for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
							image_files.extend([str(f) for f in folder_path.glob(f"*{ext}")])
						if image_files:
							# Clear current dataset and add images
							dataset_files.clear()
							dataset_files.extend(image_files)
							
							# Store dataset path for auto-saving
							det_dataset_path["value"] = str(folder_path)
							
							# Load saved annotations and classes if they exist
							import json
							annotations_file = folder_path / "annotations.json"
							classes_file = folder_path / "classes.json"
							
							if annotations_file.exists():
								try:
									with open(annotations_file, 'r', encoding='utf-8') as f:
										loaded_annotations = json.load(f)
										# Convert filenames back to full paths
										det_annotations.clear()
										for img_name, anns in loaded_annotations.items():
											# Find matching image in image_files
											for img_path in image_files:
												if Path(img_path).name == img_name:
													det_annotations[img_path] = anns
													break
								except Exception as e:
									print(f"Error cargando anotaciones: {e}")
							
							if classes_file.exists():
								try:
									with open(classes_file, 'r', encoding='utf-8') as f:
										det_classes.clear()
										det_classes.extend(json.load(f))
								except Exception as e:
									print(f"Error cargando clases: {e}")
							
							# Enable navigation to skip "Cargar contenido" tab
							can_advance[1] = True  # Tipo de modelo
							can_advance[2] = True  # Cargar contenido (to allow skipping it)
							
							# Move directly to Deteccion tab
							# visible_tabs = [0, 1, 2, 5, 6, 7, 8, 9, 10]
							# Tab 5 is Deteccion (note: current_step is the actual tab index, not visible_tabs index)
							nonlocal current_step
							current_step = 5  # Move to tab 5 (Deteccion)
							
							update_tabs_state()
							update_nav_state()
							
							# Call on_deteccion_tab_focus to load and display images
							on_deteccion_tab_focus()
							
							# Refresh classes buttons to display loaded classes
							refresh_classes_buttons()
			
			# Recreate buttons with new content
			button_refs["left"] = make_inicio_button(
				tipo_tab, 
				"NUEVO DATASET", 
				"Define un nombre único e iniciar la\ndetección desde cero.​",
				nuevo_dataset_icon,
				on_nuevo_dataset,
				1, 0,
				padx=(300, 20), pady=(20, 170)
			)
			
			button_refs["right"] = make_inicio_button(
				tipo_tab, 
				"DATASET EXISTENTE", 
				"Selecciona un dataset ya creado para\nagregar más videos o seguir con imágenes\npendientes.​",
				dataset_existente_icon,
				on_dataset_existente,
				1, 1,
				padx=(0, 300), pady=(20, 170)
			)
		
		# Load icon paths for model type buttons
		clasificacion_icon = assets_root / "NGUI" / "clasificacion_acciones.png"
		deteccion_icon = assets_root / "NGUI" / "deteccion.png"
		
		# Create two large square buttons (reuse make_inicio_button function)
		button_refs["left"] = make_inicio_button(
			tipo_tab, 
			"CLASIFICACIÓN\nDE ACCIONES", 
			"Identifica a que categoría pertenece un\nobjeto dentro de una imagen​",
			clasificacion_icon,
			select_classification, 
			1, 0,  # Row 1 (después del label)
			padx=(300, 20), pady=(20, 170)
		)
		
		button_refs["right"] = make_inicio_button(
			tipo_tab, 
			"DETECCIÓN", 
			"Encontrar la ubicación exacta de objetos\nespecíficos en una imagen​",
			deteccion_icon,
			select_detection, 
			1, 1,  # Row 1 (después del label)
			padx=(0, 300), pady=(20, 170)
		)
		
		# ============================================================
		# TAB 2: Cargar contenido para dataset
		# ============================================================
		dataset_tab = tabs[2]
		dataset_tab.configure(bg=PROC_CONTENT_BG)
		dataset_tab.grid_rowconfigure(0, weight=0)  # Name label
		dataset_tab.grid_rowconfigure(1, weight=0)  # Name entry + save button
		dataset_tab.grid_rowconfigure(2, weight=0)  # Folder label
		dataset_tab.grid_rowconfigure(3, weight=0)  # Folder entry + button
		dataset_tab.grid_rowconfigure(4, weight=1)  # Drop area
		dataset_tab.grid_rowconfigure(5, weight=0)  # Save status
		dataset_tab.grid_columnconfigure(0, weight=1)
		
		# State for dataset files
		dataset_files = []
		dataset_folder = {"path": ""}
		
		# Row 0: Nombre del dataset label
		dataset_name_label = tk.Label(
			dataset_tab,
			text="Nombre de Dataset",
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG,
			font=("Arial", 14, "bold")
		)
		dataset_name_label.grid(row=0, column=0, sticky="w", padx=40, pady=(30, 10))
		
		# Row 1: Nombre del dataset textbox + Guardar button
		name_row_frame = tk.Frame(dataset_tab, bg=PROC_CONTENT_BG)
		name_row_frame.grid(row=1, column=0, sticky="ew", padx=40, pady=(0, 20))
		
		dataset_name_entry = tk.Entry(name_row_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000", relief=tk.FLAT, bd=1, highlightthickness=1, highlightbackground="#CCCCCC", highlightcolor="#015aca")
		dataset_name_entry.grid(row=0, column=0, sticky="w", padx=(0, 5), ipadx=5, ipady=5)
		dataset_name_entry.config(width=35)
		
		# Custom rounded button for "Guardar Dataset"
		guardar_btn_canvas = tk.Canvas(name_row_frame, width=150, height=40, bg=PROC_CONTENT_BG, highlightthickness=0, cursor="hand2")
		guardar_btn_canvas.grid(row=0, column=1, sticky="w")
		
		def draw_guardar_button(color):
			guardar_btn_canvas.delete("all")
			# Draw rounded rectangle
			x1, y1, x2, y2 = 5, 5, 145, 35
			r = 8
			guardar_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
			guardar_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
			guardar_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
			guardar_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
			guardar_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
			guardar_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
			guardar_btn_canvas.create_text(75, 20, text="Guardar Dataset", fill="white", font=("Arial", 11, "bold"))
		
		guardar_btn_enabled = {"value": False}
		
		def update_guardar_button_appearance():
			if guardar_btn_enabled["value"]:
				draw_guardar_button("#015aca")
				guardar_btn_canvas.config(cursor="hand2")
			else:
				draw_guardar_button("#cccccc")
				guardar_btn_canvas.config(cursor="arrow")
		
		# Initialize button appearance based on initial state
		update_guardar_button_appearance()
		
		def on_guardar_enter(e):
			if guardar_btn_enabled["value"]:
				draw_guardar_button("#0470d4")
		
		def on_guardar_leave(e):
			if guardar_btn_enabled["value"]:
				draw_guardar_button("#015aca")
			else:
				draw_guardar_button("#cccccc")
		
		def on_guardar_click(e):
			if guardar_btn_enabled["value"]:
				save_dataset()
		
		guardar_btn_canvas.bind("<Enter>", on_guardar_enter)
		guardar_btn_canvas.bind("<Leave>", on_guardar_leave)
		guardar_btn_canvas.bind("<Button-1>", on_guardar_click)
		
		# Row 2: Carpeta para Guardar Dataset label
		dataset_folder_label = tk.Label(
			dataset_tab,
			text="Carpeta para Guardar Dataset",
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG,
			font=("Arial", 14, "bold")
		)
		dataset_folder_label.grid(row=2, column=0, sticky="w", padx=40, pady=(0, 10))
		
		# Row 3: Carpeta textbox + Buscar button
		folder_row_frame = tk.Frame(dataset_tab, bg=PROC_CONTENT_BG)
		folder_row_frame.grid(row=3, column=0, sticky="ew", padx=40, pady=(0, 20))
		
		dataset_folder_entry = tk.Entry(folder_row_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000", relief=tk.FLAT, bd=1, highlightthickness=1, highlightbackground="#CCCCCC", highlightcolor="#015aca")
		dataset_folder_entry.grid(row=0, column=0, sticky="w", padx=(0, 5), ipadx=5, ipady=5)
		dataset_folder_entry.config(width=35)
		
		# Custom rounded button for "Buscar"
		buscar_btn_canvas = tk.Canvas(folder_row_frame, width=100, height=40, bg=PROC_CONTENT_BG, highlightthickness=0, cursor="hand2")
		buscar_btn_canvas.grid(row=0, column=1, sticky="w")
		
		def draw_buscar_button(color):
			buscar_btn_canvas.delete("all")
			# Draw rounded rectangle
			x1, y1, x2, y2 = 5, 5, 95, 35
			r = 8
			buscar_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
			buscar_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
			buscar_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
			buscar_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
			buscar_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
			buscar_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
			buscar_btn_canvas.create_text(50, 20, text="Buscar", fill="white", font=("Arial", 11, "bold"))
		
		draw_buscar_button("#015aca")
		
		def on_buscar_enter(e):
			draw_buscar_button("#0470d4")
		
		def on_buscar_leave(e):
			draw_buscar_button("#015aca")
		
		def on_buscar_click(e):
			on_browse_dataset_folder()
		
		buscar_btn_canvas.bind("<Enter>", on_buscar_enter)
		buscar_btn_canvas.bind("<Leave>", on_buscar_leave)
		buscar_btn_canvas.bind("<Button-1>", on_buscar_click)
		
		# Row 4: Drop area
		dataset_drop_container = tk.Frame(dataset_tab, bg="#002e66", bd=3, relief=tk.FLAT, highlightthickness=3, highlightbackground="#002e66")
		dataset_drop_container.grid(row=4, column=0, sticky="nsew", padx=40, pady=(0, 30))
		dataset_drop_container.grid_rowconfigure(0, weight=1)
		dataset_drop_container.grid_columnconfigure(0, weight=1)
		
		dataset_canvas = tk.Canvas(dataset_drop_container, bg="#002858", highlightthickness=0)
		dataset_canvas.grid(row=0, column=0, sticky="nsew")
		
		dataset_center_label = tk.Label(dataset_canvas, text="Carga el contenido aquí. Sin cables, sin drama.", font=("Arial", 16, "bold"), fg="#00a6e5", bg="#002858")
		dataset_center_label_window = dataset_canvas.create_window(0, 0, window=dataset_center_label, tags="center")
		
		dataset_list_frame = tk.Frame(dataset_canvas, bg="#002858")
		dataset_listbox = tk.Listbox(dataset_list_frame, bg="#002858", fg=FG_COLOR, selectbackground=GRAY_HOVER, borderwidth=0, highlightthickness=0)
		dataset_scrollbar = tk.Scrollbar(dataset_list_frame, orient=tk.VERTICAL, command=dataset_listbox.yview)
		dataset_listbox.configure(yscrollcommand=dataset_scrollbar.set)
		dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		dataset_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		dataset_list_window = dataset_canvas.create_window(0, 0, window=dataset_list_frame, tags="list", state="hidden")
		
		def layout_dataset_canvas(_=None):
			w = dataset_canvas.winfo_width()
			h = dataset_canvas.winfo_height()
			if w > 1 and h > 1:
				dataset_canvas.coords(dataset_center_label_window, w//2, h//2)
				dataset_canvas.coords(dataset_list_window, w//2, h//2)
		
		dataset_canvas.bind("<Configure>", layout_dataset_canvas)
		
		# Video/image extensions
		IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
		VIDEO_EXTS_DATASET = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
		
		def parse_dataset_files(s: str) -> list[str]:
			s = s.strip()
			if not s:
				return []
			if s.startswith("{"):
				s = s[1:]
			if s.endswith("}"):
				s = s[:-1]
			parts = []
			in_quote = False
			current = []
			for ch in s:
				if ch == '"':
					in_quote = not in_quote
				elif ch == ' ' and not in_quote:
					if current:
						parts.append("".join(current))
						current = []
				else:
					current.append(ch)
			if current:
				parts.append("".join(current))
			return parts
		
		def filter_dataset_files(paths: list[str]) -> list[str]:
			valid = []
			for p in paths:
				if Path(p).suffix.lower() in IMAGE_EXTS or Path(p).suffix.lower() in VIDEO_EXTS_DATASET:
					valid.append(p)
			return valid
		
		def refresh_dataset_listbox():
			dataset_listbox.delete(0, tk.END)
			for f in dataset_files:
				dataset_listbox.insert(tk.END, Path(f).name)
			if dataset_files:
				dataset_canvas.itemconfig(dataset_center_label_window, state="hidden")
				dataset_canvas.itemconfig(dataset_list_window, state="normal")
			else:
				dataset_canvas.itemconfig(dataset_center_label_window, state="normal")
				dataset_canvas.itemconfig(dataset_list_window, state="hidden")
		
		def check_for_videos_and_extract():
			"""Check if any videos are in dataset_files and open extraction popup"""
			video_files = [f for f in dataset_files if Path(f).suffix.lower() in VIDEO_EXTS_DATASET]
			if video_files:
				open_video_extraction_popup(video_files)
		
		def accept_dataset_files(paths: list[str]):
			valid = filter_dataset_files(paths)
			if not valid:
				return
			for v in valid:
				if v not in dataset_files:
					dataset_files.append(v)
			refresh_dataset_listbox()
			can_advance[2] = True
			update_nav_state()
			update_save_button_state()
			# Check for videos and open extraction popup
			check_for_videos_and_extract()
		
		def on_dataset_drop(event):
			paths = parse_dataset_files(event.data)
			accept_dataset_files(paths)
		
		def on_dataset_click(_):
			paths = filedialog.askopenfilenames(title="Seleccionar imágenes o videos")
			if paths:
				accept_dataset_files(list(paths))
		
		def on_browse_dataset_folder():
			folder = filedialog.askdirectory(title="Seleccionar carpeta de contenido")
			if folder:
				dataset_folder["path"] = folder
				dataset_folder_entry.delete(0, tk.END)
				dataset_folder_entry.insert(0, folder)
				# Scan folder for images/videos
				folder_path = Path(folder)
				if folder_path.exists() and folder_path.is_dir():
					files_in_folder = []
					for ext in IMAGE_EXTS | VIDEO_EXTS_DATASET:
						files_in_folder.extend([str(f) for f in folder_path.glob(f"*{ext}")])
					if files_in_folder:
						accept_dataset_files(files_in_folder)
		
		# Enable drag-and-drop
		if DND_AVAILABLE:
			dataset_canvas.drop_target_register(DND_FILES)
			dataset_canvas.dnd_bind("<<Drop>>", on_dataset_drop)
		
		dataset_canvas.bind("<Button-1>", on_dataset_click)
		
		# Status label for save operations (below drop area)
		dataset_save_status = tk.Label(
			dataset_tab,
			text="",
			fg="#4CAF50",
			bg=PROC_CONTENT_BG,
			font=("Arial", 10)
		)
		dataset_save_status.grid(row=5, column=0, sticky="w", padx=40, pady=(10, 20))
		
		def save_dataset():
			"""Save current dataset files to organized folder structure (excluding videos)"""
			if not dataset_files:
				dataset_save_status.configure(text="No hay archivos para guardar", fg="#F44336")
				return
			
			dataset_name = dataset_name_entry.get().strip()
			if not dataset_name:
				dataset_save_status.configure(text="Por favor ingresa un nombre para el dataset", fg="#F44336")
				return
			
			if not model_type["value"]:
				dataset_save_status.configure(text="Por favor selecciona un tipo de modelo primero", fg="#F44336")
				return
			
			# Create folder name based on model type
			model_type_suffix = model_type["value"]  # "classification" or "detection"
			folder_name = f"{dataset_name}_{model_type_suffix}"
			
			# Use relative path from base root (for .exe compatibility)
			base = _base_root()
			datasets_dir = base / "datasets"
			dataset_folder_path = datasets_dir / folder_name
			
			try:
				# Create datasets directory if it doesn't exist
				datasets_dir.mkdir(parents=True, exist_ok=True)
				
				# Check if dataset folder already exists
				if dataset_folder_path.exists():
					response = messagebox.askyesno(
						"Carpeta existente",
						f"La carpeta '{folder_name}' ya existe. ¿Deseas sobrescribirla?"
					)
					if not response:
						dataset_save_status.configure(text="Guardado cancelado", fg="#FF9800")
						return
					else:
						# Remove existing folder
						import shutil
						shutil.rmtree(dataset_folder_path)
				
				# Create dataset folder
				dataset_folder_path.mkdir(parents=True, exist_ok=True)
				
				# Filter out videos - only copy images (videos are only for frame extraction)
				import shutil
				files_to_copy = [f for f in dataset_files if Path(f).suffix.lower() not in VIDEO_EXTS_DATASET]
				
				if not files_to_copy:
					dataset_save_status.configure(text="No hay imágenes para guardar (solo videos)", fg="#FF9800")
					return
				
				# Copy image files to dataset folder
				for i, file_path in enumerate(files_to_copy, 1):
					src = Path(file_path)
					if src.exists():
						dst = dataset_folder_path / src.name
						shutil.copy2(src, dst)
						dataset_save_status.configure(
							text=f"Guardando... {i}/{len(files_to_copy)}",
							fg="#2196F3"
						)
						train_win.update()
				
				dataset_save_status.configure(
					text=f"✓ Dataset guardado: datasets/{folder_name} ({len(files_to_copy)} imágenes)",
					fg="#4CAF50"
				)
				
				# Store dataset path for auto-saving in detection tab
				det_dataset_path["value"] = str(dataset_folder_path)
				
			except Exception as e:
				dataset_save_status.configure(
					text=f"Error al guardar: {str(e)}",
					fg="#F44336"
				)
		
		# Enable save button when files are added and name is entered
		def update_save_button_state():
			if dataset_files and dataset_name_entry.get().strip():
				guardar_btn_enabled["value"] = True
			else:
				guardar_btn_enabled["value"] = False
			update_guardar_button_appearance()
		
		# Bind to entry changes
		dataset_name_entry.bind("<KeyRelease>", lambda e: update_save_button_state())
		
		# ============================================================
		# Video Frame Extraction Popup
		# ============================================================
		def open_video_extraction_popup(video_files: list[str]):
			"""Open popup to configure and extract frames from videos"""
			extraction_win = tk.Toplevel(train_win)
			extraction_win.title("Extraer Frames de Videos")
			extraction_win.geometry("1000x800")
			extraction_win.configure(bg=BG_COLOR)
			
			# Make modal
			extraction_win.transient(train_win)
			extraction_win.grab_set()
			
			# Center window
			_center_window(extraction_win, 1000, 800)
			
			# State for current video
			current_video_idx = {"value": 0}
			video_info = {"fps": 0, "total_frames": 0, "duration": 0}
			
			# Header
			header_frame = tk.Frame(extraction_win, bg=BG_COLOR)
			header_frame.pack(fill=tk.X, padx=16, pady=(16, 8))
			
			title_label = tk.Label(header_frame, text="Configurar Extracción de Frames", font=("Arial", 16, "bold"), fg=FG_COLOR, bg=BG_COLOR)
			title_label.pack()
			
			# Video info frame
			info_frame = tk.Frame(extraction_win, bg=BG_COLOR)
			info_frame.pack(fill=tk.X, padx=16, pady=8)
			
			video_name_label = tk.Label(info_frame, text="", font=("Arial", 12, "bold"), fg=FG_COLOR, bg=BG_COLOR)
			video_name_label.pack()
			
			video_stats_label = tk.Label(info_frame, text="", font=("Arial", 10), fg=FG_COLOR, bg=BG_COLOR)
			video_stats_label.pack()
			
			# Video preview
			preview_frame = tk.Frame(extraction_win, bg="black", height=400)
			preview_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
			preview_frame.pack_propagate(False)
			
			preview_label = tk.Label(preview_frame, text="Cargando vista previa...", font=("Arial", 14), fg=FG_COLOR, bg="black")
			preview_label.pack(expand=True)
			
			# Store photo reference to prevent garbage collection
			preview_photo = {"current": None}
			
			# Slider frame
			slider_frame = tk.Frame(extraction_win, bg=BG_COLOR)
			slider_frame.pack(fill=tk.X, padx=16, pady=16)
			
			slider_label = tk.Label(slider_frame, text="Frames por segundo a exportar:", font=("Arial", 11, "bold"), fg=FG_COLOR, bg=BG_COLOR)
			slider_label.pack()
			
			fps_var = tk.DoubleVar(value=1.0)
			fps_display = tk.Label(slider_frame, text="1.0 FPS", font=("Arial", 12), fg=FG_COLOR, bg=BG_COLOR)
			fps_display.pack(pady=4)
			
			fps_slider = tk.Scale(slider_frame, from_=0.01, to=30, resolution=0.01, orient=tk.HORIZONTAL, variable=fps_var, bg=BG_COLOR, fg=FG_COLOR, highlightthickness=0, troughcolor=GRAY_BG)
			fps_slider.pack(fill=tk.X, padx=32)
			
			explanation_label = tk.Label(slider_frame, text="Izquierda: más frames/seg | Derecha: menos frames/seg", font=("Arial", 9), fg=FG_COLOR, bg=BG_COLOR)
			explanation_label.pack(pady=4)
			
			# Extraction info frame with editable textboxes
			extraction_info_frame = tk.Frame(slider_frame, bg=BG_COLOR)
			extraction_info_frame.pack(pady=(12, 0))
			
			# Label text with embedded entry widgets
			info_label_1 = tk.Label(extraction_info_frame, text="Se extraerá 1 frame cada", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
			info_label_1.pack(side=tk.LEFT, padx=(0, 4))
			
			# Interval spinbox (seconds) with ±0.1 increment
			interval_var = tk.StringVar(value="1.00")
			interval_entry = tk.Spinbox(extraction_info_frame, textvariable=interval_var, width=8, font=("Arial", 11), justify=tk.CENTER, from_=0.01, to=100.0, increment=0.1, format="%.2f", bg="#02224e", fg=FG_COLOR, buttonbackground="#02224e", insertbackground=FG_COLOR)
			interval_entry.pack(side=tk.LEFT, padx=2)
			# Will be updated dynamically when video is loaded
			
			info_label_2 = tk.Label(extraction_info_frame, text="segundos (", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
			info_label_2.pack(side=tk.LEFT, padx=2)
			
			# Total images spinbox with ±1 increment
			total_images_var = tk.StringVar(value="0")
			total_images_entry = tk.Spinbox(extraction_info_frame, textvariable=total_images_var, width=6, font=("Arial", 11), justify=tk.CENTER, from_=1, to=100000, increment=1, bg="#02224e", fg=FG_COLOR, buttonbackground="#02224e", insertbackground=FG_COLOR)
			total_images_entry.pack(side=tk.LEFT, padx=2)
			
			info_label_3 = tk.Label(extraction_info_frame, text="imágenes totales)", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
			info_label_3.pack(side=tk.LEFT, padx=(2, 0))
			
			# Buttons
			button_frame = tk.Frame(extraction_win, bg=BG_COLOR)
			button_frame.pack(fill=tk.X, padx=16, pady=(8, 16))
			
			# Extract button (green)
			extract_btn_canvas = tk.Canvas(button_frame, width=150, height=40, bg=BG_COLOR, highlightthickness=0, cursor="hand2")
			extract_btn_canvas.pack(side=tk.LEFT, padx=4)
			
			def draw_extract_button(color):
				extract_btn_canvas.delete("all")
				x1, y1, x2, y2 = 5, 5, 145, 35
				r = 8
				extract_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
				extract_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
				extract_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
				extract_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
				extract_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
				extract_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
				extract_btn_canvas.create_text(75, 20, text="Extraer Frames", fill="black", font=("Arial", 11, "bold"))
			
			draw_extract_button("#7ec331")
			
			def on_extract_enter(e):
				draw_extract_button("#8fd341")
			
			def on_extract_leave(e):
				draw_extract_button("#7ec331")
			
			extract_btn_canvas.bind("<Enter>", on_extract_enter)
			extract_btn_canvas.bind("<Leave>", on_extract_leave)
			
			# Skip button (blue)
			skip_btn_canvas = tk.Canvas(button_frame, width=150, height=40, bg=BG_COLOR, highlightthickness=0, cursor="hand2")
			skip_btn_canvas.pack(side=tk.LEFT, padx=4)
			
			def draw_skip_button(color):
				skip_btn_canvas.delete("all")
				x1, y1, x2, y2 = 5, 5, 145, 35
				r = 8
				skip_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
				skip_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
				skip_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
				skip_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
				skip_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
				skip_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
				skip_btn_canvas.create_text(75, 20, text="Omitir este video", fill="white", font=("Arial", 11, "bold"))
			
			draw_skip_button("#096bc9")
			
			def on_skip_enter(e):
				draw_skip_button("#1a7bd9")
			
			def on_skip_leave(e):
				draw_skip_button("#096bc9")
			
			skip_btn_canvas.bind("<Enter>", on_skip_enter)
			skip_btn_canvas.bind("<Leave>", on_skip_leave)
			
			# Close button (red)
			close_btn_canvas = tk.Canvas(button_frame, width=100, height=40, bg=BG_COLOR, highlightthickness=0, cursor="hand2")
			close_btn_canvas.pack(side=tk.RIGHT, padx=4)
			
			def draw_close_button(color):
				close_btn_canvas.delete("all")
				x1, y1, x2, y2 = 5, 5, 95, 35
				r = 8
				close_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
				close_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
				close_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
				close_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
				close_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
				close_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
				close_btn_canvas.create_text(50, 20, text="Cerrar", fill="white", font=("Arial", 11, "bold"))
			
			draw_close_button("#ec5b2d")
			
			def on_close_enter(e):
				draw_close_button("#fc6b3d")
			
			def on_close_leave(e):
				draw_close_button("#ec5b2d")
			
			close_btn_canvas.bind("<Enter>", on_close_enter)
			close_btn_canvas.bind("<Leave>", on_close_leave)
			
			def get_video_info(video_path: str):
				"""Get video information using ffmpeg"""
				try:
					import subprocess
					cmd = [
						"ffprobe",
						"-v", "error",
						"-select_streams", "v:0",
						"-count_packets",
						"-show_entries", "stream=nb_read_packets,r_frame_rate,duration",
						"-of", "csv=p=0",
						video_path
					]
					result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
					if result.returncode == 0:
						lines = result.stdout.strip().split('\n')
						if lines:
							parts = lines[0].split(',')
							if len(parts) >= 3:
								fps_str = parts[0]
								if '/' in fps_str:
									num, den = fps_str.split('/')
									fps = float(num) / float(den)
								else:
									fps = float(fps_str)
								duration = float(parts[1]) if parts[1] else 0
								total_frames = int(parts[2]) if parts[2] else int(fps * duration)
								return {"fps": fps, "duration": duration, "total_frames": total_frames}
				except Exception:
					pass
				return {"fps": 0, "duration": 0, "total_frames": 0}
			
			def load_video_preview(video_path: str):
				"""Load and display first frame of video"""
				try:
					import cv2
					# Open video
					cap = cv2.VideoCapture(video_path)
					if not cap.isOpened():
						preview_label.config(text="No se pudo abrir el video")
						return
					
					# Read first frame
					ret, frame = cap.read()
					cap.release()
					
					if not ret or frame is None:
						preview_label.config(text="No se pudo leer el frame")
						return
					
					# Convert BGR to RGB
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					
					# Get preview frame dimensions
					preview_frame.update_idletasks()
					max_width = preview_frame.winfo_width() - 20
					max_height = preview_frame.winfo_height() - 20
					
					if max_width < 100 or max_height < 100:
						max_width = 900
						max_height = 380
					
					# Calculate scaling to fit
					h, w = frame_rgb.shape[:2]
					scale = min(max_width / w, max_height / h)
					new_w = int(w * scale)
					new_h = int(h * scale)
					
					# Resize frame
					frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
					
					# Convert to PIL Image
					pil_image = Image.fromarray(frame_resized)
					photo = ImageTk.PhotoImage(pil_image)
					
					# Update label
					preview_label.config(image=photo, text="")
					preview_photo["current"] = photo  # Keep reference
					
				except ImportError:
					preview_label.config(text="OpenCV no disponible\nInstalar: pip install opencv-python")
				except Exception as e:
					preview_label.config(text=f"Error al cargar preview:\n{str(e)}")
			
			def update_video_display():
				"""Update display for current video"""
				idx = current_video_idx["value"]
				if idx >= len(video_files):
					extraction_win.destroy()
					return
				
				video_path = video_files[idx]
				video_name = Path(video_path).name
				video_name_label.config(text=f"Video {idx+1}/{len(video_files)}: {video_name}")
				
				# Load video preview
				preview_label.config(text="Cargando vista previa...")
				extraction_win.update()
				load_video_preview(video_path)
				
				# Get video info
				info = get_video_info(video_path)
				video_info["fps"] = info["fps"]
				video_info["total_frames"] = info["total_frames"]
				video_info["duration"] = info["duration"]
				
				if info["fps"] > 0:
					video_stats_label.config(text=f"FPS: {info['fps']:.2f} | Frames totales: {info['total_frames']} | Duración: {info['duration']:.2f}s")
					# Configure slider range
					max_fps = min(info["fps"], 30)
					min_interval = max(1.0 / 60.0, 1.0 / info["duration"]) if info["duration"] > 0 else 1.0 / 60.0
					fps_slider.config(from_=min_interval, to=max_fps)
					# Update interval spinbox max to video duration
					if info["duration"] > 0:
						interval_entry.config(to=info["duration"])
					fps_var.set(1.0)
				else:
					video_stats_label.config(text="No se pudo obtener información del video")
			
			def update_fps_display(*args):
				fps = fps_var.get()
				if fps >= 1:
					fps_display.config(text=f"{fps:.2f} frames/segundo")
				else:
					interval = 1.0 / fps
					fps_display.config(text=f"1 frame cada {interval:.2f} segundos")
				
				# Update extraction info textboxes
				updating_from_slider[0] = True
				interval_seconds = 1.0 / fps
				interval_var.set(f"{interval_seconds:.2f}")
				
				# Calculate total images if video info is available
				if video_info.get("duration"):
					total_images = int(video_info["duration"] / interval_seconds)
					total_images_var.set(str(total_images))
				else:
					total_images_var.set("0")
				updating_from_slider[0] = False
			
			fps_var.trace_add("write", update_fps_display)
			
			# Flag to prevent circular updates
			updating_from_slider = [False]
			
			# Flag to prevent circular updates
			updating_from_slider = [False]
			
			# Validation and synchronization for interval entry
			def validate_interval(*args):
				"""Update slider when interval textbox content changes"""
				if updating_from_slider[0]:
					return
				try:
					interval_value = float(interval_var.get())
					# Limit interval to video duration
					if video_info.get("duration") and video_info["duration"] > 0:
						if interval_value > video_info["duration"]:
							interval_value = video_info["duration"]
							updating_from_slider[0] = True
							interval_var.set(f"{interval_value:.2f}")
							updating_from_slider[0] = False
					if interval_value > 0:
						new_fps = 1.0 / interval_value
						# Constrain to slider range
						new_fps = max(0.01, min(30.0, new_fps))
						updating_from_slider[0] = True
						fps_var.set(new_fps)
						updating_from_slider[0] = False
				except ValueError:
					pass  # Ignore invalid input
			
			interval_var.trace_add("write", validate_interval)
			
			# Validation and synchronization for total images entry
			def validate_total_images(*args):
				"""Update slider when total images textbox content changes"""
				if updating_from_slider[0]:
					return
				try:
					total_value = int(total_images_var.get())
					if total_value > 0 and video_info.get("duration"):
						interval_value = video_info["duration"] / total_value
						new_fps = 1.0 / interval_value
						# Constrain to slider range
						new_fps = max(0.01, min(30.0, new_fps))
						updating_from_slider[0] = True
						fps_var.set(new_fps)
						updating_from_slider[0] = False
				except (ValueError, ZeroDivisionError):
					pass  # Ignore invalid input
			
			total_images_var.trace_add("write", validate_total_images)
			
			def extract_frames():
				"""Extract frames from current video using ffmpeg"""
				idx = current_video_idx["value"]
				if idx >= len(video_files):
					return
				
				video_path = video_files[idx]
				video_name = Path(video_path).stem
				video_dir = Path(video_path).parent
				output_dir = video_dir / video_name
				output_dir.mkdir(exist_ok=True)
				
				fps_extract = fps_var.get()
				
				try:
					import subprocess
					# ffmpeg command to extract frames
					cmd = [
						"ffmpeg",
						"-i", video_path,
						"-vf", f"fps={fps_extract}",
						"-q:v", "2",
						str(output_dir / "frame_%06d.jpg")
					]
					
					# Show progress - update button text
					extract_btn_canvas.delete("all")
					x1, y1, x2, y2 = 5, 5, 145, 35
					r = 8
					extract_btn_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill="#cccccc", outline="#cccccc")
					extract_btn_canvas.create_text(75, 20, text="Extrayendo...", fill="black", font=("Arial", 11, "bold"))
					extract_btn_canvas.config(cursor="arrow")
					
					draw_skip_button("#cccccc")
					skip_btn_canvas.config(cursor="arrow")
					draw_close_button("#cccccc")
					close_btn_canvas.config(cursor="arrow")
					extraction_win.update()
					
					result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
					
					if result.returncode == 0:
						# Success - count extracted frames
						extracted_count = len(list(output_dir.glob("frame_*.jpg")))
						video_stats_label.config(text=f"✓ Extraídos {extracted_count} frames en: {output_dir}")
						# Add extracted images to dataset
						extracted_images = [str(f) for f in output_dir.glob("frame_*.jpg")]
						for img in extracted_images:
							if img not in dataset_files:
								dataset_files.append(img)
						refresh_dataset_listbox()
					else:
						video_stats_label.config(text=f"✗ Error al extraer frames: {result.stderr[:100]}")
					
					draw_extract_button("#7ec331")
					extract_btn_canvas.config(cursor="hand2")
					draw_skip_button("#096bc9")
					skip_btn_canvas.config(cursor="hand2")
					draw_close_button("#ec5b2d")
					close_btn_canvas.config(cursor="hand2")
					
					# Move to next video
					current_video_idx["value"] += 1
					update_video_display()
					
				except Exception as e:
					video_stats_label.config(text=f"✗ Error: {str(e)[:100]}")
					draw_extract_button("#7ec331")
					extract_btn_canvas.config(cursor="hand2")
					draw_skip_button("#096bc9")
					skip_btn_canvas.config(cursor="hand2")
					draw_close_button("#ec5b2d")
					close_btn_canvas.config(cursor="hand2")
			
			def skip_video():
				"""Skip current video and move to next"""
				current_video_idx["value"] += 1
				update_video_display()
			
			def close_extraction():
				"""Close extraction window"""
				extraction_win.destroy()
			
			# Bind button click events
			extract_btn_canvas.bind("<Button-1>", lambda e: extract_frames())
			skip_btn_canvas.bind("<Button-1>", lambda e: skip_video())
			close_btn_canvas.bind("<Button-1>", lambda e: close_extraction())
			
			# Initialize with first video
			update_video_display()
		
		# ============================================================
		# TAB 3: Recorte
		# ============================================================
		recorte_tab = tabs[3]
		recorte_tab.configure(bg=PROC_CONTENT_BG)
		recorte_tab.grid_rowconfigure(0, weight=0)  # Instruction label
		recorte_tab.grid_rowconfigure(1, weight=0)  # Option label
		recorte_tab.grid_rowconfigure(2, weight=0)  # Iniciar recorte button
		recorte_tab.grid_rowconfigure(3, weight=0)  # Ruta row
		recorte_tab.grid_rowconfigure(4, weight=0)  # Progress bar
		recorte_tab.grid_rowconfigure(5, weight=1)  # Progress label
		recorte_tab.grid_columnconfigure(0, weight=1)
		
		# State for cropped images output directory
		cropped_output_dir = {"path": ""}
		
		# Label 1: Instruction
		instruction_label = tk.Label(
			recorte_tab,
			text="Proceder con el recorte de imágenes de personas de manera automática",
			font=("Arial", 14, "bold"),
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG
		)
		instruction_label.grid(row=0, column=0, sticky="w", padx=40, pady=(30, 10))
		
		# Label 2: Option description
		option_label = tk.Label(
			recorte_tab,
			text="Realizarlo con el dataset actual o agregar ruta de imágenes ya recortadas",
			font=("Arial", 12),
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG
		)
		option_label.grid(row=1, column=0, sticky="w", padx=40, pady=(0, 20))
		
		# Row 2: Iniciar recorte button (blue rounded button)
		iniciar_recorte_canvas = tk.Canvas(recorte_tab, width=150, height=40, bg=PROC_CONTENT_BG, highlightthickness=0, cursor="hand2")
		iniciar_recorte_canvas.grid(row=2, column=0, sticky="w", padx=40, pady=(0, 30))
		
		def draw_iniciar_recorte_button(color):
			iniciar_recorte_canvas.delete("all")
			x1, y1, x2, y2 = 5, 5, 145, 35
			r = 8
			iniciar_recorte_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
			iniciar_recorte_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
			iniciar_recorte_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
			iniciar_recorte_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
			iniciar_recorte_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
			iniciar_recorte_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
			iniciar_recorte_canvas.create_text(75, 20, text="Iniciar recorte", fill="white", font=("Arial", 11, "bold"))
		
		draw_iniciar_recorte_button("#015aca")
		
		def on_iniciar_recorte_enter(e):
			draw_iniciar_recorte_button("#0470d4")
		
		def on_iniciar_recorte_leave(e):
			draw_iniciar_recorte_button("#015aca")
		
		iniciar_recorte_canvas.bind("<Enter>", on_iniciar_recorte_enter)
		iniciar_recorte_canvas.bind("<Leave>", on_iniciar_recorte_leave)
		
		# Row 3: Ruta label + textbox + browse button
		ruta_label = tk.Label(
			recorte_tab,
			text="Ruta de imágenes recortadas:",
			font=("Arial", 14, "bold"),
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG
		)
		ruta_label.grid(row=3, column=0, sticky="w", padx=40, pady=(0, 10))
		
		ruta_row_frame = tk.Frame(recorte_tab, bg=PROC_CONTENT_BG)
		ruta_row_frame.grid(row=4, column=0, sticky="ew", padx=40, pady=(0, 20))
		
		ruta_entry = tk.Entry(ruta_row_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000", relief=tk.FLAT, bd=1, highlightthickness=1, highlightbackground="#CCCCCC", highlightcolor="#015aca")
		ruta_entry.grid(row=0, column=0, sticky="w", padx=(0, 5), ipadx=5, ipady=5)
		ruta_entry.config(width=150)
		
		# Custom rounded button for "Browse"
		ruta_browse_canvas = tk.Canvas(ruta_row_frame, width=100, height=40, bg=PROC_CONTENT_BG, highlightthickness=0, cursor="hand2")
		ruta_browse_canvas.grid(row=0, column=1, sticky="w")
		
		def draw_browse_button(color):
			ruta_browse_canvas.delete("all")
			x1, y1, x2, y2 = 5, 5, 95, 35
			r = 8
			ruta_browse_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=color, outline=color)
			ruta_browse_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=color, outline=color)
			ruta_browse_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=color, outline=color)
			ruta_browse_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=color, outline=color)
			ruta_browse_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
			ruta_browse_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)
			ruta_browse_canvas.create_text(50, 20, text="Browse", fill="white", font=("Arial", 11, "bold"))
		
		draw_browse_button("#015aca")
		
		def on_browse_enter(e):
			draw_browse_button("#0470d4")
		
		def on_browse_leave(e):
			draw_browse_button("#015aca")
		
		ruta_browse_canvas.bind("<Enter>", on_browse_enter)
		ruta_browse_canvas.bind("<Leave>", on_browse_leave)
		
		# Custom progress bar with rounded corners
		progress_canvas = tk.Canvas(recorte_tab, width=500, height=30, bg=PROC_CONTENT_BG, highlightthickness=0)
		progress_canvas.grid(row=5, column=0, sticky="w", padx=40, pady=(0, 0))
		
		# Store progress bar state
		progress_state = {"value": 0}
		
		def draw_progress_bar(percent):
			progress_canvas.delete("all")
			# Background (gray rounded rectangle)
			x1, y1, x2, y2 = 2, 2, 498, 28
			r = 8
			bg_color = "#002858"
			progress_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=bg_color, outline=bg_color)
			progress_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=bg_color, outline=bg_color)
			progress_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=bg_color, outline=bg_color)
			progress_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=bg_color, outline=bg_color)
			progress_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=bg_color, outline=bg_color)
			progress_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=bg_color, outline=bg_color)
			
			# Progress (blue rounded rectangle)
			if percent > 0:
				progress_width = int((496 * percent) / 100)  # 496 = 498 - 2 (margins)
				if progress_width > 0:
					px2 = x1 + progress_width
					progress_color = "#015aca"
					# Only draw right rounded corners if progress is near complete
					if progress_width > x2 - x1 - 2*r:
						# Full rounded on both sides
						progress_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_arc(px2-2*r, y1, px2, y1+2*r, start=0, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_arc(px2-2*r, y2-2*r, px2, y2, start=270, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_rectangle(x1+r, y1, px2-r, y2, fill=progress_color, outline=progress_color)
						progress_canvas.create_rectangle(x1, y1+r, px2, y2-r, fill=progress_color, outline=progress_color)
					else:
						# Rounded only on left side
						progress_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=progress_color, outline=progress_color)
						progress_canvas.create_rectangle(x1+r, y1, px2, y2, fill=progress_color, outline=progress_color)
						progress_canvas.create_rectangle(x1, y1+r, px2, y2-r, fill=progress_color, outline=progress_color)
			
			# Progress text
			progress_canvas.create_text(250, 15, text=f"{int(percent)}%", fill="white", font=("Arial", 10, "bold"))
		
		draw_progress_bar(0)
		
		# Progress label
		progress_label = tk.Label(
			recorte_tab,
			text="",
			font=("Arial", 10),
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG
		)
		progress_label.grid(row=6, column=0, sticky="w", padx=40, pady=(0, 20))
		
		# Functions for Recorte tab
		def browse_cropped_path():
			"""Browse for pre-cropped images folder"""
			folder = filedialog.askdirectory(title="Seleccionar carpeta de imágenes recortadas")
			if folder:
				cropped_output_dir["path"] = folder
				ruta_entry.delete(0, tk.END)
				ruta_entry.insert(0, folder)
				# Update dataset with cropped images
				folder_path = Path(folder)
				if folder_path.exists() and folder_path.is_dir():
					image_files = []
					for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
						image_files.extend([str(f) for f in folder_path.glob(f"*{ext}")])
					if image_files:
						# Clear current dataset and add cropped images
						dataset_files.clear()
						dataset_files.extend(image_files)
						refresh_dataset_listbox()
						can_advance[3] = True
						update_nav_state()
		
		ruta_browse_canvas.bind("<Button-1>", lambda e: browse_cropped_path())
		
		def run_person_cropping():
			"""Execute person cropping on dataset images/videos"""
			# Ask user to select input directory for images/videos to process
			selected_input_dir = filedialog.askdirectory(title="Seleccionar carpeta con imágenes/videos para recortar")
			if not selected_input_dir:
				# User cancelled
				return
			
			# Scan input directory for images and videos
			input_path = Path(selected_input_dir)
			input_files = []
			if input_path.exists() and input_path.is_dir():
				# Collect all images and videos
				for ext in IMAGE_EXTS | VIDEO_EXTS_DATASET:
					input_files.extend([str(f) for f in input_path.glob(f"*{ext}")])
			
			if not input_files:
				progress_label.config(text="Error: No se encontraron imágenes o videos en la carpeta seleccionada")
				return
			
			# Create output directory automatically in the same input folder
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			output_dir = input_path / f"crop_{timestamp}"
			output_dir.mkdir(parents=True, exist_ok=True)
			
			# Disable button during processing
			iniciar_recorte_canvas.delete("all")
			x1, y1, x2, y2 = 5, 5, 145, 35
			r = 8
			iniciar_recorte_canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_rectangle(x1+r, y1, x2-r, y2, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_rectangle(x1, y1+r, x2, y2-r, fill="#cccccc", outline="#cccccc")
			iniciar_recorte_canvas.create_text(75, 20, text="Procesando...", fill="white", font=("Arial", 11, "bold"))
			iniciar_recorte_canvas.config(cursor="arrow")
			draw_progress_bar(0)
			progress_label.config(text="Iniciando recorte...")
			train_win.update()
			
			try:
				# Import the cropping module
				import sys
				tools_path = _base_root() / "tools"
				if str(tools_path) not in sys.path:
					sys.path.insert(0, str(tools_path))
				
				# Import process_images_batch function using importlib (handles filenames with spaces)
				import importlib.util
				spec = importlib.util.spec_from_file_location(
					"recorte_personas_GUI",
					tools_path / "recorte_personas GUI.py"
				)
				if spec is None or spec.loader is None:
					raise ImportError(f"No se pudo cargar el módulo desde {tools_path / 'recorte_personas GUI.py'}")
				module = importlib.util.module_from_spec(spec)
				spec.loader.exec_module(module)
				process_images_batch = module.process_images_batch
				
				# Progress callback
				def update_progress(current, total, message):
					if total > 0:
						percent = int((current / total) * 100)
						draw_progress_bar(percent)
					progress_label.config(text=message)
					train_win.update()
				
				# Find model path
				model_path = _base_root() / "yolo11x.pt"
				if not model_path.exists():
					model_path = _base_root() / "yolo11m.pt"
				if not model_path.exists():
					model_path = "yolo11n.pt"  # Download if needed
				
				# Run cropping
				result_dir = process_images_batch(
					input_paths=input_files,
					output_dir=str(output_dir),
					model_path=str(model_path),
					conf_thres=0.5,
					skip_frames=25,
					rois=None,  # No ROI filtering for training datasets
					show=False,
					progress_callback=update_progress
				)
				
				# Update ruta entry with output directory
				cropped_output_dir["path"] = str(result_dir)
				ruta_entry.delete(0, tk.END)
				ruta_entry.insert(0, str(result_dir))
				
				# Update dataset with cropped images
				image_files = []
				for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
					image_files.extend([str(f) for f in Path(result_dir).glob(f"*{ext}")])
				
				if image_files:
					dataset_files.clear()
					dataset_files.extend(image_files)
					refresh_dataset_listbox()
				
				draw_progress_bar(100)
				progress_label.config(text=f"✓ Completado: {len(image_files)} imágenes recortadas")
				can_advance[3] = True
				update_nav_state()
				
			except Exception as e:
				progress_label.config(text=f"✗ Error: {str(e)[:100]}")
				import traceback
				traceback.print_exc()
			
			finally:
				# Re-enable button
				draw_iniciar_recorte_button("#015aca")
				iniciar_recorte_canvas.config(cursor="hand2")
		
		iniciar_recorte_canvas.bind("<Button-1>", lambda e: run_person_cropping())
		
		# ============================================================
		# TAB 4: Clasificacion
		# ============================================================
		clasificacion_tab = tabs[4]
		clasificacion_tab.configure(bg=PROC_CONTENT_BG)
		clasificacion_tab.grid_rowconfigure(0, weight=1)
		clasificacion_tab.grid_columnconfigure(0, weight=0, minsize=120)
		clasificacion_tab.grid_columnconfigure(1, weight=0)  # sash column
		clasificacion_tab.grid_columnconfigure(2, weight=1)
		
		# State for classification
		classification_classes = []  # List of {"name": str, "shortcut": str}
		current_image_index = {"value": 0}
		classification_images = []  # List of image paths to classify
		total_images_count = {"value": 0}  # Total images loaded initially
		current_photo = {"main": None, "previews": [None] * 10}  # Keep references for main + 10 previews
		classification_undo_stack = []  # List of {"img_path": str, "dest_path": str, "class_name": str, "idx": int}
		class_prev_counts = {}  # Tracks previous count per class to detect increase/decrease for flash
		
		# Left panel: Configuration menu
		left_panel = tk.Frame(clasificacion_tab, bg="#142443", width=300)
		left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 0))
		left_panel.grid_propagate(False)

		# Sash divider between left panel and right panel
		sash = tk.Frame(clasificacion_tab, bg="#2a3f6f", width=6, cursor="sb_h_double_arrow")
		sash.grid(row=0, column=1, sticky="nsew")

		_sash_drag = {"active": False, "start_x": 0, "start_w": 300}

		def _sash_press(event):
			_sash_drag["active"] = True
			_sash_drag["start_x"] = event.x_root
			_sash_drag["start_w"] = left_panel.winfo_width()

		def _sash_motion(event):
			if not _sash_drag["active"]:
				return
			delta = event.x_root - _sash_drag["start_x"]
			new_w = max(120, _sash_drag["start_w"] + delta)
			left_panel.configure(width=new_w)
			clasificacion_tab.grid_columnconfigure(0, weight=0, minsize=new_w)

		def _sash_release(event):
			_sash_drag["active"] = False

		sash.bind("<ButtonPress-1>", _sash_press)
		sash.bind("<B1-Motion>", _sash_motion)
		sash.bind("<ButtonRelease-1>", _sash_release)
		sash.bind("<Enter>", lambda e: sash.configure(bg="#4a6fa5"))
		sash.bind("<Leave>", lambda e: sash.configure(bg="#2a3f6f") if not _sash_drag["active"] else None)
		
		# Title for left panel
		config_title = tk.Label(
			left_panel,
			text="Configuración de Clases",
			font=("Arial", 18, "bold"),
			fg=FG_COLOR,
			bg="#142443"
		)
		config_title.pack(pady=(20, 10), padx=10)
		
		# Scrollable list of classes
		classes_frame = tk.Frame(left_panel, bg="#142443")
		classes_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
		
		classes_canvas = tk.Canvas(classes_frame, bg="#142443", highlightthickness=0)
		classes_scrollbar = tk.Scrollbar(classes_frame, orient="vertical", command=classes_canvas.yview)
		classes_canvas.configure(yscrollcommand=classes_scrollbar.set)
		
		classes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		classes_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		classes_inner = tk.Frame(classes_canvas, bg="#142443")
		classes_canvas_window = classes_canvas.create_window((0, 0), window=classes_inner, anchor="nw")
		
		def _update_classes_scroll(_=None):
			classes_canvas.configure(scrollregion=classes_canvas.bbox("all"))
		
		def _resize_classes_inner(_=None):
			w = classes_canvas.winfo_width()
			if w > 1:
				classes_canvas.itemconfig(classes_canvas_window, width=w)
		
		classes_inner.bind("<Configure>", _update_classes_scroll)
		classes_canvas.bind("<Configure>", _resize_classes_inner)
		
		# Load icons
		cross_icon = None
		plus_icon = None
		try:
			cross_img = Image.open(assets_root / "cross.png")
			cross_img = cross_img.resize((16, 16), Image.LANCZOS)
			cross_icon = ImageTk.PhotoImage(cross_img)
			
			plus_img = Image.open(assets_root / "plus.png")
			plus_img = plus_img.resize((16, 16), Image.LANCZOS)
			plus_icon = ImageTk.PhotoImage(plus_img)
		except Exception:
			pass
		
		class_widgets = []  # Store references to class entry widgets
		class_image_counts = {}  # Store image counts for each class
		
		def refresh_classes_list():
			"""Refresh the display of classification classes"""
			# Clear existing widgets
			for widget in classes_inner.winfo_children():
				widget.destroy()
			class_widgets.clear()
			
			# Add each class
			for idx, cls_info in enumerate(classification_classes):
				# Container for this class (no border, same bg as parent)
				class_container = tk.Frame(classes_inner, bg="#142443")
				class_container.pack(fill=tk.X, pady=(0, 10), padx=5)
				
				# Clase label and entry
				clase_row = tk.Frame(class_container, bg="#142443")
				clase_row.pack(fill=tk.X, pady=2)
				
				tk.Label(clase_row, text="Clase:", font=("Arial", 15, "bold"), fg=FG_COLOR, bg="#142443").pack(side=tk.LEFT, padx=(0, 5))
				class_entry = tk.Entry(clase_row, font=("Arial", 15), width=12, bg="#142443", fg="white", insertbackground="white", relief=tk.FLAT, bd=1)
				class_entry.insert(0, cls_info["name"])
				class_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
				
				# Tecla label and entry
				tecla_row = tk.Frame(class_container, bg="#142443")
				tecla_row.pack(fill=tk.X, pady=2)
				
				tk.Label(tecla_row, text="Tecla:", font=("Arial", 15, "bold"), fg=FG_COLOR, bg="#142443").pack(side=tk.LEFT, padx=(0, 5))
				shortcut_entry = tk.Entry(tecla_row, font=("Arial", 15), width=12, bg="#142443", fg="white", insertbackground="white", relief=tk.FLAT, bd=1)
				shortcut_entry.insert(0, cls_info["shortcut"])
				shortcut_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
				
				# Imagenes label
				imagenes_row = tk.Frame(class_container, bg="#142443")
				imagenes_row.pack(fill=tk.X, pady=2)
				
				tk.Label(imagenes_row, text="Imagenes:", font=("Arial", 15, "bold"), fg=FG_COLOR, bg="#142443").pack(side=tk.LEFT, padx=(0, 5))
				
				# Count images for this class
				image_count = class_image_counts.get(cls_info["name"], 0)
				count_label = tk.Label(imagenes_row, text=str(image_count), font=("Arial", 15), fg="white", bg="#142443")
				count_label.pack(side=tk.LEFT)
				
				# Flash the count label if the value changed
				prev_count = class_prev_counts.get(cls_info["name"])
				if prev_count is not None and image_count != prev_count:
					flash_color = "#4dabf7" if image_count > prev_count else "#f0a500"
					count_label.config(fg=flash_color)
					def _restore_label_color(lbl=count_label):
						if lbl.winfo_exists():
							lbl.config(fg="white")
					clases_inner_ref = classes_inner
					clases_inner_ref.after(450, _restore_label_color)
				class_prev_counts[cls_info["name"]] = image_count
				
				# Delimiter line
				delimiter = tk.Label(class_container, text="_________" * 6, font=("Arial", 10), fg="white", bg="#142443")
				delimiter.pack(fill=tk.X, pady=(5, 0))
				
				# Store references
				class_widgets.append({
					"frame": class_container,
					"class_entry": class_entry,
					"shortcut_entry": shortcut_entry,
					"count_label": count_label,
					"index": idx
				})
				
				# Bind update callbacks
				class_entry.bind("<FocusOut>", lambda e, i=idx: update_class_name(i))
				shortcut_entry.bind("<FocusOut>", lambda e, i=idx: update_class_shortcut(i))
		
		def delete_class(index):
			"""Delete a class from the list"""
			if 0 <= index < len(classification_classes):
				classification_classes.pop(index)
				refresh_classes_list()
		
		def update_class_name(index):
			"""Update class name from entry"""
			if 0 <= index < len(class_widgets):
				widget = class_widgets[index]
				new_name = widget["class_entry"].get().strip()
				if new_name and index < len(classification_classes):
					classification_classes[index]["name"] = new_name
		
		def update_class_shortcut(index):
			"""Update shortcut from entry"""
			if 0 <= index < len(class_widgets):
				widget = class_widgets[index]
				new_shortcut = widget["shortcut_entry"].get().strip()
				if new_shortcut and index < len(classification_classes):
					classification_classes[index]["shortcut"] = new_shortcut
		
		def add_new_class():
			"""Add a new class to the list"""
			classification_classes.append({"name": f"clase_{len(classification_classes)+1}", "shortcut": ""})
			refresh_classes_list()
		
		# Add button (Canvas-based rounded button)
		add_btn_frame = tk.Frame(left_panel, bg="#142443")
		add_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 30))
		
		add_btn_canvas = tk.Canvas(add_btn_frame, width=200, height=40, bg="#142443", highlightthickness=0)
		add_btn_canvas.pack()
		
		def draw_add_button(color):
			add_btn_canvas.delete("all")
			x0, y0, x1, y1 = 5, 5, 195, 35
			r = 10
			add_btn_canvas.create_arc(x0, y0, x0+2*r, y0+2*r, start=90, extent=90, fill=color, outline=color)
			add_btn_canvas.create_arc(x1-2*r, y0, x1, y0+2*r, start=0, extent=90, fill=color, outline=color)
			add_btn_canvas.create_arc(x0, y1-2*r, x0+2*r, y1, start=180, extent=90, fill=color, outline=color)
			add_btn_canvas.create_arc(x1-2*r, y1-2*r, x1, y1, start=270, extent=90, fill=color, outline=color)
			add_btn_canvas.create_rectangle(x0+r, y0, x1-r, y1, fill=color, outline=color)
			add_btn_canvas.create_rectangle(x0, y0+r, x1, y1-r, fill=color, outline=color)
			add_btn_canvas.create_text(100, 20, text="Nueva Clase", fill="white", font=("Arial", 11, "bold"))
		
		draw_add_button("#015aca")
		
		def on_add_btn_enter(event):
			draw_add_button("#0174e8")
		
		def on_add_btn_leave(event):
			draw_add_button("#015aca")
		
		def on_add_btn_click(event):
			add_new_class()
		
		add_btn_canvas.bind("<Enter>", on_add_btn_enter)
		add_btn_canvas.bind("<Leave>", on_add_btn_leave)
		add_btn_canvas.bind("<Button-1>", on_add_btn_click)
		
		# Finish button
		finish_btn_frame = tk.Frame(left_panel, bg="#142443")
		finish_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 30))
		
		finish_btn_canvas = tk.Canvas(finish_btn_frame, width=200, height=40, bg="#142443", highlightthickness=0)
		finish_btn_canvas.pack()
		
		def draw_finish_button(color):
			finish_btn_canvas.delete("all")
			x0, y0, x1, y1 = 5, 5, 195, 35
			r = 10
			finish_btn_canvas.create_arc(x0, y0, x0+2*r, y0+2*r, start=90, extent=90, fill=color, outline=color)
			finish_btn_canvas.create_arc(x1-2*r, y0, x1, y0+2*r, start=0, extent=90, fill=color, outline=color)
			finish_btn_canvas.create_arc(x0, y1-2*r, x0+2*r, y1, start=180, extent=90, fill=color, outline=color)
			finish_btn_canvas.create_arc(x1-2*r, y1-2*r, x1, y1, start=270, extent=90, fill=color, outline=color)
			finish_btn_canvas.create_rectangle(x0+r, y0, x1-r, y1, fill=color, outline=color)
			finish_btn_canvas.create_rectangle(x0, y0+r, x1, y1-r, fill=color, outline=color)
			finish_btn_canvas.create_text(100, 20, text="Finalizar Clasificación", fill="black", font=("Arial", 11, "bold"))
		
		draw_finish_button("#f8c644")
		
		def on_finish_btn_enter(event):
			draw_finish_button("#ffd454")
		
		def on_finish_btn_leave(event):
			draw_finish_button("#f8c644")
		
		def on_finish_btn_click(event):
			# Mark this tab as complete and jump to Data Augmentation tab (index 6)
			can_advance[4] = True
			update_nav_state()
			# Jump directly to Data Augmentation tab
			nonlocal current_step
			current_step = 6  # Data Augmentation tab index
			update_tabs_state()
		
		finish_btn_canvas.bind("<Enter>", on_finish_btn_enter)
		finish_btn_canvas.bind("<Leave>", on_finish_btn_leave)
		finish_btn_canvas.bind("<Button-1>", on_finish_btn_click)
		
		# Right panel: Main image display
		right_panel = tk.Frame(clasificacion_tab, bg=BG_COLOR)
		right_panel.grid(row=0, column=2, sticky="nsew")
		right_panel.grid_rowconfigure(0, weight=0)  # Instructions labels
		right_panel.grid_rowconfigure(1, weight=0)  # Progress label
		right_panel.grid_rowconfigure(2, weight=1)  # Main image
		right_panel.grid_rowconfigure(3, weight=0)  # Preview title
		right_panel.grid_rowconfigure(4, weight=0)  # Preview images
		right_panel.grid_rowconfigure(5, weight=0)  # Status bar
		right_panel.grid_columnconfigure(0, weight=1)
		
		# Instruction label
		instruction_label = tk.Label(
			right_panel, 
			text="Clasifica la imagen presionando la tecla correspondiente a la clase identificada",
			font=("Arial", 20),
			fg="white",
			bg=BG_COLOR
		)
		instruction_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
		
		# Progress label
		progress_label = tk.Label(
			right_panel,
			text="Imágenes clasificadas: 0 de 0 (0%)",
			font=("Arial", 15),
			fg="#ff7514",
			bg=BG_COLOR
		)
		progress_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
		
		# Main image display area
		main_image_frame = tk.Frame(right_panel, bg="black", height=700)
		main_image_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
		main_image_frame.pack_propagate(False)
		
		main_image_label = tk.Label(main_image_frame, bg="black", text="No hay imágenes para clasificar", fg=FG_COLOR, font=("Arial", 14))
		main_image_label.pack(expand=True)
		
		# Click on image to remove focus from textboxes
		def on_image_click(event):
			"""Remove focus from textboxes when clicking on image"""
			main_image_label.focus_set()
		
		main_image_label.bind("<Button-1>", on_image_click)
		main_image_frame.bind("<Button-1>", on_image_click)
		
		# Preview title
		preview_title = tk.Label(right_panel, text="Imágenes siguientes:", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
		preview_title.grid(row=3, column=0, sticky="w", padx=10, pady=(5, 5))
		
		# Preview area (bottom)
		preview_frame = tk.Frame(right_panel, bg=BG_COLOR, height=120)
		preview_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
		preview_frame.pack_propagate(False)
		
		# Container for 10 preview images
		preview_images_container = tk.Frame(preview_frame, bg=BG_COLOR)
		preview_images_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		
		# Create 10 preview labels
		preview_labels = []
		for i in range(10):
			preview_label = tk.Label(preview_images_container, bg="black", text="", fg=FG_COLOR, width=100, height=100)
			preview_label.pack(side=tk.LEFT, padx=2)
			preview_labels.append(preview_label)
		
		# Status bar frame (holds status label + undo hint)
		status_bar_frame = tk.Frame(right_panel, bg=BG_COLOR)
		status_bar_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))
		status_label = tk.Label(status_bar_frame, text="Presiona las teclas de acceso rápido para clasificar", font=("Arial", 10), fg=FG_COLOR, bg=BG_COLOR)
		status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
		undo_hint_label = tk.Label(status_bar_frame, text="Z para deshacer", font=("Arial", 10, "italic"), fg="#f0a500", bg=BG_COLOR)
		undo_hint_label.pack(side=tk.RIGHT)
		
		def load_images_for_classification():
			"""Load images from the cropped images directory"""
			classification_images.clear()
			current_image_index["value"] = 0
			class_image_counts.clear()
			
			# Get images from ruta entry in Recorte tab
			unclassified_count = 0
			total_classified = 0
			
			if cropped_output_dir["path"]:
				source_dir = Path(cropped_output_dir["path"])
				if source_dir.exists() and source_dir.is_dir():
					# Load unclassified images (images directly in the folder)
					for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
						classification_images.extend([str(f) for f in source_dir.glob(f"*{ext}")])
					
					unclassified_count = len(classification_images)
					
					# Scan class folders to count already classified images
					for item in source_dir.iterdir():
						if item.is_dir():
							class_name = item.name
							# Count images in this class folder
							class_count = 0
							for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
								class_count += len(list(item.glob(f"*{ext}")))
							
							if class_count > 0:
								class_image_counts[class_name] = class_count
								total_classified += class_count
			
			# Sort images
			classification_images.sort()
			
			# Calculate totals
			total_images = unclassified_count + total_classified
			total_images_count["value"] = total_images
			
			# Refresh classes list to show counts
			refresh_classes_list()
			
			# Update progress label
			if total_images > 0:
				percent = int((total_classified / total_images * 100)) if total_images > 0 else 0
				progress_label.config(text=f"Imágenes clasificadas: {total_classified} de {total_images} ({percent}%)")
			else:
				progress_label.config(text="Imágenes clasificadas: 0 de 0 (0%)")
			
			if classification_images:
				display_current_image()
			else:
				main_image_label.config(image="", text="No hay imágenes para clasificar")
				for preview_label in preview_labels:
					preview_label.config(image="", text="")
		
		def display_current_image():
			"""Display the current image and 5 preview images"""
			idx = current_image_index["value"]
			
			# Check if all images are classified
			if not classification_images or idx >= len(classification_images):
				# All images classified - hide everything and show completion message
				main_image_label.config(image="", text="Todas las imágenes clasificadas", fg="white", font=("Arial", 24, "bold"))
				instruction_label.config(text="")
				progress_label.config(text="")
				preview_title.config(text="")
				for preview_label in preview_labels:
					preview_label.config(image="", text="")
				status_label.config(text="")
				can_advance[4] = True
				update_nav_state()
				return
			
			# Load main image
			try:
				img_path = classification_images[idx]
				img = Image.open(img_path)
				
				# Resize to fit display (max 800x600)
				max_w, max_h = 800, 600
				ratio = min(max_w / img.width, max_h / img.height)
				new_w = int(img.width * ratio)
				new_h = int(img.height * ratio)
				img = img.resize((new_w, new_h), Image.LANCZOS)
				
				photo = ImageTk.PhotoImage(img)
				current_photo["main"] = photo
				main_image_label.config(image=photo, text="")
				
				# Update status
				status_label.config(text=f"Imagen {idx+1}/{len(classification_images)}: {Path(img_path).name}")
				
			except Exception as e:
				main_image_label.config(image="", text=f"Error al cargar imagen: {str(e)[:50]}")
			
			# Load 10 preview images
			for i in range(10):
				preview_idx = idx + i + 1
				if preview_idx < len(classification_images):
					try:
						preview_path = classification_images[preview_idx]
						preview_img = Image.open(preview_path)
						
						# Resize to thumbnail (60x60 for 10 images)
						preview_img.thumbnail((60, 60), Image.LANCZOS)
						preview_photo = ImageTk.PhotoImage(preview_img)
						current_photo["previews"][i] = preview_photo
						preview_labels[i].config(image=preview_photo, text="")
						
					except Exception:
						current_photo["previews"][i] = None
						preview_labels[i].config(image="", text="")
				else:
					current_photo["previews"][i] = None
					preview_labels[i].config(image="", text="")
		
		def classify_current_image(class_name):
			"""Classify current image and move to class folder"""
			if not classification_images:
				return
			
			idx = current_image_index["value"]
			if idx >= len(classification_images):
				return
			
			img_path = Path(classification_images[idx])
			
			# Create class folder
			class_folder = img_path.parent / class_name
			class_folder.mkdir(exist_ok=True)
			
			# Move image
			try:
				import shutil
				dest_path = class_folder / img_path.name
				shutil.move(str(img_path), str(dest_path))
				
				# Save undo record BEFORE modifying state
				classification_undo_stack.append({
					"img_path": str(img_path),
					"dest_path": str(dest_path),
					"class_name": class_name,
					"idx": idx,
				})
				
				# Update image count for this class
				class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1
				refresh_classes_list()
				
				# Remove from list
				classification_images.pop(idx)
				
				# Update progress label
				classified = total_images_count["value"] - len(classification_images)
				total = total_images_count["value"]
				percent = int((classified / total * 100)) if total > 0 else 0
				progress_label.config(text=f"Imágenes clasificadas: {classified} de {total} ({percent}%)")
				
				# Display next image (same index, since we removed current)
				display_current_image()
				
			except Exception as e:
				status_label.config(text=f"Error al mover imagen: {str(e)[:80]}")

		def undo_classification():
			"""Undo the last classification: move image back to original folder and re-insert it."""
			if not classification_undo_stack:
				status_label.config(text="No hay acciones para deshacer")
				return
			record = classification_undo_stack.pop()
			img_original = record["img_path"]
			dest_path = record["dest_path"]
			class_name = record["class_name"]
			original_idx = record["idx"]
			try:
				import shutil
				shutil.move(dest_path, img_original)
				# Re-insert at original position (clamped to valid range)
				insert_idx = min(original_idx, len(classification_images))
				classification_images.insert(insert_idx, img_original)
				current_image_index["value"] = insert_idx
				# Update class count
				if class_name in class_image_counts and class_image_counts[class_name] > 0:
					class_image_counts[class_name] -= 1
					if class_image_counts[class_name] == 0:
						del class_image_counts[class_name]
				refresh_classes_list()
				# Update progress label
				classified = total_images_count["value"] - len(classification_images)
				total = total_images_count["value"]
				percent = int((classified / total * 100)) if total > 0 else 0
				progress_label.config(text=f"Imágenes clasificadas: {classified} de {total} ({percent}%)")
				display_current_image()
				status_label.config(text=f"Deshecho: '{class_name}' → {Path(img_original).name}")
			except Exception as e:
				status_label.config(text=f"Error al deshacer: {str(e)[:80]}")
		
		def on_key_press(event):
			"""Handle keyboard shortcuts for classification"""
			# Check if focus is on an Entry widget (textbox)
			focused_widget = train_win.focus_get()
			if isinstance(focused_widget, tk.Entry):
				# Don't process shortcuts when typing in textboxes
				return
			
			key = event.char.lower()
			
			# Undo last classification
			if key == "z":
				undo_classification()
				return
			
			# Check if key matches any class shortcut
			for cls_info in classification_classes:
				if cls_info["shortcut"].lower() == key:
					classify_current_image(cls_info["name"])
					return
		
		# Bind keyboard events to classification tab
		def bind_classification_keys():
			"""Bind keyboard shortcuts when tab is active"""
			train_win.bind("<Key>", on_key_press)
		
		def unbind_classification_keys():
			"""Unbind keyboard shortcuts when leaving tab"""
			train_win.unbind("<Key>")
		
		# Add some default classes
		classification_classes.append({"name": "trabajando", "shortcut": "w"})
		classification_classes.append({"name": "idle", "shortcut": "a"})
		classification_classes.append({"name": "desconocido", "shortcut": "d"})
		classification_classes.append({"name": "trash", "shortcut": "s"})
		refresh_classes_list()
		
		# Load images when tab is displayed
		def on_clasificacion_tab_focus():
			"""Called when classification tab becomes active"""
			load_images_for_classification()
			bind_classification_keys()
		
		# Note: Tab focus binding handled in on_tab_changed callback
		
		# ============================================================
		# TAB 5: Deteccion (Object Detection Annotation Tool)
		# ============================================================
		deteccion_tab = tabs[5]
		deteccion_tab.configure(bg=PROC_CONTENT_BG)
		deteccion_tab.grid_rowconfigure(0, weight=1)
		deteccion_tab.grid_columnconfigure(0, weight=0, minsize=300)
		deteccion_tab.grid_columnconfigure(1, weight=1)
		
		# Configure combobox style for detection classes
		det_combo_style = ttk.Style()
		det_combo_style.configure(
			"DetClass.TCombobox",
			fieldbackground="#02224e",
			background="#02224e",
			foreground="white",
			selectbackground="#096bc9",
			selectforeground="white"
		)
		det_combo_style.map(
			"DetClass.TCombobox",
			fieldbackground=[("readonly", "#02224e")],
			selectbackground=[("readonly", "#02224e")]
		)
		
		def on_deteccion_tab_focus():
			"""Called when detection tab becomes active"""
			load_images_for_detection()
		
		# State for detection annotation
		det_images_list = []  # List of image paths
		det_current_image_idx = {"value": 0}
		det_annotations = {}  # Dict: image_path -> [{"class": str, "bbox": [x, y, w, h], "color": str}]
		det_classes = []  # List of {"name": str, "color": str}
		det_selected_images = set()  # Set of selected image indices
		det_selected_labels = set()  # Set of selected label indices for current image
		det_dataset_path = {"value": None}  # Path to the dataset folder for auto-saving
		
		def auto_save_detection_progress():
			"""Automatically save detection progress (annotations and classes) to dataset folder"""
			if not det_dataset_path["value"]:
				return
			
			try:
				import json
				dataset_folder = Path(det_dataset_path["value"])
				
				if not dataset_folder.exists():
					return
				
				# Save annotations (images with their labels)
				annotations_file = dataset_folder / "annotations.json"
				annotations_data = {}
				for img_path, anns in det_annotations.items():
					# Convert to relative path or just filename
					img_name = Path(img_path).name
					annotations_data[img_name] = anns
				
				with open(annotations_file, 'w', encoding='utf-8') as f:
					json.dump(annotations_data, f, indent=2)
				
				# Save classes
				classes_file = dataset_folder / "classes.json"
				with open(classes_file, 'w', encoding='utf-8') as f:
					json.dump(det_classes, f, indent=2)
				
			except Exception as e:
				print(f"Error al guardar progreso de detección: {e}")
		
		# Left panel: Vertical menu with two subsections
		det_left_panel = tk.Frame(deteccion_tab, bg=GRAY_BG, width=300)
		det_left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
		det_left_panel.grid_propagate(False)
		det_left_panel.grid_rowconfigure(0, weight=0)  # Images section
		det_left_panel.grid_rowconfigure(1, weight=1)  # Labels section
		det_left_panel.grid_columnconfigure(0, weight=1)
		
		# ===== SUBSECTION 1: Imagenes =====
		images_section = tk.Frame(det_left_panel, bg=GRAY_BG)
		images_section.grid(row=0, column=0, sticky="nsew", padx=5, pady=(10, 5))
		
		images_title = tk.Label(
			images_section,
			text="Imagenes",
			font=("Arial", 13, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		images_title.pack(pady=(5, 5))
		
		# Scrollable list of images
		images_list_frame = tk.Frame(images_section, bg=GRAY_BG)
		images_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
		
		images_canvas = tk.Canvas(images_list_frame, bg=BG_COLOR, highlightthickness=0, height=200)
		images_scrollbar = tk.Scrollbar(images_list_frame, orient="vertical", command=images_canvas.yview)
		images_canvas.configure(yscrollcommand=images_scrollbar.set)
		
		images_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		images_inner = tk.Frame(images_canvas, bg=BG_COLOR)
		images_canvas_window = images_canvas.create_window((0, 0), window=images_inner, anchor="nw")
		
		def _update_images_scroll(_=None):
			images_canvas.configure(scrollregion=images_canvas.bbox("all"))
		
		def _resize_images_inner(_=None):
			w = images_canvas.winfo_width()
			if w > 1:
				images_canvas.itemconfig(images_canvas_window, width=w)
		
		images_inner.bind("<Configure>", _update_images_scroll)
		images_canvas.bind("<Configure>", _resize_images_inner)
		
		# Buttons frame
		images_buttons_frame = tk.Frame(images_section, bg=GRAY_BG)
		images_buttons_frame.pack(fill=tk.X, pady=5)
		
		add_images_btn_canvas = create_rounded_button(
			images_buttons_frame,
			text="Agregar imagenes...",
			command=lambda: add_detection_images(),
			bg_color="#7ec331",
			fg_color=FG_COLOR,
			active_bg="#6ba829",
			width=280
		)
		add_images_btn_canvas.pack(fill=tk.X, pady=2)
		
		remove_images_btn_canvas = create_rounded_button(
			images_buttons_frame,
			text="Remover imagenes seleccionadas",
			command=lambda: remove_selected_images(),
			bg_color="#ec5b2d",
			fg_color=FG_COLOR,
			active_bg="#d44a20",
			width=280
		)
		remove_images_btn_canvas.pack(fill=tk.X, pady=2)
		
		manage_classes_btn_canvas = create_rounded_button(
			images_buttons_frame,
			text="Manejar clases",
			command=lambda: manage_detection_classes(),
			bg_color="#096bc9",
			fg_color=FG_COLOR,
			active_bg="#075a9e",
			width=280
		)
		manage_classes_btn_canvas.pack(fill=tk.X, pady=2)
		
		# ===== SUBSECTION 2: Etiquetas de la imagen =====
		labels_section = tk.Frame(det_left_panel, bg=GRAY_BG)
		labels_section.grid(row=1, column=0, sticky="nsew", padx=5, pady=(5, 10))
		labels_section.grid_rowconfigure(0, weight=0)
		labels_section.grid_rowconfigure(1, weight=1)
		labels_section.grid_rowconfigure(2, weight=0)
		labels_section.grid_columnconfigure(0, weight=1)
		
		labels_title = tk.Label(
			labels_section,
			text="Etiquetas",
			font=("Arial", 13, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		labels_title.grid(row=0, column=0, pady=(5, 5))
		
		# Scrollable list of labels
		labels_list_frame = tk.Frame(labels_section, bg=GRAY_BG)
		labels_list_frame.grid(row=1, column=0, sticky="nsew", pady=5)
		labels_list_frame.grid_rowconfigure(0, weight=1)
		labels_list_frame.grid_columnconfigure(0, weight=1)
		
		labels_canvas = tk.Canvas(labels_list_frame, bg=BG_COLOR, highlightthickness=0)
		labels_scrollbar = tk.Scrollbar(labels_list_frame, orient="vertical", command=labels_canvas.yview)
		labels_canvas.configure(yscrollcommand=labels_scrollbar.set)
		
		labels_scrollbar.grid(row=0, column=1, sticky="ns")
		labels_canvas.grid(row=0, column=0, sticky="nsew")
		
		labels_inner = tk.Frame(labels_canvas, bg=BG_COLOR)
		labels_canvas_window = labels_canvas.create_window((0, 0), window=labels_inner, anchor="nw")
		
		def _update_labels_scroll(_=None):
			labels_canvas.configure(scrollregion=labels_canvas.bbox("all"))
		
		def _resize_labels_inner(_=None):
			w = labels_canvas.winfo_width()
			if w > 1:
				labels_canvas.itemconfig(labels_canvas_window, width=w)
		
		labels_inner.bind("<Configure>", _update_labels_scroll)
		labels_canvas.bind("<Configure>", _resize_labels_inner)
		
		# Remove labels button
		remove_labels_btn_canvas = create_rounded_button(
			labels_section,
			text="Remover etiquetas seleccionadas",
			command=lambda: remove_selected_labels(),
			bg_color="#ec5b2d",
			fg_color=FG_COLOR,
			active_bg="#d44a20",
			width=280
		)
		remove_labels_btn_canvas.grid(row=2, column=0, sticky="ew", pady=5)
		
		# Right panel: Main working area
		det_right_panel = tk.Frame(deteccion_tab, bg=BG_COLOR)
		det_right_panel.grid(row=0, column=1, sticky="nsew")
		det_right_panel.grid_rowconfigure(0, weight=0)  # Top menu bar
		det_right_panel.grid_rowconfigure(1, weight=1)  # Main canvas
		det_right_panel.grid_columnconfigure(0, weight=1)
		
		# Top horizontal menu bar
		det_top_menu = tk.Frame(det_right_panel, bg=GRAY_BG, height=40)
		det_top_menu.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
		det_top_menu.grid_columnconfigure(0, weight=0)  # Left buttons
		det_top_menu.grid_columnconfigure(1, weight=1)  # Spacer
		det_top_menu.grid_columnconfigure(2, weight=0)  # Right section (classes)
		
		# Left side: Zoom controls
		zoom_frame = tk.Frame(det_top_menu, bg=GRAY_BG)
		zoom_frame.grid(row=0, column=0, sticky="w", padx=5)
		
		det_zoom_level = {"value": 1.0}  # 1.0 = 100%
		det_zoom_mode = {"value": "fit"}  # "fit" or "custom"
		
		def zoom_fit():
			det_zoom_mode["value"] = "fit"
			det_canvas_offset["x"] = 0
			det_canvas_offset["y"] = 0
			if det_images_list and det_current_image_idx["value"] < len(det_images_list):
				display_detection_image(det_current_image_idx["value"])
		
		def zoom_out():
			det_zoom_mode["value"] = "custom"
			det_zoom_level["value"] = max(0.1, det_zoom_level["value"] - 0.1)
			if det_images_list and det_current_image_idx["value"] < len(det_images_list):
				display_detection_image(det_current_image_idx["value"])
		
		def zoom_in():
			det_zoom_mode["value"] = "custom"
			det_zoom_level["value"] = min(5.0, det_zoom_level["value"] + 0.1)
			if det_images_list and det_current_image_idx["value"] < len(det_images_list):
				display_detection_image(det_current_image_idx["value"])
		
		def zoom_100():
			det_zoom_mode["value"] = "custom"
			det_zoom_level["value"] = 1.0
			if det_images_list and det_current_image_idx["value"] < len(det_images_list):
				display_detection_image(det_current_image_idx["value"])
		
		tk.Button(zoom_frame, text="Fit", command=zoom_fit, bg="#607D8B", fg=FG_COLOR, width=6, font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
		tk.Button(zoom_frame, text="-", command=zoom_out, bg="#607D8B", fg=FG_COLOR, width=3, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2)
		tk.Button(zoom_frame, text="+", command=zoom_in, bg="#607D8B", fg=FG_COLOR, width=3, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2)
		tk.Button(zoom_frame, text="100%", command=zoom_100, bg="#607D8B", fg=FG_COLOR, width=6, font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
		
		# Right side: Classes selector
		classes_frame = tk.Frame(det_top_menu, bg=GRAY_BG)
		classes_frame.grid(row=0, column=2, sticky="e", padx=5)
		
		tk.Label(classes_frame, text="Clases:", font=("Arial", 10, "bold"), fg=FG_COLOR, bg=GRAY_BG).pack(side=tk.LEFT, padx=(0, 10))
		
		det_selected_class = {"value": None}  # Currently selected class for drawing
		classes_buttons_container = tk.Frame(classes_frame, bg=GRAY_BG)
		classes_buttons_container.pack(side=tk.LEFT)
		
		def refresh_classes_buttons():
			"""Refresh the class selection buttons in the top menu"""
			for widget in classes_buttons_container.winfo_children():
				widget.destroy()
			
			for idx, cls_info in enumerate(det_classes):
				cls_name = cls_info["name"]
				cls_color = cls_info["color"]
				
				def make_class_select(class_idx):
					def select():
						det_selected_class["value"] = class_idx
						refresh_classes_buttons()
					return select
				
				is_selected = det_selected_class["value"] == idx
				btn = tk.Button(
					classes_buttons_container,
					text=cls_name,
					command=make_class_select(idx),
					bg=cls_color if is_selected else GRAY_BG,
					fg="#000000" if is_selected else FG_COLOR,
					font=("Arial", 9, "bold" if is_selected else "normal"),
					relief=tk.SUNKEN if is_selected else tk.RAISED,
					width=10
				)
				btn.pack(side=tk.LEFT, padx=2)
		
		refresh_classes_buttons()
		
		# Canvas for image display
		det_main_canvas = tk.Canvas(det_right_panel, bg="black", highlightthickness=0)
		det_main_canvas.grid(row=1, column=0, sticky="nsew")
		
		det_current_photo = {"image": None}  # Keep reference to avoid garbage collection
		det_canvas_image_id = {"id": None}  # Canvas image object ID
		det_canvas_offset = {"x": 0, "y": 0}  # Canvas pan offset
		det_drag_data = {"x": 0, "y": 0, "dragging": False}  # For right-click drag
		det_image_info = {  # Current image display information
			"original_width": 0,
			"original_height": 0,
			"display_width": 0,
			"display_height": 0,
			"canvas_x": 0,  # Top-left corner of image in canvas
			"canvas_y": 0
		}
		
		# Right-click drag handlers for panning
		def on_canvas_press(event):
			"""Start dragging with right click"""
			det_drag_data["x"] = event.x
			det_drag_data["y"] = event.y
			det_drag_data["dragging"] = True
			det_main_canvas.config(cursor="fleur")
		
		def on_canvas_drag(event):
			"""Handle dragging motion - optimized for smooth performance"""
			if det_drag_data["dragging"] and det_canvas_image_id["id"] is not None:
				dx = event.x - det_drag_data["x"]
				dy = event.y - det_drag_data["y"]
				det_drag_data["x"] = event.x
				det_drag_data["y"] = event.y
				
				# Update offset
				det_canvas_offset["x"] += dx
				det_canvas_offset["y"] += dy
				
				# Update image info canvas position for coordinate conversions
				det_image_info["canvas_x"] += dx
				det_image_info["canvas_y"] += dy
				
				# Move image and all bounding boxes together (much faster than redrawing)
				det_main_canvas.move(det_canvas_image_id["id"], dx, dy)
				det_main_canvas.move("bbox", dx, dy)  # Move all bounding boxes with image
		
		def on_canvas_release(event):
			"""Stop dragging"""
			det_drag_data["dragging"] = False
			det_main_canvas.config(cursor="")
		
		# Bind right-click drag events for panning
		det_main_canvas.bind("<ButtonPress-3>", on_canvas_press)
		det_main_canvas.bind("<B3-Motion>", on_canvas_drag)
		det_main_canvas.bind("<ButtonRelease-3>", on_canvas_release)
		
		# Left-click drawing handlers for bounding boxes
		det_bbox_draw = {"start_x": 0, "start_y": 0, "rect_id": None, "drawing": False}
		
		# Bbox editing state
		det_bbox_edit = {
			"active": False,  # Whether we're in edit mode
			"bbox_idx": None,  # Index of bbox being edited
			"handle": None,  # Which handle is being dragged (0-3 for corners, "move" for moving entire bbox)
			"start_x": 0,
			"start_y": 0,
			"original_bbox": None  # Copy of original bbox coords
		}
		
		def on_bbox_press(event):
			"""Start drawing bounding box or enter edit mode if clicking inside existing bbox"""
			# First check if clicking on an edit handle
			if det_bbox_edit["active"]:
				handle_idx = check_handle_click(event.x, event.y)
				if handle_idx is not None:
					# Start dragging a handle
					det_bbox_edit["handle"] = handle_idx
					det_bbox_edit["start_x"] = event.x
					det_bbox_edit["start_y"] = event.y
					return
				
				# Check if clicking inside the currently edited bbox
				bbox_idx = check_bbox_click(event.x, event.y)
				if bbox_idx == det_bbox_edit["bbox_idx"]:
					# Clicked inside the same bbox but not on handle - start moving entire bbox
					det_bbox_edit["handle"] = "move"
					det_bbox_edit["start_x"] = event.x
					det_bbox_edit["start_y"] = event.y
					det_main_canvas.config(cursor="fleur")  # Change cursor to move cursor
					return
				elif bbox_idx is not None:
					# Clicked on a different bbox - switch to editing that one
					enter_bbox_edit_mode(bbox_idx)
					return
				else:
					# Clicked outside all bboxes - exit edit mode
					exit_bbox_edit_mode()
					# Don't return yet, fall through to normal drawing mode
			
			# Check if clicking inside an existing bbox (when not in edit mode)
			if not det_bbox_edit["active"]:
				bbox_idx = check_bbox_click(event.x, event.y)
				if bbox_idx is not None:
					# Enter edit mode for this bbox
					enter_bbox_edit_mode(bbox_idx)
					return
			
			# Normal drawing mode
			if det_selected_class["value"] is None:
				return  # No class selected, can't draw
			
			det_bbox_draw["start_x"] = event.x
			det_bbox_draw["start_y"] = event.y
			det_bbox_draw["drawing"] = True
			det_bbox_draw["rect_id"] = None
		
		def on_bbox_drag(event):
			"""Draw temporary bounding box while dragging or resize bbox if editing"""
			# Handle bbox editing (dragging a handle or moving entire bbox)
			if det_bbox_edit["active"] and det_bbox_edit["handle"] is not None:
				if det_bbox_edit["handle"] == "move":
					move_bbox_by_drag(event.x, event.y)
				else:
					resize_bbox_by_handle(event.x, event.y)
				return
			
			if not det_bbox_draw["drawing"]:
				return
			
			# Delete previous temporary rectangle
			if det_bbox_draw["rect_id"] is not None:
				det_main_canvas.delete(det_bbox_draw["rect_id"])
			
			# Get selected class color
			if det_selected_class["value"] is not None and det_selected_class["value"] < len(det_classes):
				color = det_classes[det_selected_class["value"]]["color"]
			else:
				color = "#00FF00"
			
			# Draw new temporary rectangle with semi-transparent fill
			det_bbox_draw["rect_id"] = det_main_canvas.create_rectangle(
				det_bbox_draw["start_x"], det_bbox_draw["start_y"],
				event.x, event.y,
				outline=color,
				fill=color,
				stipple="gray12",  # Creates ~10% opacity effect (90% transparency)
				width=2,
				tags="temp_bbox"
			)
		
		def on_bbox_release(event):
			"""Finalize bounding box on release or finish handle drag"""
			# Handle bbox editing (releasing a handle or finishing move)
			if det_bbox_edit["active"] and det_bbox_edit["handle"] is not None:
				det_bbox_edit["handle"] = None
				det_main_canvas.config(cursor="")  # Restore cursor
				# Redraw with handles
				draw_edit_handles()
				# Refresh labels list to show updated coords
				refresh_labels_list()
				return
			
			if not det_bbox_draw["drawing"]:
				return
			
			det_bbox_draw["drawing"] = False
			
			# Delete temporary rectangle
			if det_bbox_draw["rect_id"] is not None:
				det_main_canvas.delete(det_bbox_draw["rect_id"])
				det_bbox_draw["rect_id"] = None
			
			# Check if we have a valid selection
			if det_selected_class["value"] is None or not det_images_list:
				return
			
			# Calculate bbox coordinates (ensure min < max)
			x1 = min(det_bbox_draw["start_x"], event.x)
			y1 = min(det_bbox_draw["start_y"], event.y)
			x2 = max(det_bbox_draw["start_x"], event.x)
			y2 = max(det_bbox_draw["start_y"], event.y)
			
			# Ignore tiny boxes (likely accidental clicks)
			if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
				return
			
			# Get current image path
			current_img = det_images_list[det_current_image_idx["value"]]
			
			# Get class info
			cls_info = det_classes[det_selected_class["value"]]
			cls_name = cls_info["name"]
			cls_color = cls_info["color"]
			
			# Convert canvas coordinates to image coordinates
			# Account for display scale and position
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return  # Can't convert without image info
			
			# Calculate scale factor
			scale_x = det_image_info["original_width"] / det_image_info["display_width"]
			scale_y = det_image_info["original_height"] / det_image_info["display_height"]
			
			# Convert canvas coords to image coords relative to displayed image
			img_x1 = (x1 - det_image_info["canvas_x"]) * scale_x
			img_y1 = (y1 - det_image_info["canvas_y"]) * scale_y
			img_x2 = (x2 - det_image_info["canvas_x"]) * scale_x
			img_y2 = (y2 - det_image_info["canvas_y"]) * scale_y
			
			# Clamp to image boundaries
			img_x1 = max(0, min(img_x1, det_image_info["original_width"]))
			img_y1 = max(0, min(img_y1, det_image_info["original_height"]))
			img_x2 = max(0, min(img_x2, det_image_info["original_width"]))
			img_y2 = max(0, min(img_y2, det_image_info["original_height"]))
			
			# Store as [x, y, width, height] in original image coordinates
			bbox = [img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1]
			
			# Add annotation
			if current_img not in det_annotations:
				det_annotations[current_img] = []
			
			det_annotations[current_img].append({
				"class": cls_name,
				"bbox": bbox,
				"color": cls_color
			})
			
			# Redraw image with all bounding boxes
			redraw_bounding_boxes()
			
			# Refresh labels list
			refresh_labels_list()
			
			# Refresh images list to update color (green if has labels)
			refresh_images_list()
			
			# Check if all images are annotated to enable next button
			check_all_images_annotated()
			
			# Auto-save progress after adding annotation
			auto_save_detection_progress()
		
		def check_all_images_annotated():
			"""Check if all images have at least one annotation and update can_advance[5]"""
			if not det_images_list:
				can_advance[5] = False
			else:
				# Check if all images have at least one annotation
				all_annotated = all(
					img_path in det_annotations and len(det_annotations[img_path]) > 0
					for img_path in det_images_list
				)
				can_advance[5] = all_annotated
			
			# Update navigation buttons state
			update_nav_state()
		
		def check_bbox_click(x, y):
			"""Check if click is inside any existing bbox. Returns bbox index or None."""
			if not det_images_list:
				return None
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return None
			
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return None
			
			# Calculate scale factor
			scale_x = det_image_info["display_width"] / det_image_info["original_width"]
			scale_y = det_image_info["display_height"] / det_image_info["original_height"]
			
			# Check each bbox (in reverse order so we select the top-most one)
			for idx in range(len(det_annotations[current_img]) - 1, -1, -1):
				ann = det_annotations[current_img][idx]
				bbox = ann["bbox"]
				
				# Convert bbox to canvas coords
				canvas_x1 = bbox[0] * scale_x + det_image_info["canvas_x"]
				canvas_y1 = bbox[1] * scale_y + det_image_info["canvas_y"]
				canvas_x2 = (bbox[0] + bbox[2]) * scale_x + det_image_info["canvas_x"]
				canvas_y2 = (bbox[1] + bbox[3]) * scale_y + det_image_info["canvas_y"]
				
				# Check if click is inside
				if canvas_x1 <= x <= canvas_x2 and canvas_y1 <= y <= canvas_y2:
					return idx
			
			return None
		
		def enter_bbox_edit_mode(bbox_idx):
			"""Enter edit mode for a specific bbox"""
			det_bbox_edit["active"] = True
			det_bbox_edit["bbox_idx"] = bbox_idx
			det_bbox_edit["handle"] = None
			
			# Store original bbox coords for reference
			current_img = det_images_list[det_current_image_idx["value"]]
			det_bbox_edit["original_bbox"] = det_annotations[current_img][bbox_idx]["bbox"].copy()
			
			# Redraw with handles
			draw_edit_handles()
		
		def exit_bbox_edit_mode():
			"""Exit bbox edit mode"""
			det_bbox_edit["active"] = False
			det_bbox_edit["bbox_idx"] = None
			det_bbox_edit["handle"] = None
			det_bbox_edit["original_bbox"] = None
			
			# Remove handles and redraw normal
			det_main_canvas.delete("edit_handle")
			redraw_bounding_boxes()
		
		def draw_edit_handles():
			"""Draw edit handles (corner points) for the bbox being edited"""
			if not det_bbox_edit["active"] or det_bbox_edit["bbox_idx"] is None:
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return
			
			if det_bbox_edit["bbox_idx"] >= len(det_annotations[current_img]):
				return
			
			# Remove old handles
			det_main_canvas.delete("edit_handle")
			
			# Get bbox coords
			ann = det_annotations[current_img][det_bbox_edit["bbox_idx"]]
			bbox = ann["bbox"]
			
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return
			
			# Calculate scale factor
			scale_x = det_image_info["display_width"] / det_image_info["original_width"]
			scale_y = det_image_info["display_height"] / det_image_info["original_height"]
			
			# Convert to canvas coords
			canvas_x1 = bbox[0] * scale_x + det_image_info["canvas_x"]
			canvas_y1 = bbox[1] * scale_y + det_image_info["canvas_y"]
			canvas_x2 = (bbox[0] + bbox[2]) * scale_x + det_image_info["canvas_x"]
			canvas_y2 = (bbox[1] + bbox[3]) * scale_y + det_image_info["canvas_y"]
			
			# Redraw the bbox being edited with a different style
			redraw_bounding_boxes()
			
			# Draw corner handles
			handle_size = 8
			color = ann.get("color", "#00FF00")
			
			# Four corners: top-left, top-right, bottom-right, bottom-left
			corners = [
				(canvas_x1, canvas_y1),  # 0: top-left
				(canvas_x2, canvas_y1),  # 1: top-right
				(canvas_x2, canvas_y2),  # 2: bottom-right
				(canvas_x1, canvas_y2),  # 3: bottom-left
			]
			
			for cx, cy in corners:
				# Draw filled circle as handle
				det_main_canvas.create_oval(
					cx - handle_size, cy - handle_size,
					cx + handle_size, cy + handle_size,
					fill="#FFFFFF",
					outline=color,
					width=2,
					tags="edit_handle"
				)
		
		def check_handle_click(x, y):
			"""Check if click is on a handle. Returns handle index or None."""
			if not det_bbox_edit["active"] or det_bbox_edit["bbox_idx"] is None:
				return None
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return None
			
			if det_bbox_edit["bbox_idx"] >= len(det_annotations[current_img]):
				return None
			
			ann = det_annotations[current_img][det_bbox_edit["bbox_idx"]]
			bbox = ann["bbox"]
			
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return None
			
			# Calculate scale factor
			scale_x = det_image_info["display_width"] / det_image_info["original_width"]
			scale_y = det_image_info["display_height"] / det_image_info["original_height"]
			
			# Convert to canvas coords
			canvas_x1 = bbox[0] * scale_x + det_image_info["canvas_x"]
			canvas_y1 = bbox[1] * scale_y + det_image_info["canvas_y"]
			canvas_x2 = (bbox[0] + bbox[2]) * scale_x + det_image_info["canvas_x"]
			canvas_y2 = (bbox[1] + bbox[3]) * scale_y + det_image_info["canvas_y"]
			
			handle_size = 8
			corners = [
				(canvas_x1, canvas_y1),  # 0: top-left
				(canvas_x2, canvas_y1),  # 1: top-right
				(canvas_x2, canvas_y2),  # 2: bottom-right
				(canvas_x1, canvas_y2),  # 3: bottom-left
			]
			
			for idx, (cx, cy) in enumerate(corners):
				if abs(x - cx) <= handle_size and abs(y - cy) <= handle_size:
					return idx
			
			return None
		
		def move_bbox_by_drag(x, y):
			"""Move entire bbox by dragging"""
			if not det_bbox_edit["active"] or det_bbox_edit["bbox_idx"] is None:
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return
			
			if det_bbox_edit["bbox_idx"] >= len(det_annotations[current_img]):
				return
			
			ann = det_annotations[current_img][det_bbox_edit["bbox_idx"]]
			bbox = ann["bbox"]  # [x, y, width, height] in original image coordinates
			
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return
			
			# Calculate how much we moved in canvas coordinates
			dx_canvas = x - det_bbox_edit["start_x"]
			dy_canvas = y - det_bbox_edit["start_y"]
			
			# Convert canvas delta to image delta
			scale_x = det_image_info["original_width"] / det_image_info["display_width"]
			scale_y = det_image_info["original_height"] / det_image_info["display_height"]
			
			dx_img = dx_canvas * scale_x
			dy_img = dy_canvas * scale_y
			
			# Calculate new position
			new_x = bbox[0] + dx_img
			new_y = bbox[1] + dy_img
			
			# Clamp to image boundaries (keep bbox fully inside image)
			new_x = max(0, min(new_x, det_image_info["original_width"] - bbox[2]))
			new_y = max(0, min(new_y, det_image_info["original_height"] - bbox[3]))
			
			# Update bbox position
			bbox[0] = new_x
			bbox[1] = new_y
			
			# Update start position for next drag event
			det_bbox_edit["start_x"] = x
			det_bbox_edit["start_y"] = y
			
			# Redraw with handles
			draw_edit_handles()
		
		def resize_bbox_by_handle(x, y):
			"""Resize bbox by dragging a handle"""
			if not det_bbox_edit["active"] or det_bbox_edit["bbox_idx"] is None:
				return
			if det_bbox_edit["handle"] is None or det_bbox_edit["handle"] == "move":
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return
			
			if det_bbox_edit["bbox_idx"] >= len(det_annotations[current_img]):
				return
			
			ann = det_annotations[current_img][det_bbox_edit["bbox_idx"]]
			bbox = ann["bbox"]  # [x, y, width, height] in original image coordinates
			
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return
			
			# Calculate scale factor
			scale_x = det_image_info["original_width"] / det_image_info["display_width"]
			scale_y = det_image_info["original_height"] / det_image_info["display_height"]
			
			# Convert current canvas position to image coords
			img_x = (x - det_image_info["canvas_x"]) * scale_x
			img_y = (y - det_image_info["canvas_y"]) * scale_y
			
			# Clamp to image boundaries
			img_x = max(0, min(img_x, det_image_info["original_width"]))
			img_y = max(0, min(img_y, det_image_info["original_height"]))
			
			# Get current bbox corners in image coords
			x1 = bbox[0]
			y1 = bbox[1]
			x2 = bbox[0] + bbox[2]
			y2 = bbox[1] + bbox[3]
			
			# Update corners based on which handle is being dragged
			handle_idx = det_bbox_edit["handle"]
			
			if handle_idx == 0:  # top-left
				x1 = img_x
				y1 = img_y
			elif handle_idx == 1:  # top-right
				x2 = img_x
				y1 = img_y
			elif handle_idx == 2:  # bottom-right
				x2 = img_x
				y2 = img_y
			elif handle_idx == 3:  # bottom-left
				x1 = img_x
				y2 = img_y
			
			# Ensure x1 < x2 and y1 < y2 (normalize)
			if x1 > x2:
				x1, x2 = x2, x1
			if y1 > y2:
				y1, y2 = y2, y1
			
			# Ensure minimum size (5 pixels)
			if x2 - x1 < 5:
				x2 = x1 + 5
			if y2 - y1 < 5:
				y2 = y1 + 5
			
			# Update bbox
			bbox[0] = x1
			bbox[1] = y1
			bbox[2] = x2 - x1
			bbox[3] = y2 - y1
			
			# Redraw with handles
			draw_edit_handles()
		
		# Bind left-click events for bounding box drawing
		det_main_canvas.bind("<ButtonPress-1>", on_bbox_press)
		det_main_canvas.bind("<B1-Motion>", on_bbox_drag)
		det_main_canvas.bind("<ButtonRelease-1>", on_bbox_release)
		
		# Mouse wheel zoom handler with debouncing for smooth performance
		det_zoom_timer = {"id": None}
		
		def apply_zoom():
			"""Actually apply the zoom (called after debounce delay)"""
			if det_images_list and det_current_image_idx["value"] < len(det_images_list):
				display_detection_image(det_current_image_idx["value"])
			det_zoom_timer["id"] = None
		
		def on_mouse_wheel(event):
			"""Handle mouse wheel zoom with debouncing for smooth performance"""
			if not det_images_list:
				return
			
			# Switch to custom zoom mode
			det_zoom_mode["value"] = "custom"
			
			# Determine zoom direction (Windows/Linux compatibility)
			if event.delta > 0 or event.num == 4:
				# Zoom in (scroll up)
				det_zoom_level["value"] = min(5.0, det_zoom_level["value"] + 0.1)
			elif event.delta < 0 or event.num == 5:
				# Zoom out (scroll down)
				det_zoom_level["value"] = max(0.1, det_zoom_level["value"] - 0.1)
			
			# Cancel previous timer if exists
			if det_zoom_timer["id"] is not None:
				det_main_canvas.after_cancel(det_zoom_timer["id"])
			
			# Schedule redraw after a short delay (debouncing)
			# This prevents redrawing on every scroll event, making it much smoother
			det_zoom_timer["id"] = det_main_canvas.after(50, apply_zoom)
		
		# Bind mouse wheel events (Windows and Linux)
		det_main_canvas.bind("<MouseWheel>", on_mouse_wheel)  # Windows
		det_main_canvas.bind("<Button-4>", on_mouse_wheel)    # Linux scroll up
		det_main_canvas.bind("<Button-5>", on_mouse_wheel)    # Linux scroll down
		
		def redraw_bounding_boxes():
			"""Redraw all bounding boxes for current image"""
			# Delete existing bbox rectangles (but not edit handles)
			det_main_canvas.delete("bbox")
			
			if not det_images_list:
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return
			
			# Check if we have valid image info
			if det_image_info["display_width"] == 0 or det_image_info["original_width"] == 0:
				return
			
			# Calculate scale factor (image coords -> canvas coords)
			scale_x = det_image_info["display_width"] / det_image_info["original_width"]
			scale_y = det_image_info["display_height"] / det_image_info["original_height"]
			
			# Draw each annotation
			for idx, ann in enumerate(det_annotations[current_img]):
				bbox = ann["bbox"]  # [x, y, width, height] in original image coordinates
				color = ann.get("color", "#00FF00")
				
				# Use white outline if this label is selected
				outline_color = "#FFFFFF" if idx in det_selected_labels else color
				
				# Convert image coordinates to canvas coordinates
				canvas_x1 = bbox[0] * scale_x + det_image_info["canvas_x"]
				canvas_y1 = bbox[1] * scale_y + det_image_info["canvas_y"]
				canvas_x2 = (bbox[0] + bbox[2]) * scale_x + det_image_info["canvas_x"]
				canvas_y2 = (bbox[1] + bbox[3]) * scale_y + det_image_info["canvas_y"]
				
				# Draw rectangle with semi-transparent fill
				det_main_canvas.create_rectangle(
					canvas_x1, canvas_y1, canvas_x2, canvas_y2,
					outline=outline_color,
					fill=color,
					stipple="gray12",  # Creates ~10% opacity effect (90% transparency)
					width=2,
					tags="bbox"
				)
		
		# Functions
		def load_images_for_detection():
			"""Load images from dataset_files into detection tab"""
			# Store existing annotations and classes if they were loaded from existing dataset
			preserve_annotations = len(det_annotations) > 0
			preserve_classes = len(det_classes) > 0
			saved_annotations = det_annotations.copy() if preserve_annotations else {}
			saved_classes = det_classes.copy() if preserve_classes else []
			
			# Clear existing detection images
			det_images_list.clear()
			det_annotations.clear()
			det_current_image_idx["value"] = 0
			det_selected_images.clear()
			det_selected_labels.clear()
			
			# Get image files from dataset_files
			image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
			for file_path in dataset_files:
				file_ext = Path(file_path).suffix.lower()
				if file_ext in image_extensions:
					if file_path not in det_images_list:
						det_images_list.append(file_path)
						# Initialize empty annotations for new image
						if file_path not in det_annotations:
							det_annotations[file_path] = []
			
			# Restore annotations and classes if they were loaded from existing dataset
			if preserve_annotations:
				for img_path, anns in saved_annotations.items():
					if img_path in det_images_list:
						det_annotations[img_path] = anns
			
			if preserve_classes:
				det_classes.clear()
				det_classes.extend(saved_classes)
			
			# Refresh the images list display
			refresh_images_list()
			
			# Display first image if available (delayed to ensure canvas is sized)
			if det_images_list:
				det_main_canvas.after(100, lambda: display_detection_image(0))
			
			# Check if all images are annotated
			check_all_images_annotated()
		
		def add_detection_images():
			"""Add images for annotation"""
			files = filedialog.askopenfilenames(
				title="Seleccionar imágenes",
				filetypes=[
					("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
					("Todos los archivos", "*.*")
				]
			)
			if files:
				for f in files:
					if f not in det_images_list:
						det_images_list.append(f)
						# Initialize empty annotations for new image
						if f not in det_annotations:
							det_annotations[f] = []
				refresh_images_list()
				if det_images_list and det_current_image_idx["value"] == 0:
					det_main_canvas.after(50, lambda: display_detection_image(0))
				
				# Check if all images are annotated
				check_all_images_annotated()
				
				# Auto-save progress after adding images
				auto_save_detection_progress()
		
		def remove_selected_images():
			"""Remove selected images"""
			if not det_selected_images:
				return
			# Remove from list in reverse order to maintain indices
			for idx in sorted(det_selected_images, reverse=True):
				if 0 <= idx < len(det_images_list):
					img_path = det_images_list[idx]
					det_images_list.pop(idx)
					# Remove annotations too
					if img_path in det_annotations:
						del det_annotations[img_path]
			
			det_selected_images.clear()
			refresh_images_list()
			
			# Check if all remaining images are annotated
			check_all_images_annotated()
			
			# Update current image if needed
			if det_images_list:
				if det_current_image_idx["value"] >= len(det_images_list):
					det_current_image_idx["value"] = len(det_images_list) - 1
				display_detection_image(det_current_image_idx["value"])
			else:
				det_current_image_idx["value"] = 0
				det_main_canvas.delete("all")
				det_current_photo["image"] = None
				refresh_labels_list()
			
			# Auto-save progress after removing images
			auto_save_detection_progress()
		
		def manage_detection_classes():
			"""Open dialog to manage detection classes"""
			import random
			import tkinter.colorchooser
			
			# Backup of original classes (for cancel)
			original_classes = [cls.copy() for cls in det_classes]
			
			# Create popup window
			classes_win = tk.Toplevel(train_win)
			classes_win.title("Manejar Clases")
			classes_win.geometry("620x520")
			classes_win.configure(bg=BG_COLOR)
			
			tk.Label(
				classes_win,
				text="Clases de Detección",
				font=("Arial", 14, "bold"),
				fg=FG_COLOR,
				bg=BG_COLOR
			).pack(pady=10)
			
			# Frame with canvas for scrollable list with color squares
			list_frame = tk.Frame(classes_win, bg=BG_COLOR, height=300)
			list_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)
			list_frame.pack_propagate(False)
			
			canvas = tk.Canvas(list_frame, bg=BG_COLOR, highlightthickness=0)
			scrollbar = tk.Scrollbar(list_frame, command=canvas.yview)
			canvas.configure(yscrollcommand=scrollbar.set)
			
			scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
			canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
			
			classes_inner_frame = tk.Frame(canvas, bg=BG_COLOR)
			canvas_window = canvas.create_window((0, 0), window=classes_inner_frame, anchor="nw")
			
			selected_class_idx = {"value": None}
			
			def _update_scroll(_=None):
				canvas.configure(scrollregion=canvas.bbox("all"))
			
			def _resize_inner(_=None):
				w = canvas.winfo_width()
				if w > 1:
					canvas.itemconfig(canvas_window, width=w)
			
			classes_inner_frame.bind("<Configure>", _update_scroll)
			canvas.bind("<Configure>", _resize_inner)
			
			def generate_random_color():
				"""Generate a random bright color"""
				return "#{:02x}{:02x}{:02x}".format(
					random.randint(100, 255),
					random.randint(100, 255),
					random.randint(100, 255)
				)
			
			def refresh_classes_display():
				"""Refresh the display of classes with color squares"""
				for widget in classes_inner_frame.winfo_children():
					widget.destroy()
				
				for idx, cls_info in enumerate(det_classes):
					cls_name = cls_info["name"]
					cls_color = cls_info["color"]
					
					# Frame for each class
					is_selected = selected_class_idx["value"] == idx
					cls_frame = tk.Frame(
						classes_inner_frame,
						bg="#0B5ED7" if is_selected else BG_COLOR,
						relief=tk.RAISED if is_selected else tk.FLAT,
						bd=2 if is_selected else 1
					)
					cls_frame.pack(fill=tk.X, pady=2, padx=5)
					
					# Color square (20x20)
					color_canvas = tk.Canvas(cls_frame, width=20, height=20, bg=cls_color, highlightthickness=1, highlightbackground="#FFFFFF")
					color_canvas.pack(side=tk.LEFT, padx=(5, 10), pady=5)
					
					# Class name label
					cls_label = tk.Label(
						cls_frame,
						text=cls_name,
						font=("Arial", 11),
						fg=FG_COLOR,
						bg="#0B5ED7" if is_selected else BG_COLOR,
						anchor="w"
					)
					cls_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
					
					# Make frame clickable to select
					def make_select_callback(index):
						def select(_=None):
							selected_class_idx["value"] = index
							refresh_classes_display()
						return select
					
					cls_frame.bind("<Button-1>", make_select_callback(idx))
					color_canvas.bind("<Button-1>", make_select_callback(idx))
					cls_label.bind("<Button-1>", make_select_callback(idx))
			
			refresh_classes_display()
			
			# Action buttons frame (anchored with list)
			action_btn_frame = tk.Frame(classes_win, bg=BG_COLOR)
			action_btn_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
			
			# Exit buttons frame (anchored at bottom right)
			exit_btn_frame = tk.Frame(classes_win, bg=BG_COLOR)
			exit_btn_frame.pack(side=tk.BOTTOM, anchor=tk.E, padx=20, pady=10)
			
			def add_class():
				new_class = tk.simpledialog.askstring("Nueva Clase", "Nombre de la clase:", parent=classes_win)
				if new_class and new_class.strip():
					new_class = new_class.strip()
					# Check if class name already exists
					if not any(cls["name"] == new_class for cls in det_classes):
						det_classes.append({
							"name": new_class,
							"color": generate_random_color()
						})
						refresh_classes_display()
			
			def remove_class():
				if selected_class_idx["value"] is not None:
					idx = selected_class_idx["value"]
					if 0 <= idx < len(det_classes):
						det_classes.pop(idx)
						selected_class_idx["value"] = None
						refresh_classes_display()
			
			def rename_class():
				if selected_class_idx["value"] is not None:
					idx = selected_class_idx["value"]
					if 0 <= idx < len(det_classes):
						old_name = det_classes[idx]["name"]
						new_name = tk.simpledialog.askstring(
							"Renombrar Clase",
							f"Nuevo nombre para '{old_name}':",
							parent=classes_win,
							initialvalue=old_name
						)
						if new_name and new_name.strip():
							new_name = new_name.strip()
							# Check if new name doesn't exist in other classes
							if not any(cls["name"] == new_name for i, cls in enumerate(det_classes) if i != idx):
								det_classes[idx]["name"] = new_name
								refresh_classes_display()
			
			def change_color():
				if selected_class_idx["value"] is not None:
					idx = selected_class_idx["value"]
					if 0 <= idx < len(det_classes):
						current_color = det_classes[idx]["color"]
						# Open color chooser
						color = tkinter.colorchooser.askcolor(
							color=current_color,
							title="Elegir Color",
							parent=classes_win
						)
						if color[1]:  # color[1] is the hex string
							det_classes[idx]["color"] = color[1]
							refresh_classes_display()
			
			def on_ok():
				"""Save changes and close"""
				classes_win.destroy()
				refresh_classes_buttons()  # Update class buttons in top menu
				
				# Auto-save progress after managing classes
				auto_save_detection_progress()
			
			def on_cancel():
				"""Discard changes and close"""
				# Restore original classes
				det_classes.clear()
				det_classes.extend(original_classes)
				classes_win.destroy()
				refresh_classes_buttons()  # Update class buttons in top menu
			
			# Action buttons (Agregar, Remover, Renombrar, Color)
			create_rounded_button(action_btn_frame, text="Agregar Clase", command=add_class, bg_color="#78b82f", fg_color=FG_COLOR, active_bg="#66a026").pack(side=tk.LEFT, padx=2)
			create_rounded_button(action_btn_frame, text="Remover Clase", command=remove_class, bg_color="#ec5b2d", fg_color=FG_COLOR, active_bg="#d44a20").pack(side=tk.LEFT, padx=2)
			create_rounded_button(action_btn_frame, text="Renombrar Clase", command=rename_class, bg_color="#096bc9", fg_color=FG_COLOR, active_bg="#075a9e").pack(side=tk.LEFT, padx=2)
			create_rounded_button(action_btn_frame, text="Color", command=change_color, bg_color="#9C27B0", fg_color=FG_COLOR, active_bg="#7B1FA2").pack(side=tk.LEFT, padx=2)
			
			# Exit buttons (OK and Cancel) - anchored at bottom right
			create_rounded_button(exit_btn_frame, text="OK", command=on_ok, bg_color="#2196F3", fg_color=FG_COLOR, active_bg="#1976D2").pack(side=tk.LEFT, padx=2)
			create_rounded_button(exit_btn_frame, text="Cancel", command=on_cancel, bg_color="#757575", fg_color=FG_COLOR, active_bg="#5c5c5c").pack(side=tk.LEFT, padx=2)
		
		def remove_selected_labels():
			"""Remove selected labels from current image"""
			if not det_selected_labels or not det_images_list:
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img in det_annotations:
				# Remove in reverse order
				for idx in sorted(det_selected_labels, reverse=True):
					if 0 <= idx < len(det_annotations[current_img]):
						det_annotations[current_img].pop(idx)
			
			det_selected_labels.clear()
			refresh_labels_list()
			display_detection_image(det_current_image_idx["value"])
			
			# Refresh images list to update color (white if no labels left)
			refresh_images_list()
			
			# Check if all images still have annotations
			check_all_images_annotated()
			
			# Auto-save progress after removing labels
			auto_save_detection_progress()
		
		def refresh_images_list():
			"""Refresh the list of images"""
			for widget in images_inner.winfo_children():
				widget.destroy()
			
			for idx, img_path in enumerate(det_images_list):
				img_name = Path(img_path).name
				
				# Check if image has annotations
				has_annotations = img_path in det_annotations and len(det_annotations[img_path]) > 0
				text_color = "#00FF00" if has_annotations else FG_COLOR  # Green if has labels, white otherwise
				
				# Create frame for each image
				img_frame = tk.Frame(images_inner, bg=BG_COLOR if idx not in det_selected_images else "#0B5ED7")
				img_frame.pack(fill=tk.X, pady=2, padx=2)
				
				# Checkbox for selection
				var = tk.BooleanVar(value=idx in det_selected_images)
				
				def make_toggle(index):
					def toggle():
						if index in det_selected_images:
							det_selected_images.remove(index)
						else:
							det_selected_images.add(index)
						refresh_images_list()
					return toggle
				
				chk = tk.Checkbutton(
					img_frame,
					variable=var,
					command=make_toggle(idx),
					bg=BG_COLOR if idx not in det_selected_images else "#0B5ED7",
					fg=FG_COLOR,
					selectcolor=BG_COLOR,
					activebackground=BG_COLOR
				)
				chk.pack(side=tk.LEFT)
				
				# Image name label (clickable to display)
				def make_display_callback(index):
					return lambda e: display_detection_image(index)
				
				lbl = tk.Label(
					img_frame,
					text=img_name,
					font=("Arial", 9),
					fg=text_color,  # Use text_color based on annotations
					bg=BG_COLOR if idx not in det_selected_images else "#0B5ED7",
					cursor="hand2",
					anchor="w"
				)
				lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
				lbl.bind("<Button-1>", make_display_callback(idx))
		
		def refresh_labels_list():
			"""Refresh the list of labels for current image"""
			for widget in labels_inner.winfo_children():
				widget.destroy()
			
			if not det_images_list:
				return
			
			current_img = det_images_list[det_current_image_idx["value"]]
			if current_img not in det_annotations:
				return
			
			annotations = det_annotations[current_img]
			
			# Count instances per class
			class_counts = {}
			for ann in annotations:
				cls = ann["class"]
				class_counts[cls] = class_counts.get(cls, 0) + 1
			
			# Display each annotation
			class_instances = {}
			for idx, ann in enumerate(annotations):
				cls = ann["class"]
				bbox = ann["bbox"]
				color = ann.get("color", "#00FF00")
				
				# Get instance number for this class
				if cls not in class_instances:
					class_instances[cls] = 0
				class_instances[cls] += 1
				instance_num = class_instances[cls]
				
				# Create frame for label
				label_frame = tk.Frame(labels_inner, bg=BG_COLOR if idx not in det_selected_labels else "#0B5ED7")
				label_frame.pack(fill=tk.X, pady=2, padx=2)
				
				# Checkbox
				var = tk.BooleanVar(value=idx in det_selected_labels)
				
				def make_label_toggle(index):
					def toggle():
						if index in det_selected_labels:
							det_selected_labels.remove(index)
						else:
							det_selected_labels.add(index)
						refresh_labels_list()
						redraw_bounding_boxes()  # Redraw to show white outline on selected
					return toggle
				
				chk = tk.Checkbutton(
					label_frame,
					variable=var,
					command=make_label_toggle(idx),
					bg=BG_COLOR if idx not in det_selected_labels else "#0B5ED7",
					fg=FG_COLOR,
					selectcolor=BG_COLOR,
					activebackground=BG_COLOR
				)
				chk.pack(side=tk.LEFT)
				
				# Color indicator
				color_indicator = tk.Label(
					label_frame,
					text="■",
					font=("Arial", 14),
					fg=color,
					bg=BG_COLOR if idx not in det_selected_labels else "#0B5ED7"
				)
				color_indicator.pack(side=tk.LEFT, padx=(0, 5))
				
				# Combobox for class selection (editable)
				class_names = [c["name"] for c in det_classes]
				
				def make_class_change(index):
					def on_class_change(event):
						# Get value directly from the widget that triggered the event
						new_class = event.widget.get()
						# Update annotation class
						if current_img in det_annotations and index < len(det_annotations[current_img]):
							# Find the color for the new class
							new_color = "#00FF00"  # default
							for c in det_classes:
								if c["name"] == new_class:
									new_color = c["color"]
									break
							det_annotations[current_img][index]["class"] = new_class
							det_annotations[current_img][index]["color"] = new_color
							# Auto-save progress
							auto_save_detection_progress()
							# Refresh display
							refresh_labels_list()
							redraw_bounding_boxes()
					return on_class_change
				
				class_combo = ttk.Combobox(
					label_frame,
					values=class_names,
					state="readonly",
					width=12,
					font=("Arial", 9),
					style="DetClass.TCombobox"
				)
				class_combo.set(cls)  # Set current value without StringVar
				class_combo.bind("<<ComboboxSelected>>", make_class_change(idx))
				class_combo.pack(side=tk.LEFT, padx=5)
				
				# Instance and bbox info
				info_text = f"#{instance_num} ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
				info_lbl = tk.Label(
					label_frame,
					text=info_text,
					font=("Arial", 9),
					fg=FG_COLOR,
					bg=BG_COLOR if idx not in det_selected_labels else "#0B5ED7",
					anchor="w"
				)
				info_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
		
		def display_detection_image(idx):
			"""Display image at given index"""
			if not det_images_list or idx >= len(det_images_list):
				return
			
			# Exit edit mode when changing images
			if det_current_image_idx["value"] != idx:
				exit_bbox_edit_mode()
			
			# Reset pan offset when changing images (unless dragging)
			if det_current_image_idx["value"] != idx and not det_drag_data["dragging"]:
				det_canvas_offset["x"] = 0
				det_canvas_offset["y"] = 0
			
			det_current_image_idx["value"] = idx
			img_path = det_images_list[idx]
			
			try:
				# Load image
				img = Image.open(img_path)
				original_width, original_height = img.size
				
				# Force canvas to update its size
				det_main_canvas.update_idletasks()
				
				# Get canvas size
				canvas_width = det_main_canvas.winfo_width()
				canvas_height = det_main_canvas.winfo_height()
				
				# Use reasonable defaults if canvas not yet rendered
				if canvas_width <= 1:
					canvas_width = 800
				if canvas_height <= 1:
					canvas_height = 600
				
				# Calculate new size based on zoom mode
				if det_zoom_mode["value"] == "fit":
					# Resize image to fit canvas while preserving aspect ratio
					img_ratio = img.width / img.height
					canvas_ratio = canvas_width / canvas_height
					
					if img_ratio > canvas_ratio:
						# Image is wider
						new_width = canvas_width
						new_height = int(canvas_width / img_ratio)
					else:
						# Image is taller
						new_height = canvas_height
						new_width = int(canvas_height * img_ratio)
				else:
					# Custom zoom level
					new_width = int(original_width * det_zoom_level["value"])
					new_height = int(original_height * det_zoom_level["value"])
				
				img = img.resize((new_width, new_height), Image.LANCZOS)
				
				# Convert to PhotoImage
				photo = ImageTk.PhotoImage(img)
				det_current_photo["image"] = photo
				
				# Calculate image position in canvas (top-left corner)
				img_center_x = canvas_width // 2 + det_canvas_offset["x"]
				img_center_y = canvas_height // 2 + det_canvas_offset["y"]
				img_top_left_x = img_center_x - new_width // 2
				img_top_left_y = img_center_y - new_height // 2
				
				# Store image display information for coordinate conversions
				det_image_info["original_width"] = original_width
				det_image_info["original_height"] = original_height
				det_image_info["display_width"] = new_width
				det_image_info["display_height"] = new_height
				det_image_info["canvas_x"] = img_top_left_x
				det_image_info["canvas_y"] = img_top_left_y
				
				# Display on canvas with offset (for panning)
				det_main_canvas.delete("all")
				image_id = det_main_canvas.create_image(img_center_x, img_center_y, image=photo, anchor="center")
				det_canvas_image_id["id"] = image_id  # Store for smooth dragging
				
				# Draw bounding boxes on top of image
				redraw_bounding_boxes()
				
				# Refresh labels list
				refresh_labels_list()
				
			except Exception as e:
				det_main_canvas.delete("all")
				det_main_canvas.create_text(
					canvas_width // 2, canvas_height // 2,
					text=f"Error al cargar imagen:\n{str(e)[:100]}",
					fill=FG_COLOR,
					font=("Arial", 12)
				)
		
		# ============================================================
		# TAB 5,9-10: Placeholder tabs (to be implemented)
		# ============================================================
		for i in range(5, len(tabs)):
			if i == 5:  # Skip Deteccion tab (implemented above)
				continue
			if i == 6:  # Skip Data Augmentation tab (implemented below)
				continue
			if i == 7:  # Skip Configuracion tab (implemented below)
				continue
			if i == 8:  # Skip Entrenamiento tab (implemented below)
				continue
			if i == 9:  # Skip Estadisticas tab (implemented below)
				continue
			if i == 10:  # Skip Pruebas tab (implemented below)
				continue
			placeholder = tk.Label(
				tabs[i],
				text=f"Contenido de '{tab_titles[i]}'\n(En desarrollo)",
				font=("Arial", 16),
				fg=FG_COLOR,
				bg=BG_COLOR
			)
			placeholder.pack(expand=True)
		
		# Track tab initialization
		aug_tab_initialized = {"value": False}  # Track if Data Augmentation tab was initialized
		
		# Enhanced tab navigation to handle classification tab focus
		def enhanced_go_next():
			nonlocal current_step
			if current_step < len(tabs) - 1 and can_advance[current_step]:
				# Find the next visible tab
				next_step = None
				for i in range(current_step + 1, len(tabs)):
					if i in visible_tabs:
						next_step = i
						break
				
				if next_step is not None:
					current_step = next_step
					# Handle special focus logic when entering specific tabs
					if current_step == 4:  # Clasificacion tab
						on_clasificacion_tab_focus()
					elif current_step == 5:  # Deteccion tab
						# Auto-load images from dataset_files when model_type is "detection"
						if model_type["value"] == "detection":
							on_deteccion_tab_focus()
					elif current_step == 6:  # Data Augmentation tab
						# Auto-load labeled images from Detection tab when model_type is "detection"
						if model_type["value"] == "detection" and not aug_tab_initialized["value"]:
							aug_tab_initialized["value"] = True
							# Automatically trigger loading of labeled images
							load_images_from_detection_tab()
						# Always enable next button in Data Augmentation tab
						can_advance[6] = True
						update_nav_state()
					elif current_step == 7:  # Configuracion tab
						on_config_tab_focus()
					else:
						unbind_classification_keys()
				update_tabs_state()
		
		# Update the next button to use enhanced version
		next_btn_canvas.configure(command=enhanced_go_next)
		
		# Override go_prev to unbind keys when leaving classification tab
		def enhanced_go_prev():
			nonlocal current_step
			if current_step > 0:
				if current_step == 4:  # Leaving Clasificacion tab
					unbind_classification_keys()
				# Find the previous visible tab
				for i in range(current_step - 1, -1, -1):
					if i in visible_tabs:
						current_step = i
						break
				update_tabs_state()
		
		# Update the prev button to use enhanced version
		prev_btn_canvas.configure(command=enhanced_go_prev)
		
		# ============================================================
		# TAB 6: Data Augmentation
		# ============================================================
		augmentation_tab = tabs[6]
		augmentation_tab.configure(bg=PROC_CONTENT_BG)
		
		# Scrollable container
		aug_scroll_canvas = tk.Canvas(augmentation_tab, bg=BG_COLOR, highlightthickness=0)
		aug_vsb = tk.Scrollbar(augmentation_tab, orient="vertical", command=aug_scroll_canvas.yview)
		aug_scroll_canvas.configure(yscrollcommand=aug_vsb.set)
		aug_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		aug_vsb.pack(side=tk.RIGHT, fill=tk.Y)
		
		aug_content = tk.Frame(aug_scroll_canvas, bg=BG_COLOR)
		aug_scroll_window = aug_scroll_canvas.create_window((0, 0), window=aug_content, anchor="nw")
		
		def _update_aug_scroll(_=None):
			aug_scroll_canvas.configure(scrollregion=aug_scroll_canvas.bbox("all"))
		
		def _resize_aug(_=None):
			w = aug_scroll_canvas.winfo_width()
			aug_scroll_canvas.itemconfig(aug_scroll_window, width=w)
		
		aug_content.bind("<Configure>", _update_aug_scroll)
		aug_scroll_canvas.bind("<Configure>", _resize_aug)
		
		# Main container with padding
		aug_main = tk.Frame(aug_content, bg=BG_COLOR)
		aug_main.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
		
		# State variables
		aug_source_images = []  # List of image paths
		aug_image_classes = {}  # Dict mapping image path -> class name (for classification)
		aug_detected_classes = []  # List of detected class names (for classification)
		aug_section_states = [False, False, False, False, False]  # Track which sections are enabled
		aug_train_split = {"train": 70, "val": 20, "test": 10}  # Default split percentages
		
		# Placeholder for callback function (will be defined after all sections are created)
		show_all_sections_callback = None
		
		# ===== SECTION 1: Imágenes Fuente =====
		section1_frame = tk.Frame(aug_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		section1_frame.pack(fill=tk.X, pady=(0, 20))
		
		section1_title = tk.Label(
			section1_frame,
			text="1. Imágenes Fuente",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		section1_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Drop area for images
		drop_frame = tk.Frame(section1_frame, bg=GRAY_BG)
		drop_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
		
		drop_canvas = tk.Canvas(drop_frame, bg=BG_COLOR, highlightthickness=0, height=200)
		drop_canvas.pack(fill=tk.BOTH, expand=True)
		
		def draw_dashed_rect(canvas, x1, y1, x2, y2, dash=(10, 5), color=FG_COLOR, width=2):
			# Draw dashed rectangle
			canvas.create_line(x1, y1, x2, y1, dash=dash, fill=color, width=width)
			canvas.create_line(x2, y1, x2, y2, dash=dash, fill=color, width=width)
			canvas.create_line(x2, y2, x1, y2, dash=dash, fill=color, width=width)
			canvas.create_line(x1, y2, x1, y1, dash=dash, fill=color, width=width)
		
		# Drop label text changes based on model type
		def get_drop_label_text():
			if model_type["value"] == "classification":
				return "Arrastrar carpeta con subfolders (clases) o hacer clic para seleccionar"
			elif model_type["value"] == "detection":
				return "Etiqueta imagenes para hacer data augmentation"
			else:
				return "Arrastrar imágenes aquí o hacer clic para seleccionar carpeta"
		
		drop_label = tk.Label(
			drop_canvas,
			text=get_drop_label_text(),
			font=("Arial", 12),
			fg=FG_COLOR,
			bg=BG_COLOR
		)
		drop_label_window = drop_canvas.create_window(0, 0, window=drop_label, tags="drop_label")
		
		# Preview frame (hidden initially)
		preview_frame = tk.Frame(section1_frame, bg=GRAY_BG)
		preview_label = tk.Label(preview_frame, text="Primeras 5 imágenes:", font=("Arial", 11, "bold"), fg=FG_COLOR, bg=GRAY_BG)
		preview_label.pack(anchor="w", padx=20, pady=(5, 5))
		
		preview_images_frame = tk.Frame(preview_frame, bg=GRAY_BG)
		preview_images_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
		
		# Buttons frame (hidden initially)
		section1_buttons = tk.Frame(section1_frame, bg=GRAY_BG)
		continue_btn1 = ttk.Button(section1_buttons, text="Continuar", state="disabled")
		add_more_btn1 = ttk.Button(section1_buttons, text="Agregar más imágenes")
		
		def layout_drop_canvas(_=None):
			w = drop_canvas.winfo_width()
			h = drop_canvas.winfo_height()
			if w > 1 and h > 1:
				drop_canvas.delete("dashed")
				margin = 20
				draw_dashed_rect(drop_canvas, margin, margin, w-margin, h-margin, dash=(10, 5), color=FG_COLOR)
				drop_canvas.coords(drop_label_window, w//2, h//2)
		
		drop_canvas.bind("<Configure>", layout_drop_canvas)
		
		def parse_aug_files(s: str) -> list[str]:
			# Parse dropped files from DND string
			s = s.strip()
			if not s:
				return []
			if s.startswith("{") and s.endswith("}"):
				s = s[1:-1]
			parts = []
			current = []
			in_brace = 0
			for c in s:
				if c == "{":
					in_brace += 1
					current.append(c)
				elif c == "}":
					in_brace -= 1
					current.append(c)
				elif c == " " and in_brace == 0:
					if current:
						parts.append("".join(current))
						current = []
				else:
					current.append(c)
			if current:
				parts.append("".join(current))
			result = []
			for p in parts:
				p = p.strip()
				if p.startswith("{") and p.endswith("}"):
					p = p[1:-1]
				if p:
					result.append(p)
			return result
		
		def load_images_from_paths(paths: list[str]):
			nonlocal aug_source_images, aug_image_classes, aug_detected_classes
			new_images = []
			
			if model_type["value"] == "classification":
				# For classification: expect folder with subfolders (classes)
				for p in paths:
					p_obj = Path(p)
					if p_obj.is_dir():
						# Check for subfolders (classes)
						subfolders = [d for d in p_obj.iterdir() if d.is_dir()]
						if subfolders:
							# Load images from each subfolder (class)
							for subfolder in subfolders:
								class_name = subfolder.name
								if class_name not in aug_detected_classes:
									aug_detected_classes.append(class_name)
								for ext in IMAGE_EXTS:
									for img_file in subfolder.glob(f"*{ext}"):
										img_path = str(img_file)
										new_images.append(img_path)
										aug_image_classes[img_path] = class_name
						else:
							# No subfolders found, load all images directly
							for ext in IMAGE_EXTS:
								new_images.extend([str(f) for f in p_obj.glob(f"*{ext}")])
			elif model_type["value"] == "detection":
				# For detection: use only labeled images from Detection tab
				# Filter det_images_list to only include images with at least 1 annotation
				for img_path in det_images_list:
					if img_path in det_annotations and len(det_annotations[img_path]) > 0:
						new_images.append(img_path)
			else:
				# For other types: load all images from directory
				for p in paths:
					p_obj = Path(p)
					if p_obj.is_dir():
						for ext in IMAGE_EXTS:
							new_images.extend([str(f) for f in p_obj.glob(f"*{ext}")])
					elif p_obj.suffix.lower() in IMAGE_EXTS:
						new_images.append(str(p_obj))
			
			aug_source_images.extend(new_images)
			aug_source_images = list(set(aug_source_images))  # Remove duplicates
			show_image_previews()
			# Show all sections after previews are shown
			if len(aug_source_images) > 0 and show_all_sections_callback is not None:
				show_all_sections_callback()
		
		def show_image_previews():
			# Check if we have images to show
			if len(aug_source_images) == 0:
				# No images loaded - keep drop area visible and update message
				if model_type["value"] == "detection":
					# Update drop label to show message for detection mode
					drop_label.config(text="Etiqueta imagenes para hacer data augmentation")
				# Don't proceed with showing preview
				return
			
			# Hide drop area completely (including the frame that holds it)
			drop_frame.pack_forget()
			
			# Show preview frame
			preview_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
			
			# Clear previous previews
			for widget in preview_images_frame.winfo_children():
				widget.destroy()
			
			# Show first 5 images
			for i, img_path in enumerate(aug_source_images[:5]):
				try:
					img = Image.open(img_path)
					img.thumbnail((100, 100))
					photo = ImageTk.PhotoImage(img)
					
					img_label = tk.Label(preview_images_frame, image=photo, bg=GRAY_BG)
					img_label.image = photo  # Keep reference
					img_label.pack(side=tk.LEFT, padx=5)
				except Exception:
					pass
			
			# Show count label
			count_text = f"Total: {len(aug_source_images)} imágenes cargadas"
			if model_type["value"] == "classification" and aug_detected_classes:
				count_text += f" ({len(aug_detected_classes)} clases: {', '.join(sorted(aug_detected_classes))})"
			
			count_label = tk.Label(
				preview_frame,
				text=count_text,
				font=("Arial", 10),
				fg=FG_COLOR,
				bg=GRAY_BG
			)
			count_label.pack(anchor="w", padx=20, pady=(5, 10))
			
			# Buttons removed - sections show automatically
			add_more_btn1.pack(side=tk.RIGHT, padx=5)
			
			aug_section_states[0] = True
			
			# Note: show_all_sections_callback will be defined later, call it at the end
		
		def on_drop_aug(event):
			paths = parse_aug_files(event.data)
			load_images_from_paths(paths)
		
		def on_click_drop_area(_):
			# For detection model, load from Detection tab automatically
			if model_type["value"] == "detection":
				load_images_from_detection_tab()
			else:
				folder = filedialog.askdirectory(title="Seleccionar carpeta de imágenes")
				if folder:
					load_images_from_paths([folder])
		
		def load_images_from_detection_tab():
			"""Load only labeled images from Detection tab"""
			load_images_from_paths([])  # Pass empty list to trigger detection logic
		
		def add_more_images():
			# For detection model, reload from Detection tab
			if model_type["value"] == "detection":
				load_images_from_detection_tab()
			else:
				folder = filedialog.askdirectory(title="Seleccionar carpeta adicional")
				if folder:
					load_images_from_paths([folder])
		
		drop_canvas.bind("<Button-1>", on_click_drop_area)
		add_more_btn1.configure(command=add_more_images)
		
		if DND_AVAILABLE:
			drop_canvas.drop_target_register(DND_FILES)
			drop_canvas.dnd_bind("<<Drop>>", on_drop_aug)
		
		# ===== SECTION 2: Split de Entrenamiento/Validación/Testeo =====
		section2_frame = tk.Frame(aug_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		# Initially hidden, will be packed when section 1 is complete
		
		section2_title = tk.Label(
			section2_frame,
			text="2. Split de Entrenamiento / Validación / Testeo",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		section2_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Info boxes container
		info_boxes_frame = tk.Frame(section2_frame, bg=GRAY_BG)
		info_boxes_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
		info_boxes_frame.columnconfigure(0, weight=1)
		info_boxes_frame.columnconfigure(1, weight=1)
		info_boxes_frame.columnconfigure(2, weight=1)
		
		def create_split_info_box(parent, title, color, row, col):
			box = tk.Frame(parent, bg=color, relief=tk.RAISED, bd=3, highlightbackground=color, highlightthickness=2)
			box.grid(row=row, column=col, sticky="nsew", padx=10, pady=5)
			box.config(borderwidth=3, relief=tk.RAISED)
			
			title_lbl = tk.Label(box, text=title, font=("Arial", 13, "bold"), fg=FG_COLOR, bg=color)
			title_lbl.pack(pady=(10, 5))
			
			percent_lbl = tk.Label(box, text="0%", font=("Arial", 24, "bold"), fg=FG_COLOR, bg=color)
			percent_lbl.pack(pady=5)
			
			count_lbl = tk.Label(box, text="0 imágenes", font=("Arial", 10), fg=FG_COLOR, bg=color)
			count_lbl.pack(pady=(5, 10))
			
			return box, percent_lbl, count_lbl
		
		train_box, train_percent, train_count = create_split_info_box(info_boxes_frame, "Set de Entrenamiento", "#0B5ED7", 0, 0)
		val_box, val_percent, val_count = create_split_info_box(info_boxes_frame, "Set de Validación", "#6610F2", 0, 1)
		test_box, test_percent, test_count = create_split_info_box(info_boxes_frame, "Set de Testeo", "#DC3545", 0, 2)
		
		# Range slider container
		slider_container = tk.Frame(section2_frame, bg=GRAY_BG)
		slider_container.pack(fill=tk.X, padx=20, pady=(0, 20))
		
		slider_label = tk.Label(
			slider_container,
			text="Ajustar proporciones:",
			font=("Arial", 12, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		slider_label.pack(anchor="w", pady=(0, 10))
		
		# Train/Val slider
		train_val_frame = tk.Frame(slider_container, bg=GRAY_BG)
		train_val_frame.pack(fill=tk.X, pady=5)
		
		tk.Label(train_val_frame, text="Train/Val:", font=("Arial", 10), fg=FG_COLOR, bg=GRAY_BG).pack(side=tk.LEFT, padx=(0, 10))
		train_val_slider = tk.Scale(
			train_val_frame,
			from_=50,
			to=90,
			orient=tk.HORIZONTAL,
			bg=GRAY_BG,
			fg=FG_COLOR,
			highlightthickness=0,
			troughcolor=GRAY_BG,
			length=400
		)
		train_val_slider.set(70)
		train_val_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
		
		# Val/Test slider
		val_test_frame = tk.Frame(slider_container, bg=GRAY_BG)
		val_test_frame.pack(fill=tk.X, pady=5)
		
		tk.Label(val_test_frame, text="Val/Test:", font=("Arial", 10), fg=FG_COLOR, bg=GRAY_BG).pack(side=tk.LEFT, padx=(0, 10))
		val_test_slider = tk.Scale(
			val_test_frame,
			from_=10,
			to=90,
			orient=tk.HORIZONTAL,
			bg=GRAY_BG,
			fg=FG_COLOR,
			highlightthickness=0,
			troughcolor=GRAY_BG,
			length=400
		)
		val_test_slider.set(67)  # 20/(20+10) * 100
		val_test_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
		
		def update_split_display():
			total = len(aug_source_images)
			if total == 0:
				return
			
			# Get train percentage
			train_pct = train_val_slider.get()
			remaining = 100 - train_pct
			
			# Get val percentage from remaining
			val_ratio = val_test_slider.get() / 100.0
			val_pct = remaining * val_ratio
			test_pct = remaining * (1 - val_ratio)
			
			# Update display
			train_percent.configure(text=f"{train_pct}%")
			val_percent.configure(text=f"{val_pct:.0f}%")
			test_percent.configure(text=f"{test_pct:.0f}%")
			
			train_count.configure(text=f"{int(total * train_pct / 100)} imágenes")
			val_count.configure(text=f"{int(total * val_pct / 100)} imágenes")
			test_count.configure(text=f"{int(total * test_pct / 100)} imágenes")
			
			# Update state
			aug_train_split["train"] = train_pct
			aug_train_split["val"] = val_pct
			aug_train_split["test"] = test_pct
		
		train_val_slider.configure(command=lambda _: update_split_display())
		val_test_slider.configure(command=lambda _: update_split_display())
		
		# Buttons removed - sections show automatically
		
		def show_section2():
			section2_frame.pack(fill=tk.X, pady=(0, 20))
			update_split_display()
			aug_section_states[1] = True
		
		def hide_section2():
			section2_frame.pack_forget()
			aug_section_states[1] = False
		
		def go_back_to_section1():
			hide_section2()
			# Re-enable section 1 for editing
		
		# Buttons configuration removed - sections show automatically
		
		# ===== SECTION 3: Preprocesamiento =====
		section3_frame = tk.Frame(aug_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		# Initially hidden
		
		section3_title = tk.Label(
			section3_frame,
			text="3. Preprocesamiento",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		section3_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Preprocessing options
		preproc_container = tk.Frame(section3_frame, bg=GRAY_BG)
		preproc_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
		
		preproc_options = [
			"Auto-Orient",
			"Resize (640x640)",
			"Grayscale",
			"Normalize",
			"Histogram Equalization",
			"Denoise",
			"Edge Enhancement",
			"Contrast Adjustment"
		]
		
		preproc_vars = {}
		for i, option in enumerate(preproc_options):
			var = tk.BooleanVar(value=False)
			preproc_vars[option] = var
			check = tk.Checkbutton(
				preproc_container,
				text=option,
				variable=var,
				font=("Arial", 11),
				fg=FG_COLOR,
				bg=GRAY_BG,
				selectcolor=BG_COLOR,
				activebackground=GRAY_BG,
				activeforeground=FG_COLOR
			)
			check.pack(anchor="w", pady=3)
		
		# Buttons removed - sections show automatically
		
		# ===== SECTION 4: Data Augmentation =====
		section4_frame = tk.Frame(aug_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		# Initially hidden
		
		section4_title = tk.Label(
			section4_frame,
			text="4. Data Augmentation",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		section4_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Augmentation options
		aug_container = tk.Frame(section4_frame, bg=GRAY_BG)
		aug_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
		
		aug_options = [
			"Flip (Horizontal)",
			"Flip (Vertical)",
			"90° Rotate",
			"Crop",
			"Rotation (Random)",
			"Shear",
			"Grayscale",
			"Hue",
			"Saturation",
			"Brightness",
			"Exposure",
			"Blur",
			"Noise"
		]
		
		aug_vars = {}
		for i, option in enumerate(aug_options):
			var = tk.BooleanVar(value=False)
			aug_vars[option] = var
			check = tk.Checkbutton(
				aug_container,
				text=option,
				variable=var,
				font=("Arial", 11),
				fg=FG_COLOR,
				bg=GRAY_BG,
				selectcolor=BG_COLOR,
				activebackground=GRAY_BG,
				activeforeground=FG_COLOR
			)
			check.pack(anchor="w", pady=3)
		
		# Buttons removed - sections show automatically
		
		# ===== SECTION 5: Crear Versión de Dataset =====
		section5_frame = tk.Frame(aug_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		# Initially hidden
		
		section5_title = tk.Label(
			section5_frame,
			text="5. Crear Versión de Dataset",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		section5_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Version name
		name_container = tk.Frame(section5_frame, bg=GRAY_BG)
		name_container.pack(fill=tk.X, padx=20, pady=(10, 15))
		
		name_label = tk.Label(
			name_container,
			text="Nombre de la versión:",
			font=("Arial", 12, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		name_label.pack(anchor="w", pady=(0, 5))
		
		version_name_entry = tk.Entry(
			name_container,
			font=("Arial", 11),
			bg=BG_COLOR,
			fg=FG_COLOR,
			insertbackground=FG_COLOR
		)
		version_name_entry.pack(fill=tk.X)
		
		# Version notes
		notes_container = tk.Frame(section5_frame, bg=GRAY_BG)
		notes_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
		
		notes_label = tk.Label(
			notes_container,
			text="Notas de la versión:",
			font=("Arial", 12, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		notes_label.pack(anchor="w", pady=(0, 5))
		
		version_notes_text = tk.Text(
			notes_container,
			font=("Arial", 10),
			bg=BG_COLOR,
			fg=FG_COLOR,
			height=8,
			insertbackground=FG_COLOR,
			wrap=tk.WORD
		)
		version_notes_text.pack(fill=tk.BOTH, expand=True)
		
		# Buttons
		section5_buttons = tk.Frame(section5_frame, bg=GRAY_BG)
		section5_buttons.pack(fill=tk.X, padx=20, pady=(15, 15))
		
		# Create dataset button as Canvas with rounded corners
		create_dataset_canvas = tk.Canvas(section5_buttons, width=150, height=40, bg=GRAY_BG, highlightthickness=0)
		create_dataset_canvas.pack(side=tk.RIGHT, padx=5)
		
		def draw_create_dataset_button(color):
			create_dataset_canvas.delete("all")
			x0, y0, x1, y1 = 5, 5, 145, 35
			r = 10
			create_dataset_canvas.create_arc(x0, y0, x0+2*r, y0+2*r, start=90, extent=90, fill=color, outline=color)
			create_dataset_canvas.create_arc(x1-2*r, y0, x1, y0+2*r, start=0, extent=90, fill=color, outline=color)
			create_dataset_canvas.create_arc(x0, y1-2*r, x0+2*r, y1, start=180, extent=90, fill=color, outline=color)
			create_dataset_canvas.create_arc(x1-2*r, y1-2*r, x1, y1, start=270, extent=90, fill=color, outline=color)
			create_dataset_canvas.create_rectangle(x0+r, y0, x1-r, y1, fill=color, outline=color)
			create_dataset_canvas.create_rectangle(x0, y0+r, x1, y1-r, fill=color, outline=color)
			create_dataset_canvas.create_text(75, 20, text="Crear Dataset", fill="white", font=("Arial", 11, "bold"))
		
		draw_create_dataset_button("#015aca")
		
		def on_create_dataset_enter(event):
			draw_create_dataset_button("#0174e8")
		
		def on_create_dataset_leave(event):
			draw_create_dataset_button("#015aca")
		
		def on_create_dataset_click(event):
			create_dataset_version()
		
		create_dataset_canvas.bind("<Enter>", on_create_dataset_enter)
		create_dataset_canvas.bind("<Leave>", on_create_dataset_leave)
		create_dataset_canvas.bind("<Button-1>", on_create_dataset_click)
		
		# Navigation functions
		def show_section3():
			section3_frame.pack(fill=tk.X, pady=(0, 20))
			aug_section_states[2] = True
		
		def hide_section3():
			section3_frame.pack_forget()
			aug_section_states[2] = False
		
		def go_back_to_section2():
			hide_section3()
		
		def show_section4():
			section4_frame.pack(fill=tk.X, pady=(0, 20))
			aug_section_states[3] = True
		
		def hide_section4():
			section4_frame.pack_forget()
			aug_section_states[3] = False
		
		def go_back_to_section3():
			hide_section4()
		
		def show_section5():
			section5_frame.pack(fill=tk.X, pady=(0, 20))
			aug_section_states[4] = True
		
		def hide_section5():
			section5_frame.pack_forget()
			aug_section_states[4] = False
		
		def go_back_to_section4():
			hide_section5()
		
		# Callback function to show all sections after images are loaded
		def _show_all_sections():
			show_section2()
			show_section3()
			show_section4()
			show_section5()
		
		# Assign the callback to the nonlocal variable
		show_all_sections_callback = _show_all_sections
		
		# Dataset creation function
		def create_dataset_version():
			# Validate inputs
			version_name = version_name_entry.get().strip()
			if not version_name:
				messagebox.showerror("Error", "Por favor ingresa un nombre para la versión del dataset")
				return
			
			if not aug_source_images:
				messagebox.showerror("Error", "No hay imágenes cargadas")
				return
			
			# Get version notes
			version_notes = version_notes_text.get("1.0", tk.END).strip()
			
			# Create output directory
			base_datasets_dir = _base_root() / "datasets"
			base_datasets_dir.mkdir(exist_ok=True)
			dataset_dir = base_datasets_dir / version_name
			
			if dataset_dir.exists():
				if not messagebox.askyesno("Confirmar", f"El dataset '{version_name}' ya existe. ¿Deseas sobrescribirlo?"):
					return
				import shutil
				shutil.rmtree(dataset_dir)
			
			dataset_dir.mkdir(exist_ok=True)
			
			# Create split directories based on model type
			if model_type["value"] == "detection":
				# For detection: images/train, images/val, images/test + labels/train, labels/val, labels/test
				images_dir = dataset_dir / "images"
				labels_dir = dataset_dir / "labels"
				images_dir.mkdir(exist_ok=True)
				labels_dir.mkdir(exist_ok=True)
				
				train_dir = images_dir / "train"
				val_dir = images_dir / "val"
				test_dir = images_dir / "test"
				train_dir.mkdir(exist_ok=True)
				val_dir.mkdir(exist_ok=True)
				test_dir.mkdir(exist_ok=True)
				
				train_labels_dir = labels_dir / "train"
				val_labels_dir = labels_dir / "val"
				test_labels_dir = labels_dir / "test"
				train_labels_dir.mkdir(exist_ok=True)
				val_labels_dir.mkdir(exist_ok=True)
				test_labels_dir.mkdir(exist_ok=True)
			else:
				# For classification: train, val, test
				train_dir = dataset_dir / "train"
				val_dir = dataset_dir / "val"
				test_dir = dataset_dir / "test"
				train_dir.mkdir(exist_ok=True)
				val_dir.mkdir(exist_ok=True)
				test_dir.mkdir(exist_ok=True)
			
			# Save version notes
			from datetime import datetime
			notes_file = dataset_dir / "version_notes.txt"
			with open(notes_file, 'w', encoding='utf-8') as f:
				f.write(f"Dataset Version: {version_name}\n")
				# Add dataset type
				dataset_type = "Classification" if model_type["value"] == "classification" else "Detection"
				f.write(f"Dataset Type: {dataset_type}\n")
				f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
				f.write(f"Total Images: {len(aug_source_images)}\n")
				f.write(f"Train Split: {aug_train_split['train']:.0f}%\n")
				f.write(f"Val Split: {aug_train_split['val']:.0f}%\n")
				f.write(f"Test Split: {aug_train_split['test']:.0f}%\n")
				f.write("\n--- Preprocessing Options ---\n")
				for opt, var in preproc_vars.items():
					f.write(f"  {opt}: {'Yes' if var.get() else 'No'}\n")
				f.write("\n--- Data Augmentation Options ---\n")
				for opt, var in aug_vars.items():
					f.write(f"  {opt}: {'Yes' if var.get() else 'No'}\n")
				f.write("\n--- Notes ---\n")
				f.write(version_notes if version_notes else "No notes provided.")
			
			# Create progress window
			progress_win = tk.Toplevel(train_win)
			progress_win.title("Creando Dataset")
			progress_win.geometry("600x200")
			progress_win.transient(train_win)
			progress_win.grab_set()
			
			progress_frame = tk.Frame(progress_win, bg=BG_COLOR, padx=30, pady=30)
			progress_frame.pack(fill=tk.BOTH, expand=True)
			
			progress_title = tk.Label(
				progress_frame,
				text="Procesando Dataset...",
				font=("Arial", 14, "bold"),
				fg=FG_COLOR,
				bg=BG_COLOR
			)
			progress_title.pack(pady=(0, 20))
			
			progress_bar = ttk.Progressbar(
				progress_frame,
				mode='determinate',
				length=500
			)
			progress_bar.pack(fill=tk.X, pady=(0, 10))
			
			progress_label = tk.Label(
				progress_frame,
				text="Iniciando...",
				font=("Arial", 10),
				fg=FG_COLOR,
				bg=BG_COLOR
			)
			progress_label.pack()
			
			progress_details = tk.Label(
				progress_frame,
				text="",
				font=("Arial", 9),
				fg=FG_COLOR,
				bg=BG_COLOR
			)
			progress_details.pack(pady=(5, 0))
			
			# Function to update progress
			def update_progress(current, total, message, details=""):
				progress_bar['value'] = (current / total) * 100
				progress_label.configure(text=message)
				progress_details.configure(text=details)
				progress_win.update()
			
			# Process images in thread
			import threading
			import random
			import shutil
			
			def process_images():
				try:
					if Image is None or ImageOps is None:
						messagebox.showerror("Error", "PIL/Pillow no está disponible")
						progress_win.destroy()
						return
					
					# Shuffle and split images
					is_classification = model_type["value"] == "classification" and aug_detected_classes
					
					if is_classification:
						# For classification: split per class to maintain class balance
						# Create class folders
						for class_name in aug_detected_classes:
							(train_dir / class_name).mkdir(exist_ok=True)
							(val_dir / class_name).mkdir(exist_ok=True)
							(test_dir / class_name).mkdir(exist_ok=True)
						
						# Group images by class
						images_by_class = {}
						for img_path in aug_source_images:
							if img_path in aug_image_classes:
								class_name = aug_image_classes[img_path]
								if class_name not in images_by_class:
									images_by_class[class_name] = []
								images_by_class[class_name].append(img_path)
						
						# Split each class independently
						train_images = []
						val_images = []
						test_images = []
						
						for class_name, class_images in images_by_class.items():
							random.shuffle(class_images)
							total_class = len(class_images)
							
							# Calculate split counts per class
							train_count = round(total_class * aug_train_split['train'] / 100)
							val_count = round(total_class * aug_train_split['val'] / 100)
							# Ensure all images are used
							test_count = total_class - train_count - val_count
							
							# Adjust if rounding caused issues
							if test_count < 0:
								if train_count > 0:
									train_count += test_count
									test_count = 0
								elif val_count > 0:
									val_count += test_count
									test_count = 0
							
							train_images.extend(class_images[:train_count])
							val_images.extend(class_images[train_count:train_count + val_count])
							test_images.extend(class_images[train_count + val_count:])
						
						total_images = len(aug_source_images)
					else:
						# For detection: split all images together
						random.shuffle(aug_source_images)
						total_images = len(aug_source_images)
						
						train_count = int(total_images * aug_train_split['train'] / 100)
						val_count = int(total_images * aug_train_split['val'] / 100)
						test_count = total_images - train_count - val_count
						
						train_images = aug_source_images[:train_count]
						val_images = aug_source_images[train_count:train_count + val_count]
						test_images = aug_source_images[train_count + val_count:]
					
					# Get selected preprocessing options
					selected_preproc = [opt for opt, var in preproc_vars.items() if var.get()]
					
					# Get selected augmentation options
					selected_aug = [opt for opt, var in aug_vars.items() if var.get()]
					
					# Helper function to apply preprocessing
					def apply_preprocessing(img, options):
						for opt in options:
							if opt == "Auto-Orient":
								img = ImageOps.exif_transpose(img)
							elif opt == "Resize (640x640)":
								img = img.resize((640, 640), Image.LANCZOS)
							elif opt == "Grayscale":
								img = ImageOps.grayscale(img)
								img = img.convert("RGB")  # Convert back to RGB
							elif opt == "Normalize":
								# Simple normalization
								pass
							elif opt == "Histogram Equalization":
								img = ImageOps.equalize(img)
							elif opt == "Denoise":
								# Simple blur for denoising
								from PIL import ImageFilter
								img = img.filter(ImageFilter.MedianFilter(size=3))
							elif opt == "Edge Enhancement":
								from PIL import ImageFilter
								img = img.filter(ImageFilter.EDGE_ENHANCE)
							elif opt == "Contrast Adjustment":
								from PIL import ImageEnhance
								enhancer = ImageEnhance.Contrast(img)
								img = enhancer.enhance(1.2)
						return img
					
					# Helper function to apply augmentation
					def apply_augmentation(img, aug_type):
						if aug_type == "Flip (Horizontal)":
							return ImageOps.mirror(img)
						elif aug_type == "Flip (Vertical)":
							return ImageOps.flip(img)
						elif aug_type == "90° Rotate":
							return img.rotate(90, expand=True)
						elif aug_type == "Crop":
							w, h = img.size
							crop_size = int(min(w, h) * 0.8)
							left = (w - crop_size) // 2
							top = (h - crop_size) // 2
							return img.crop((left, top, left + crop_size, top + crop_size))
						elif aug_type == "Rotation (Random)":
							angle = random.uniform(-15, 15)
							return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))
						elif aug_type == "Shear":
							# Simple shear using affine transform
							from PIL import Image as PILImage
							return img.transform(img.size, PILImage.AFFINE, (1, 0.2, 0, 0, 1, 0))
						elif aug_type == "Grayscale":
							gray = ImageOps.grayscale(img)
							return gray.convert("RGB")
						elif aug_type == "Hue":
							from PIL import ImageEnhance
							converter = ImageEnhance.Color(img)
							return converter.enhance(random.uniform(0.7, 1.3))
						elif aug_type == "Saturation":
							from PIL import ImageEnhance
							converter = ImageEnhance.Color(img)
							return converter.enhance(random.uniform(0.5, 1.5))
						elif aug_type == "Brightness":
							from PIL import ImageEnhance
							enhancer = ImageEnhance.Brightness(img)
							return enhancer.enhance(random.uniform(0.7, 1.3))
						elif aug_type == "Exposure":
							from PIL import ImageEnhance
							enhancer = ImageEnhance.Brightness(img)
							return enhancer.enhance(random.uniform(0.8, 1.2))
						elif aug_type == "Blur":
							from PIL import ImageFilter
							return img.filter(ImageFilter.GaussianBlur(radius=2))
						elif aug_type == "Noise":
							import numpy as np
							arr = np.array(img)
							noise = np.random.randint(-20, 20, arr.shape, dtype='int16')
							arr = np.clip(arr.astype('int16') + noise, 0, 255).astype('uint8')
							return Image.fromarray(arr)
						return img
					
					# Process all splits
					total_operations = len(train_images) * (1 + len(selected_aug) * 2) + len(val_images) + len(test_images)
					current_op = 0
					
					# STEP 1: Process and split images
					update_progress(0, 100, "Paso 1/3: Aplicando preprocesamiento y dividiendo dataset", f"0/{total_images} imágenes procesadas")
					
					# Process train images
					for idx, img_path in enumerate(train_images):
						try:
							img = Image.open(img_path)
							img = apply_preprocessing(img, selected_preproc)
							
							# Save original processed image
							stem = Path(img_path).stem
							ext = Path(img_path).suffix
							
							# Determine output directory based on model type
							if is_classification and img_path in aug_image_classes:
								class_name = aug_image_classes[img_path]
								output_path = train_dir / class_name / f"{stem}{ext}"
							else:
								output_path = train_dir / f"{stem}{ext}"
							
							img.save(output_path)
							
							current_op += 1
							update_progress(current_op, total_operations, 
								f"Paso 1/3: Procesando imágenes de entrenamiento",
								f"{idx + 1}/{len(train_images)} imágenes de train procesadas")
						except Exception as e:
							print(f"Error processing {img_path}: {e}")
					
					# Process val images
					for idx, img_path in enumerate(val_images):
						try:
							img = Image.open(img_path)
							img = apply_preprocessing(img, selected_preproc)
							
							# Save original processed image
							stem = Path(img_path).stem
							ext = Path(img_path).suffix
							
							# Determine output directory based on model type
							if is_classification and img_path in aug_image_classes:
								class_name = aug_image_classes[img_path]
								output_path = val_dir / class_name / f"{stem}{ext}"
							else:
								output_path = val_dir / f"{stem}{ext}"
							
							img.save(output_path)
							
							# For detection model, copy label file
							if not is_classification and model_type["value"] == "detection":
								# Look for corresponding .txt label file
								label_file = img_path.with_suffix('.txt')
								if label_file.exists():
									label_output = val_labels_dir / f"{stem}.txt"
									shutil.copy2(label_file, label_output)
							
							current_op += 1
							update_progress(current_op, total_operations,
								f"Paso 1/3: Procesando imágenes de validación",
								f"{idx + 1}/{len(val_images)} imágenes de val procesadas")
						except Exception as e:
							print(f"Error processing {img_path}: {e}")
					
					# Process test images
					for idx, img_path in enumerate(test_images):
						try:
							img = Image.open(img_path)
							img = apply_preprocessing(img, selected_preproc)
							
							# Save original processed image
							stem = Path(img_path).stem
							ext = Path(img_path).suffix
							
							# Determine output directory based on model type
							if is_classification and img_path in aug_image_classes:
								class_name = aug_image_classes[img_path]
								output_path = test_dir / class_name / f"{stem}{ext}"
							else:
								output_path = test_dir / f"{stem}{ext}"
							
							img.save(output_path)
							
							# For detection model, copy label file
							if not is_classification and model_type["value"] == "detection":
								# Look for corresponding .txt label file
								label_file = img_path.with_suffix('.txt')
								if label_file.exists():
									label_output = test_labels_dir / f"{stem}.txt"
									shutil.copy2(label_file, label_output)
							
							current_op += 1
							update_progress(current_op, total_operations,
								f"Paso 1/3: Procesando imágenes de testeo",
								f"{idx + 1}/{len(test_images)} imágenes de test procesadas")
						except Exception as e:
							print(f"Error processing {img_path}: {e}")
					
					# STEP 2: Apply data augmentation to train set
					if selected_aug:
						update_progress(current_op, total_operations,
							f"Paso 2/3: Aplicando data augmentation al set de entrenamiento",
							"Generando variantes aumentadas...")
						
						# Get train files based on model type
						if is_classification:
							train_files = []
							for class_name in aug_detected_classes:
								class_dir = train_dir / class_name
								train_files.extend(list(class_dir.glob("*")))
						else:
							train_files = list(train_dir.glob("*"))
						
						for idx, img_path in enumerate(train_files):
							try:
								img = Image.open(img_path)
								stem = img_path.stem
								ext = img_path.suffix
								
								# Generate 2 augmented variants
								for variant in range(1, 3):
									aug_img = img.copy()
									
									# Apply random augmentations
									num_augs = min(len(selected_aug), random.randint(1, 3))
									chosen_augs = random.sample(selected_aug, num_augs)
									
									for aug_type in chosen_augs:
										aug_img = apply_augmentation(aug_img, aug_type)
									
									# Save augmented image in same folder as original
									output_path = img_path.parent / f"{stem}_aug{variant}{ext}"
									aug_img.save(output_path)
									
									current_op += 1
									update_progress(current_op, total_operations,
										f"Paso 2/3: Aplicando data augmentation",
										f"{idx + 1}/{len(train_files)} imágenes originales, variante {variant}/2")
							except Exception as e:
								print(f"Error augmenting {img_path}: {e}")
					
					# STEP 3: Finalize
					update_progress(total_operations, total_operations,
						"Paso 3/3: Finalizando...",
						"Dataset creado exitosamente")
					
					# Count images in each split
					def count_images_in_dir(directory):
						if is_classification:
							total = 0
							for class_name in aug_detected_classes:
								class_dir = directory / class_name
								if class_dir.exists():
									total += len(list(class_dir.glob("*")))
							return total
						else:
							return len(list(directory.glob("*")))
					
					train_img_count = count_images_in_dir(train_dir)
					val_img_count = count_images_in_dir(val_dir)
					test_img_count = count_images_in_dir(test_dir)
					
					# For detection model, create data.yaml
					if not is_classification and model_type["value"] == "detection":
						import yaml
						
						# Get unique classes from det_classes
						class_names = [cls["name"] for cls in det_classes]
						
						# Create data.yaml content
						data_yaml_content = {
							'path': str(dataset_dir.absolute()),
							'train': 'images/train',
							'val': 'images/val',
							'test': 'images/test',
							'nc': len(class_names),
							'names': class_names
						}
						
						# Write data.yaml
						data_yaml_path = dataset_dir / "data.yaml"
						with open(data_yaml_path, 'w') as f:
							yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
					
					# Show completion message
					completion_msg = f"Dataset '{version_name}' creado exitosamente!\n\n"
					completion_msg += f"Ubicación: {dataset_dir}\n"
					if is_classification:
						completion_msg += f"Clases: {len(aug_detected_classes)} ({', '.join(sorted(aug_detected_classes))})\n"
					else:
						# For detection, show class info from det_classes
						completion_msg += f"Clases: {len(det_classes)} ({', '.join([cls['name'] for cls in det_classes])})\n"
					completion_msg += f"Train: {train_img_count} imágenes\n"
					completion_msg += f"Val: {val_img_count} imágenes\n"
					completion_msg += f"Test: {test_img_count} imágenes"
					
					train_win.after(500, lambda: [
						progress_win.destroy(),
						messagebox.showinfo("Éxito", completion_msg)
					])
					
				except Exception as e:
					train_win.after(0, lambda: [
						progress_win.destroy(),
						messagebox.showerror("Error", f"Error al crear dataset:\n{str(e)}")
					])
			
			# Start processing in background thread
			thread = threading.Thread(target=process_images, daemon=True)
			thread.start()
		
		# Button command already configured via bind in Canvas creation
		
		# Initialize UI state
		update_tabs_state()
		update_nav_state()
		
		# ============================================================
		# TAB 7: Configuracion
		# ============================================================
		config_tab = tabs[7]
		config_tab.configure(bg=PROC_CONTENT_BG)
		config_tab.grid_rowconfigure(0, weight=1)
		config_tab.grid_columnconfigure(0, weight=1)
		
		# Scrollable area for config tab
		config_scroll_canvas = tk.Canvas(config_tab, bg=BG_COLOR, highlightthickness=0)
		config_vsb = tk.Scrollbar(config_tab, orient="vertical", command=config_scroll_canvas.yview)
		config_scroll_canvas.configure(yscrollcommand=config_vsb.set)
		config_scroll_canvas.grid(row=0, column=0, sticky="nsew")
		config_vsb.grid(row=0, column=1, sticky="ns")
		
		config_content = tk.Frame(config_scroll_canvas, bg=BG_COLOR)
		config_scroll_window = config_scroll_canvas.create_window((0, 0), window=config_content, anchor="nw")
		
		def _update_config_scroll(_=None):
			config_scroll_canvas.configure(scrollregion=config_scroll_canvas.bbox("all"))
		
		def _resize_config(_=None):
			w = config_scroll_canvas.winfo_width()
			if w > 1:
				config_scroll_canvas.itemconfig(config_scroll_window, width=w)
		
		config_content.bind("<Configure>", _update_config_scroll)
		config_scroll_canvas.bind("<Configure>", _resize_config)
		
		# Main config frame
		config_main = tk.Frame(config_content, bg=BG_COLOR)
		config_main.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
		
		# Title
		config_title = tk.Label(
			config_main,
			text="Configuración de Entrenamiento",
			font=("Arial", 20, "bold"),
			fg=FG_COLOR,
			bg=BG_COLOR
		)
		config_title.pack(pady=(0, 30))
		
		# Dataset selection section
		dataset_section = tk.Frame(config_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		dataset_section.pack(fill=tk.X, pady=(0, 20))
		
		dataset_section_title = tk.Label(
			dataset_section,
			text="Selección de Dataset",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		dataset_section_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Dataset selection row
		dataset_select_frame = tk.Frame(dataset_section, bg=GRAY_BG)
		dataset_select_frame.pack(fill=tk.X, padx=20, pady=(10, 15))
		
		dataset_label = tk.Label(
			dataset_select_frame,
			text="Dataset:",
			font=("Arial", 12, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		dataset_label.pack(side=tk.LEFT, padx=(0, 10))
		
		# ComboBox for dataset selection with custom style
		selected_dataset = tk.StringVar()
		
		# Create custom style for combobox
		combo_style = ttk.Style()
		combo_style.configure(
			"Dataset.TCombobox",
			fieldbackground="#02224e",
			background="#02224e",
			foreground="white",
			selectbackground="#015bcb",
			selectforeground="white",
			arrowcolor="white"
		)
		combo_style.map("Dataset.TCombobox",
			fieldbackground=[("readonly", "#02224e")],
			foreground=[("readonly", "white")]
		)
		
		dataset_combobox = ttk.Combobox(
			dataset_select_frame,
			textvariable=selected_dataset,
			state="readonly",
			width=50,
			font=("Arial", 11),
			style="Dataset.TCombobox"
		)
		dataset_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
		
		# Function to scan datasets folder and filter by type
		def refresh_dataset_list():
			"""Scan datasets folder and populate combobox with matching datasets"""
			datasets_dir = _base_root() / "datasets"
			
			if not datasets_dir.exists():
				dataset_combobox['values'] = []
				selected_dataset.set("")
				return
			
			# Get current model type
			current_model_type = model_type["value"]
			if current_model_type is None:
				dataset_combobox['values'] = []
				selected_dataset.set("")
				return
			
			# Update model file to match current model type
			# (This ensures the correct model variant is selected)
			update_model_file()
			
			# Expected dataset type string
			expected_type = "Classification" if current_model_type == "classification" else "Detection"
			
			# Scan datasets folder
			matching_datasets = []
			for folder in datasets_dir.iterdir():
				if not folder.is_dir():
					continue
				
				# Look for version_notes.txt
				notes_file = folder / "version_notes.txt"
				if not notes_file.exists():
					continue
				
				try:
					# Read and check dataset type
					with open(notes_file, 'r', encoding='utf-8') as f:
						content = f.read()
						# Look for "Dataset Type: Classification" or "Dataset Type: Detection"
						if f"Dataset Type: {expected_type}" in content:
							matching_datasets.append(folder.name)
				except Exception:
					continue
			
			# Sort alphabetically
			matching_datasets.sort()
			
			# Update combobox
			dataset_combobox['values'] = matching_datasets
			
			# Select first if available
			if matching_datasets:
				selected_dataset.set(matching_datasets[0])
			else:
				selected_dataset.set("")
		
		# Refresh button (canvas-based rounded button)
		refresh_btn_canvas = tk.Canvas(
			dataset_select_frame,
			width=130,
			height=35,
			bg=GRAY_BG,
			highlightthickness=0
		)
		refresh_btn_canvas.pack(side=tk.LEFT, padx=(10, 0))
		
		def draw_refresh_btn(hover=False):
			refresh_btn_canvas.delete("all")
			color = "#0168d6" if hover else "#015bcb"
			r = 10
			w, h = 130, 35
			refresh_btn_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
			refresh_btn_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
			refresh_btn_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
			refresh_btn_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
			refresh_btn_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
			refresh_btn_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
			refresh_btn_canvas.create_text(w/2, h/2, text="Actualizar", fill="white", font=("Arial", 10, "bold"))
		
		draw_refresh_btn()
		refresh_btn_canvas.bind("<Enter>", lambda e: draw_refresh_btn(True))
		refresh_btn_canvas.bind("<Leave>", lambda e: draw_refresh_btn(False))
		refresh_btn_canvas.bind("<Button-1>", lambda e: refresh_dataset_list())
		refresh_btn_canvas.config(cursor="hand2")
		
		# Info label
		dataset_info = tk.Label(
			dataset_section,
			text="Los datasets mostrados coinciden con el tipo de modelo seleccionado",
			font=("Arial", 9, "italic"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		dataset_info.pack(anchor="w", padx=20, pady=(0, 15))
		
		# Auto-refresh when tab is shown
		def on_config_tab_focus():
			refresh_dataset_list()
		
		# Initial refresh
		refresh_dataset_list()
		
		# ============================================================
		# Model Size Selection Section
		# ============================================================
		model_size_section = tk.Frame(config_main, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
		model_size_section.pack(fill=tk.X, pady=(0, 20))
		
		model_size_title = tk.Label(
			model_size_section,
			text="Selección del Tamaño del Modelo",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=GRAY_BG
		)
		model_size_title.pack(anchor="w", padx=20, pady=(15, 10))
		
		# Secret feature: Click counter for unlocking Extra Grande
		secret_click_counter = {"count": 0, "unlocked": False}
		
		def on_title_click(event):
			"""Secret feature: Triple-click to unlock Extra Grande model"""
			secret_click_counter["count"] += 1
			
			# Reset counter after 1 second of inactivity
			def reset_counter():
				secret_click_counter["count"] = 0
			
			# Cancel previous reset timer if exists
			if hasattr(on_title_click, "reset_timer"):
				model_size_title.after_cancel(on_title_click.reset_timer)
			
			# Set new reset timer
			on_title_click.reset_timer = model_size_title.after(1000, reset_counter)
			
			# Check if triple-clicked
			if secret_click_counter["count"] >= 3 and not secret_click_counter["unlocked"]:
				secret_click_counter["unlocked"] = True
				unlock_extra_grande()
		
		model_size_title.bind("<Button-1>", on_title_click)
		
		# State for selected model size
		selected_model_size = {"value": "Mediano"}  # Default
		selected_model_file = {"value": None}
		
		# Container for model size buttons
		model_buttons_container = tk.Frame(model_size_section, bg=GRAY_BG)
		model_buttons_container.pack(fill=tk.X, padx=20, pady=(10, 15))
		
		# Configure grid for 4 columns
		for i in range(4):
			model_buttons_container.grid_columnconfigure(i, weight=1, uniform="model_size")
		
		# Model size options
		model_sizes = [
			{
				"name": "Nano",
				"description": "El más rápido de entrenar pero\ntambién el menos preciso.",
				"detection_file": "yolo11n.pt",
				"classification_file": "yolo11n-cls.pt"
			},
			{
				"name": "Pequeño",
				"description": "Rápido para entrenar pero\nmenos preciso.",
				"detection_file": "yolo11s.pt",
				"classification_file": "yolo11s-cls.pt"
			},
			{
				"name": "Mediano",
				"description": "Promedio tanto en tiempo de\nentrenamiento y precisión.",
				"detection_file": "yolo11m.pt",
				"classification_file": "yolo11m-cls.pt"
			},
			{
				"name": "Grande",
				"description": "Lento para entrenar pero\nmás preciso.",
				"detection_file": "yolo11l.pt",
				"classification_file": "yolo11l-cls.pt"
			}
		]
		
		model_size_frames = {}
		
		def unlock_extra_grande():
			"""Secret function: Transform 'Grande' button into 'Extra Grande'"""
			# Find the Grande button and update it
			if "Grande" in model_size_frames:
				old_frame_data = model_size_frames["Grande"]
				old_canvas = old_frame_data["canvas"]
				
				# Store the grid position before destroying
				grid_info = old_canvas.grid_info()
				row = grid_info.get("row", 0)
				column = grid_info.get("column", 3)
				padx = grid_info.get("padx", 5)
				
				# Destroy the old canvas
				old_canvas.destroy()
				
				# Update the model sizes list
				for model in model_sizes:
					if model["name"] == "Grande":
						model["name"] = "Extra Grande"
						model["description"] = "El más lento de entrenar pero\ntambién el más preciso."
						model["detection_file"] = "yolo11x.pt"
						model["classification_file"] = "yolo11x-cls.pt"
						break
				
				# Create new canvas for Extra Grande
				new_canvas = tk.Canvas(
					model_buttons_container,
					height=120,
					bg=GRAY_BG,
					highlightthickness=0
				)
				new_canvas.grid(row=row, column=column, padx=padx, sticky="nsew")
				
				# Update frame data
				model_size_frames["Extra Grande"] = {
					"canvas": new_canvas,
					"name": "Extra Grande",
					"desc": "El más lento de entrenar pero\ntambién el más preciso."
				}
				
				# Remove old entry
				model_size_frames.pop("Grande", None)
				
				# Initial draw (delayed to allow proper sizing)
				def delayed_draw():
					selected = (selected_model_size["value"] == "Extra Grande")
					draw_model_btn(new_canvas, "Extra Grande", "El más lento de entrenar pero\ntambién el más preciso.", selected, False)
				
				new_canvas.after(10, delayed_draw)
				
				# Bind configure event to redraw when canvas is resized
				def on_canvas_resize(e):
					selected = (selected_model_size["value"] == "Extra Grande")
					draw_model_btn(new_canvas, "Extra Grande", "El más lento de entrenar pero\ntambién el más preciso.", selected, False)
				
				new_canvas.bind("<Configure>", on_canvas_resize)
				
				# Hover effects
				def on_enter(e):
					if selected_model_size["value"] != "Extra Grande":
						draw_model_btn(new_canvas, "Extra Grande", "El más lento de entrenar pero\ntambién el más preciso.", False, True)
				
				def on_leave(e):
					if selected_model_size["value"] != "Extra Grande":
						draw_model_btn(new_canvas, "Extra Grande", "El más lento de entrenar pero\ntambién el más preciso.", False, False)
				
				# Bind hover events
				new_canvas.bind("<Enter>", on_enter)
				new_canvas.bind("<Leave>", on_leave)
				
				# Bind click events
				new_canvas.bind("<Button-1>", lambda e: select_model_size("Extra Grande"))
				new_canvas.config(cursor="hand2")
				
				# Update current selection if Grande was selected
				if selected_model_size["value"] == "Grande":
					selected_model_size["value"] = "Extra Grande"
					update_model_file()
					# Redraw with selected state
					new_canvas.after(20, lambda: draw_model_btn(new_canvas, "Extra Grande", "El más lento de entrenar pero\ntambién el más preciso.", True, False))
		
		def update_model_file():
			"""Update the selected model file based on model type and size"""
			size_name = selected_model_size["value"]
			model_data = next((m for m in model_sizes if m["name"] == size_name), None)
			
			if model_data:
				# Only update if model_type is set
				if model_type["value"] == "classification":
					selected_model_file["value"] = model_data["classification_file"]
				elif model_type["value"] == "detection":
					selected_model_file["value"] = model_data["detection_file"]
				else:
					# If model_type not set yet, default to detection
					selected_model_file["value"] = model_data["detection_file"]
		
		def select_model_size(size_name):
			"""Select a model size and update UI"""
			selected_model_size["value"] = size_name
			update_model_file()
			
			# Update all canvas buttons to show selection
			for name, frame_data in model_size_frames.items():
				canvas = frame_data["canvas"]
				button_name = frame_data["name"]
				button_desc = frame_data["desc"]
				selected = (name == size_name)
				draw_model_btn(canvas, button_name, button_desc, selected=selected, hover=False)
		
		# Create model size buttons (canvas-based)
		for idx, model_info in enumerate(model_sizes):
			# Create canvas for rounded button
			canvas = tk.Canvas(
				model_buttons_container,
				height=120,
				bg=GRAY_BG,
				highlightthickness=0
			)
			canvas.grid(row=0, column=idx, padx=5, sticky="nsew")
			
			def draw_model_btn(cv, name, desc, selected=False, hover=False):
				cv.delete("all")
				if selected:
					color = "#015bcb"
				elif hover:
					color = "#0168d6"
				else:
					color = "#02234e"
				
				# Get actual canvas width
				w = cv.winfo_width()
				h = 120
				
				# Ensure canvas has been rendered
				if w <= 1:
					w = 150  # Fallback width
				
				r = 10
				
				# Draw rounded rectangle
				cv.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
				cv.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
				cv.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
				cv.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
				cv.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
				cv.create_rectangle(0, r, w, h-r, fill=color, outline=color)
				
				# Draw text
				cv.create_text(w/2, 30, text=name, fill="white", font=("Arial", 13, "bold"))
				cv.create_text(w/2, 75, text=desc, fill="white", font=("Arial", 9), width=w-20, justify=tk.CENTER)
			
			# Store canvas reference
			model_size_frames[model_info["name"]] = {
				"canvas": canvas,
				"name": model_info["name"],
				"desc": model_info["description"]
			}
			
			# Initial draw (delayed to allow proper sizing)
			def delayed_draw(cv=canvas, name=model_info["name"], desc=model_info["description"]):
				draw_model_btn(cv, name, desc)
			
			canvas.after(10, delayed_draw)
			
			# Bind configure event to redraw when canvas is resized
			def on_canvas_resize(e, cv=canvas, name=model_info["name"], desc=model_info["description"]):
				if selected_model_size["value"] == name:
					draw_model_btn(cv, name, desc, selected=True, hover=False)
				else:
					draw_model_btn(cv, name, desc, selected=False, hover=False)
			
			canvas.bind("<Configure>", on_canvas_resize)
			
			# Hover effects
			def on_enter(e, cv=canvas, name=model_info["name"], desc=model_info["description"]):
				if selected_model_size["value"] != name:
					draw_model_btn(cv, name, desc, False, True)
			
			def on_leave(e, cv=canvas, name=model_info["name"], desc=model_info["description"]):
				if selected_model_size["value"] != name:
					draw_model_btn(cv, name, desc, False, False)
			
			# Bind hover events
			canvas.bind("<Enter>", on_enter)
			canvas.bind("<Leave>", on_leave)
			
			# Bind click events
			canvas.bind("<Button-1>", lambda e, name=model_info["name"]: select_model_size(name))
			canvas.config(cursor="hand2")
		
		# Set default selection
		select_model_size("Mediano")
		
		# ============================================================
		# TAB 8: Entrenamiento
		# ============================================================
		entrenamiento_tab = tabs[8]
		entrenamiento_tab.configure(bg=PROC_CONTENT_BG)
		
		# Main container
		entrenamiento_container = tk.Frame(entrenamiento_tab, bg=BG_COLOR)
		entrenamiento_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
		
		# Title
		entrenamiento_title = tk.Label(
			entrenamiento_container,
			text="Entrenamiento del Modelo",
			font=("Arial", 16, "bold"),
			fg=FG_COLOR,
			bg=BG_COLOR
		)
		entrenamiento_title.pack(anchor="w", pady=(0, 15))
		
		# Blue CMD-style output area
		cmd_frame = tk.Frame(entrenamiento_container, bg="#0e2f64", relief=tk.SUNKEN, bd=2)
		cmd_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
		cmd_frame.grid_rowconfigure(0, weight=1)
		cmd_frame.grid_columnconfigure(0, weight=1)
		
		# Text widget for CMD output with scrollbar
		cmd_scrollbar = tk.Scrollbar(cmd_frame, orient="vertical")
		cmd_scrollbar.grid(row=0, column=1, sticky="ns")
		
		cmd_text = tk.Text(
			cmd_frame,
			bg="#0e2f64",
			fg="#FFFFFF",
			font=("Consolas", 10),
			state="disabled",
			wrap=tk.WORD,
			yscrollcommand=cmd_scrollbar.set,
			relief=tk.FLAT,
			borderwidth=0
		)
		cmd_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
		cmd_scrollbar.config(command=cmd_text.yview)
		
		# Initial message in CMD
		cmd_text.config(state="normal")
		cmd_text.insert(tk.END, "Esperando inicio del entrenamiento...\n")
		cmd_text.config(state="disabled")
		
		# Training process state
		training_process = {"proc": None, "thread": None}
		
		def append_to_cmd(text: str):
			"""Append text to the CMD output area"""
			cmd_text.config(state="normal")
			cmd_text.insert(tk.END, text)
			cmd_text.see(tk.END)
			cmd_text.config(state="disabled")
		
		def run_training_process():
			"""Run the training directly by importing the training function"""
			import threading
			import sys
			
			# Get configuration values
			dataset_name = selected_dataset.get()
			if not dataset_name:
				append_to_cmd("\nError: No se ha seleccionado ningún dataset.\n")
				return
			
			model_file = selected_model_file.get("value")
			if not model_file:
				append_to_cmd("\nError: No se ha seleccionado ningún modelo.\n")
				return
			
			# Build dataset path
			dataset_path = _base_root() / "datasets" / dataset_name
			if not dataset_path.exists():
				append_to_cmd(f"\nError: El dataset no existe en la ruta: {dataset_path}\n")
				return
			
			# Build absolute model path
			model_path = _base_root() / model_file
			
			# Determine model type from file extension
			is_classification_model = "-cls.pt" in model_file
			current_model_type = model_type["value"]
			
			# Validate model type matches dataset type
			if current_model_type == "classification" and not is_classification_model:
				append_to_cmd(f"\n⚠️ ADVERTENCIA: El modelo seleccionado '{model_file}' no es un modelo de clasificación.\n")
				append_to_cmd(f"Los modelos de clasificación deben terminar en '-cls.pt'.\n")
				return
			elif current_model_type == "detection" and is_classification_model:
				append_to_cmd(f"\n⚠️ ADVERTENCIA: El modelo seleccionado '{model_file}' es un modelo de clasificación.\n")
				append_to_cmd(f"Para detección, usa modelos sin '-cls.pt'.\n")
				return
			
			if not model_path.exists():
				append_to_cmd(f"\n⚠️ El archivo del modelo no existe: {model_path}\n")
				append_to_cmd(f"YOLO intentará descargar el modelo automáticamente...\n\n")
			
			# Clear CMD and show starting message
			cmd_text.config(state="normal")
			cmd_text.delete("1.0", tk.END)
			cmd_text.config(state="disabled")
			
			# Show command info
			cmd_display = f'Iniciando entrenamiento: {current_model_type}\n'
			cmd_display += f'Dataset: "{dataset_path}"\n'
			cmd_display += f'Model: "{model_path}"\n'
			cmd_display += f'Trials: 5\n'
			cmd_display += f'Name: "{dataset_name}"\n\n'
			append_to_cmd(cmd_display)
			
			def run_in_thread():
				import subprocess
				
				try:
					# Build command based on model type
					if current_model_type == "classification":
						cmd = [
							sys.executable,
							str(_base_root() / "consolidation_tool" / "Train_Classifier_GUI.py"),
							"--data", str(dataset_path),
							"--model", str(model_path),
							"--name", dataset_name,
							"--trials", "5",
							"--timeout", "50400"  # 14 hours
						]
					else:  # detection
						data_yaml_path = dataset_path / "data.yaml"
						if not data_yaml_path.exists():
							train_win.after(0, lambda: append_to_cmd(f"\n❌ Error: No se encontró data.yaml en {dataset_path}\n"))
							return
						
						cmd = [
							sys.executable,
							str(_base_root() / "consolidation_tool" / "Train_Detection_GUI.py"),
							"--data", str(data_yaml_path),
							"--model", str(model_path),
							"--name", dataset_name,
							"--trials", "5",
							"--timeout", "3600"
						]
					
					# Start process with output capture
					process = subprocess.Popen(
						cmd,
						stdout=subprocess.PIPE,
						stderr=subprocess.STDOUT,
						text=True,
						bufsize=1,
						universal_newlines=True,
						encoding='utf-8',
						errors='replace'
					)
					
					training_process["proc"] = process
					
					# Read output line by line in real-time
					for line in iter(process.stdout.readline, ''):
						if line:
							train_win.after(0, lambda l=line: append_to_cmd(l))
					
					# Wait for process to complete
					process.wait()
					
					# Check exit code
					if process.returncode == 0:
						train_win.after(0, lambda: append_to_cmd("\n\n=== Entrenamiento completado exitosamente ===\n"))
						train_win.after(100, show_training_results_popup)
					else:
						train_win.after(0, lambda: append_to_cmd(f"\n\n=== Entrenamiento fallido con código: {process.returncode} ===\n"))
					
				except Exception as e:
					import traceback
					error_msg = f"{str(e)}\n{traceback.format_exc()}"
					train_win.after(0, lambda: append_to_cmd(f"\n\n❌ Error al ejecutar el entrenamiento:\n{error_msg}\n"))
				finally:
					training_process["proc"] = None
			
			def show_training_results_popup():
				"""Show popup with training results and cleanup option"""
				
				# Try to load training results
				dataset_name = selected_dataset.get()
				results_file = _base_root() / "runs" / "optuna" / dataset_name / "training_results.json"
				
				if not results_file.exists():
					messagebox.showwarning(
						"Resultados no encontrados",
						"No se pudieron cargar los resultados del entrenamiento."
					)
					return
				
				try:
					with open(results_file, 'r', encoding='utf-8') as f:
						stats = json.load(f)
					
					if not stats.get("completed", False):
						messagebox.showerror(
							"Error en entrenamiento",
							stats.get("error", "El entrenamiento no se completó correctamente.")
						)
						return
					
					# Create popup window
					popup = tk.Toplevel(train_win)
					popup.title("🎉 Entrenamiento Completado")
					popup.configure(bg=BG_COLOR)
					popup.geometry("650x600")
					popup.resizable(True, True)
					
					# Center popup
					popup.transient(train_win)
					popup.grab_set()
					
					# Configure grid for popup
					popup.grid_rowconfigure(0, weight=1)
					popup.grid_columnconfigure(0, weight=1)
					
					# Create canvas with scrollbar
					canvas = tk.Canvas(popup, bg=BG_COLOR, highlightthickness=0)
					scrollbar = tk.Scrollbar(popup, orient="vertical", command=canvas.yview)
					scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)
					
					scrollable_frame.bind(
						"<Configure>",
						lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
					)
					
					canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
					canvas.configure(yscrollcommand=scrollbar.set)
					
					canvas.grid(row=0, column=0, sticky="nsew")
					scrollbar.grid(row=0, column=1, sticky="ns")
					
					# Enable mouse wheel scrolling
					def on_mousewheel(event):
						canvas.yview_scroll(int(-1*(event.delta/120)), "units")
					
					canvas.bind_all("<MouseWheel>", on_mousewheel)
					
					# Main container with padding
					main_container = tk.Frame(scrollable_frame, bg=BG_COLOR)
					main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
					
					# Title
					title_label = tk.Label(
						main_container,
						text="🎉 Entrenamiento Completado",
						font=("Arial", 18, "bold"),
						fg="#4CAF50",
						bg=BG_COLOR
					)
					title_label.pack(pady=(0, 20))
					
					# Results frame
					results_frame = tk.Frame(main_container, bg=GRAY_BG, relief=tk.RIDGE, bd=2)
					results_frame.pack(fill=tk.X, pady=(0, 20))
					
					# Validation Accuracy
					val_acc = stats.get("val_accuracy", 0)
					val_label = tk.Label(
						results_frame,
						text=f"📊 Validation Accuracy: {val_acc:.2%}",
						font=("Arial", 14, "bold"),
						fg=FG_COLOR,
						bg=GRAY_BG
					)
					val_label.pack(anchor="w", padx=20, pady=(20, 10))
					
					# Train Accuracy
					train_acc = stats.get("train_accuracy", 0)
					train_label = tk.Label(
						results_frame,
						text=f"📈 Train Accuracy: {train_acc:.2%}",
						font=("Arial", 12),
						fg=FG_COLOR,
						bg=GRAY_BG
					)
					train_label.pack(anchor="w", padx=20, pady=(0, 10))
					
					# Test Accuracy (if available)
					test_acc = stats.get("test_accuracy")
					if test_acc is not None:
						test_label = tk.Label(
							results_frame,
							text=f"🧪 Test Accuracy: {test_acc:.2%}",
							font=("Arial", 12),
							fg=FG_COLOR,
							bg=GRAY_BG
						)
						test_label.pack(anchor="w", padx=20, pady=(0, 10))
					
					# Divider
					tk.Frame(results_frame, bg=FG_COLOR, height=1).pack(fill=tk.X, padx=20, pady=10)
					
					# Overfitting Analysis
					overfitting_level = stats.get("overfitting_level", "")
					overfitting_diff = stats.get("overfitting_diff", 0)
					
					overfitting_title = tk.Label(
						results_frame,
						text="🔍 Análisis de Overfitting",
						font=("Arial", 13, "bold"),
						fg=FG_COLOR,
						bg=GRAY_BG
					)
					overfitting_title.pack(anchor="w", padx=20, pady=(0, 10))
					
					overfitting_status = tk.Label(
						results_frame,
						text=f"Nivel: {overfitting_level}",
						font=("Arial", 11),
						fg=FG_COLOR,
						bg=GRAY_BG
					)
					overfitting_status.pack(anchor="w", padx=20, pady=(0, 5))
					
					overfitting_diff_label = tk.Label(
						results_frame,
						text=f"Diferencia Train-Val: {overfitting_diff:.2%}",
						font=("Arial", 11),
						fg=FG_COLOR,
						bg=GRAY_BG
					)
					overfitting_diff_label.pack(anchor="w", padx=20, pady=(0, 10))
					
					recommendation = stats.get("recommendation", "")
					recommendation_label = tk.Label(
						results_frame,
						text=recommendation,
						font=("Arial", 10, "italic"),
						fg="#FFD700",
						bg=GRAY_BG,
						wraplength=500,
						justify=tk.LEFT
					)
					recommendation_label.pack(anchor="w", padx=20, pady=(0, 20))
					
					# Question about cleaning up models
					question_frame = tk.Frame(main_container, bg=BG_COLOR)
					question_frame.pack(fill=tk.X, pady=(0, 20))
					
					question_label = tk.Label(
						question_frame,
						text="¿Desea eliminar los modelos intermedios generados durante el entrenamiento?\n(Solo se conservará el mejor modelo)",
						font=("Arial", 11, "bold"),
						fg=FG_COLOR,
						bg=BG_COLOR,
						justify=tk.CENTER
					)
					question_label.pack(pady=(0, 15))
					
					# Buttons frame
					buttons_frame = tk.Frame(main_container, bg=BG_COLOR)
					buttons_frame.pack(fill=tk.X)
					
					def cleanup_and_advance():
						"""Delete intermediate models and advance to next tab"""
						try:
							optuna_dir = Path(stats.get("optuna_dir"))
							best_trial_num = stats.get("best_trial_number")
							best_model_path = Path(stats.get("best_model_path"))
							
							if optuna_dir.exists():
								# List all trial folders
								deleted_count = 0
								for folder in optuna_dir.iterdir():
									if folder.is_dir() and f"trial_{best_trial_num}_" not in folder.name:
										shutil.rmtree(folder)
										deleted_count += 1
								
								append_to_cmd(f"\n\n🗑️  Se eliminaron {deleted_count} modelos intermedios.")
								append_to_cmd(f"\n✅ Mejor modelo conservado: {best_model_path}\n")
						except Exception as e:
							append_to_cmd(f"\n⚠️  Error al limpiar modelos: {str(e)}\n")
						
						popup.destroy()
						canvas.unbind_all("<MouseWheel>")
						# Advance to next tab (Estadisticas)
						can_advance[8] = True
						update_tabs_state()
						go_next()
					
					def skip_cleanup_and_advance():
						"""Skip cleanup and advance to next tab"""
						append_to_cmd("\n\nℹ️  Se conservaron todos los modelos generados.\n")
						popup.destroy()
						canvas.unbind_all("<MouseWheel>")
						# Advance to next tab (Estadisticas)
						can_advance[8] = True
						update_tabs_state()
						go_next()
					
					yes_btn = tk.Button(
						buttons_frame,
						text="Sí, eliminar modelos intermedios",
						font=("Arial", 12, "bold"),
						fg=FG_COLOR,
						bg="#D32F2F",
						activebackground="#B71C1C",
						activeforeground=FG_COLOR,
						relief=tk.RAISED,
						padx=20,
						pady=10,
						command=cleanup_and_advance
					)
					yes_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
					
					no_btn = tk.Button(
						buttons_frame,
						text="No, conservar todos",
						font=("Arial", 12, "bold"),
						fg=FG_COLOR,
						bg="#1976D2",
						activebackground="#0D47A1",
						activeforeground=FG_COLOR,
						relief=tk.RAISED,
						padx=20,
						pady=10,
						command=skip_cleanup_and_advance
					)
					no_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
					
					# Handle window close (treat as "No")
					popup.protocol("WM_DELETE_WINDOW", skip_cleanup_and_advance)
					
				except Exception as e:
					messagebox.showerror(
						"Error",
						f"Error al cargar los resultados del entrenamiento:\n{str(e)}"
					)
			
			# Start training in background thread
			thread = threading.Thread(target=run_in_thread, daemon=True)
			thread.start()
			training_process["thread"] = thread
		
		# ===== Train Model Button =====
		train_model_btn_frame = tk.Frame(config_content, bg=BG_COLOR)
		train_model_btn_frame.pack(fill=tk.X, pady=(30, 20))
		
		def start_training():
			"""Switch to training tab and start the training process"""
			nonlocal current_step
			current_step = 8  # Training tab
			update_tabs_state()
			# Wait a moment for tab to switch, then start training
			train_win.after(100, run_training_process)
		
		# Canvas-based rounded button for Entrenar Modelo
		train_model_canvas = tk.Canvas(
			train_model_btn_frame,
			width=250,
			height=60,
			bg=BG_COLOR,
			highlightthickness=0
		)
		train_model_canvas.pack()
		
		def draw_train_btn(hover=False):
			train_model_canvas.delete("all")
			color = "#0168d6" if hover else "#015bcb"
			r = 15
			w, h = 250, 60
			train_model_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
			train_model_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
			train_model_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
			train_model_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
			train_model_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
			train_model_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
			train_model_canvas.create_text(w/2, h/2, text="Entrenar modelo", fill="white", font=("Arial", 14, "bold"))
		
		draw_train_btn()
		train_model_canvas.bind("<Enter>", lambda e: draw_train_btn(True))
		train_model_canvas.bind("<Leave>", lambda e: draw_train_btn(False))
		train_model_canvas.bind("<Button-1>", lambda e: start_training())
		train_model_canvas.config(cursor="hand2")
		
		# ============================================================
		# TAB 9: Estadisticas
		# ============================================================
		estadisticas_tab = tabs[9]
		estadisticas_tab.configure(bg=PROC_CONTENT_BG)
		estadisticas_tab.grid_rowconfigure(0, weight=1)
		estadisticas_tab.grid_columnconfigure(0, weight=1)
		
		# Scrollable area for statistics
		stats_scroll_canvas = tk.Canvas(estadisticas_tab, bg=BG_COLOR, highlightthickness=0)
		stats_vsb = tk.Scrollbar(estadisticas_tab, orient="vertical", command=stats_scroll_canvas.yview)
		stats_scroll_canvas.configure(yscrollcommand=stats_vsb.set)
		stats_scroll_canvas.grid(row=0, column=0, sticky="nsew")
		stats_vsb.grid(row=0, column=1, sticky="ns")
		
		stats_content = tk.Frame(stats_scroll_canvas, bg=BG_COLOR)
		stats_scroll_window = stats_scroll_canvas.create_window((0, 0), window=stats_content, anchor="nw")
		
		def _update_stats_scroll(_=None):
			stats_scroll_canvas.configure(scrollregion=stats_scroll_canvas.bbox("all"))
		
		def _resize_stats(_=None):
			w = stats_scroll_canvas.winfo_width()
			stats_scroll_canvas.itemconfigure(stats_scroll_window, width=w)
			_update_stats_scroll()
		
		stats_content.bind("<Configure>", _update_stats_scroll)
		stats_scroll_canvas.bind("<Configure>", _resize_stats)
		
		# Title
		stats_title = tk.Label(
			stats_content,
			text="Estadísticas de Entrenamiento",
			font=("Arial", 18, "bold"),
			fg=FG_COLOR,
			bg=BG_COLOR
		)
		stats_title.pack(pady=(20, 10))
		
		stats_subtitle = tk.Label(
			stats_content,
			text="Resultados de entrenamientos previos encontrados en el dataset",
			font=("Arial", 12),
			fg=FG_COLOR,
			bg=BG_COLOR
		)
		stats_subtitle.pack(pady=(0, 20))
		
		# Container for statistics cards
		stats_cards_container = tk.Frame(stats_content, bg=BG_COLOR)
		stats_cards_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))
		
		# Refresh button
		refresh_btn_frame = tk.Frame(stats_content, bg=BG_COLOR)
		refresh_btn_frame.pack(pady=(0, 20))
		
		def find_training_results():
			"""Recursively find all training_results.json files in datasets"""
			results = []
			try:
				# Base directory for datasets
				base_dir = _base_root() / "runs" / "optuna"
				if not base_dir.exists():
					return results
				
				# Search recursively for training_results.json files
				for json_path in base_dir.rglob("training_results.json"):
					try:
						# Get dataset name (parent directory of the json file)
						dataset_name = json_path.parent.name
						
						# Read JSON file
						with open(json_path, "r", encoding="utf-8") as f:
							data = json.load(f)
						
						results.append({
							"dataset_name": dataset_name,
							"json_path": json_path,
							"data": data
						})
					except Exception as e:
						print(f"Error reading {json_path}: {e}")
						continue
			except Exception as e:
				print(f"Error searching for training results: {e}")
			
			return results
		
		def create_stat_card(parent, result, row, col):
			"""Create a card displaying training statistics"""
			dataset_name = result["dataset_name"]
			data = result["data"]
			json_path = result["json_path"]
			
			# Fixed width for cards (approximately half of typical window width)
			CARD_WIDTH = 500
			
			# Container frame for the card with fixed width
			card_container = tk.Frame(parent, bg=BG_COLOR, width=CARD_WIDTH)
			card_container.grid(row=row, column=col, padx=10, pady=10, sticky="n")
			card_container.grid_propagate(False)  # Prevent resizing based on content
			
			# Canvas for rounded border with fixed width
			card_canvas = tk.Canvas(
				card_container,
				bg=BG_COLOR,
				highlightthickness=0,
				width=CARD_WIDTH,
				height=10  # Will be updated after content is added
			)
			card_canvas.pack(fill=tk.BOTH, expand=True)
			
			# Card frame with border (inside canvas) with fixed width
			card_frame = tk.Frame(card_canvas, bg=GRAY_BG, width=CARD_WIDTH-10)
			card_window = card_canvas.create_window(5, 5, window=card_frame, anchor="nw")
			
			# Header with dataset name (with rounded top corners)
			header_canvas = tk.Canvas(
				card_frame,
				bg=GRAY_BG,
				highlightthickness=0,
				width=CARD_WIDTH-10,
				height=50
			)
			header_canvas.pack(fill=tk.X)
			
			def draw_header():
				header_canvas.delete("all")
				w = CARD_WIDTH - 10
				h = 50
				r = 10
				color = "#015bcb"
				
				# Draw rounded rectangle (only top corners rounded)
				header_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
				header_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
				header_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
				header_canvas.create_rectangle(0, r, w, h, fill=color, outline=color)
				
				# Draw text
				header_canvas.create_text(15, h/2, text=f"📊 Dataset: {dataset_name}", 
					fill="white", font=("Arial", 14, "bold"), anchor="w")
			
			header_canvas.after(10, draw_header)
			
			# Content area
			content_frame = tk.Frame(card_frame, bg=GRAY_BG)
			content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
			
			# Display statistics in a grid
			row = 0
			
			# Model information
			if "model" in data:
				model_info = tk.Label(
					content_frame,
					text=f"Modelo: {data['model']}",
					font=("Arial", 11, "bold"),
					fg="#4CAF50",
					bg=GRAY_BG,
					justify=tk.LEFT,
					wraplength=450,
					anchor="w"
				)
				model_info.grid(row=row, column=0, sticky="w", pady=5)
				row += 1
			
			# Training parameters
			if "epochs" in data:
				epochs_label = tk.Label(
					content_frame,
					text=f"Épocas entrenadas: {data['epochs']}",
					font=("Arial", 10),
					fg=FG_COLOR,
					bg=GRAY_BG,
					wraplength=450,
					anchor="w"
				)
				epochs_label.grid(row=row, column=0, sticky="w", pady=2)
				row += 1
			
			if "imgsz" in data:
				imgsz_label = tk.Label(
					content_frame,
					text=f"Tamaño de imagen: {data['imgsz']}",
					font=("Arial", 10),
					fg=FG_COLOR,
					bg=GRAY_BG,
					wraplength=450,
					anchor="w"
				)
				imgsz_label.grid(row=row, column=0, sticky="w", pady=2)
				row += 1
			
			if "batch" in data:
				batch_label = tk.Label(
					content_frame,
					text=f"Batch size: {data['batch']}",
					font=("Arial", 10),
					fg=FG_COLOR,
					bg=GRAY_BG,
					wraplength=450,
					anchor="w"
				)
				batch_label.grid(row=row, column=0, sticky="w", pady=2)
				row += 1
			
			# Separator
			separator = tk.Frame(content_frame, bg=FG_COLOR, height=2)
			separator.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
			row += 1
			
			# Metrics title
			metrics_title = tk.Label(
				content_frame,
				text="Métricas de Rendimiento:",
				font=("Arial", 11, "bold"),
				fg=FG_COLOR,
				bg=GRAY_BG
			)
			metrics_title.grid(row=row, column=0, sticky="w", pady=5)
			row += 1
			
			# Display all available metrics
			metrics_displayed = False
			for key, value in data.items():
				# Skip already displayed fields
				if key in ["model", "epochs", "imgsz", "batch", "dataset_name"]:
					continue
				
				# Format metric name (convert snake_case to Title Case)
				metric_name = key.replace("_", " ").title()
				
				# Format value
				if isinstance(value, (int, float)):
					if isinstance(value, float):
						value_str = f"{value:.4f}"
					else:
						value_str = str(value)
				else:
					value_str = str(value)
				
				metric_label = tk.Label(
					content_frame,
					text=f"  • {metric_name}: {value_str}",
					font=("Arial", 10),
					fg=FG_COLOR,
					bg=GRAY_BG,
					justify=tk.LEFT,
					wraplength=450,
					anchor="w"
				)
				metric_label.grid(row=row, column=0, sticky="w", pady=2)
				row += 1
				metrics_displayed = True
			
			if not metrics_displayed:
				no_metrics_label = tk.Label(
					content_frame,
					text="  No hay métricas adicionales disponibles",
					font=("Arial", 10, "italic"),
					fg="#888888",
					bg=GRAY_BG
				)
				no_metrics_label.grid(row=row, column=0, sticky="w", pady=2)
				row += 1
			
			# Footer with file path
			footer_frame = tk.Frame(card_frame, bg=GRAY_BG)
			footer_frame.pack(fill=tk.X, padx=15, pady=(10, 15))
			
			path_label = tk.Label(
				footer_frame,
				text=f"📁 {json_path}",
				font=("Arial", 8),
				fg="#888888",
				bg=GRAY_BG,
				wraplength=450,
				anchor="w",
				justify=tk.LEFT
			)
			path_label.pack(anchor="w")
			
			# Update canvas size after content is added
			def update_card_size():
				card_frame.update_idletasks()
				w = CARD_WIDTH - 10
				h = card_frame.winfo_reqheight() + 10
				card_canvas.config(width=CARD_WIDTH, height=h)
				
				# Draw rounded border
				card_canvas.delete("border")
				r = 10
				border_color = "#1A3A7A"
				
				# Draw rounded rectangle border
				card_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, outline=border_color, width=2, tags="border", style=tk.ARC)
				card_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, outline=border_color, width=2, tags="border", style=tk.ARC)
				card_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, outline=border_color, width=2, tags="border", style=tk.ARC)
				card_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, outline=border_color, width=2, tags="border", style=tk.ARC)
				card_canvas.create_line(r, 0, w-r, 0, fill=border_color, width=2, tags="border")
				card_canvas.create_line(r, h, w-r, h, fill=border_color, width=2, tags="border")
				card_canvas.create_line(0, r, 0, h-r, fill=border_color, width=2, tags="border")
				card_canvas.create_line(w, r, w, h-r, fill=border_color, width=2, tags="border")
			
			card_frame.after(50, update_card_size)
		
		def refresh_statistics():
			"""Refresh the statistics display"""
			# Clear existing cards
			for widget in stats_cards_container.winfo_children():
				widget.destroy()
			
			# Find all training results
			results = find_training_results()
			
			if not results:
				# No results found
				no_results_label = tk.Label(
					stats_cards_container,
					text="No se encontraron resultados de entrenamiento.\n\nLos archivos 'training_results.json' se generan automáticamente\nal completar un entrenamiento.",
					font=("Arial", 12),
					fg="#888888",
					bg=BG_COLOR,
					justify=tk.CENTER
				)
				no_results_label.grid(row=0, column=0, columnspan=3, pady=50)
			else:
				# Display results count
				count_label = tk.Label(
					stats_cards_container,
					text=f"Se encontraron {len(results)} resultado(s) de entrenamiento:",
					font=("Arial", 11, "bold"),
					fg=FG_COLOR,
					bg=BG_COLOR
				)
				count_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10), padx=10)
				
				# Configure grid columns to be equal width (3 columns)
				stats_cards_container.grid_columnconfigure(0, weight=1, uniform="cards")
				stats_cards_container.grid_columnconfigure(1, weight=1, uniform="cards")
				stats_cards_container.grid_columnconfigure(2, weight=1, uniform="cards")
				
				# Create a card for each result in a 3-column layout
				for idx, result in enumerate(results):
					row = (idx // 3) + 1  # Start from row 1 (row 0 is the count label)
					col = idx % 3
					create_stat_card(stats_cards_container, result, row, col)
		
		refresh_btn = tk.Button(
			refresh_btn_frame,
			text="Actualizar Estadísticas",
			font=("Arial", 12, "bold"),
			fg=FG_COLOR,
			bg="#0B5ED7",
			activebackground="#0A4EB5",
			activeforeground=FG_COLOR,
			relief=tk.RAISED,
			padx=30,
			pady=10,
			command=refresh_statistics
		)
		refresh_btn.pack()
		
		# Load statistics on tab initialization
		train_win.after(100, refresh_statistics)

		# ============================================================
		# TAB 10: Pruebas (Model Testing)
		# ============================================================
		pruebas_tab = tabs[10]
		pruebas_tab.configure(bg=PROC_CONTENT_BG)
	
		# Main container with grid layout
		pruebas_tab.grid_rowconfigure(0, weight=1)
		pruebas_tab.grid_columnconfigure(0, weight=2, minsize=300)  # Left menu
		pruebas_tab.grid_columnconfigure(1, weight=8)  # Right display
	
		# Left menu frame
		left_frame = tk.Frame(pruebas_tab, bg=BG_COLOR, padx=20, pady=20)
		left_frame.grid(row=0, column=0, sticky="nsew")
	
		# Right display frame
		right_frame = tk.Frame(pruebas_tab, bg=BG_COLOR, padx=10, pady=10)
		right_frame.grid(row=0, column=1, sticky="nsew")
	
		# ==== LEFT MENU COMPONENTS ====
	
		# Title
		title_label = tk.Label(
			left_frame,
			text="Pruebas de Modelo",
			font=("Arial", 18, "bold"),
			bg=BG_COLOR,
			fg=FG_COLOR
		)
		title_label.pack(pady=(0, 20))
	
		# Find available models
		def find_models():
			"""Scan for trained models in best_models/ and runs/optuna/"""
			import os
			from pathlib import Path
		
			models = []
			base_root = _base_root()
		
			# Search best_models/
			best_models_dir = base_root / "best_models"
			if best_models_dir.exists():
				for file in best_models_dir.iterdir():
					if file.suffix == ".pt":
						models.append(("best_models/" + file.name, str(file)))
		
			# Search runs/optuna/
			optuna_dir = base_root / "runs" / "optuna"
			if optuna_dir.exists():
				for best_pt in optuna_dir.rglob("best.pt"):
					rel_path = best_pt.relative_to(base_root)
					display_name = str(rel_path).replace("\\", "/")
					models.append((display_name, str(best_pt)))
		
			return models
	
		available_models = find_models()
		model_display_names = [name for name, _ in available_models]
		model_paths_dict = {name: path for name, path in available_models}

		# Multi-model selection list (max 9)
		models_list_state = []  # List of {"path_key": str, "name": str}

		models_label = tk.Label(
			left_frame,
			text="Modelos (máx. 9):",
			font=("Arial", 12),
			bg=BG_COLOR,
			fg=FG_COLOR
		)
		models_label.pack(anchor="w", pady=(0, 5))

		models_list_frame = tk.Frame(left_frame, bg=BG_COLOR)
		models_list_frame.pack(fill="x", pady=(0, 5))

		models_list_canvas = tk.Canvas(models_list_frame, bg=BG_COLOR, highlightthickness=0, height=200)
		models_list_scrollbar = tk.Scrollbar(models_list_frame, orient="vertical", command=models_list_canvas.yview)
		models_list_canvas.configure(yscrollcommand=models_list_scrollbar.set)
		models_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		models_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		models_list_inner = tk.Frame(models_list_canvas, bg=BG_COLOR)
		models_list_canvas_win = models_list_canvas.create_window((0, 0), window=models_list_inner, anchor="nw")
		models_list_inner.bind("<Configure>", lambda e: models_list_canvas.configure(scrollregion=models_list_canvas.bbox("all")))
		models_list_canvas.bind("<Configure>", lambda e: models_list_canvas.itemconfigure(models_list_canvas_win, width=e.width))

		def refresh_models_list_ui():
			for w in models_list_inner.winfo_children():
				w.destroy()
			for idx, entry in enumerate(models_list_state):
				row = tk.Frame(models_list_inner, bg="#0a1e40")
				row.pack(fill=tk.X, pady=2)
				name_entry = tk.Entry(row, font=("Arial", 9), width=12, bg="#02234f", fg="white", insertbackground="white", relief=tk.FLAT, bd=1)
				name_entry.insert(0, entry["name"])
				name_entry.pack(side=tk.LEFT, padx=(4, 2), pady=2)
				def _on_name_change(e, i=idx, ne=name_entry):
					if i < len(models_list_state):
						models_list_state[i]["name"] = ne.get().strip()
				name_entry.bind("<FocusOut>", _on_name_change)
				path_lbl = tk.Label(row, text=entry["path_key"], font=("Arial", 8), fg="#aaaaaa", bg="#0a1e40", anchor="w")
				path_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
				def _remove(i=idx):
					if i < len(models_list_state):
						models_list_state.pop(i)
						refresh_models_list_ui()
						check_enable_test_button()
				rm_btn = tk.Label(row, text="✕", font=("Arial", 10, "bold"), fg="#ff4444", bg="#0a1e40", cursor="hand2")
				rm_btn.pack(side=tk.RIGHT, padx=(2, 4))
				rm_btn.bind("<Button-1>", lambda e, fn=_remove: fn())
			check_enable_test_button()

		# Model selector combobox + Add button
		add_model_frame = tk.Frame(left_frame, bg=BG_COLOR)
		add_model_frame.pack(fill="x", pady=(0, 5))

		# Create custom style for model combobox
		model_combo_style = ttk.Style()
		model_combo_style.configure(
			"ModelTest.TCombobox",
			fieldbackground="#02234f",
			background="#02234f",
			foreground="white",
			selectbackground="#015bcb",
			selectforeground="white",
			arrowcolor="white"
		)
		model_combo_style.map("ModelTest.TCombobox",
			fieldbackground=[("readonly", "#02234f")],
			foreground=[("readonly", "white")]
		)

		model_var = tk.StringVar()
		model_combo = ttk.Combobox(
			add_model_frame,
			textvariable=model_var,
			values=model_display_names,
			state="readonly",
			width=25,
			font=("Arial", 9),
			style="ModelTest.TCombobox"
		)
		model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

		def add_selected_model():
			key = model_var.get()
			if not key or key not in model_paths_dict:
				return
			if len(models_list_state) >= 9:
				return
			base_name = Path(key).stem if "/" not in Path(key).stem else Path(key).parent.name
			models_list_state.append({"path_key": key, "name": base_name})
			model_var.set("")
			refresh_models_list_ui()

		add_model_btn = tk.Label(add_model_frame, text="+", font=("Arial", 14, "bold"), fg="white", bg="#015bcb", cursor="hand2", padx=8, pady=0)
		add_model_btn.pack(side=tk.LEFT)
		add_model_btn.bind("<Button-1>", lambda e: add_selected_model())

		# Separator
		tk.Frame(left_frame, bg="#444444", height=1).pack(fill=tk.X, pady=(10, 10))
	
		# Video selection
		video_label = tk.Label(
			left_frame,
			text="Video:",
			font=("Arial", 12),
			bg=BG_COLOR,
			fg=FG_COLOR
		)
		video_label.pack(anchor="w", pady=(0, 5))
	
		video_path_var = tk.StringVar()
		video_entry = tk.Entry(
			left_frame,
			textvariable=video_path_var,
			font=("Arial", 10),
			state="readonly",
			bg="#02234f",
			fg="white",
			readonlybackground="#02234f",
			insertbackground="white"
		)
		video_entry.pack(fill="x", pady=(0, 10))
	
		def browse_video():
			"""Open file dialog to select video"""
			video_file = filedialog.askopenfilename(
				title="Seleccionar video",
				filetypes=[
					("Video files", "*.mp4 *.avi *.mov *.mkv"),
					("All files", "*.*")
				]
			)
			if video_file:
				video_path_var.set(video_file)
				check_enable_test_button()
	
		# Browse button (canvas-based rounded button)
		browse_btn_canvas = tk.Canvas(
			left_frame,
			height=40,
			bg=BG_COLOR,
			highlightthickness=0
		)
		browse_btn_canvas.pack(fill="x", pady=(0, 30))
		
		def draw_browse_btn(hover=False):
			browse_btn_canvas.delete("all")
			color = "#0168d6" if hover else "#015bcb"
			w = browse_btn_canvas.winfo_width()
			if w <= 1:
				w = 300  # Fallback width
			h = 40
			r = 10
			
			# Draw rounded rectangle
			browse_btn_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
			browse_btn_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
			browse_btn_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
			browse_btn_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
			browse_btn_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
			browse_btn_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
			browse_btn_canvas.create_text(w/2, h/2, text="Buscar", fill="white", font=("Arial", 11, "bold"))
		
		browse_btn_canvas.after(10, draw_browse_btn)
		browse_btn_canvas.bind("<Configure>", lambda e: draw_browse_btn())
		browse_btn_canvas.bind("<Enter>", lambda e: draw_browse_btn(True))
		browse_btn_canvas.bind("<Leave>", lambda e: draw_browse_btn(False))
		browse_btn_canvas.bind("<Button-1>", lambda e: browse_video())
		browse_btn_canvas.config(cursor="hand2")
	
		# Test button (canvas-based rounded button)
		test_btn_canvas = tk.Canvas(
			left_frame,
			height=50,
			bg=BG_COLOR,
			highlightthickness=0
		)
		test_btn_canvas.pack(fill="x", pady=(0, 20))
		
		test_btn_state = {"enabled": False}
		
		def draw_test_btn(hover=False):
			test_btn_canvas.delete("all")
			if test_btn_state["enabled"]:
				color = "#0168d6" if hover else "#015bcb"
			else:
				color = "#555555"  # Disabled color
			
			w = test_btn_canvas.winfo_width()
			if w <= 1:
				w = 300  # Fallback width
			h = 50
			r = 10
			
			# Draw rounded rectangle
			test_btn_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
			test_btn_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
			test_btn_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
			test_btn_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
			test_btn_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
			test_btn_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
			test_btn_canvas.create_text(w/2, h/2, text="Iniciar pruebas", fill="white", font=("Arial", 12, "bold"))
		
		test_btn_canvas.after(10, draw_test_btn)
		test_btn_canvas.bind("<Configure>", lambda e: draw_test_btn())
		
		def on_test_btn_hover(e):
			if test_btn_state["enabled"]:
				draw_test_btn(True)
		
		def on_test_btn_leave(e):
			if test_btn_state["enabled"]:
				draw_test_btn(False)
		
		def on_test_btn_click(e):
			if test_btn_state["enabled"]:
				pass  # Will be set below with actual command
		
		test_btn_canvas.bind("<Enter>", on_test_btn_hover)
		test_btn_canvas.bind("<Leave>", on_test_btn_leave)
		test_btn_canvas.bind("<Button-1>", on_test_btn_click)

		# Preview Frame button (small)
		preview_btn_canvas = tk.Canvas(
			left_frame,
			height=30,
			bg=BG_COLOR,
			highlightthickness=0
		)
		preview_btn_canvas.pack(fill="x", pady=(0, 10))

		preview_state = {"active": False, "last_grid_tk": None}

		def draw_preview_btn(hover=False):
			preview_btn_canvas.delete("all")
			if preview_state["active"]:
				color = "#d63031" if not hover else "#e74c3c"
				label = "Quit Preview"
			else:
				color = "#444444" if not hover else "#555555"
				label = "Preview Frame"
			w = preview_btn_canvas.winfo_width()
			if w <= 1:
				w = 300
			h = 30
			r = 8
			preview_btn_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
			preview_btn_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
			preview_btn_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
			preview_btn_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
			preview_btn_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
			preview_btn_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
			preview_btn_canvas.create_text(w/2, h/2, text=label, fill="white", font=("Arial", 9))

		preview_btn_canvas.after(10, draw_preview_btn)
		preview_btn_canvas.bind("<Configure>", lambda e: draw_preview_btn())
		preview_btn_canvas.bind("<Enter>", lambda e: draw_preview_btn(True))
		preview_btn_canvas.bind("<Leave>", lambda e: draw_preview_btn(False))
		preview_btn_canvas.config(cursor="hand2")
	
		# Status label
		status_label = tk.Label(
			left_frame,
			text="",
			font=("Arial", 10),
			bg=BG_COLOR,
			fg=FG_COLOR,
			wraplength=250,
			justify="left"
		)
		status_label.pack(anchor="w", pady=(10, 0))

		# Note label
		note_label = tk.Label(
			left_frame,
			text="Notas:\n"
			"- Considerar que al momento de la inferencia los resultados dependen de la prediccion mas comun en un conjunto de frames.\n"
			"- Considerar que al momento de la inferencia se agrega +-10% a la clase VA dependiendo de el ROI actual.",
			font=("Arial", 10),
			bg=BG_COLOR,
			fg="#fffd7a",
			wraplength=250,
			justify="left"
		)
		note_label.pack(anchor="w", pady=(10, 0))
	
		def check_enable_test_button():
			"""Enable test button only when at least one model and a video are selected"""
			if len(models_list_state) > 0 and video_path_var.get():
				test_btn_state["enabled"] = True
				test_btn_canvas.config(cursor="hand2")
			else:
				test_btn_state["enabled"] = False
				test_btn_canvas.config(cursor="")
			draw_test_btn()
	
		# Bind model selection to enable check
		model_combo.bind("<<ComboboxSelected>>", lambda e: check_enable_test_button())
	
		# ==== RIGHT DISPLAY COMPONENTS ====

		right_frame.grid_rowconfigure(0, weight=1)
		right_frame.grid_rowconfigure(1, weight=0)
		right_frame.grid_columnconfigure(0, weight=1)

		# Frame display canvas
		display_canvas = tk.Canvas(
			right_frame,
			bg="#000000",
			highlightthickness=0
		)
		display_canvas.grid(row=0, column=0, sticky="nsew")

		# Placeholder text
		placeholder_text = display_canvas.create_text(
			400, 300,
			text="Seleccione modelos y video\npara comenzar la prueba",
			font=("Arial", 16),
			fill="#666666",
			justify="center"
		)

		# Navigation bar
		nav_bar = tk.Frame(right_frame, bg=BG_COLOR, height=40)
		nav_bar.grid(row=1, column=0, sticky="ew", pady=(5, 0))

		nav_left_btn = tk.Label(nav_bar, text="◀", font=("Arial", 18, "bold"), fg="#555555", bg=BG_COLOR, padx=20, cursor="arrow")
		nav_left_btn.pack(side=tk.LEFT, padx=(10, 5))

		nav_frame_container = tk.Frame(nav_bar, bg=BG_COLOR)
		nav_frame_container.pack(side=tk.LEFT, expand=True)

		nav_frame_lbl_pre = tk.Label(nav_frame_container, text="Frame ", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
		nav_frame_lbl_pre.pack(side=tk.LEFT)

		nav_frame_entry_var = tk.StringVar(value="")
		nav_frame_entry = tk.Entry(
			nav_frame_container,
			textvariable=nav_frame_entry_var,
			font=("Arial", 11),
			width=6,
			bg="#02234f", fg="white",
			insertbackground="white",
			justify="center",
			relief=tk.FLAT,
			bd=1,
			state="disabled",
			disabledbackground="#02234f",
			disabledforeground="#555555"
		)
		nav_frame_entry.pack(side=tk.LEFT)

		nav_frame_lbl_total = tk.Label(nav_frame_container, text="", font=("Arial", 11), fg=FG_COLOR, bg=BG_COLOR)
		nav_frame_lbl_total.pack(side=tk.LEFT)

		def _on_frame_entry_enter(e=None):
			if not test_state["ready"] or processing_active["running"]:
				return
			try:
				val = int(nav_frame_entry_var.get())
			except (ValueError, TypeError):
				nav_frame_entry_var.set(str(test_state["current_frame"] + 1))
				return
			target = max(0, min(test_state["total_frames"] - 1, val - 1))
			if target != test_state["current_frame"]:
				_process_and_display_frame(target)
			else:
				nav_frame_entry_var.set(str(target + 1))

		nav_frame_entry.bind("<Return>", _on_frame_entry_enter)

		nav_right_btn = tk.Label(nav_bar, text="▶", font=("Arial", 18, "bold"), fg="#555555", bg=BG_COLOR, padx=20, cursor="arrow")
		nav_right_btn.pack(side=tk.RIGHT, padx=(5, 10))

		# Step size row
		step_bar = tk.Frame(right_frame, bg=BG_COLOR)
		step_bar.grid(row=2, column=0, sticky="ew", pady=(3, 0))

		step_lbl = tk.Label(step_bar, text="Salto de frames:", font=("Arial", 10), fg=FG_COLOR, bg=BG_COLOR)
		step_lbl.pack(side=tk.LEFT, padx=(10, 5))

		step_var = tk.IntVar(value=1)
		step_spinbox = tk.Spinbox(
			step_bar,
			from_=1, to=1000,
			textvariable=step_var,
			width=6,
			font=("Arial", 11),
			bg="#02234f", fg="white",
			buttonbackground="#015bcb",
			insertbackground="white",
			justify="center"
		)
		step_spinbox.pack(side=tk.LEFT, padx=(0, 10))

		# Persistent test state (models loaded once, reused across frames)
		test_state = {
			"person_det": None,
			"classifiers": [],  # [{"name": str, "model": YOLO}, ...]
			"video_path": None,
			"total_frames": 0,
			"current_frame": 0,
			"ready": False,
		}

		# Processing state
		processing_active = {"running": False, "stop": False}

		def _update_nav_ui():
			"""Update navigation buttons and frame entry based on test_state"""
			if not test_state["ready"]:
				nav_frame_entry.config(state="disabled")
				nav_frame_entry_var.set("")
				nav_frame_lbl_total.config(text="")
				nav_left_btn.config(fg="#555555", cursor="arrow")
				nav_right_btn.config(fg="#555555", cursor="arrow")
				return
			cur = test_state["current_frame"]
			total = test_state["total_frames"]
			nav_frame_entry.config(state="normal")
			nav_frame_entry_var.set(str(cur + 1))
			nav_frame_lbl_total.config(text=f" / {total}")
			if cur > 0:
				nav_left_btn.config(fg="white", cursor="hand2")
			else:
				nav_left_btn.config(fg="#555555", cursor="arrow")
			if cur < total - 1:
				nav_right_btn.config(fg="white", cursor="hand2")
			else:
				nav_right_btn.config(fg="#555555", cursor="arrow")

		def _get_class_color(class_name):
			if not class_name:
				return (128, 128, 128)
			cn = class_name.lower()
			if cn in ["working", "trabajando", "trabajo"]:
				return (0, 255, 0)
			elif cn in ["idle", "parado", "ocioso"]:
				return (0, 0, 255)
			return (128, 128, 128)

		def _preview_current_frame():
			"""Show the current frame without annotations; toggle preview state."""
			if preview_state["active"]:
				_quit_preview()
				return
			import os
			vpath = video_path_var.get()
			if not vpath or not os.path.exists(vpath):
				status_label.config(text="Seleccione un video primero")
				return
			if processing_active["running"]:
				return

			# Save current grid image before overwriting
			preview_state["last_grid_tk"] = getattr(display_canvas, "image", None)

			processing_active["running"] = True

			def _pw():
				try:
					target = test_state["current_frame"] if test_state["ready"] else 0
					cap = cv2.VideoCapture(vpath)
					total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
					if total <= 0:
						cap.release()
						status_label.config(text="Error: video sin frames")
						processing_active["running"] = False
						return
					cap.set(cv2.CAP_PROP_POS_FRAMES, target)
					ret, frame = cap.read()
					cap.release()
					if not ret or frame is None:
						status_label.config(text=f"No se pudo leer frame {target + 1}")
						processing_active["running"] = False
						return

					frame_rgb = cv2.cvtColor(cv2.resize(frame, (1280, 720)), cv2.COLOR_BGR2RGB)

					cw = display_canvas.winfo_width()
					ch = display_canvas.winfo_height()
					if cw <= 1 or ch <= 1:
						cw, ch = 900, 600
					h_f, w_f = frame_rgb.shape[:2]
					scale = min(cw / w_f, ch / h_f)
					resized = cv2.resize(frame_rgb, (int(w_f * scale), int(h_f * scale)))

					img_pil = Image.fromarray(resized)
					img_tk = ImageTk.PhotoImage(img_pil)
					display_canvas.delete("all")
					display_canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor="center")
					display_canvas.image = img_tk
					status_label.config(text=f"Preview: frame {target + 1}/{total}")

					preview_state["active"] = True
					draw_preview_btn()
				except Exception as e:
					status_label.config(text=f"Error: {e}")
				finally:
					processing_active["running"] = False

			thread = threading.Thread(target=_pw, daemon=True)
			thread.start()

		def _quit_preview():
			"""Restore the grid view and reset preview button."""
			preview_state["active"] = False
			draw_preview_btn()
			grid_tk = preview_state.get("last_grid_tk")
			if grid_tk is not None:
				cw = display_canvas.winfo_width()
				ch = display_canvas.winfo_height()
				if cw <= 1 or ch <= 1:
					cw, ch = 900, 600
				display_canvas.delete("all")
				display_canvas.create_image(cw // 2, ch // 2, image=grid_tk, anchor="center")
				display_canvas.image = grid_tk
				status_label.config(text=f"Frame {test_state['current_frame'] + 1}/{test_state['total_frames']} — {len(test_state['classifiers'])} modelo(s)")
			else:
				display_canvas.delete("all")
				status_label.config(text="")

		preview_btn_canvas.bind("<Button-1>", lambda e: _preview_current_frame())

		def _process_and_display_frame(frame_idx, max_skip=10):
			"""Read frame at frame_idx, process with all loaded models, display grid.
			If a frame cannot be read, auto-advance forward up to max_skip times."""
			if not test_state["ready"] or processing_active["running"]:
				return

			processing_active["running"] = True
			video_path = test_state["video_path"]
			total = test_state["total_frames"]

			def _worker():
				try:
					target = frame_idx
					skipped = 0
					frame = None

					# Try to read the target frame, auto-advance on failure
					while skipped <= max_skip and 0 <= target < total:
						cap = cv2.VideoCapture(video_path)
						cap.set(cv2.CAP_PROP_POS_FRAMES, target)
						ret, f = cap.read()
						cap.release()
						if ret and f is not None:
							frame = f
							break
						target += 1
						skipped += 1
						status_label.config(text=f"Frame {target} ilegible, avanzando...")

					if frame is None:
						status_label.config(text="No se pudo leer ningún frame cercano")
						processing_active["running"] = False
						return

					test_state["current_frame"] = target
					_update_nav_ui()

					frame_resized = cv2.resize(frame, (1280, 720))
					status_label.config(text=f"Procesando frame {target + 1}/{total}...")

					person_det = test_state["person_det"]
					person_results = person_det.predict(
						frame_resized,
						conf=0.60,
						classes=[0],
						verbose=False
					)

					person_crops = []
					person_coords = []
					if person_results and len(person_results) > 0:
						boxes = person_results[0].boxes
						if boxes is not None and len(boxes) > 0:
							for box in boxes:
								x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
								x1, y1 = max(0, x1), max(0, y1)
								x2, y2 = min(frame_resized.shape[1], x2), min(frame_resized.shape[0], y2)
								if x2 > x1 and y2 > y1:
									crop = frame_resized[y1:y2, x1:x2]
									person_crops.append(crop)
									person_coords.append((x1, y1, x2, y2))

					cell_images = []
					for c_idx, cls_info in enumerate(test_state["classifiers"]):
						status_label.config(text=f"Clasificando con {cls_info['name']} ({c_idx+1}/{len(test_state['classifiers'])})...")
						display_frame = frame_resized.copy()
						classifier = cls_info["model"]

						if person_crops:
							classifications = []
							for crop in person_crops:
								cls_result = classifier(crop, verbose=False)
								if cls_result and len(cls_result) > 0:
									probs = cls_result[0].probs
									if probs is not None:
										top_idx = int(probs.top1)
										class_name = classifier.names.get(top_idx, "unknown")
										conf = float(probs.top1conf)
										classifications.append((class_name, conf))
									else:
										classifications.append(("unknown", 0.0))
								else:
									classifications.append(("unknown", 0.0))

							for (x1, y1, x2, y2), (class_name, conf) in zip(person_coords, classifications):
								color = _get_class_color(class_name)
								cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
								label = f"{class_name}: {conf:.2f}"
								(lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
								cv2.rectangle(display_frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
								cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

						cv2.putText(display_frame, cls_info["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
						cell_images.append(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

					# Build grid
					canvas_width = display_canvas.winfo_width()
					canvas_height = display_canvas.winfo_height()
					if canvas_width <= 1 or canvas_height <= 1:
						canvas_width, canvas_height = 900, 600

					n = len(cell_images)
					if n <= 1:
						cols, rows_g = 1, 1
					elif n <= 2:
						cols, rows_g = 2, 1
					elif n <= 4:
						cols, rows_g = 2, 2
					elif n <= 6:
						cols, rows_g = 3, 2
					else:
						cols, rows_g = 3, 3

					cell_w = canvas_width // cols
					cell_h = canvas_height // rows_g

					grid_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

					for i in range(min(n, 9)):
						r = i // cols
						c = i % cols
						x_off = c * cell_w
						y_off = r * cell_h
						cell = cell_images[i]
						h_c, w_c = cell.shape[:2]
						scale = min(cell_w / w_c, cell_h / h_c)
						new_w = int(w_c * scale)
						new_h = int(h_c * scale)
						resized_cell = cv2.resize(cell, (new_w, new_h))
						pad_x = (cell_w - new_w) // 2
						pad_y = (cell_h - new_h) // 2
						grid_img[y_off + pad_y:y_off + pad_y + new_h, x_off + pad_x:x_off + pad_x + new_w] = resized_cell

					# Grid lines
					for i in range(1, cols):
						cv2.line(grid_img, (i * cell_w, 0), (i * cell_w, canvas_height), (80, 80, 80), 1)
					for i in range(1, rows_g):
						cv2.line(grid_img, (0, i * cell_h), (canvas_width, i * cell_h), (80, 80, 80), 1)

					img_pil = Image.fromarray(grid_img)
					img_tk = ImageTk.PhotoImage(img_pil)

					display_canvas.delete("all")
					display_canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk, anchor="center")
					display_canvas.image = img_tk
					display_canvas.update_idletasks()

					status_label.config(text=f"Frame {target + 1}/{total} — {n} modelo(s)")
				except Exception as e:
					import traceback
					traceback.print_exc()
					status_label.config(text=f"Error: {str(e)}")
				finally:
					processing_active["running"] = False

			thread = threading.Thread(target=_worker, daemon=True)
			thread.start()

		def _nav_left(e=None):
			if not test_state["ready"] or processing_active["running"]:
				return
			step = max(1, min(1000, step_var.get()))
			new_frame = max(0, test_state["current_frame"] - step)
			if new_frame != test_state["current_frame"]:
				_process_and_display_frame(new_frame)

		def _nav_right(e=None):
			if not test_state["ready"] or processing_active["running"]:
				return
			step = max(1, min(1000, step_var.get()))
			new_frame = min(test_state["total_frames"] - 1, test_state["current_frame"] + step)
			if new_frame != test_state["current_frame"]:
				_process_and_display_frame(new_frame)

		nav_left_btn.bind("<Button-1>", _nav_left)
		nav_right_btn.bind("<Button-1>", _nav_right)

		def process_video():
			"""Load models, open video, process first frame"""
			import os

			if processing_active["running"]:
				return

			video_path = video_path_var.get()

			if not models_list_state or not video_path:
				return

			if not os.path.exists(video_path):
				status_label.config(text="Error: Video no encontrado")
				return

			model_entries = []
			for entry in models_list_state:
				full_path = model_paths_dict.get(entry["path_key"])
				if not full_path or not os.path.exists(full_path):
					status_label.config(text=f"Error: Modelo no encontrado: {entry['path_key']}")
					return
				model_entries.append({"name": entry["name"], "path": full_path})

			display_canvas.delete("all")

			processing_active["running"] = True
			processing_active["stop"] = False

			# Change button to "Detener"
			def draw_stop_btn(hover=False):
				test_btn_canvas.delete("all")
				color = "#d63031" if hover else "#e74c3c"
				w = test_btn_canvas.winfo_width()
				if w <= 1:
					w = 300
				h = 50
				r = 10
				test_btn_canvas.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline=color)
				test_btn_canvas.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline=color)
				test_btn_canvas.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline=color)
				test_btn_canvas.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline=color)
				test_btn_canvas.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
				test_btn_canvas.create_rectangle(0, r, w, h-r, fill=color, outline=color)
				test_btn_canvas.create_text(w/2, h/2, text="Detener", fill="white", font=("Arial", 12, "bold"))

			def on_stop_hover(e):
				draw_stop_btn(True)
			def on_stop_leave(e):
				draw_stop_btn(False)
			def on_stop_click(e):
				stop_processing()

			test_btn_canvas.unbind("<Enter>")
			test_btn_canvas.unbind("<Leave>")
			test_btn_canvas.unbind("<Button-1>")
			test_btn_canvas.bind("<Enter>", on_stop_hover)
			test_btn_canvas.bind("<Leave>", on_stop_leave)
			test_btn_canvas.bind("<Button-1>", on_stop_click)
			draw_stop_btn()

			status_label.config(text="Cargando modelos...")

			def loading_thread():
				try:
					status_label.config(text="Cargando detector de personas...")
					train_win.update()

					base_root = _base_root()
					person_det_path = base_root / "yolo11x.pt"
					if not person_det_path.exists():
						person_det_path = "yolo11x.pt"

					test_state["person_det"] = YOLO(str(person_det_path))

					classifiers = []
					for i, m_entry in enumerate(model_entries):
						if processing_active["stop"]:
							break
						status_label.config(text=f"Cargando clasificador {i+1}/{len(model_entries)}: {m_entry['name']}...")
						train_win.update()
						classifiers.append({
							"name": m_entry["name"],
							"model": YOLO(m_entry["path"]),
						})
					test_state["classifiers"] = classifiers

					if processing_active["stop"]:
						status_label.config(text="Carga detenida")
						reset_processing_state()
						return

					# Open video to get total frames
					cap = cv2.VideoCapture(video_path)
					if not cap.isOpened():
						status_label.config(text="Error: No se pudo abrir el video")
						reset_processing_state()
						return
					test_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
					cap.release()

					test_state["video_path"] = video_path
					test_state["current_frame"] = 0
					test_state["ready"] = True

					# Restore button to normal
					reset_processing_state()

					# Process frame 0
					_process_and_display_frame(0)

				except Exception as e:
					import traceback
					traceback.print_exc()
					status_label.config(text=f"Error: {str(e)}")
					reset_processing_state()

			thread = threading.Thread(target=loading_thread, daemon=True)
			thread.start()

		def stop_processing():
			processing_active["stop"] = True
			status_label.config(text="Deteniendo...")

		def reset_processing_state():
			processing_active["running"] = False
			processing_active["stop"] = False

			test_btn_canvas.unbind("<Enter>")
			test_btn_canvas.unbind("<Leave>")
			test_btn_canvas.unbind("<Button-1>")
			test_btn_canvas.bind("<Enter>", on_test_btn_hover)
			test_btn_canvas.bind("<Leave>", on_test_btn_leave)
			test_btn_canvas.bind("<Button-1>", lambda e: process_video() if test_btn_state["enabled"] else None)

			check_enable_test_button()
			_update_nav_ui()

		# Set test button click handler
		test_btn_canvas.unbind("<Button-1>")
		test_btn_canvas.bind("<Button-1>", lambda e: process_video() if test_btn_state["enabled"] else None)

	def open_procesamiento_window(preset_type=None, pre_scanned_ips=None, auto_scan_on_open=False):
		if pre_scanned_ips is None:
			pre_scanned_ips = []
		# New color scheme for processing window
		PROC_BG = "#01326a"  # Main background
		PROC_CONTENT_BG = "#02234e"  # Content area background
		PROC_DRAG_BG = "#002858"  # Drag & drop area background
		PROC_BTN_NORMAL = "#015aca"  # Normal button color
		PROC_BTN_CONFIRM = "#ffc735"  # Confirm/Continue button color
		PROC_TAB_ACTIVE = "#5BA8C9"  # Active tab text color (blue)
		PROC_TAB_PREVIOUS = "#ffc735"  # Previous tab text color (yellow)
		PROC_TAB_FUTURE = "#E6EEF9"  # Future tab text color (white)
		
		win = tk.Toplevel(root)
		win.title("Procesamiento")
		# Hide main window while processing window is open
		root.withdraw()
		def _on_close_processing():
			# Ensure RT processing is interrupted before closing the window
			try:
				_stop_rt_processing()
			except Exception:
				pass
			# Restore main window then destroy processing window
			try:
				root.deiconify()
				root.state("zoomed")
			except Exception:
				pass
			win.destroy()
		win.protocol("WM_DELETE_WINDOW", _on_close_processing)
		# Start maximized
		try:
			win.state("zoomed")
		except Exception:
			try:
				win.attributes("-zoomed", True)
			except Exception:
				pass
		# Keep a fallback geometry if zoom not supported
		_center_window(win, 720, 520)
		win.configure(bg=PROC_BG)
		
		wrapper = tk.Frame(win, bg=PROC_BG)
		wrapper.pack(fill=tk.BOTH, expand=True)
		
		# ========== TOP HEADER: Logo + Breadcrumb Tabs ==========
		header_frame = tk.Frame(wrapper, bg=PROC_BG, height=80)
		header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
		header_frame.pack_propagate(False)
		
		# Mini logo on the left
		mini_logo_path = assets_root / "NGUI" / "ArnesisMiniLogo.png"
		menorque_path = assets_root / "NGUI" / "menorque.png"
		mini_logo_img = load_icon(mini_logo_path, 200, 100, invert=False)
		if mini_logo_img:
			logo_label = tk.Label(header_frame, image=mini_logo_img, bg=PROC_BG)
			logo_label.image = mini_logo_img
			logo_label.pack(side=tk.LEFT, padx=(0, 20))
		
		# Breadcrumb tabs container
		breadcrumb_frame = tk.Frame(header_frame, bg=PROC_BG)
		breadcrumb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		# Tab titles change based on processing type
		if preset_type == "rt":
			tab_titles = [
				"Configurar\nCámaras",
				"Delimitar\nZonas",
				"Cargar\nModelo",
				"Configuración",
				"Procesamiento",
				"Fin",
			]
		else:
			tab_titles = [
				"Cargar\nVideos",
				"Delimitar\nZonas",
				"Cargar\nModelo",
				"Configuración",
				"Procesamiento",
				"Fin",
			]
		
		# Create tab labels with separators
		breadcrumb_labels = []
		separator_imgs = []
		menorque_img = load_icon(menorque_path, 12, 12, invert=False) if menorque_path.exists() else None
		
		for i, title in enumerate(tab_titles):
			if i > 0:
				# Add separator
				if menorque_img:
					sep_label = tk.Label(breadcrumb_frame, image=menorque_img, bg=PROC_BG)
					sep_label.image = menorque_img
					separator_imgs.append(sep_label)
				else:
					sep_label = tk.Label(breadcrumb_frame, text=">", font=("Arial", 14), 
					                     fg=FG_COLOR, bg=PROC_BG)
					separator_imgs.append(sep_label)
				sep_label.pack(side=tk.LEFT, padx=15)
			
			# Tab label
			tab_label = tk.Label(breadcrumb_frame, text=title, font=("Arial", 12, "bold"), 
			                     fg=PROC_TAB_FUTURE, bg=PROC_BG, justify=tk.CENTER)
			tab_label.pack(side=tk.LEFT, padx=15)
			breadcrumb_labels.append(tab_label)
		
		# ========== CONTENT AREA ==========
		content_frame = tk.Frame(wrapper, bg=PROC_CONTENT_BG)
		content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
		
		# Create tabs (hidden notebook for content management)
		tabs = []
		for _ in tab_titles:
			frm = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			tabs.append(frm)
		
		current_step = 0
		can_advance = [False] * len(tab_titles)
		sync_rt_controls_state = {"func": None}
		
		# Set processing type based on preset or None
		processing_type = {"value": preset_type}
		
		video_list: list[str] = []
		selected_folder: list[str] = [""]
		NP_ACTUAL = {"value": None}
		MODEL_FILE = {"value": None}
		VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
		def make_half(parent: tk.Widget, bg: str, hover_bg: str, text: str, command):
			frame = tk.Frame(parent, bg=bg, highlightthickness=0, cursor="hand2")
			inner = tk.Frame(frame, bg=bg)
			inner.place(relx=0.5, rely=0.5, anchor="center")
			label = tk.Label(inner, text=text, font=("Arial", 16, "bold"), fg=FG_COLOR, bg=bg)
			label.pack(padx=8, pady=8)
			def on_enter(_):
				frame.configure(bg=hover_bg)
				inner.configure(bg=hover_bg)
				label.configure(bg=hover_bg)
			def on_leave(_):
				frame.configure(bg=bg)
				inner.configure(bg=bg)
				label.configure(bg=bg)
			frame.bind("<Enter>", on_enter)
			frame.bind("<Leave>", on_leave)
			frame.bind("<Button-1>", lambda e: command())
			label.bind("<Button-1>", lambda e: command())
			return frame
		
		# Function to create rounded button
		def make_rounded_button(parent, text, command, bg_color, width=120, height=40, fg=None):
			"""Create a button with rounded corners"""
			container = tk.Frame(parent, bg=PROC_CONTENT_BG)
			canvas = tk.Canvas(container, width=width, height=height, bg=PROC_CONTENT_BG, 
			                   highlightthickness=0, cursor="hand2")
			canvas.pack()
			
			# Store command reference so it can be updated
			command_ref = {"func": command}
			
			def draw_button(color, state="normal"):
				canvas.delete("all")
				# Adjust color for disabled state
				if state == "disabled":
					color = "#666666"
				# Draw rounded rectangle
				radius = 8
				canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(width-radius*2, 0, width, radius*2, start=0, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(0, height-radius*2, radius*2, height, start=180, extent=90, 
				                  fill=color, outline=color)
				canvas.create_arc(width-radius*2, height-radius*2, width, height, start=270, extent=90, 
				                  fill=color, outline=color)
				canvas.create_rectangle(radius, 0, width-radius, height, fill=color, outline=color)
				canvas.create_rectangle(0, radius, width, height-radius, fill=color, outline=color)
				# Draw text
				if fg:
					text_color = fg
				else:
					text_color = "#000000" if bg_color == PROC_BTN_CONFIRM else "#FFFFFF"
				if state == "disabled":
					text_color = "#999999"
				canvas.create_text(width/2, height/2, text=text, font=("Arial", 11, "bold"), 
				                   fill=text_color)
				
				# Redraw icon if it exists (for buttons with icons like undo button)
				if hasattr(canvas, '_redraw_icon'):
					canvas._redraw_icon()
			
			# Store state
			button_state = {"state": "normal"}
			
			def redraw():
				draw_button(bg_color, button_state["state"])
			
			redraw()
			
			def on_enter(e):
				if button_state["state"] == "normal":
					draw_button(bg_color if bg_color == PROC_BTN_CONFIRM else "#0174E8", "normal")
			
			def on_leave(e):
				redraw()
			
			def on_click(e):
				if button_state["state"] == "normal" and command_ref["func"]:
					command_ref["func"]()
			
			canvas.bind("<Enter>", on_enter)
			canvas.bind("<Leave>", on_leave)
			canvas.bind("<Button-1>", on_click)
			
			# Add methods to container to mimic button behavior
			def configure(command=None, state=None):
				if command is not None:
					command_ref["func"] = command
				if state is not None:
					button_state["state"] = state
					redraw()
			
			def config(**kwargs):
				configure(**kwargs)
			
			container.configure = configure
			container.config = config
			
			return container
		
		# Skip the first tab setup since we removed "Tipo de Procesamiento"
		# Now tabs[0] is "Cargar Videos" which has custom UI
		
		# for i in range(1, len(tabs)):
		# 	# Skip custom tabs (1st: upload videos, 2nd: ROIs, 3rd: Cargar modelo, 4th: Configuracion)
		# 	if i in (0, 1, 2, 3):
		# 		continue
		# 	lbl = tk.Label(tabs[i], text=f"Paso {i+1}: {tab_titles[i]}", font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		# 	lbl.pack(pady=24)
		
		def update_breadcrumb():
			"""Update breadcrumb colors based on current step"""
			for i, label in enumerate(breadcrumb_labels):
				if i < current_step:
					label.config(fg=PROC_TAB_PREVIOUS)  # Yellow for previous
				elif i == current_step:
					label.config(fg=PROC_TAB_ACTIVE)  # Blue for current
				else:
					label.config(fg=PROC_TAB_FUTURE)  # White for future
		
		def show_current_tab():
			"""Show only the current tab content"""
			for i, tab in enumerate(tabs):
				if i == current_step:
					tab.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
					if i == 4 and processing_type["value"] == "rt" and sync_rt_controls_state["func"]:
						sync_rt_controls_state["func"]()
					# Update model checkbox when showing Configuration tab
					if i == 3:  # Configuration tab
						model_path = MODEL_FILE.get("value", "")
						if model_path:
							model_name = Path(model_path).name
							model_loaded_chk.config(text=f"Modelo cargado: {model_name}")
						else:
							model_loaded_chk.config(text="Modelo cargado: Sin modelo")
				else:
					tab.pack_forget()
		
		def update_tabs_state():
			update_breadcrumb()
			show_current_tab()
			update_nav_state()
			
		def update_nav_state():
			prev_btn_container.pack_forget() if current_step == 0 else prev_btn_container.pack(side=tk.LEFT, padx=10)
			# Update next button text
			if current_step == len(tabs) - 1:
				# Last tab - no next button or change to "Finish"
				next_btn_container.pack_forget()
			else:
				next_btn_container.pack(side=tk.RIGHT, padx=10)
				
		def go_prev():
			nonlocal current_step
			if current_step > 0:
				current_step -= 1
				update_tabs_state()
				
		def go_next():
			nonlocal current_step
			if current_step == 0:
				val = np_entry.get().strip()
				NP_ACTUAL["value"] = val if val else None
			if current_step < len(tabs) - 1:
				current_step += 1
				# Cambiar UI del tab 0 según tipo de procesamiento
				if current_step == 0:
					if processing_type["value"] == "videos":
						show_video_mode()
					elif processing_type["value"] == "rt":
						show_rt_mode()
				update_tabs_state()
				
		# ========== NAVIGATION BUTTONS ==========
		nav_frame = tk.Frame(wrapper, bg=PROC_BG, height=60)
		nav_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
		nav_frame.pack_propagate(False)
		
		prev_btn_container = make_rounded_button(nav_frame, "Regresar", go_prev, PROC_BTN_NORMAL, width=140)
		prev_btn_container.pack(side=tk.LEFT, padx=10)
		
		next_btn_container = make_rounded_button(nav_frame, "Confirmar", go_next, PROC_BTN_CONFIRM, width=140)
		next_btn_container.pack(side=tk.RIGHT, padx=10)
		
		# tabs[0] is now "Subir videos/fuente"
		second = tabs[0]
		second.configure(bg=PROC_CONTENT_BG)
		second.grid_rowconfigure(0, weight=1)
		second.grid_columnconfigure(0, weight=1)
		
		# Contenedor para modo videos
		video_mode_frame = tk.Frame(second, bg=PROC_CONTENT_BG)
		# Contenedor para modo RT (RTSP)
		rt_mode_frame = tk.Frame(second, bg=PROC_CONTENT_BG)
		
		# ========== VIDEO MODE LAYOUT (NO SCROLLBAR) ==========
		video_content = tk.Frame(video_mode_frame, bg=PROC_CONTENT_BG)
		video_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
		
		# Configure rows for layout
		video_content.rowconfigure(0, weight=0)  # Title
		video_content.rowconfigure(1, weight=0)  # Path input row
		video_content.rowconfigure(2, weight=0)  # NP label
		video_content.rowconfigure(3, weight=0)  # NP entry
		video_content.rowconfigure(4, weight=1)  # Drop area (expandable)
		video_content.columnconfigure(0, weight=1)
		
		# Row 0: Title label
		title_container = tk.Frame(video_content, bg=PROC_CONTENT_BG)
		title_container.grid(row=0, column=0, sticky="w", pady=(0, 15))
		
		tk.Label(title_container, text="Cargar Videos desde Carpeta ", 
		         font=("Arial", 13, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG).pack(side=tk.LEFT)
		tk.Label(title_container, text="(.MP4)", 
		         font=("Arial", 13, "bold"), fg="#006cba", bg=PROC_CONTENT_BG).pack(side=tk.LEFT)
		
		# Row 1: Path input + Browse button
		path_container = tk.Frame(video_content, bg=PROC_CONTENT_BG)
		path_container.grid(row=1, column=0, sticky="ew", pady=(0, 20))
		path_container.columnconfigure(0, weight=1)
		
		# Rounded textbox for folder path
		folder_entry_frame = tk.Frame(path_container, bg="#FFFFFF", highlightthickness=1, 
		                              highlightbackground="#D0D0D0", highlightcolor="#D0D0D0")
		folder_entry_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10))
		
		folder_entry = tk.Entry(folder_entry_frame, font=("Arial", 10), bg="#FFFFFF", fg="#000000", 
		                        relief=tk.FLAT, bd=0)
		folder_entry.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
		
		def validate_video_datetime(video_path):
			"""Check if video filename contains valid datetime. Returns (is_valid, datetime_obj)"""
			basename = os.path.basename(video_path)
			parts = basename.split('_')
			try:
				try:
					start_dt = datetime.strptime(parts[-4], "%Y%m%d%H%M%S")
					print(f"[DEBUG] Valid datetime found in parts[-4] for {basename}")
				except Exception:
					try:
						start_dt = datetime.strptime(basename, "%Y%m%d%H%M%S")
						print(f"[DEBUG] Valid datetime found in basename for {basename}")
					except Exception:
						start_dt = datetime.strptime(parts[-3], "%Y%m%d%H%M%S")
						print(f"[DEBUG] Valid datetime found in parts[-3] for {basename}")
				return (True, start_dt)
			except Exception as e:
				print(f"[DEBUG] No valid datetime found for {basename}: {e}")
				return (False, None)
		
		def show_datetime_input_dialog(video_path):
			"""Show dialog to manually input datetime for video with frame navigation"""
			print(f"[DEBUG] show_datetime_input_dialog called for: {os.path.basename(video_path)}")
			
			# Create modal dialog
			try:
				dialog = tk.Toplevel(win)
				dialog.title("Agregar Fecha y Hora Manualmente")
				dialog.geometry("850x800")
				dialog.configure(bg=PROC_CONTENT_BG)
				dialog.transient(win)
				
				# Center dialog on screen
				dialog.update_idletasks()
				x = (dialog.winfo_screenwidth() // 2) - (850 // 2)
				y = (dialog.winfo_screenheight() // 2) - (800 // 2)
				dialog.geometry(f"850x800+{x}+{y}")
				
				dialog.grab_set()
				dialog.focus_force()
				
				print(f"[DEBUG] Dialog created successfully")
			except Exception as e:
				print(f"[ERROR] Failed to create dialog: {e}")
				import traceback
				traceback.print_exc()
				return None
			
			result = {"datetime": None, "cancelled": False}
			
			# Load video frames
			cap = cv2.VideoCapture(video_path)
			total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			frames = []
			
			# Load first frame and last frame for navigation
			for frame_idx in [0, total_frames - 1]:
				cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
				ret, frame = cap.read()
				if ret:
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					frames.append(frame_rgb)
			cap.release()
			
			if not frames:
				messagebox.showerror("Error", "No se pudo cargar el video")
				dialog.destroy()
				return None
			
			current_frame_idx = {"value": 0}
			
			# Zoom and pan state
			zoom_state = {"scale": 1.0, "pan_x": 0, "pan_y": 0, "drag_start_x": 0, "drag_start_y": 0}
			
			# Content frame
			content_frame = tk.Frame(dialog, bg=PROC_CONTENT_BG)
			content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
			
			# Warning message
			warning_label = tk.Label(content_frame, 
				text="Oops, parece que este video no tiene la fecha y hora en la que empezó a grabar en su nombre,\nes necesario que lo agregues de manera manual",
				font=("Arial", 10), fg=FG_COLOR, bg=PROC_CONTENT_BG, wraplength=750, justify=tk.LEFT)
			warning_label.pack(pady=(0, 10))
			
			# Frame navigation container
			nav_frame = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			nav_frame.pack(pady=5)
			
			# Canvas for frame display
			canvas_width, canvas_height = 640, 360
			frame_canvas = tk.Canvas(nav_frame, width=canvas_width, height=canvas_height, 
			                        bg="#000000", highlightthickness=1, highlightbackground="#666666")
			
			def update_frame_display():
				"""Update the displayed frame"""
				frame = frames[current_frame_idx["value"]]
				# Resize frame to fit canvas while preserving aspect ratio
				h, w = frame.shape[:2]
				base_scale = min(canvas_width / w, canvas_height / h)
				
				# Apply zoom scale
				final_scale = base_scale * zoom_state["scale"]
				new_w, new_h = int(w * final_scale), int(h * final_scale)
				
				# Use INTER_AREA for better quality when scaling down
				resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
				
				# Convert to PhotoImage
				img = Image.fromarray(resized)
				photo = ImageTk.PhotoImage(image=img)
				
				# Clear canvas and draw black background
				frame_canvas.delete("all")
				frame_canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#000000", outline="")
				
				# Apply pan and center the image on canvas
				x_offset = (canvas_width - new_w) // 2 + zoom_state["pan_x"]
				y_offset = (canvas_height - new_h) // 2 + zoom_state["pan_y"]
				frame_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
				frame_canvas.image = photo  # Keep reference
			
			def update_nav_buttons():
				"""Update navigation button states"""
				left_btn.config_state(state_value="disabled" if current_frame_idx["value"] == 0 else "normal")
				right_btn.config_state(state_value="disabled" if current_frame_idx["value"] == len(frames) - 1 else "normal")
			
			def go_prev_frame():
				if current_frame_idx["value"] > 0:
					current_frame_idx["value"] -= 1
					update_frame_display()
					update_nav_buttons()
			
			def go_next_frame():
				if current_frame_idx["value"] < len(frames) - 1:
					current_frame_idx["value"] += 1
					update_frame_display()
					update_nav_buttons()
			
			def on_mouse_wheel(event):
				"""Handle mouse wheel for zoom"""
				# Zoom in/out
				zoom_factor = 1.1 if event.delta > 0 else 0.9
				new_scale = zoom_state["scale"] * zoom_factor
				
				# Limit zoom range
				if 0.5 <= new_scale <= 5.0:
					zoom_state["scale"] = new_scale
					update_frame_display()
			
			def on_mouse_press(event):
				"""Handle mouse button press for pan"""
				zoom_state["drag_start_x"] = event.x
				zoom_state["drag_start_y"] = event.y
				frame_canvas.config(cursor="fleur")
			
			def on_mouse_drag(event):
				"""Handle mouse drag for pan"""
				dx = event.x - zoom_state["drag_start_x"]
				dy = event.y - zoom_state["drag_start_y"]
				
				zoom_state["pan_x"] += dx
				zoom_state["pan_y"] += dy
				
				zoom_state["drag_start_x"] = event.x
				zoom_state["drag_start_y"] = event.y
				
				update_frame_display()
			
			def on_mouse_release(event):
				"""Handle mouse button release"""
				frame_canvas.config(cursor="")
			
			# Helper function to create rounded buttons
			def create_dialog_rounded_button(parent, text, command, bg_color, fg_color="#FFFFFF", 
			                                  hover_color=None, width=80, height=40, corner_radius=8, initial_state="normal"):
				"""Create a rounded button using Canvas for the dialog"""
				if hover_color is None:
					# Slightly lighter version of bg_color for hover
					try:
						r = int(bg_color[1:3], 16)
						g = int(bg_color[3:5], 16)
						b = int(bg_color[5:7], 16)
						r = min(255, r + 30)
						g = min(255, g + 30)
						b = min(255, b + 30)
						hover_color = f"#{r:02x}{g:02x}{b:02x}"
					except:
						hover_color = bg_color
				
				disabled_color = "#555555"
				
				state = {
					"enabled": initial_state == "normal",
					"bg_color": bg_color,
					"hover_color": hover_color,
					"fg_color": fg_color
				}
				
				canvas = tk.Canvas(parent, width=width, height=height, 
				                   bg=PROC_CONTENT_BG, highlightthickness=0, cursor="hand2")
				
				def draw_button(fill_color, text_color=None):
					if text_color is None:
						text_color = fg_color
					canvas.delete("all")
					# Draw rounded rectangle
					canvas.create_arc(0, 0, corner_radius*2, corner_radius*2, 
					                 start=90, extent=90, fill=fill_color, outline="")
					canvas.create_arc(width-corner_radius*2, 0, width, corner_radius*2, 
					                 start=0, extent=90, fill=fill_color, outline="")
					canvas.create_arc(0, height-corner_radius*2, corner_radius*2, height, 
					                 start=180, extent=90, fill=fill_color, outline="")
					canvas.create_arc(width-corner_radius*2, height-corner_radius*2, width, height, 
					                 start=270, extent=90, fill=fill_color, outline="")
					canvas.create_rectangle(corner_radius, 0, width-corner_radius, height, 
					                       fill=fill_color, outline="")
					canvas.create_rectangle(0, corner_radius, width, height-corner_radius, 
					                       fill=fill_color, outline="")
					# Draw text
					canvas.create_text(width//2, height//2, text=text, 
					                  fill=text_color, font=("Arial", 11, "bold"))
				
				def on_enter(e):
					if state["enabled"]:
						draw_button(state["hover_color"])
				
				def on_leave(e):
					if state["enabled"]:
						draw_button(state["bg_color"])
					else:
						draw_button(disabled_color, "#888888")
				
				def on_click(e):
					if state["enabled"] and command:
						command()
				
				canvas.bind("<Enter>", on_enter)
				canvas.bind("<Leave>", on_leave)
				canvas.bind("<Button-1>", on_click)
				
				# Initial draw
				if state["enabled"]:
					draw_button(state["bg_color"])
				else:
					draw_button(disabled_color, "#888888")
				
				# Add config method to change state
				def config_button(state_value=None, **kwargs):
					if state_value is not None:
						state["enabled"] = (state_value == "normal")
						if state["enabled"]:
							draw_button(state["bg_color"])
						else:
							draw_button(disabled_color, "#888888")
				
				canvas.config_state = config_button
				return canvas
			
			# Left arrow button (rounded)
			left_btn = create_dialog_rounded_button(nav_frame, "◀", go_prev_frame, 
			                                         BUTTON_BG, FG_COLOR, width=50, height=50)
			left_btn.pack(side=tk.LEFT, padx=10)
			
			# Frame canvas
			frame_canvas.pack(side=tk.LEFT)
			
			# Bind mouse events for zoom and pan
			frame_canvas.bind("<MouseWheel>", on_mouse_wheel)
			frame_canvas.bind("<Button-1>", on_mouse_press)
			frame_canvas.bind("<B1-Motion>", on_mouse_drag)
			frame_canvas.bind("<ButtonRelease-1>", on_mouse_release)
			
			# Right arrow button (rounded)
			right_btn = create_dialog_rounded_button(nav_frame, "▶", go_next_frame, 
			                                          BUTTON_BG, FG_COLOR, width=50, height=50)
			right_btn.pack(side=tk.LEFT, padx=10)
			
			# DateTime input frame
			datetime_frame = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			datetime_frame.pack(pady=10)
			
			tk.Label(datetime_frame, text="Fecha y Hora de Inicio:", font=("Arial", 11, "bold"),
			        fg=FG_COLOR, bg=PROC_CONTENT_BG).pack(anchor=tk.W, pady=(0, 5))
			
			# Date entry
			date_row = tk.Frame(datetime_frame, bg=PROC_CONTENT_BG)
			date_row.pack(pady=5)
			
			tk.Label(date_row, text="Fecha (yyyy/MM/dd):", font=("Arial", 10), fg=FG_COLOR, bg=PROC_CONTENT_BG).pack(side=tk.LEFT, padx=(0, 10))
			date_entry = tk.Entry(date_row, font=("Arial", 11), width=15, justify=tk.CENTER)
			date_entry.pack(side=tk.LEFT)
			now = datetime.now()
			date_entry.insert(0, f"{now.year}/{now.month:02d}/{now.day:02d}")
			
			# Time entry
			time_row = tk.Frame(datetime_frame, bg=PROC_CONTENT_BG)
			time_row.pack(pady=5)
			
			tk.Label(time_row, text="Hora (hh:mm:ss):  ", font=("Arial", 10), fg=FG_COLOR, bg=PROC_CONTENT_BG).pack(side=tk.LEFT, padx=(0, 10))
			time_entry = tk.Entry(time_row, font=("Arial", 11), width=15, justify=tk.CENTER)
			time_entry.pack(side=tk.LEFT)
			time_entry.insert(0, "00:00:00")
			
			# Validation message
			validation_label = tk.Label(content_frame, text="", font=("Arial", 9), fg="#FF6B6B", bg=PROC_CONTENT_BG)
			validation_label.pack(pady=(5, 10))
			
			# Buttons frame
			button_frame = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			button_frame.pack(pady=15)
			
			def validate_inputs(*args):
				"""Validate inputs in real-time and enable/disable continue button"""
				try:
					# Get date and time strings
					date_str = date_entry.get().strip()
					time_str = time_entry.get().strip()
					
					# Validate date format yyyy/MM/dd
					date_parts = date_str.split('/')
					if len(date_parts) != 3:
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					
					try:
						year = int(date_parts[0])
						month = int(date_parts[1])
						day = int(date_parts[2])
					except ValueError:
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					
					# Validate time format hh:mm:ss
					time_parts = time_str.split(':')
					if len(time_parts) != 3:
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					
					try:
						hour = int(time_parts[0])
						minute = int(time_parts[1])
						second = int(time_parts[2])
					except ValueError:
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					
					# Validate ranges
					if not (1900 <= year <= 2100):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					if not (1 <= month <= 12):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					if not (1 <= day <= 31):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					if not (0 <= hour <= 23):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					if not (0 <= minute <= 59):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					if not (0 <= second <= 59):
						validation_label.config(text="")
						continue_btn.config_state(state_value="disabled")
						return
					
					# Try to create datetime to ensure it's valid
					dt = datetime(year, month, day, hour, minute, second)
					
					# All validations passed
					validation_label.config(text="✓ Fecha y hora válidas", fg="#4CAF50")
					continue_btn.config_state(state_value="normal")
				except Exception as e:
					validation_label.config(text="", fg="#FF6B6B")
					continue_btn.config_state(state_value="disabled")
			
			def validate_and_continue():
				"""Validate datetime input and close dialog"""
				try:
					# Get date and time strings
					date_str = date_entry.get().strip()
					time_str = time_entry.get().strip()
					
					# Parse date
					date_parts = date_str.split('/')
					year = int(date_parts[0])
					month = int(date_parts[1])
					day = int(date_parts[2])
					
					# Parse time
					time_parts = time_str.split(':')
					hour = int(time_parts[0])
					minute = int(time_parts[1])
					second = int(time_parts[2])
					
					# Create datetime object
					dt = datetime(year, month, day, hour, minute, second)
					result["datetime"] = dt
					dialog.destroy()
				except Exception as e:
					validation_label.config(text=f"Error: {str(e)}", fg="#FF6B6B")
			
			def cancel_action():
				result["cancelled"] = True
				dialog.destroy()
			
			# Create rounded buttons for Continue and Cancel
			continue_btn = create_dialog_rounded_button(button_frame, "Continuar", validate_and_continue,
			                                             PROC_BTN_CONFIRM, "#000000", width=120, height=40,
			                                             initial_state="disabled")
			continue_btn.pack(side=tk.LEFT, padx=10)
			
			cancel_btn = create_dialog_rounded_button(button_frame, "Cancelar", cancel_action,
			                                           BUTTON_BG, FG_COLOR, width=120, height=40)
			cancel_btn.pack(side=tk.LEFT, padx=10)
			
			# Bind validation to entry fields
			date_entry.bind("<KeyRelease>", validate_inputs)
			time_entry.bind("<KeyRelease>", validate_inputs)
			
			# Show video name
			video_name_label = tk.Label(content_frame, text=f"Video: {os.path.basename(video_path)}",
			                           font=("Arial", 9), fg="#999999", bg=PROC_CONTENT_BG)
			video_name_label.pack(pady=(5, 10))
			
			# Initialize display
			update_frame_display()
			
			# Update navigation button states
			update_nav_buttons()
			
			# Validate initial values
			validate_inputs()
			
			# Ensure dialog is visible and on top
			dialog.update()
			dialog.deiconify()
			dialog.lift()
			dialog.focus_set()
			dialog.attributes('-topmost', True)
			dialog.after(10, lambda: dialog.attributes('-topmost', False))
			dialog.after(20, lambda: dialog.lift())
			dialog.after(30, lambda: dialog.focus_force())
			
			print(f"[DEBUG] Dialog should be visible now")
			
			# Wait for dialog to close
			dialog.wait_window()
			
			print(f"[DEBUG] Dialog closed")
			
			if result["cancelled"]:
				return None
			return result["datetime"]
		
		# Dictionary to store manual datetimes for videos without valid datetime in filename
		manual_datetimes = {}
		
		def load_videos_from_folder(folder):
			"""Load all video files from selected folder and validate datetime in filenames"""
			files = []
			for p in Path(folder).iterdir():
				if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
					files.append(str(p))
			
			print(f"[DEBUG] Found {len(files)} video files")
			
			# Validate datetime for each video
			invalid_videos = []
			for video_path in files:
				is_valid, dt_obj = validate_video_datetime(video_path)
				if not is_valid:
					invalid_videos.append(video_path)
					print(f"[DEBUG] Video without valid datetime: {os.path.basename(video_path)}")
			
			print(f"[DEBUG] Found {len(invalid_videos)} videos without valid datetime")
			
			# Ask for manual datetime for invalid videos
			if invalid_videos:
				print(f"[DEBUG] Opening datetime dialog for {len(invalid_videos)} videos")
				for video_path in invalid_videos:
					print(f"[DEBUG] Showing dialog for: {os.path.basename(video_path)}")
					dt = show_datetime_input_dialog(video_path)
					if dt is None:
						# User cancelled, don't load any videos
						messagebox.showwarning("Operación Cancelada", 
							"Debe proporcionar fecha y hora para todos los videos. No se cargaron videos.")
						return
					manual_datetimes[video_path] = dt
					print(f"[DEBUG] Manual datetime saved: {dt}")
			
			nonlocal video_list
			video_list = files
			refresh_listbox()
			update_drop_enabled(False)
			set_next_enabled_for_second()
		
		def validate_folder_path(event=None):
			"""Validate and load videos when path is typed manually"""
			path_text = folder_entry.get().strip()
			if path_text and Path(path_text).exists() and Path(path_text).is_dir():
				load_videos_from_folder(path_text)
		
		def browse_folder_action():
			folder = filedialog.askdirectory(title="Seleccionar carpeta con videos")
			if folder:
				folder_entry.delete(0, tk.END)
				folder_entry.insert(0, folder)
				load_videos_from_folder(folder)
		
		# Bind validation to folder_entry changes - triggers on any text change
		folder_entry.bind("<KeyRelease>", validate_folder_path)
		folder_entry.bind("<Return>", validate_folder_path)
		folder_entry.bind("<FocusOut>", validate_folder_path)
		
		folder_browse = make_rounded_button(path_container, "Buscar", browse_folder_action, 
		                                     PROC_BTN_NORMAL, width=100, height=36)
		folder_browse.grid(row=0, column=1, sticky="e")
		
		# Row 2: NP label
		tk.Label(video_content, text="Escribir Número de Parte Asociado a los Videos", 
		         font=("Arial", 11, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG).grid(row=2, column=0, sticky="w", pady=(0, 8))
		
		# Row 3: NP entry with rounded corners
		np_entry_frame = tk.Frame(video_content, bg="#FFFFFF", highlightthickness=1, 
		                          highlightbackground="#D0D0D0", highlightcolor="#D0D0D0")
		np_entry_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))
		
		np_entry = tk.Entry(np_entry_frame, font=("Arial", 10), bg="#FFFFFF", fg="#000000", 
		                    relief=tk.FLAT, bd=0)
		np_entry.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
		
		# Row 4: Drop area (bottom) with rounded corners
		drop_container = tk.Frame(video_content, bg=PROC_CONTENT_BG)
		drop_container.grid(row=4, column=0, sticky="nsew")
		
		# Canvas for rounded rectangle with solid background
		canvas = tk.Canvas(drop_container, bg=PROC_DRAG_BG, highlightthickness=0, cursor="hand2")
		canvas.pack(fill=tk.BOTH, expand=True)
		
		def draw_rounded_drop_area(width, height):
			"""Draw solid rounded rectangle for drop area"""
			canvas.delete("bg")
			radius = 12
			# Draw rounded corners
			canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
			                  fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
			canvas.create_arc(width-radius*2, 0, width, radius*2, start=0, extent=90, 
			                  fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
			canvas.create_arc(0, height-radius*2, radius*2, height, start=180, extent=90, 
			                  fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
			canvas.create_arc(width-radius*2, height-radius*2, width, height, start=270, extent=90, 
			                  fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
			# Fill rectangles
			canvas.create_rectangle(radius, 0, width-radius, height, fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
			canvas.create_rectangle(0, radius, width, height-radius, fill=PROC_DRAG_BG, outline=PROC_DRAG_BG, tags="bg")
		
		# Center label for drag & drop message AND video list
		center_frame = tk.Frame(drop_container, bg=PROC_DRAG_BG)
		center_frame.place(relx=0.5, rely=0.5, anchor="center")
		center_label = tk.Label(center_frame, text="¡Arrastra y suelta videos aquí para cargarlos!", 
		                        font=("Arial", 13, "bold"), fg="#006cba", bg=PROC_DRAG_BG, justify=tk.LEFT)
		center_label.pack()
		
		def layout_canvas(_=None):
			w = canvas.winfo_width()
			h = canvas.winfo_height()
			if w > 1 and h > 1:
				draw_rounded_drop_area(w, h)
		
		canvas.bind("<Configure>", layout_canvas)
		
		def set_next_enabled_for_second():
			can_advance[0] = len(video_list) > 0
			update_nav_state()
		
		def update_folder_controls(enabled: bool):
			"""Enable/disable folder path controls"""
			if enabled:
				folder_entry.configure(state="normal", bg="#FFFFFF")
				folder_entry_frame.configure(bg="#FFFFFF")
			else:
				folder_entry.configure(state="disabled", bg="#E0E0E0")
				folder_entry_frame.configure(bg="#E0E0E0")
		def update_drop_enabled(enabled: bool):
			if enabled:
				canvas.configure(cursor="hand2")
				center_label.configure(fg="#006cba")
				if DND_AVAILABLE:
					canvas.drop_target_register(DND_FILES)
					canvas.dnd_bind("<<Drop>>", on_drop)
				canvas.bind("<Button-1>", on_click_select_files)
			else:
				canvas.configure(cursor="arrow")
				center_label.configure(fg="#5BA8C9")
				if DND_AVAILABLE:
					try:
						canvas.drop_target_unregister()
					except Exception:
						pass
				canvas.unbind("<Button-1>")
		
		def parse_dnd_files(s: str) -> list[str]:
			files = []
			buf = ""
			in_brace = False
			for ch in s:
				if ch == "{":
					in_brace = True
					buf = ""
				elif ch == "}":
					in_brace = False
					files.append(buf)
					buf = ""
				elif ch == " " and not in_brace:
					if buf:
						files.append(buf)
						buf = ""
				else:
					buf += ch
			if buf:
				files.append(buf)
			return files
		
		def filter_video_files(paths: list[str]) -> list[str]:
			out = []
			for p in paths:
				if Path(p).suffix.lower() in VIDEO_EXTS and Path(p).is_file():
					out.append(str(Path(p)))
			return out
		
		def refresh_listbox():
			if video_list:
				# Show list of videos
				video_names = [Path(p).name for p in video_list]
				list_text = "\n".join(video_names)
				center_label.config(text=list_text, fg="#0070c0", font=("Arial", 10))
			else:
				# Show drag & drop message
				center_label.config(text="¡Arrastra y suelta videos aquí para cargarlos!", 
				                    fg="#006cba", font=("Arial", 13, "bold"))
		
		def accept_files(paths: list[str]):
			nonlocal video_list
			files = filter_video_files(paths)
			if not files:
				return
			
			print(f"[DEBUG] accept_files called with {len(files)} video files")
			
			# Validate datetime for each video
			invalid_videos = []
			for video_path in files:
				is_valid, dt_obj = validate_video_datetime(video_path)
				if not is_valid:
					invalid_videos.append(video_path)
					print(f"[DEBUG] Video without valid datetime: {os.path.basename(video_path)}")
			
			print(f"[DEBUG] Found {len(invalid_videos)} videos without valid datetime")
			
			# Ask for manual datetime for invalid videos
			if invalid_videos:
				print(f"[DEBUG] Opening datetime dialog for {len(invalid_videos)} videos")
				validated_files = []
				for video_path in files:
					if video_path in invalid_videos:
						print(f"[DEBUG] Showing dialog for: {os.path.basename(video_path)}")
						dt = show_datetime_input_dialog(video_path)
						if dt is None:
							# User cancelled for this video, skip it
							print(f"[DEBUG] User cancelled datetime input for: {os.path.basename(video_path)}")
							continue
						manual_datetimes[video_path] = dt
						print(f"[DEBUG] Manual datetime saved: {dt}")
					validated_files.append(video_path)
				
				# Update files list to only include validated files
				files = validated_files
			
			# Add validated files to video list
			existing = set(video_list)
			for f in files:
				if f not in existing:
					video_list.append(f)
					existing.add(f)
			refresh_listbox()
			update_folder_controls(False)
			set_next_enabled_for_second()
		
		def on_drop(event):
			paths = parse_dnd_files(event.data)
			accept_files(paths)
		
		def on_click_select_files(_):
			paths = filedialog.askopenfilenames(title="Seleccionar videos", 
			                                     filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv *.webm")])
			accept_files(list(paths))
		
		update_folder_controls(True)
		update_drop_enabled(True)
		
		# ---------------- RT Mode UI (RTSP Multi-Camera) - NEW DESIGN ----------------
		rt_mode_frame.grid_rowconfigure(0, weight=0)  # Label row - no expansion
		rt_mode_frame.grid_rowconfigure(1, weight=1)  # Content row - expandable
		rt_mode_frame.grid_columnconfigure(0, weight=0)  # Left panel
		rt_mode_frame.grid_columnconfigure(1, weight=1)  # Right panel
		
		# Lista para almacenar datos de cada cámara
		camera_widgets = []
		rtsp_urls_list = []
		selected_camera_index = [0]  # Index of currently selected camera for editing
		
		# Label "Cámaras" outside the panel - spans both columns
		cameras_label = tk.Label(rt_mode_frame, text="Cámaras", font=("Arial", 14, "bold"), 
		                         fg=FG_COLOR, bg=PROC_CONTENT_BG)
		cameras_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 0))
		
		# Secret feature: Triple-click on "Cámaras" label to auto-configure cameras
		secret_click_state = {"count": 0, "last_time": 0}
		
		def on_cameras_label_click(event):
			"""Handle triple-click secret feature"""
			import time
			current_time = time.time()
			
			# Reset counter if more than 1 second has passed since last click
			if current_time - secret_click_state["last_time"] > 1.0:
				secret_click_state["count"] = 0
			
			secret_click_state["count"] += 1
			secret_click_state["last_time"] = current_time
			
			# Check for triple-click
			if secret_click_state["count"] >= 3:
				secret_click_state["count"] = 0  # Reset
				activate_secret_camera_config()
		
		def activate_secret_camera_config():
			"""Secret function: Auto-configure 3 cameras with specific IPs and password"""
			try:
				# Clear existing cameras
				camera_widgets.clear()
				for item in camera_list_items:
					item["frame"].destroy()
				camera_list_items.clear()
				
				# Add 3 cameras with specific IPs
				camera_ips = ["192.168.1.12", "192.168.1.14", "192.168.1.15"]
				password = "Nortec10"
				
				for i, ip in enumerate(camera_ips):
					camera_data = {
						"name": f"Cámara {i+1}",
						"ip": ip,
						"port": "554",
						"user": "admin",
						"password": password,
						"status": "NOK",
						"connected": False,
						"url": None
					}
					camera_widgets.append(camera_data)
					
					# Create list item
					list_item = create_camera_list_item(list_container, i, camera_data)
					camera_list_items.append(list_item)
				
				# Select first camera
				selected_camera_index[0] = 0
				load_camera_to_form(0)
				update_list_selection()
				update_camera_button_state()
				
				# Test connections for all 3 cameras sequentially
				def test_all_cameras(index=0):
					if index >= len(camera_widgets):
						update_advance_state()
						error_label.config(text="✓ ¡3 cámaras configuradas y probadas exitosamente!", fg="#7ec331")
						return
					
					# Select camera
					selected_camera_index[0] = index
					load_camera_to_form(index)
					update_list_selection()
					
					# Test connection
					camera_data = camera_widgets[index]
					ip = camera_data["ip"]
					port = camera_data["port"]
					user = camera_data["user"]
					password = camera_data["password"]
					
					# Construct RTSP URL
					rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/101"
					
					def test_camera_thread():
						import cv2
						try:
							cap = cv2.VideoCapture(rtsp_url)
							if cap.isOpened():
								ret, frame = cap.read()
								cap.release()
								if ret and frame is not None:
									# Success
									win.after(0, lambda: on_test_success(index, rtsp_url))
								else:
									win.after(0, lambda: on_test_failure(index))
							else:
								win.after(0, lambda: on_test_failure(index))
						except Exception:
							win.after(0, lambda: on_test_failure(index))
					
					def on_test_success(idx, url):
						camera_widgets[idx]["connected"] = True
						camera_widgets[idx]["status"] = "OK"
						camera_widgets[idx]["url"] = url
						camera_list_items[idx]["status_btn"].configure(text="OK", bg="#7ec331")
						# Test next camera
						win.after(100, lambda: test_all_cameras(idx + 1))
					
					def on_test_failure(idx):
						camera_widgets[idx]["connected"] = False
						camera_widgets[idx]["status"] = "NOK"
						camera_widgets[idx]["url"] = None
						camera_list_items[idx]["status_btn"].configure(text="NOK", bg="#ec5b2d")
						# Test next camera anyway
						win.after(100, lambda: test_all_cameras(idx + 1))
					
					# Start test in thread
					import threading
					thread = threading.Thread(target=test_camera_thread, daemon=True)
					thread.start()
				
				# Start testing all cameras
				win.after(500, lambda: test_all_cameras(0))
				
			except Exception as e:
				messagebox.showerror("Error", f"Error en configuración secreta:\n{str(e)}")
		
		cameras_label.bind("<Button-1>", on_cameras_label_click)
		
		# LEFT PANEL: Camera list box
		left_panel = tk.Frame(rt_mode_frame, bg="#001234")
		left_panel.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
		left_panel.grid_rowconfigure(0, weight=1)  # List (scrollable)
		left_panel.grid_rowconfigure(1, weight=0)  # Button
		left_panel.grid_columnconfigure(0, weight=1)
		
		# Scrollable list area in a frame
		list_frame = tk.Frame(left_panel, bg="#001234")
		list_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 12))
		list_frame.grid_rowconfigure(0, weight=1)
		list_frame.grid_columnconfigure(0, weight=1)
		
		list_canvas = tk.Canvas(list_frame, bg="#001234", highlightthickness=0, width=250, height=400)
		list_scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=list_canvas.yview)
		list_canvas.configure(yscrollcommand=list_scrollbar.set)
		
		list_canvas.grid(row=0, column=0, sticky="nsew")
		list_scrollbar.grid(row=0, column=1, sticky="ns")
		
		list_container = tk.Frame(list_canvas, bg="#001234")
		list_window = list_canvas.create_window((0, 0), window=list_container, anchor="nw")
		
		def update_list_scroll(_=None):
			list_canvas.configure(scrollregion=list_canvas.bbox("all"))
		
		def resize_list(_=None):
			list_canvas.itemconfig(list_window, width=list_canvas.winfo_width())
		
		list_container.bind("<Configure>", update_list_scroll)
		list_canvas.bind("<Configure>", resize_list)
		
		# Add camera button at bottom of left panel
		add_camera_btn_frame = tk.Frame(left_panel, bg="#001234")
		add_camera_btn_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
		
		# RIGHT PANEL: Camera configuration form
		right_panel = tk.Frame(rt_mode_frame, bg=PROC_CONTENT_BG)
		right_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
		right_panel.grid_columnconfigure(0, weight=1)
		
		# Form fields
		form_y_padding = 8
		
		# Nombre de la cámara
		name_label = tk.Label(right_panel, text="Nombre de la cámara", font=("Arial", 11),
		                     fg=FG_COLOR, bg=PROC_CONTENT_BG)
		name_label.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))
		
		name_entry = tk.Entry(right_panel, font=("Arial", 11), fg="black", bg="white")
		name_entry.insert(0, "Cámara")
		name_entry.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		
		# IP / Dominio
		ip_label = tk.Label(right_panel, text="IP / Dominio", font=("Arial", 11),
		                   fg=FG_COLOR, bg=PROC_CONTENT_BG)
		ip_label.grid(row=2, column=0, sticky="w", padx=16, pady=(form_y_padding, 4))
		
		ip_entry = tk.Entry(right_panel, font=("Arial", 11), fg="black", bg="white")
		ip_entry.insert(0, "192.168.1.12")
		ip_entry.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		
		# Lista de IPs detectadas
		discovered_label = tk.Label(right_panel, text="IPs detectadas en red", font=("Arial", 11),
		                           fg=FG_COLOR, bg=PROC_CONTENT_BG)
		discovered_label.grid(row=4, column=0, sticky="w", padx=16, pady=(form_y_padding, 4))
		
		discovered_frame = tk.Frame(right_panel, bg=PROC_CONTENT_BG)
		discovered_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		discovered_frame.grid_columnconfigure(0, weight=1)
		
		discovered_ips_var = tk.StringVar()
		discovered_ips_combo = ttk.Combobox(discovered_frame, textvariable=discovered_ips_var,
		                                   font=("Arial", 11), state="readonly")
		discovered_ips_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
		ip_blacklist = {"192.168.1.64"}
		if pre_scanned_ips:
			filtered_pre_scanned_ips = [ip for ip in pre_scanned_ips if ip not in ip_blacklist]
			discovered_ips_combo["values"] = filtered_pre_scanned_ips
			if filtered_pre_scanned_ips:
				discovered_ips_var.set(filtered_pre_scanned_ips[0])
				ip_entry.delete(0, tk.END)
				ip_entry.insert(0, filtered_pre_scanned_ips[0])
			else:
				discovered_ips_var.set("")
				ip_entry.delete(0, tk.END)
				ip_entry.insert(0, "192.168.1.12")
		
		def apply_selected_discovered_ip(_event=None):
			selected_ip = discovered_ips_var.get().strip()
			if not selected_ip:
				return
			ip_entry.delete(0, tk.END)
			ip_entry.insert(0, selected_ip)
			if selected_camera_index[0] < len(camera_widgets):
				idx = selected_camera_index[0]
				camera_widgets[idx]["ip"] = selected_ip
				if idx < len(camera_list_items):
					camera_list_items[idx]["ip_label"].configure(text=selected_ip)
		
		discovered_ips_combo.bind("<<ComboboxSelected>>", apply_selected_discovered_ip)
		
		def discover_connected_camera_ips():
			def _scan_prefix_from_form():
				try:
					ip_text = ip_entry.get().strip()
					parts = ip_text.split(".")
					if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
						return f"{parts[0]}.{parts[1]}.{parts[2]}"
				except Exception:
					pass
				return "192.168.1"
			
			def _finish_scan(found_ips, prefix):
				discover_btn.configure(state="normal")
				discovered_ips_combo["values"] = found_ips
				if found_ips:
					discovered_ips_var.set(found_ips[0])
					apply_selected_discovered_ip()
					error_label.config(text=f"✓ Detectadas {len(found_ips)} IPs activas en {prefix}.x", fg="#7ec331")
				else:
					discovered_ips_var.set("")
					error_label.config(text=f"No se detectaron IPs activas en {prefix}.x", fg="#ec5b2d")
			
			def _worker():
				import socket
				from concurrent.futures import ThreadPoolExecutor
				prefix = _scan_prefix_from_form()
				candidates = [f"{prefix}.{i}" for i in range(1, 255)]
				
				def _is_camera_candidate(ip):
					for p in (554, 8000):
						sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
						sock.settimeout(0.15)
						try:
							if sock.connect_ex((ip, p)) == 0:
								return True
						except Exception:
							pass
						finally:
							sock.close()
					return False
			
				def _set_first_found_ip(ip):
					ip_entry.delete(0, tk.END)
					ip_entry.insert(0, ip)
					if selected_camera_index[0] < len(camera_widgets):
						idx = selected_camera_index[0]
						camera_widgets[idx]["ip"] = ip
						if idx < len(camera_list_items):
							camera_list_items[idx]["ip_label"].configure(text=ip)
				
				found = []
				first_found_sent = [False]
				try:
					with ThreadPoolExecutor(max_workers=64) as executor:
						for ip, ok in zip(candidates, executor.map(_is_camera_candidate, candidates)):
							if ok and ip not in ip_blacklist:
								found.append(ip)
								if not first_found_sent[0]:
									first_found_sent[0] = True
									win.after(0, lambda ip=ip: _set_first_found_ip(ip))
				except Exception:
					found = []
				
				win.after(0, lambda: _finish_scan(found, prefix))
			
			discover_btn.configure(state="disabled")
			error_label.config(text="Escaneando red local...", fg=FG_COLOR)
			t = threading.Thread(target=_worker, daemon=True)
			t.start()
		
		discover_btn = make_rounded_button(discovered_frame, "Buscar", discover_connected_camera_ips, "#015aca",
		                               width=90, height=32, fg="white")
		discover_btn.grid(row=0, column=1, sticky="e")
		if preset_type == "rt" and auto_scan_on_open:
			win.after(150, discover_connected_camera_ips)
		
		# Puerto
		port_label = tk.Label(right_panel, text="Puerto", font=("Arial", 11),
		                     fg=FG_COLOR, bg=PROC_CONTENT_BG)
		port_label.grid(row=6, column=0, sticky="w", padx=16, pady=(form_y_padding, 4))
		
		port_entry = tk.Entry(right_panel, font=("Arial", 11), fg="black", bg="white")
		port_entry.insert(0, "554")
		port_entry.grid(row=7, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		
		# Usuario
		user_label = tk.Label(right_panel, text="Usuario", font=("Arial", 11),
		                     fg=FG_COLOR, bg=PROC_CONTENT_BG)
		user_label.grid(row=8, column=0, sticky="w", padx=16, pady=(form_y_padding, 4))
		
		user_entry = tk.Entry(right_panel, font=("Arial", 11), fg="black", bg="white")
		user_entry.insert(0, "admin")
		user_entry.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		
		# Contraseña
		password_label = tk.Label(right_panel, text="Contraseña", font=("Arial", 11),
		                         fg=FG_COLOR, bg=PROC_CONTENT_BG)
		password_label.grid(row=10, column=0, sticky="w", padx=16, pady=(form_y_padding, 4))
		
		password_entry = tk.Entry(right_panel, font=("Arial", 11), fg="black", bg="white", show="*")
		password_entry.grid(row=11, column=0, sticky="ew", padx=16, pady=(0, form_y_padding))
		
		# Buttons row
		buttons_frame = tk.Frame(right_panel, bg=PROC_CONTENT_BG)
		buttons_frame.grid(row=12, column=0, sticky="w", padx=16, pady=(20, 16))
		
		# Error message label (hidden by default)
		error_label = tk.Label(right_panel, text="", font=("Arial", 10), 
		                      fg="#ec5b2d", bg=PROC_CONTENT_BG, anchor="w")
		error_label.grid(row=13, column=0, sticky="w", padx=16, pady=(0, 8))
		
		# Camera list item widget references
		camera_list_items = []  # List of {"frame": tk.Frame, "data": camera_data_dict}
		
		def create_camera_list_item(parent, index, camera_data):
			"""Creates a camera item in the left list"""
			item_frame = tk.Frame(parent, bg="#001234", cursor="hand2", relief=tk.FLAT, bd=1)
			item_frame.pack(fill=tk.X, padx=8, pady=4)
			
			content_frame = tk.Frame(item_frame, bg="#001234")
			content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
			content_frame.grid_columnconfigure(0, weight=1)
			content_frame.grid_columnconfigure(1, weight=0)
			
			# Left side: Camera name and IP
			info_frame = tk.Frame(content_frame, bg="#001234")
			info_frame.grid(row=0, column=0, sticky="w")
			
			name_lbl = tk.Label(info_frame, text=camera_data.get("name", f"Cámara {index+1}"),
			                   font=("Arial", 12, "bold"), fg=FG_COLOR, bg="#001234", anchor="w")
			name_lbl.pack(anchor="w")
			
			ip_lbl = tk.Label(info_frame, text=camera_data.get("ip", "192.168.1.12"),
			                 font=("Arial", 10), fg=FG_COLOR, bg="#001234", anchor="w")
			ip_lbl.pack(anchor="w")
			
			# Right side: Status button
			status = camera_data.get("status", "NOK")
			status_color = "#7ec331" if status == "OK" else "#ec5b2d"
			status_btn = create_rounded_button(content_frame, status, 
			                                   bg_color=status_color, active_bg=status_color,
			                                   font=("Arial", 10, "bold"), padx=16, pady=4,
			                                   corner_radius=6)
			status_btn.grid(row=0, column=1, sticky="e", padx=(12, 0))
			
			# Click handler to select camera
			def select_camera(event=None):
				selected_camera_index[0] = index
				load_camera_to_form(index)
				update_list_selection()
			
			item_frame.bind("<Button-1>", select_camera)
			for widget in [content_frame, info_frame, name_lbl, ip_lbl]:
				widget.bind("<Button-1>", select_camera)
			
			return {
				"frame": item_frame,
				"name_label": name_lbl,
				"ip_label": ip_lbl,
				"status_btn": status_btn,
				"data": camera_data
			}
		
		def update_list_selection():
			"""Highlights the selected camera in the list"""
			for i, item in enumerate(camera_list_items):
				if i == selected_camera_index[0]:
					item["frame"].configure(bg="#043c86", relief=tk.SUNKEN)
					item["name_label"].configure(bg="#043c86")
					item["ip_label"].configure(bg="#043c86")
					item["status_btn"].configure(canvas_bg="#043c86")
					for child in item["frame"].winfo_children():
						if isinstance(child, tk.Frame):
							child.configure(bg="#043c86")
							for subchild in child.winfo_children():
								if isinstance(subchild, (tk.Label, tk.Frame)):
									subchild.configure(bg="#043c86")
				else:
					item["frame"].configure(bg="#001234", relief=tk.FLAT)
					item["name_label"].configure(bg="#001234")
					item["ip_label"].configure(bg="#001234")
					item["status_btn"].configure(canvas_bg="#001234")
					for child in item["frame"].winfo_children():
						if isinstance(child, tk.Frame):
							child.configure(bg="#001234")
							for subchild in child.winfo_children():
								if isinstance(subchild, (tk.Label, tk.Frame)):
									subchild.configure(bg="#001234")
		
		def load_camera_to_form(index):
			"""Loads the selected camera data into the form"""
			if index >= len(camera_widgets):
				return
			camera_data = camera_widgets[index]
			
			# Clear error message
			error_label.config(text="")
			
			# Clear and populate form fields
			name_entry.delete(0, tk.END)
			name_entry.insert(0, camera_data.get("name", f"Cámara {index+1}"))
			
			ip_entry.delete(0, tk.END)
			ip_entry.insert(0, camera_data.get("ip", "192.168.1.12"))
			
			port_entry.delete(0, tk.END)
			port_entry.insert(0, camera_data.get("port", "554"))
			
			user_entry.delete(0, tk.END)
			user_entry.insert(0, camera_data.get("user", "admin"))
			
			password_entry.delete(0, tk.END)
			password_entry.insert(0, camera_data.get("password", ""))
		
		def save_form_to_camera():
			"""Saves the current form data to the selected camera"""
			if selected_camera_index[0] >= len(camera_widgets):
				return
			
			index = selected_camera_index[0]
			camera_widgets[index].update({
				"name": name_entry.get().strip() or f"Cámara {index+1}",
				"ip": ip_entry.get().strip() or "192.168.1.12",
				"port": port_entry.get().strip() or "554",
				"user": user_entry.get().strip() or "admin",
				"password": password_entry.get().strip()
			})
			
			# Update list item display
			if index < len(camera_list_items):
				item = camera_list_items[index]
				item["name_label"].configure(text=camera_widgets[index]["name"])
				item["ip_label"].configure(text=camera_widgets[index]["ip"])
				item["data"] = camera_widgets[index]
		
		def add_new_camera():
			"""Adds a new camera to the list"""
			if len(camera_widgets) >= 10:
				messagebox.showwarning("Advertencia", "El máximo de cámaras permitidas es 10.")
				return
			
			new_index = len(camera_widgets)
			default_ip = pre_scanned_ips[0] if pre_scanned_ips else "192.168.1.12"
			new_camera = {
				"name": f"Cámara {new_index+1}",
				"ip": default_ip,
				"port": "554",
				"user": "admin",
				"password": "",
				"status": "NOK",
				"connected": False,
				"url": None
			}
			camera_widgets.append(new_camera)
			
			# Create list item
			list_item = create_camera_list_item(list_container, new_index, new_camera)
			camera_list_items.append(list_item)
			
			# Select the new camera
			selected_camera_index[0] = new_index
			load_camera_to_form(new_index)
			update_list_selection()
			update_camera_button_state()
			update_advance_state()
		
		def delete_current_camera():
			"""Deletes the currently selected camera"""
			if len(camera_widgets) <= 1:
				messagebox.showwarning("Advertencia", "Debe haber al menos una cámara configurada.")
				return
			
			index = selected_camera_index[0]
			
			# Remove from data list
			camera_widgets.pop(index)
			
			# Remove from UI list
			camera_list_items[index]["frame"].destroy()
			camera_list_items.pop(index)
			
			# Rebuild list items with updated indices
			for i, item in enumerate(camera_list_items):
				item["data"] = camera_widgets[i]
				# Update index in data if needed
			
			# Select previous camera or first
			selected_camera_index[0] = max(0, index - 1)
			if camera_widgets:
				load_camera_to_form(selected_camera_index[0])
				update_list_selection()
			
			update_camera_button_state()
			update_advance_state()
		
		def update_camera_button_state():
			"""Updates the add camera button state based on camera count"""
			if len(camera_widgets) >= 10:
				add_camera_btn.configure(state="disabled")
			else:
				add_camera_btn.configure(state="normal")
		
		def test_camera_connection():
			"""Tests the connection of the currently selected camera"""
			save_form_to_camera()
			index = selected_camera_index[0]
			camera_data = camera_widgets[index]
			
			# Clear error message
			error_label.config(text="")
			
			# Check for duplicate IPs in already connected cameras
			for i, cam in enumerate(camera_widgets):
				if i != index and cam["status"] == "OK" and cam["ip"] == camera_data["ip"]:
					error_msg = f"Error: La IP {camera_data['ip']} ya está en uso por la cámara '{cam['name']}'"
					error_label.config(text=error_msg)
					camera_data["connected"] = False
					camera_data["status"] = "NOK"
					camera_data["url"] = None
					camera_list_items[index]["status_btn"].configure(text="NOK", bg="#ec5b2d")
					return
			
			test_btn.configure(text="Conectando...", state="disabled")
			win.update_idletasks()
			
			ip = camera_data["ip"]
			port = camera_data["port"]
			user = camera_data["user"]
			password = camera_data["password"]
			
			# Construct RTSP URL (assuming Hikvision format with channel 101)
			rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/101"
			
			try:
				import cv2
				cap = cv2.VideoCapture(rtsp_url)
				if cap.isOpened():
					ret, frame = cap.read()
					cap.release()
					if ret and frame is not None:
						# Success
						error_label.config(text=f"✓ Cámara {camera_data['name']} conectada correctamente.", fg="#7ec331")
						camera_data["connected"] = True
						camera_data["status"] = "OK"
						camera_data["url"] = rtsp_url
						camera_list_items[index]["status_btn"].configure(text="OK", bg="#7ec331")
					else:
						raise Exception("No se pudo leer frame")
				else:
					raise Exception("No se pudo abrir stream")
			except Exception as e:
				# Failure
				camera_data["connected"] = False
				camera_data["status"] = "NOK"
				camera_data["url"] = None
				camera_list_items[index]["status_btn"].configure(text="NOK", bg="#ec5b2d")
				error_label.config(text=f"✗ Error de conexión: {str(e)[:80]}", fg="#ec5b2d")
			
			test_btn.configure(text="Probar Conexión", state="normal")
			update_advance_state()
		
		def update_advance_state():
			"""Updates the can_advance state based on all cameras being connected"""
			rtsp_urls_list.clear()
			all_connected = all(cam["connected"] for cam in camera_widgets)
			
			if all_connected and len(camera_widgets) > 0:
				# Build RTSP URLs list
				for cam in camera_widgets:
					rtsp_urls_list.append({
						"url": cam["url"],
						"np": cam.get("name", ""),
						"linea": str(cam.get("linea", "")).strip()
					})
				can_advance[0] = True
			else:
				can_advance[0] = False
			
			update_nav_state()
		
		# Bind form field changes to save on change
		def on_form_change(event=None):
			save_form_to_camera()
		
		for entry in [name_entry, ip_entry, port_entry, user_entry, password_entry]:
			entry.bind("<KeyRelease>", on_form_change)
		
		# Create buttons with commands
		add_camera_btn = create_rounded_button(add_camera_btn_frame, "Añadir Nueva Cámara", 
		                                       bg_color="#015aca", active_bg="#0147a0",
		                                       padx=20, pady=10, command=add_new_camera)
		add_camera_btn.pack()
		
		delete_btn = create_rounded_button(buttons_frame, "Eliminar", 
		                                   bg_color="#ec5b2d", active_bg="#d94a1f",
		                                   padx=20, pady=10, command=delete_current_camera)
		delete_btn.pack(side=tk.LEFT, padx=(0, 8))
		
		test_btn = create_rounded_button(buttons_frame, "Probar Conexión",
		                                 bg_color="#015aca", active_bg="#0147a0",
		                                 padx=20, pady=10, command=test_camera_connection)
		test_btn.pack(side=tk.LEFT)

		
		# Add "Utilizar grupo guardado" button - loads config and starts processing
		load_group_btn_frame = tk.Frame(left_panel, bg="#001234")
		load_group_btn_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 16))
		
		def load_camera_group():
			"""Load saved camera group and start processing automatically"""
			import json
			from pathlib import Path
			
			# Get list of saved groups
			groups_dir = Path("camera_groups")
			if not groups_dir.exists():
				messagebox.showinfo("Sin grupos", "No hay grupos de cámaras guardados.")
				return
			
			group_files = list(groups_dir.glob("*.json"))
			if not group_files:
				messagebox.showinfo("Sin grupos", "No hay grupos de cámaras guardados.")
				return
			
			# Create popup
			popup = tk.Toplevel(win)
			popup.title("Cargar Grupo de Cámaras")
			popup.geometry("700x450")
			popup.configure(bg=PROC_CONTENT_BG)
			popup.transient(win)
			popup.grab_set()
			
			# Title
			title_lbl = tk.Label(popup, text="Selecciona un grupo guardado", 
			                     font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			title_lbl.pack(pady=(20, 10))
			
			# Table frame
			table_frame = tk.Frame(popup, bg=PROC_CONTENT_BG)
			table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
			
			# Create Treeview for table
			columns = ("Línea", "Segmento", "Área", "Planta", "#_camaras")
			tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12, selectmode="extended")
			
			# Define column headings with custom widths
			column_widths = {
				"Línea": 120,
				"Segmento": 100,
				"Área": 90,
				"Planta": 120,
				"#_camaras": 80
			}
			for col in columns:
				tree.heading(col, text=col)
				tree.column(col, width=column_widths.get(col, 100), anchor="center")
			
			# Scrollbar for table
			scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
			tree.configure(yscrollcommand=scrollbar.set)
			scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
			tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
			
			# Populate table with group data
			group_data_map = {}  # Map tree item IDs to file paths
			for gfile in group_files:
				try:
					with open(gfile, 'r', encoding='utf-8') as f:
						data = json.load(f)
					metadata = data.get("metadata", {})
					linea = metadata.get("linea", "N/A")
					segmento = metadata.get("segmento", "N/A")
					area = metadata.get("area", "N/A")
					planta = metadata.get("planta", "N/A")
					num_camaras = len(data.get("cameras", []))
					
					item_id = tree.insert("", tk.END, values=(linea, segmento, area, planta, num_camaras))
					group_data_map[item_id] = gfile
				except Exception as e:
					print(f"[WARNING] Error loading metadata from {gfile}: {e}")
			
			# Buttons frame
			btn_frame = tk.Frame(popup, bg=PROC_CONTENT_BG)
			btn_frame.pack(pady=(0, 20))
			
			def on_load():
				selection = tree.selection()
				if not selection:
					messagebox.showwarning("Selección requerida", "Por favor selecciona uno o más grupos de la tabla.")
					return

				selected_group_files = []
				for item_id in selection:
					group_file = group_data_map.get(item_id)
					if not group_file:
						messagebox.showerror("Error", "No se pudo encontrar uno de los archivos de grupo seleccionados.")
						return
					selected_group_files.append(group_file)

				try:
					# Reset current camera and ROI state
					camera_widgets.clear()
					for item in camera_list_items:
						item["frame"].destroy()
					camera_list_items.clear()
					roi_state["rois_map"].clear()

					last_valid_model_path = ""
					last_settings = {}
					loaded_group_metadata = []
					camera_offset = 0

					for group_file in selected_group_files:
						with open(group_file, 'r', encoding='utf-8') as f:
							data = json.load(f)

						group_metadata_item = data.get("metadata", {})
						group_linea = str(group_metadata_item.get("linea", "")).strip()
						cameras = data.get("cameras", [])
						for cam_data in cameras:
							cam_payload = cam_data.copy()
							if group_linea and not str(cam_payload.get("linea", "")).strip():
								cam_payload["linea"] = group_linea
							camera_widgets.append(cam_payload)
							idx = len(camera_widgets) - 1
							list_item = create_camera_list_item(list_container, idx, cam_payload)
							camera_list_items.append(list_item)

						for cam_index, cam_data in enumerate(cameras):
							if "rois" in cam_data:
								merged_index = camera_offset + cam_index
								roi_state["rois_map"][merged_index] = cam_data["rois"]
								print(f"[DEBUG] Cargados {len(cam_data['rois'])} ROIs para cámara {merged_index}")

						camera_offset += len(cameras)

						model_path = data.get("model_path", "")
						if model_path and Path(model_path).exists():
							last_valid_model_path = model_path

						settings = data.get("settings", {})
						if settings:
							last_settings = settings

						metadata = data.get("metadata", {})
						if metadata:
							loaded_group_metadata.append(metadata)

					if camera_widgets:
						selected_camera_index[0] = 0
						load_camera_to_form(0)
						update_list_selection()
						update_camera_button_state()

					if last_valid_model_path:
						MODEL_FILE["value"] = last_valid_model_path
						model_name = Path(last_valid_model_path).name
						model_loaded_chk.config(text=f"Modelo cargado: {model_name}")
						print(f"[DEBUG] Modelo cargado: {model_name}")
					else:
						model_loaded_chk.config(text="Modelo cargado: Sin modelo")

					if last_settings:
						security_var.set(last_settings.get("security", False))
						face_blur_var.set(last_settings.get("face_blur", False))
						background_processing_var.set(last_settings.get("background_processing", False))
						processing_level_var.set(last_settings.get("processing_level", 1))
						videos_entry.delete(0, tk.END)
						videos_entry.insert(0, last_settings.get("videos_output", r"C:\Arnesis\videos_procesados"))
						data_entry.delete(0, tk.END)
						data_entry.insert(0, last_settings.get("data_output", r"C:\Arnesis\videos_procesados\datos"))

					group_metadata.clear()
					if len(loaded_group_metadata) == 1:
						group_metadata.update(loaded_group_metadata[0])
					elif len(loaded_group_metadata) > 1:
						group_metadata.update({"groups": loaded_group_metadata})
					print(f"[DEBUG] Metadata cargada: {group_metadata}")

					# Update RTSP URLs list and advance state
					update_advance_state()

					# Iniciar procesamiento automáticamente con los datos cargados
					# perform_process() construye la configuración desde las variables UI
					win.after(100, lambda: perform_process())

					# Cerrar la ventana pop-up después de cargar exitosamente
					popup.destroy()

				except Exception as e:
					messagebox.showerror("Error", f"Error al cargar grupos:\n{str(e)}")
				
			def on_cancel():
				"""Close popup without loading"""
				popup.destroy()
			
			def on_update():
				"""Update the selected group"""
				selection = tree.selection()
				if not selection:
					messagebox.showwarning("Selección requerida", "Por favor selecciona un grupo de la tabla para actualizar.")
					return
				
				# Get selected item
				item_id = selection[0]
				group_file = group_data_map.get(item_id)
				if not group_file:
					messagebox.showerror("Error", "No se pudo encontrar el archivo del grupo.")
					return
				
				try:
					with open(group_file, 'r', encoding='utf-8') as f:
						group_data = json.load(f)
				except Exception as e:
					messagebox.showerror("Error", f"Error al leer el grupo:\n{str(e)}")
					return
				
				# Create update popup
				update_popup = tk.Toplevel(popup)
				update_popup.title("Actualizar Grupo de Cámaras")
				update_popup.geometry("500x350")
				update_popup.configure(bg=PROC_CONTENT_BG)
				update_popup.transient(popup)
				update_popup.grab_set()
				
				# Title
				title_lbl = tk.Label(update_popup, text="Actualizar datos del grupo de cámaras", 
				                     font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
				title_lbl.pack(pady=(20, 10))
				
				# Form frame
				form_frame = tk.Frame(update_popup, bg=PROC_CONTENT_BG)
				form_frame.pack(pady=10, padx=40, fill=tk.X)
				
				# Load existing metadata
				metadata = group_data.get("metadata", {})
				
				# Linea
				linea_lbl = tk.Label(form_frame, text="Línea:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
				linea_lbl.grid(row=0, column=0, sticky="w", pady=10)
				linea_entry = tk.Entry(form_frame, font=("Arial", 11), fg="black", bg="white")
				linea_entry.insert(0, metadata.get("linea", ""))
				linea_entry.grid(row=0, column=1, sticky="ew", pady=10, padx=(10, 0))
				linea_entry.focus()
				
				# Segmento
				segmento_lbl = tk.Label(form_frame, text="Segmento:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
				segmento_lbl.grid(row=1, column=0, sticky="w", pady=10)
				segmento_var = tk.StringVar(value=metadata.get("segmento", "1"))
				segmento_combo = ttk.Combobox(form_frame, textvariable=segmento_var, font=("Arial", 11), 
				                              values=[str(i) for i in range(1, 7)], state="readonly", width=18)
				segmento_combo.grid(row=1, column=1, sticky="ew", pady=10, padx=(10, 0))
				
				# Area
				area_lbl = tk.Label(form_frame, text="Área:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
				area_lbl.grid(row=2, column=0, sticky="w", pady=10)
				area_entry = tk.Entry(form_frame, font=("Arial", 11), fg="black", bg="white")
				area_entry.insert(0, metadata.get("area", ""))
				area_entry.grid(row=2, column=1, sticky="ew", pady=10, padx=(10, 0))
				
				# Planta
				planta_lbl = tk.Label(form_frame, text="Planta:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
				planta_lbl.grid(row=3, column=0, sticky="w", pady=10)
				planta_var = tk.StringVar(value=metadata.get("planta", "1"))
				planta_combo = ttk.Combobox(form_frame, textvariable=planta_var, font=("Arial", 11), 
				                            values=["1", "2"], state="readonly", width=18)
				planta_combo.grid(row=3, column=1, sticky="ew", pady=10, padx=(10, 0))
				
				# Configure column weights
				form_frame.columnconfigure(1, weight=1)
				
				# Buttons
				btn_frame = tk.Frame(update_popup, bg=PROC_CONTENT_BG)
				btn_frame.pack(pady=(20, 20))
				
				def save_update():
					linea = linea_entry.get().strip()
					segmento = segmento_var.get()
					area = area_entry.get().strip()
					planta = planta_var.get()
					
					if not linea or not area:
						messagebox.showwarning("Datos requeridos", "Por favor completa los campos Línea y Área.")
						return
					
					try:
						# Update metadata
						group_data["metadata"] = {
							"linea": linea,
							"segmento": segmento,
							"area": area,
							"planta": planta
						}
						
						# Save updated data to same file
						with open(group_file, 'w', encoding='utf-8') as f:
							json.dump(group_data, f, indent=2, ensure_ascii=False)
						
						# Refresh table
						refresh_table()
						
						update_popup.destroy()
						messagebox.showinfo("Éxito", "Grupo actualizado correctamente.")
						
					except Exception as e:
						messagebox.showerror("Error", f"Error al actualizar el grupo:\n{str(e)}")
				
				def cancel_update():
					update_popup.destroy()
				
				cancel_btn = make_rounded_button(btn_frame, "Cancelar", cancel_update, "#666666", 
				                                 width=120, height=40, fg="white")
				cancel_btn.pack(side=tk.LEFT, padx=(0, 10))
				
				update_btn = make_rounded_button(btn_frame, "Guardar cambios", save_update, "#7ec331", 
				                                 width=150, height=40, fg="white")
				update_btn.pack(side=tk.LEFT)
			
			def on_delete():
				"""Delete the selected group"""
				selection = tree.selection()
				if not selection:
					messagebox.showwarning("Selección requerida", "Por favor selecciona un grupo de la tabla para eliminar.")
					return
				
				# Get selected item
				item_id = selection[0]
				group_file = group_data_map.get(item_id)
				if not group_file:
					messagebox.showerror("Error", "No se pudo encontrar el archivo del grupo.")
					return
				
				# Get group info for confirmation
				try:
					with open(group_file, 'r', encoding='utf-8') as f:
						group_data = json.load(f)
					metadata = group_data.get("metadata", {})
					group_name = f"L{metadata.get('linea', 'N/A')}_S{metadata.get('segmento', 'N/A')}_A{metadata.get('area', 'N/A')}_P{metadata.get('planta', 'N/A')}"
				except Exception:
					group_name = "Grupo seleccionado"
				
				# Confirm deletion
				result = messagebox.askyesno("Confirmar eliminación", 
				                           f"¿Estás seguro de que deseas eliminar el grupo '{group_name}'?\n\nEsta acción no se puede deshacer.")
				if not result:
					return
				
				try:
					import os
					os.remove(group_file)
					refresh_table()
					messagebox.showinfo("Éxito", f"Grupo '{group_name}' eliminado correctamente.")
				except Exception as e:
					messagebox.showerror("Error", f"Error al eliminar el grupo:\n{str(e)}")
			
			def refresh_table():
				"""Refresh the table with updated group data"""
				# Clear existing items
				for item in tree.get_children():
					tree.delete(item)
				group_data_map.clear()
				
				# Reload data
				group_files = list(Path("camera_groups").glob("*.json")) if Path("camera_groups").exists() else []
				for gfile in group_files:
					try:
						with open(gfile, 'r', encoding='utf-8') as f:
							data = json.load(f)
						metadata = data.get("metadata", {})
						linea = metadata.get("linea", "N/A")
						segmento = metadata.get("segmento", "N/A")
						area = metadata.get("area", "N/A")
						planta = metadata.get("planta", "N/A")
						num_camaras = len(data.get("cameras", []))
						
						item_id = tree.insert("", tk.END, values=(linea, segmento, area, planta, num_camaras))
						group_data_map[item_id] = gfile
					except Exception as e:
						print(f"[WARNING] Error loading metadata from {gfile}: {e}")
			
			# Update button layout
			cancel_btn = make_rounded_button(btn_frame, "Cancelar", on_cancel, "#666666", 
			                                 width=100, height=40, fg="white")
			cancel_btn.pack(side=tk.LEFT, padx=(0, 8))
			
			update_btn = make_rounded_button(btn_frame, "Actualizar", on_update, "#ff9800", 
			                                 width=100, height=40, fg="white")
			update_btn.pack(side=tk.LEFT, padx=(0, 8))
			
			delete_btn = make_rounded_button(btn_frame, "Borrar", on_delete, "#f44336", 
			                                 width=100, height=40, fg="white")
			delete_btn.pack(side=tk.LEFT, padx=(0, 8))
			
			load_btn = make_rounded_button(btn_frame, "Cargar y procesar", on_load, "#7ec331", 
			                               width=140, height=40, fg="white")
			load_btn.pack(side=tk.LEFT)
		
		load_group_btn = create_rounded_button(load_group_btn_frame, "Utilizar grupo guardado",
		                                       bg_color="#7ec331", active_bg="#6ba828",
		                                       padx=20, pady=10, command=load_camera_group)
		load_group_btn.pack()
		
		# Initialize with one camera
		add_new_camera()
		
		# Función para alternar entre modos
		def show_video_mode():
			rt_mode_frame.grid_remove()
			video_mode_frame.grid(row=0, column=0, sticky="nsew")
		
		def show_rt_mode():
			video_mode_frame.grid_remove()
			rt_mode_frame.grid(row=0, column=0, sticky="nsew")
		
		# Inicializar según el tipo predefinido o mostrar modo videos por defecto
		if preset_type == "rt":
			show_rt_mode()
		else:
			show_video_mode()
		
		update_tabs_state()

		# ---------------- Tab 3: ROIs ----------------
		third = tabs[1]
		third.configure(bg=PROC_CONTENT_BG)
		third.grid_rowconfigure(0, weight=1)
		third.grid_columnconfigure(0, weight=1)

		# State for ROI building
		roi_state = {
			"source_index": 0,  # índice para videos o cámaras
			"source_type": None,  # "videos" o "cameras"
			"points": [],
			"naming_mode": False,
			"finalize_after_name": False,
			"scale_x": 1.0,
			"scale_y": 1.0,
			"orig_size": (0, 0),
			"photo": None,
			"canvas_img_id": None,
			"rois_map": {},  # Para videos: basename -> [{name, coords}] | Para cámaras: camera_index -> [{name, coords}]
			"drawing_enabled": False,  # Drawing mode disabled by default
			"selected_roi": None,  # Selected ROI: {"index": int, "source_key": str/int}
			"dragging": None,  # Drag state: {"type": "roi"/"vertex", "vertex_idx": int, "start_x": int, "start_y": int}
		}

		# State for loaded group metadata
		group_metadata = {}

		# State for video list items (define before functions that use them)
		video_items = []  # List of dicts with {frame, video_path, thumbnail, selected}
		selected_video_index = {"value": 0}

		# Video list helper functions (must be defined before on_yes)
		def get_video_duration(video_path: str) -> str:
			"""Get video duration in format MM:SS"""
			try:
				import cv2
				cap = cv2.VideoCapture(video_path)
				fps = cap.get(cv2.CAP_PROP_FPS)
				frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
				cap.release()
				if fps > 0:
					duration_sec = int(frame_count / fps)
					minutes = duration_sec // 60
					seconds = duration_sec % 60
					return f"{minutes}:{seconds:02d}"
				return "0:00"
			except Exception:
				return "0:00"

		def create_video_item(video_path: str, index: int):
			"""Create a video list item with thumbnail, name, and duration"""
			# This function will be properly defined after video_list_frame is created
			pass

		def select_video(index: int):
			"""Select a video from the list and load it in the canvas"""
			# This function will be properly defined after video_items is populated
			pass

		def populate_video_list():
			"""Populate the left column with all loaded videos"""
			# This function will be properly defined after video_list_frame is created
			pass

		# Initial choice: add ROIs? - Centered layout with new design
		choice_frame = tk.Frame(third, bg=PROC_CONTENT_BG)
		choice_frame.grid(row=0, column=0, sticky="nsew")
		choice_frame.grid_rowconfigure(0, weight=1)
		choice_frame.grid_columnconfigure(0, weight=1)
		
		# Center container
		center_container = tk.Frame(choice_frame, bg=PROC_CONTENT_BG)
		center_container.grid(row=0, column=0)
		
		# Main question label
		q_lbl = tk.Label(center_container, 
		                 text="¿Deseas delimitar las zonas de interés\n(ROI) en los vídeos para el modelo?", 
		                 font=("Arial", 16, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG, justify=tk.CENTER)
		q_lbl.pack(pady=(0, 30))
		
		# Buttons container
		btns = tk.Frame(center_container, bg=PROC_CONTENT_BG)
		btns.pack(pady=(0, 20))
		
		# Helper function for rounded buttons with custom colors
		def make_roi_button(parent, text, bg_color, fg_color, command):
			"""Create a rounded button for ROI choice"""
			btn_container = tk.Frame(parent, bg=PROC_CONTENT_BG)
			
			canvas = tk.Canvas(btn_container, width=280, height=50, bg=PROC_CONTENT_BG, 
			                   highlightthickness=0, cursor="hand2")
			canvas.pack()
			
			# Draw rounded rectangle
			radius = 8
			def draw_rect(fill_color):
				canvas.delete("bg")  # Only delete background shapes, not text window
				# Rounded corners
				canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
				                  fill=fill_color, outline=fill_color, tags="bg")
				canvas.create_arc(280-radius*2, 0, 280, radius*2, start=0, extent=90, 
				                  fill=fill_color, outline=fill_color, tags="bg")
				canvas.create_arc(0, 50-radius*2, radius*2, 50, start=180, extent=90, 
				                  fill=fill_color, outline=fill_color, tags="bg")
				canvas.create_arc(280-radius*2, 50-radius*2, 280, 50, start=270, extent=90, 
				                  fill=fill_color, outline=fill_color, tags="bg")
				# Rectangles
				canvas.create_rectangle(radius, 0, 280-radius, 50, fill=fill_color, outline=fill_color, tags="bg")
				canvas.create_rectangle(0, radius, 280, 50-radius, fill=fill_color, outline=fill_color, tags="bg")
			
			draw_rect(bg_color)
			
			# Text label
			text_label = tk.Label(canvas, text=text, font=("Arial", 12, "bold"), 
			                      fg=fg_color, bg=bg_color)
			canvas.create_window(140, 25, window=text_label)
			
			# Hover effects
			def on_enter(e):
				# Darken color on hover
				hover_color = "#014a9e" if bg_color == "#015aca" else "#e5b62f"
				draw_rect(hover_color)
				text_label.configure(bg=hover_color)
			
			def on_leave(e):
				draw_rect(bg_color)
				text_label.configure(bg=bg_color)
			
			def on_click(e):
				command()
			
			canvas.bind("<Enter>", on_enter)
			canvas.bind("<Leave>", on_leave)
			canvas.bind("<Button-1>", on_click)
			text_label.bind("<Button-1>", on_click)
			
			return btn_container
		
		# Define button commands first
		def on_yes():
			"""Enable ROI builder when user chooses Yes"""
			choice_frame.grid_remove()
			builder.grid(row=0, column=0, sticky="nsew")
			# Determine source type based on processing_type
			if processing_type["value"] == "rt":
				roi_state["source_type"] = "cameras"
				videos_title.config(text="Cámaras configuradas")
				if not rtsp_urls_list:
					info_lbl.config(text="No hay cámaras configuradas.")
					can_advance[1] = True
					update_nav_state()
					return
				# Populate camera list on left column
				populate_video_list()
			else:
				roi_state["source_type"] = "videos"
				videos_title.config(text="Videos Cargados")
				if not video_list:
					info_lbl.config(text="No hay videos cargados.")
					can_advance[1] = True
					update_nav_state()
					return
				# Populate video list on left column
				populate_video_list()
			roi_state["source_index"] = 0
			set_nav_enabled_for_third()

		def on_no():
			# Skip ROIs and advance
			can_advance[1] = True
			update_nav_state()
			go_next()
		
		no_btn = make_roi_button(btns, "No, analiza todas las áreas", "#015aca", "white", on_no)
		yes_btn = make_roi_button(btns, "¡Sí, quiero delimitar zonas!", "#ffc735", "black", on_yes)
		
		no_btn.pack(side=tk.LEFT, padx=10)
		yes_btn.pack(side=tk.LEFT, padx=10)
		
		# Info label below buttons
		info_yellow = tk.Label(center_container, 
		                       text="Si eliges procesar todo el video, el modelo analizará todas las áreas sin restricciones", 
		                       font=("Arial", 11), fg="#ffff00", bg=PROC_CONTENT_BG, justify=tk.CENTER, wraplength=600)
		info_yellow.pack(pady=(0, 20))

		# ROI builder UI (hidden until 'Sí') - NEW TWO-COLUMN LAYOUT
		builder = tk.Frame(third, bg=PROC_CONTENT_BG)
		builder.grid_rowconfigure(0, weight=1)
		builder.grid_columnconfigure(0, weight=0, minsize=350)  # Left column (video list)
		builder.grid_columnconfigure(1, weight=1)  # Right column (ROI editor)

		# ============ LEFT COLUMN: Video List ============
		left_column = tk.Frame(builder, bg=PROC_CONTENT_BG, width=350)
		left_column.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
		left_column.grid_rowconfigure(1, weight=1)
		left_column.grid_columnconfigure(0, weight=1)
		left_column.grid_propagate(False)

		# Title label (dynamic based on processing type)
		videos_title = tk.Label(left_column, text="", 
		                        font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		videos_title.grid(row=0, column=0, sticky="w", pady=(0, 15))

		# Scrollable video list
		video_list_container = tk.Frame(left_column, bg=PROC_CONTENT_BG)
		video_list_container.grid(row=1, column=0, sticky="nsew")
		video_list_container.grid_rowconfigure(0, weight=1)
		video_list_container.grid_columnconfigure(0, weight=1)
		video_list_container.grid_columnconfigure(1, weight=0)
		
		# Create canvas for scrolling
		video_canvas = tk.Canvas(video_list_container, bg=PROC_CONTENT_BG, highlightthickness=0)
		video_canvas.grid(row=0, column=0, sticky="nsew")
		
		# Scrollbar with ttk style for better visibility
		style_sb = ttk.Style()
		style_sb.theme_use('clam')
		style_sb.configure("Blue.Vertical.TScrollbar",
			background="#015aca",
			troughcolor="#001f3f",
			bordercolor="#015aca",
			arrowcolor="white",
			relief=tk.FLAT,
			borderwidth=0,
			arrowsize=14,
			width=18
		)
		style_sb.map("Blue.Vertical.TScrollbar",
			background=[('active', '#0B5ED7'), ('!active', '#015aca')],
			arrowcolor=[('active', 'white'), ('!active', 'white')]
		)
		
		# Create ttk scrollbar
		video_scrollbar = ttk.Scrollbar(
			video_list_container,
			orient="vertical",
			command=video_canvas.yview,
			style="Blue.Vertical.TScrollbar"
		)
		video_scrollbar.grid(row=0, column=1, sticky="ns", padx=(3, 0))
		video_canvas.configure(yscrollcommand=video_scrollbar.set)
		
		video_list_frame = tk.Frame(video_canvas, bg=PROC_CONTENT_BG)
		video_canvas_window = video_canvas.create_window((0, 0), window=video_list_frame, anchor="nw")
		
		def update_video_list_scroll(_=None):
			video_canvas.configure(scrollregion=video_canvas.bbox("all"))
		
		def resize_video_list(_=None):
			w = video_canvas.winfo_width()
			video_canvas.itemconfigure(video_canvas_window, width=w)
			update_video_list_scroll()
		
		video_list_frame.bind("<Configure>", update_video_list_scroll)
		video_canvas.bind("<Configure>", resize_video_list)

		# Optimized mouse wheel scrolling with faster speed
		def on_mouse_wheel(event):
			# Scroll 3 units at a time for smoother and faster scrolling
			video_canvas.yview_scroll(int(-1 * (event.delta / 40)), "units")
		
		# Bind mouse wheel to canvas and all children for better responsiveness
		def bind_mouse_wheel(widget):
			widget.bind("<MouseWheel>", on_mouse_wheel)
			for child in widget.winfo_children():
				bind_mouse_wheel(child)
		
		video_canvas.bind("<MouseWheel>", on_mouse_wheel)
		video_list_frame.bind("<MouseWheel>", on_mouse_wheel)
		
		# Bind Enter/Leave events to enable/disable scrolling
		def on_enter(event):
			video_canvas.bind_all("<MouseWheel>", on_mouse_wheel)
		
		def on_leave(event):
			video_canvas.unbind_all("<MouseWheel>")
		
		video_list_container.bind("<Enter>", on_enter)
		video_list_container.bind("<Leave>", on_leave)

		# Drag-and-drop state for reordering
		drag_state = {
			"dragging": False,
			"drag_item_index": None,
			"drag_start_y": 0,
			"placeholder": None
		}

		def start_drag(item_index, event):
			"""Start dragging a video item"""
			drag_state["dragging"] = True
			drag_state["drag_item_index"] = item_index
			drag_state["drag_start_y"] = event.y_root
			
			# Highlight the dragged item
			if item_index < len(video_items):
				video_items[item_index]["frame"].configure(bg="#043c86")
		
		def on_drag(event):
			"""Handle drag motion"""
			if not drag_state["dragging"]:
				return
			
			# Calculate which position the item should be inserted at
			drag_index = drag_state["drag_item_index"]
			if drag_index is None or drag_index >= len(video_items):
				return
			
			# Get mouse position relative to the list
			mouse_y = event.y_root
			
			# Find which item we're hovering over
			for i, item in enumerate(video_items):
				if i == drag_index:
					continue
				try:
					frame_widget = item["frame"]
					frame_y = frame_widget.winfo_rooty()
					frame_height = frame_widget.winfo_height()
					
					if frame_y <= mouse_y <= frame_y + frame_height:
						# Hovering over this item - show visual feedback
						frame_widget.configure(highlightbackground="#ffc735", highlightthickness=3)
					else:
						# Reset to normal
						update_video_item_border(i)
				except Exception:
					pass
		
		def end_drag(event):
			"""End dragging and reorder items"""
			if not drag_state["dragging"]:
				return
			
			drag_index = drag_state["drag_item_index"]
			if drag_index is None or drag_index >= len(video_items):
				drag_state["dragging"] = False
				return
			
			# Find drop position
			mouse_y = event.y_root
			drop_index = drag_index
			
			for i, item in enumerate(video_items):
				if i == drag_index:
					continue
				try:
					frame_widget = item["frame"]
					frame_y = frame_widget.winfo_rooty()
					frame_height = frame_widget.winfo_height()
					
					if frame_y <= mouse_y <= frame_y + frame_height:
						drop_index = i
						break
				except Exception:
					pass
			
			# Reorder the list
			if drop_index != drag_index:
				# Move item in video_items list
				item_to_move = video_items.pop(drag_index)
				video_items.insert(drop_index, item_to_move)
				
				# Update the corresponding list (video_list or rtsp_urls_list)
				if roi_state["source_type"] == "videos":
					video_to_move = video_list.pop(drag_index)
					video_list.insert(drop_index, video_to_move)
				else:
					cam_to_move = rtsp_urls_list.pop(drag_index)
					rtsp_urls_list.insert(drop_index, cam_to_move)
				
				# Rebuild the visual list
				populate_video_list()
				
				# Select the moved item at its new position
				select_video(drop_index)
			else:
				# Just reset the dragged item appearance
				update_video_item_border(drag_index)
			
			drag_state["dragging"] = False
			drag_state["drag_item_index"] = None

		# Now define the actual implementations of video list functions
		def load_frame_at_index(video_path: str, frame_idx: int):
			"""Load a specific frame from a video file"""
			try:
				import cv2
				cap = cv2.VideoCapture(video_path)
				
				# Set frame position
				cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
				ret, frame = cap.read()
				total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				cap.release()
				
				if ret and frame is not None:
					return frame, total_frames
				return None, total_frames
			except Exception as e:
				return None, 0
		
		def navigate_frame(item_index: int, direction: int):
			"""Navigate to previous (-1) or next (+1) frame for a video item"""
			FRAME_SKIP_AMOUNT = 25  # Number of frames to skip per button press
			
			if item_index >= len(video_items):
				return
			
			item_data = video_items[item_index]
			video_path = item_data["video_path"]
			
			# Skip cameras
			if video_path.startswith("rtsp://"):
				return
			
			# Get current frame index
			current_frame = item_data.get("frame_index", 0)
			total_frames = item_data.get("total_frames", 1)
			
			# Calculate new frame index with configurable skip amount
			new_frame = max(0, min(current_frame + (direction * FRAME_SKIP_AMOUNT), total_frames - 1))
			
			if new_frame == current_frame:
				return  # No change
			
			# Load new frame
			frame, _ = load_frame_at_index(video_path, new_frame)
			if frame is None:
				return
			
			# Update frame index
			item_data["frame_index"] = new_frame
			
			# Store original frame for ROI overlay
			item_data["original_frame"] = frame.copy()
			
			# Update thumbnail
			thumb_label = item_data.get("thumbnail_label")
			if thumb_label:
				try:
					import cv2
					from PIL import Image, ImageTk as _ImageTk
					
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					img = Image.fromarray(frame_rgb)
					img.thumbnail((80, 60), Image.LANCZOS)
					photo = _ImageTk.PhotoImage(img)
					
					thumb_label.configure(image=photo)
					thumb_label.image = photo
					item_data["thumbnail"] = photo
				except Exception:
					pass
			
			# If this video is currently selected, reload it in the canvas
			if item_data.get("selected", False):
				load_first_frame_of(item_index)
		
		def create_video_item(video_path: str, index: int):
			"""Create a video/camera list item with thumbnail, name, and duration"""
			item_frame = tk.Frame(video_list_frame, bg="#002858", cursor="hand2", 
			                      highlightthickness=2, highlightbackground="#002858")
			item_frame.pack(fill=tk.X, pady=(0, 10))
			item_frame.grid_columnconfigure(1, weight=1)
			
			# Store reference
			item_data = {"frame": item_frame, "video_path": video_path, "index": index, 
			             "selected": False, "thumbnail_label": None, "original_frame": None,
			             "frame_index": 0, "total_frames": 1}
			video_items.append(item_data)
			
			# Check if this is a camera (RTSP URL) or video file
			is_camera = video_path.startswith("rtsp://")
			
			# Create thumbnail container frame for buttons overlay
			thumb_container = tk.Frame(item_frame, bg="#002858")
			thumb_container.grid(row=0, column=0, rowspan=2, padx=10, pady=5)
			
			# Try to get thumbnail
			thumb_label = None
			try:
				import cv2
				cap = cv2.VideoCapture(video_path)
				
				# For cameras, wait a bit for stream to stabilize
				if is_camera:
					for _ in range(5):  # Skip first few frames
						cap.read()
				else:
					# For videos, get total frame count
					total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
					item_data["total_frames"] = max(1, total_frames)
				
				
				# Try to read a valid frame, skipping corrupted ones
				ret, frame = False, None
				frame_attempt = 0
				max_attempts = 10  # Try up to 10 positions (0, 25, 50, ..., 225)
				
				while not ret or frame is None:
					if frame_attempt >= max_attempts:
						break
					
					# Set frame position
					if frame_attempt > 0 and not is_camera:
						cap.set(cv2.CAP_PROP_POS_FRAMES, frame_attempt * 25)
					
					ret, frame = cap.read()
					frame_attempt += 1
				
				# Update frame_index to reflect the actual frame used
				if ret and frame is not None and not is_camera:
					item_data["frame_index"] = (frame_attempt - 1) * 25
				
				cap.release()
				
				if ret and frame is not None:
					# Store original frame for ROI overlay
					item_data["original_frame"] = frame.copy()
					
					# Resize to thumbnail
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					from PIL import Image, ImageTk as _ImageTk
					img = Image.fromarray(frame_rgb)
					img.thumbnail((80, 60), Image.LANCZOS)
					photo = _ImageTk.PhotoImage(img)
					
					thumb_label = tk.Label(thumb_container, image=photo, bg="#002858")
					thumb_label.image = photo  # Keep reference
					thumb_label.pack()
					item_data["thumbnail"] = photo
					item_data["thumbnail_label"] = thumb_label
					
					# Add navigation buttons for videos only
					if not is_camera:
						# Left button "<"
						left_btn = tk.Canvas(thumb_container, width=20, height=30, bg="#002858", 
						                     highlightthickness=0, cursor="hand2")
						left_btn.place(x=0, y=15)
						# Semi-transparent background (50% opacity simulation with gray)
						left_btn.create_rectangle(0, 0, 20, 30, fill="#404040", outline="", stipple="gray50")
						left_btn.create_text(10, 15, text="<", font=("Arial", 16, "bold"), fill="white")
						left_btn.bind("<Button-1>", lambda e: navigate_frame(index, -1))
						
						# Right button ">"
						right_btn = tk.Canvas(thumb_container, width=20, height=30, bg="#002858", 
						                      highlightthickness=0, cursor="hand2")
						right_btn.place(x=60, y=15)
						# Semi-transparent background (50% opacity simulation with gray)
						right_btn.create_rectangle(0, 0, 20, 30, fill="#404040", outline="", stipple="gray50")
						right_btn.create_text(10, 15, text=">", font=("Arial", 16, "bold"), fill="white")
						right_btn.bind("<Button-1>", lambda e: navigate_frame(index, 1))
				else:
					# Placeholder if no frame
					thumb_label = tk.Label(thumb_container, text="🎬", font=("Arial", 24), 
					                       bg="#002858", fg=FG_COLOR, width=5, height=2)
					thumb_label.pack()
					item_data["thumbnail_label"] = thumb_label
			except Exception as e:
				# Fallback placeholder
				thumb_label = tk.Label(thumb_container, text="🎬", font=("Arial", 24), 
				                       bg="#002858", fg=FG_COLOR, width=5, height=2)
				thumb_label.pack()
				item_data["thumbnail_label"] = thumb_label
			
			# Display name
			if is_camera:
				# For cameras, show a friendly name from rtsp_urls_list
				if index < len(rtsp_urls_list):
					cam_data = rtsp_urls_list[index]
					video_name = cam_data.get("np", f"Cámara {index + 1}")
				else:
					video_name = f"Cámara {index + 1}"
			else:
				video_name = Path(video_path).name
			
			name_label = tk.Label(item_frame, text=video_name, font=("Arial", 11, "bold"),
			                      fg=FG_COLOR, bg="#002858", anchor="w")
			name_label.grid(row=0, column=1, sticky="w", padx=5, pady=(5, 0))
			
			# Duration or connection info
			if is_camera:
				duration_text = "Cámara en vivo"
			else:
				duration_text = get_video_duration(video_path)
			
			duration_label = tk.Label(item_frame, text=duration_text, font=("Arial", 9),
			                          fg="#5BA8C9", bg="#002858", anchor="w")
			duration_label.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 5))
			
			# Add drag handle (hamburger menu icon) on the right
			drag_handle = tk.Canvas(item_frame, width=30, height=50, bg="#002858", 
			                        highlightthickness=0, cursor="hand2")
			drag_handle.grid(row=0, column=2, rowspan=2, padx=(0, 10), pady=5)
			
			# Draw three horizontal lines for hamburger menu
			for i in range(3):
				y_pos = 15 + (i * 8)
				drag_handle.create_line(5, y_pos, 25, y_pos, fill="#5BA8C9", width=2)
			
			# Bind drag events to the handle
			drag_handle.bind("<Button-1>", lambda e, idx=index: start_drag(idx, e))
			drag_handle.bind("<B1-Motion>", on_drag)
			drag_handle.bind("<ButtonRelease-1>", end_drag)
			
			# Bind click to select video
			def on_item_click(_=None):
				select_video(index)
			
			item_frame.bind("<Button-1>", on_item_click)
			if thumb_label:
				thumb_label.bind("<Button-1>", on_item_click)
			name_label.bind("<Button-1>", on_item_click)
			duration_label.bind("<Button-1>", on_item_click)
			
			# Update border color based on ROI status
			update_video_item_border(index)
			
			return item_frame

		def update_video_item_border(index: int):
			"""Update video item border color based on ROI status"""
			if index >= len(video_items):
				return
			
			item_data = video_items[index]
			video_path = item_data["video_path"]
			bn = Path(video_path).name
			
			# Check if video has ROIs
			has_rois = bn in roi_state["rois_map"] and len(roi_state["rois_map"][bn]) > 0
			
			# Update border color
			border_color = "#00F4B0" if has_rois else "#002858"
			item_data["frame"].configure(highlightbackground=border_color)

		def update_video_thumbnail_with_rois(index: int):
			"""Redraw video thumbnail with ROI overlays"""
			if index >= len(video_items):
				return
			
			item_data = video_items[index]
			original_frame = item_data.get("original_frame")
			thumb_label = item_data.get("thumbnail_label")
			
			if original_frame is None or thumb_label is None:
				return
			
			try:
				import cv2
				import numpy as np
				from PIL import Image, ImageTk as _ImageTk, ImageDraw
				
				# Get ROIs for this video
				video_path = item_data["video_path"]
				bn = Path(video_path).name
				rois = roi_state["rois_map"].get(bn, [])
				
				# Create a copy to draw on
				frame_with_rois = original_frame.copy()
				h, w = frame_with_rois.shape[:2]
				
				# Draw ROIs
				for roi in rois:
					coords = roi.get("coords", [])
					if len(coords) >= 6:  # At least 3 points
						# Convert to points list
						points = []
						for i in range(0, len(coords), 2):
							x = int(coords[i])
							y = int(coords[i+1])
							points.append((x, y))
						
						# Draw polygon
						pts = np.array(points, np.int32)
						cv2.polylines(frame_with_rois, [pts], True, (0, 244, 176), 2)  # #00F4B0 in BGR
				
				# Convert to RGB and thumbnail
				frame_rgb = cv2.cvtColor(frame_with_rois, cv2.COLOR_BGR2RGB)
				img = Image.fromarray(frame_rgb)
				img.thumbnail((80, 60), Image.LANCZOS)
				photo = _ImageTk.PhotoImage(img)
				
				# Update thumbnail
				thumb_label.configure(image=photo)
				thumb_label.image = photo  # Keep reference
				item_data["thumbnail"] = photo
				
			except Exception as e:
				pass  # Silently fail if can't update thumbnail

		# Stub functions - will be redefined after canvas widgets are created
		def select_video(index: int):
			pass

		def populate_video_list():
			pass

		def set_nav_enabled_for_third():
			pass

		def save_all_delimitaciones():
			pass

		# Save all button at bottom
		save_all_btn = make_rounded_button(left_column, "Guardar Delimitaciones", 
		                                    lambda: None, "#ffc735", fg="#000000", width=310, height=40)
		save_all_btn.grid(row=2, column=0, sticky="ew", pady=(15, 0))

		# ============ RIGHT COLUMN: ROI Editor ============
		right_column = tk.Frame(builder, bg=PROC_CONTENT_BG)
		right_column.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
		right_column.grid_rowconfigure(2, weight=1)
		right_column.grid_columnconfigure(0, weight=1)

		# Title label
		roi_title = tk.Label(right_column, text="Zonas de Interés", 
		                     font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		roi_title.grid(row=0, column=0, sticky="w", pady=(0, 15))

		# Button row
		button_row = tk.Frame(right_column, bg=PROC_CONTENT_BG)
		button_row.grid(row=1, column=0, sticky="ew", pady=(0, 15))

		use_prev_btn = make_rounded_button(button_row, "Usar ROI Anterior", lambda: None, "#7030a0", width=150, height=36)
		delete_roi_btn = make_rounded_button(button_row, "Eliminar ROI", lambda: None, "#ec5b2d", width=150, height=36)
		add_roi_btn = make_rounded_button(button_row, "Agregar ROI", lambda: None, "#015aca", width=150, height=36)
		save_video_btn = make_rounded_button(button_row, "Guardar Video", lambda: None, "#ffc735", fg="#000000", width=150, height=36)

		use_prev_btn.pack(side=tk.LEFT, padx=(0, 10))
		delete_roi_btn.pack(side=tk.LEFT, padx=(0, 10))
		add_roi_btn.pack(side=tk.LEFT, padx=(0, 10))
		save_video_btn.pack(side=tk.LEFT)

		# Canvas for video frame and ROI drawing
		canvas_container = tk.Frame(right_column, bg="#000000")
		canvas_container.grid(row=2, column=0, sticky="nsew")
		
		roi_canvas = tk.Canvas(canvas_container, bg="#000000", highlightthickness=0, cursor="arrow")
		roi_canvas.pack(fill=tk.BOTH, expand=True)

		# ROI naming panel (shown when adding ROI)
		roi_naming_panel = tk.Frame(right_column, bg=PROC_CONTENT_BG)
		roi_naming_panel.grid(row=3, column=0, sticky="ew", pady=(10, 0))
		roi_naming_panel.grid_remove()  # Hidden by default

		# Cancel button (X)
		cancel_roi_btn = make_rounded_button(roi_naming_panel, "✕", lambda: None, "#ec5b2d", width=40, height=40)
		cancel_roi_btn.pack(side=tk.LEFT, padx=(0, 10))

		# Undo button (with arrow icon) - using same style as cancel/confirm buttons
		undo_icon_path = assets_root / "NGUI" / "deshacer-flecha.png"
		undo_icon_img = load_icon(undo_icon_path, 20, 20, invert=False)
		
		# Create undo button with make_rounded_button for consistent styling
		undo_roi_btn = make_rounded_button(roi_naming_panel, "", lambda: None, "#7030a0", width=40, height=40)
		undo_roi_btn.pack(side=tk.LEFT, padx=(0, 10))
		
		# Add icon to the button canvas with persistent reference
		undo_btn_canvas = None
		if undo_icon_img:
			# Find the canvas widget inside the button
			for widget in undo_roi_btn.winfo_children():
				if isinstance(widget, tk.Canvas):
					undo_btn_canvas = widget
					# Store the icon reference to prevent garbage collection
					widget.icon_img = undo_icon_img
					# Add icon centered on the canvas
					widget.create_image(20, 20, image=undo_icon_img, tags="undo_icon")
					
					# Override the canvas's internal redraw to preserve the icon
					original_redraw = widget.configure if hasattr(widget, '_original_configure') else None
					
					def redraw_with_icon():
						# Redraw the icon after any canvas updates
						if hasattr(widget, 'icon_img') and widget.icon_img:
							# Remove old icon and redraw
							widget.delete("undo_icon")
							widget.create_image(20, 20, image=widget.icon_img, tags="undo_icon")
					
					# Store original configure for later use
					widget._redraw_icon = redraw_with_icon
					break

		# Name input
		roi_name_entry = tk.Entry(roi_naming_panel, font=("Arial", 11), bg="#FFFFFF", fg="#000000", 
		                          insertbackground="#000000", relief="solid", bd=1)
		roi_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

		# Confirm button (checkmark) - disabled initially
		confirm_roi_btn = make_rounded_button(roi_naming_panel, "✓", lambda: None, "#7ec331", width=40, height=40)
		confirm_roi_btn.pack(side=tk.LEFT)
		confirm_roi_btn.config(state="disabled")  # Disabled until 3 points

		# Error label for validation messages
		error_lbl = tk.Label(right_column, text="", font=("Arial", 9), fg="#FF0000", bg=PROC_CONTENT_BG)
		error_lbl.grid(row=4, column=0, sticky="w", pady=(5, 0))

		# Info label for instructions/status
		info_lbl = tk.Label(right_column, text="", font=("Arial", 10), fg="#5BA8C9", bg=PROC_CONTENT_BG)
		info_lbl.grid(row=5, column=0, sticky="w", pady=(5, 0))

		# Hide builder initially
		builder.grid_remove()

		# Now define canvas-related functions (after roi_canvas, canvas_container, info_lbl exist)
		def get_current_saved_rois():
			"""Get saved ROIs for current source"""
			src_idx = roi_state.get("source_index", 0)
			if roi_state["source_type"] == "videos" and 0 <= src_idx < len(video_list):
				bn = Path(video_list[src_idx]).name
				return roi_state["rois_map"].get(bn, []), bn
			elif roi_state["source_type"] == "cameras" and 0 <= src_idx < len(rtsp_urls_list):
				return roi_state["rois_map"].get(src_idx, []), src_idx
			return [], None

		def point_in_polygon(x, y, poly_points):
			"""Check if point (x,y) is inside polygon using ray casting"""
			n = len(poly_points)
			inside = False
			j = n - 1
			for i in range(n):
				xi, yi = poly_points[i]
				xj, yj = poly_points[j]
				if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
					inside = not inside
				j = i
			return inside

		def distance_to_point(x1, y1, x2, y2):
			"""Calculate distance between two points"""
			return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

		def find_clicked_roi(x, y):
			"""Find which ROI was clicked (returns index and type: 'vertex' or 'inside')"""
			saved, _ = get_current_saved_rois()
			if not saved:
				return None, None, None
			
			sx = roi_state.get("scale_x", 1.0) or 1.0
			sy = roi_state.get("scale_y", 1.0) or 1.0
			
			# Check from last to first (top to bottom in draw order)
			for roi_idx in range(len(saved) - 1, -1, -1):
				roi = saved[roi_idx]
				coords = roi.get("coords", [])
				pts_disp = []
				for i in range(0, len(coords), 2):
					xd = int(round(coords[i] * sx))
					yd = int(round(coords[i+1] * sy))
					pts_disp.append((xd, yd))
				
				# Check if clicked on vertex (within 8 pixels)
				for v_idx, (px, py) in enumerate(pts_disp):
					if distance_to_point(x, y, px, py) <= 8:
						return roi_idx, "vertex", v_idx
				
				# Check if clicked inside polygon
				if point_in_polygon(x, y, pts_disp):
					return roi_idx, "inside", None
			
			return None, None, None

		def redraw_overlay():
			roi_canvas.delete("overlay")
			# Draw previously saved ROIs for current source (scaled)
			saved, source_key = get_current_saved_rois()
			selected = roi_state.get("selected_roi")
			
			if saved:
				sx = roi_state.get("scale_x", 1.0) or 1.0
				sy = roi_state.get("scale_y", 1.0) or 1.0
				for roi_idx, roi in enumerate(saved):
					coords = roi.get("coords", [])
					name = roi.get("name", "ROI")
					pts_disp = []
					for i in range(0, len(coords), 2):
						xd = int(round(coords[i] * sx))
						yd = int(round(coords[i+1] * sy))
						pts_disp.append((xd, yd))
					
					# Check if this ROI is selected
					is_selected = (selected and selected.get("index") == roi_idx and 
					               selected.get("source_key") == source_key)
					
					# Draw lines
					line_color = "#00FF00" if is_selected else "#FFD54A"
					line_width = 3 if is_selected else 2
					for j in range(len(pts_disp)):
						x0, y0 = pts_disp[j]
						x1, y1 = pts_disp[(j+1) % len(pts_disp)]
						roi_canvas.create_line(x0, y0, x1, y1, fill=line_color, width=line_width, tags="overlay")
					
					# Draw vertices
					vertex_size = 6 if is_selected else 2
					for x, y in pts_disp:
						roi_canvas.create_oval(x-vertex_size, y-vertex_size, x+vertex_size, y+vertex_size, 
						                       fill=line_color, outline="", tags="overlay")
					
					# Draw name
					if pts_disp:
						xc = sum(p[0] for p in pts_disp) // len(pts_disp)
						yc = sum(p[1] for p in pts_disp) // len(pts_disp)
						roi_canvas.create_text(xc, yc, text=name, fill="#FFFFFF", 
						                       font=("Arial", 10, "bold"), tags="overlay")
			
			# In-progress polygon on stretched image
			pts = roi_state["points"]
			for i, (x, y) in enumerate(pts):
				roi_canvas.create_oval(x-3, y-3, x+3, y+3, fill="#00FFFF", outline="", tags="overlay")
				if i > 0:
					x0, y0 = pts[i-1]
					roi_canvas.create_line(x0, y0, x, y, fill="#00FFFF", width=2, tags="overlay")
			if roi_state["naming_mode"] and len(pts) >= 3:
				x0, y0 = pts[0]
				x1, y1 = pts[-1]
				roi_canvas.create_line(x1, y1, x0, y0, fill="#00FFFF", width=2, tags="overlay")

		def on_canvas_click(event):
			x, y = event.x, event.y
			
			# If drawing mode is active, add points
			if roi_state.get("drawing_enabled", False):
				if roi_state["naming_mode"]:
					return
				roi_state["points"].append((x, y))
				redraw_overlay()
				
				# Enable confirm button when 3+ points
				if len(roi_state["points"]) >= 3:
					confirm_roi_btn.config(state="normal")
				else:
					confirm_roi_btn.config(state="disabled")
				return
			
			# Otherwise, check for ROI selection
			roi_idx, click_type, vertex_idx = find_clicked_roi(x, y)
			
			if roi_idx is not None:
				# Select the ROI
				_, source_key = get_current_saved_rois()
				roi_state["selected_roi"] = {"index": roi_idx, "source_key": source_key}
				
				# Start dragging
				roi_state["dragging"] = {
					"type": click_type,
					"vertex_idx": vertex_idx,
					"start_x": x,
					"start_y": y
				}
				redraw_overlay()
			else:
				# Deselect if clicked on empty space
				if roi_state["selected_roi"]:
					roi_state["selected_roi"] = None
					redraw_overlay()

		def on_canvas_drag(event):
			"""Handle dragging of ROI or vertex"""
			if not roi_state.get("dragging"):
				return
			
			drag_info = roi_state["dragging"]
			selected = roi_state.get("selected_roi")
			if not selected:
				return
			
			saved, source_key = get_current_saved_rois()
			if not saved or selected["index"] >= len(saved):
				return
			
			roi = saved[selected["index"]]
			coords = roi["coords"]
			sx = roi_state.get("scale_x", 1.0) or 1.0
			sy = roi_state.get("scale_y", 1.0) or 1.0
			
			dx = event.x - drag_info["start_x"]
			dy = event.y - drag_info["start_y"]
			
			if drag_info["type"] == "vertex":
				# Move single vertex
				v_idx = drag_info["vertex_idx"]
				# Convert display delta to original coords
				orig_dx = dx / sx
				orig_dy = dy / sy
				coords[v_idx * 2] += orig_dx
				coords[v_idx * 2 + 1] += orig_dy
			elif drag_info["type"] == "inside":
				# Move entire ROI
				orig_dx = dx / sx
				orig_dy = dy / sy
				for i in range(0, len(coords), 2):
					coords[i] += orig_dx
					coords[i + 1] += orig_dy
			
			# Update start position for next drag event
			drag_info["start_x"] = event.x
			drag_info["start_y"] = event.y
			
			redraw_overlay()

		def on_canvas_release(event):
			"""Stop dragging and update thumbnails"""
			if roi_state.get("dragging"):
				# Update thumbnails after finishing the drag
				if roi_state["source_type"] == "videos":
					src_idx = roi_state.get("source_index")
					if src_idx is not None:
						update_video_item_border(src_idx)
						update_video_thumbnail_with_rois(src_idx)
			
			roi_state["dragging"] = None

		roi_canvas.bind("<Button-1>", on_canvas_click)
		roi_canvas.bind("<B1-Motion>", on_canvas_drag)
		roi_canvas.bind("<ButtonRelease-1>", on_canvas_release)

		def delete_selected_roi():
			"""Delete the currently selected ROI"""
			selected = roi_state.get("selected_roi")
			if not selected:
				info_lbl.config(text="No hay ROI seleccionado", fg="#FF0000")
				return
			
			saved, source_key = get_current_saved_rois()
			if not saved or selected["index"] >= len(saved):
				return
			
			# Get ROI name before deleting
			roi_name = saved[selected["index"]].get("name", "ROI")
			
			# Remove from list
			del saved[selected["index"]]
			
			# Clear selection
			roi_state["selected_roi"] = None
			
			# Update display
			redraw_overlay()
			info_lbl.config(text=f"✓ ROI '{roi_name}' eliminado", fg="#5BA8C9")
			
			# Update video item thumbnail and border
			if roi_state["source_type"] == "videos":
				src_idx = roi_state.get("source_index", 0)
				if 0 <= src_idx < len(video_list):
					update_video_item_border(src_idx)
					update_video_thumbnail_with_rois(src_idx)

		def start_drawing_roi():
			"""Enable drawing mode and show naming panel"""
			roi_state["drawing_enabled"] = True
			roi_state["points"] = []
			roi_state["naming_mode"] = False
			roi_state["selected_roi"] = None  # Deselect any selected ROI
			roi_canvas.config(cursor="crosshair")
			roi_naming_panel.grid()
			roi_name_entry.delete(0, tk.END)
			error_lbl.config(text="")
			confirm_roi_btn.config(state="disabled")
			info_lbl.config(text="Haga clic en el canvas para agregar puntos al ROI (mínimo 3)", fg="#5BA8C9")
			redraw_overlay()

		def cancel_drawing_roi():
			"""Cancel ROI drawing and hide naming panel"""
			roi_state["drawing_enabled"] = False
			roi_state["points"] = []
			roi_state["naming_mode"] = False
			roi_canvas.config(cursor="arrow")
			roi_naming_panel.grid_remove()
			error_lbl.config(text="")
			info_lbl.config(text="", fg="#5BA8C9")
			redraw_overlay()

		def undo_last_point():
			"""Remove the last point from the current ROI being drawn"""
			if not roi_state.get("drawing_enabled", False):
				return
			
			if roi_state["naming_mode"]:
				return
			
			pts = roi_state["points"]
			if len(pts) > 0:
				# Remove last point
				pts.pop()
				redraw_overlay()
				
				# Update confirm button state
				if len(pts) >= 3:
					confirm_roi_btn.config(state="normal")
				else:
					confirm_roi_btn.config(state="disabled")
				
				# Update info message
				if len(pts) == 0:
					info_lbl.config(text="Haga clic en el canvas para agregar puntos al ROI (mínimo 3)", fg="#5BA8C9")
				else:
					info_lbl.config(text=f"Puntos actuales: {len(pts)} (mínimo 3 requeridos)", fg="#5BA8C9")

		def confirm_roi():
			"""Save the ROI with the entered name"""
			if len(roi_state["points"]) < 3:
				return
			
			# Validate name
			roi_name = roi_name_entry.get().strip()
			if not roi_name:
				error_lbl.config(text="⚠ El nombre del ROI no puede estar vacío")
				return
			
			# Save ROI
			pts = roi_state["points"]
			scale_x = roi_state.get("scale_x", 1.0)
			scale_y = roi_state.get("scale_y", 1.0)
			
			# Convert display coords to original image coords
			coords = []
			for x, y in pts:
				orig_x = int(round(x / scale_x))
				orig_y = int(round(y / scale_y))
				coords.extend([orig_x, orig_y])
			
			# Add to saved ROIs
			src_idx = roi_state.get("source_index", 0)
			if roi_state["source_type"] == "videos" and 0 <= src_idx < len(video_list):
				bn = Path(video_list[src_idx]).name
				if bn not in roi_state["rois_map"]:
					roi_state["rois_map"][bn] = []
				roi_state["rois_map"][bn].append({"name": roi_name, "coords": coords})
			elif roi_state["source_type"] == "cameras" and 0 <= src_idx < len(rtsp_urls_list):
				if src_idx not in roi_state["rois_map"]:
					roi_state["rois_map"][src_idx] = []
				roi_state["rois_map"][src_idx].append({"name": roi_name, "coords": coords})
			
			# Reset drawing mode
			cancel_drawing_roi()
			info_lbl.config(text=f"✓ ROI '{roi_name}' guardado exitosamente", fg="#5BA8C9")
			
			# Update video item thumbnail and border
			if roi_state["source_type"] == "videos":
				update_video_item_border(src_idx)
				update_video_thumbnail_with_rois(src_idx)

		def use_previous_roi():
			"""Copy ROIs from the previous video to the current one"""
			src_idx = roi_state.get("source_index", 0)
			
			if roi_state["source_type"] == "videos":
				# Find previous video index
				if src_idx <= 0 or src_idx >= len(video_list):
					info_lbl.config(text="⚠ No hay video anterior disponible", fg="#FF0000")
					return
				
				prev_idx = src_idx - 1
				current_bn = Path(video_list[src_idx]).name
				prev_bn = Path(video_list[prev_idx]).name
				
				# Check if previous video has ROIs
				prev_rois = roi_state["rois_map"].get(prev_bn, [])
				if not prev_rois:
					info_lbl.config(text=f"⚠ El video anterior '{prev_bn}' no tiene ROIs definidos", fg="#FF0000")
					return
				
				# Copy ROIs (deep copy to avoid reference issues)
				import copy
				roi_state["rois_map"][current_bn] = copy.deepcopy(prev_rois)
				
				# Deselect any selected ROI
				roi_state["selected_roi"] = None
				
				# Update display
				redraw_overlay()
				info_lbl.config(text=f"✓ {len(prev_rois)} ROI(s) copiados desde '{prev_bn}'", fg="#5BA8C9")
				
				# Update video item thumbnail and border
				update_video_item_border(src_idx)
				update_video_thumbnail_with_rois(src_idx)
			
			elif roi_state["source_type"] == "cameras":
				# For cameras, copy from previous camera index
				if src_idx <= 0 or src_idx >= len(rtsp_urls_list):
					info_lbl.config(text="⚠ No hay cámara anterior disponible", fg="#FF0000")
					return
				
				prev_idx = src_idx - 1
				
				# Check if previous camera has ROIs
				prev_rois = roi_state["rois_map"].get(prev_idx, [])
				if not prev_rois:
					info_lbl.config(text=f"⚠ La cámara anterior (#{prev_idx + 1}) no tiene ROIs definidos", fg="#FF0000")
					return
				
				# Copy ROIs
				import copy
				roi_state["rois_map"][src_idx] = copy.deepcopy(prev_rois)
				
				# Deselect any selected ROI
				roi_state["selected_roi"] = None
				
				# Update display
				redraw_overlay()
				info_lbl.config(text=f"✓ {len(prev_rois)} ROI(s) copiados desde cámara #{prev_idx + 1}", fg="#5BA8C9")

		# Connect buttons to functions
		add_roi_btn.configure(command=start_drawing_roi)
		delete_roi_btn.configure(command=delete_selected_roi)
		use_prev_btn.configure(command=use_previous_roi)
		cancel_roi_btn.configure(command=cancel_drawing_roi)
		undo_roi_btn.configure(command=undo_last_point)
		confirm_roi_btn.configure(command=confirm_roi)

		def check_all_videos_have_rois():
			"""Check if all videos have at least one ROI defined"""
			if roi_state["source_type"] != "videos":
				return True  # Not in video mode
			
			for video_path in video_list:
				bn = Path(video_path).name
				if bn not in roi_state["rois_map"] or len(roi_state["rois_map"][bn]) == 0:
					return False
			return True

		def apply_last_rois_to_remaining_videos():
			"""Apply ROIs from last defined video to all videos without ROIs"""
			if roi_state["source_type"] != "videos":
				return
			
			# Find the last video with ROIs
			last_video_with_rois = None
			last_rois = None
			
			for video_path in video_list:
				bn = Path(video_path).name
				if bn in roi_state["rois_map"] and len(roi_state["rois_map"][bn]) > 0:
					last_video_with_rois = bn
					last_rois = roi_state["rois_map"][bn]
			
			if not last_rois:
				return  # No video has ROIs
			
			# Apply to all videos without ROIs
			import copy
			applied_count = 0
			for idx, video_path in enumerate(video_list):
				bn = Path(video_path).name
				if bn not in roi_state["rois_map"] or len(roi_state["rois_map"][bn]) == 0:
					roi_state["rois_map"][bn] = copy.deepcopy(last_rois)
					# Update visual indicators
					update_video_item_border(idx)
					update_video_thumbnail_with_rois(idx)
					applied_count += 1
			
			# Show notification
			if applied_count > 0:
				info_lbl.config(text=f"✓ ROIs aplicados automáticamente a {applied_count} video(s)", fg="#5BA8C9")

		def save_all_delimitaciones():
			"""Save all ROI delimitaciones to perma_rois.json"""
			# Check if all videos have ROIs
			if roi_state["source_type"] == "videos" and not check_all_videos_have_rois():
				# Show confirmation dialog
				show_incomplete_rois_dialog()
			else:
				# All videos have ROIs or not in video mode, save directly
				perform_save_delimitaciones()

		def show_incomplete_rois_dialog():
			"""Show dialog asking user what to do when not all videos have ROIs"""
			dialog = tk.Toplevel(third)
			dialog.title("Delimitaciones Incompletas")
			dialog.configure(bg=PROC_CONTENT_BG)
			dialog.resizable(False, False)
			
			# Center on parent
			dialog.transient(third)
			dialog.grab_set()
			
			# Calculate position
			dialog.update_idletasks()
			parent_x = third.winfo_rootx()
			parent_y = third.winfo_rooty()
			parent_w = third.winfo_width()
			parent_h = third.winfo_height()
			
			dialog_w = 850
			dialog_h = 180
			
			x = parent_x + (parent_w - dialog_w) // 2
			y = parent_y + (parent_h - dialog_h) // 2
			
			dialog.geometry(f"{dialog_w}x{dialog_h}+{x}+{y}")
			
			# Content frame
			content = tk.Frame(dialog, bg=PROC_CONTENT_BG)
			content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
			
			# Message
			msg_label = tk.Label(content, 
			                     text="No todos los videos tienen delimitaciones,\n¿Deseas utilizar las últimas delimitaciones para los videos restantes?",
			                     font=("Arial", 11, "bold"),
			                     fg=FG_COLOR,
			                     bg=PROC_CONTENT_BG,
			                     justify=tk.CENTER)
			msg_label.pack(pady=(0, 20))
			
			# Button frame
			btn_frame = tk.Frame(content, bg=PROC_CONTENT_BG)
			btn_frame.pack(fill=tk.X)
			
			def on_auto_delimit():
				apply_last_rois_to_remaining_videos()
				dialog.destroy()
				perform_save_delimitaciones()
			
			def on_manual():
				dialog.destroy()
			
			def make_left_aligned_button(parent, text, command, bg_color, fg_color, width, height):
				"""Create button with left-aligned text"""
				container = tk.Frame(parent, bg=PROC_CONTENT_BG)
				canvas = tk.Canvas(container, width=width, height=height, bg=PROC_CONTENT_BG, 
				                   highlightthickness=0, cursor="hand2")
				canvas.pack()
				
				def draw_button(color):
					canvas.delete("all")
					radius = 8
					canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
					                  fill=color, outline=color)
					canvas.create_arc(width-radius*2, 0, width, radius*2, start=0, extent=90, 
					                  fill=color, outline=color)
					canvas.create_arc(0, height-radius*2, radius*2, height, start=180, extent=90, 
					                  fill=color, outline=color)
					canvas.create_arc(width-radius*2, height-radius*2, width, height, start=270, extent=90, 
					                  fill=color, outline=color)
					canvas.create_rectangle(radius, 0, width-radius, height, fill=color, outline=color)
					canvas.create_rectangle(0, radius, width, height-radius, fill=color, outline=color)
					# Text aligned to left with padding
					canvas.create_text(20, height/2, text=text, font=("Arial", 11, "bold"), 
					                   fill=fg_color, anchor="w")
				
				draw_button(bg_color)
				
				def on_enter(e):
					draw_button(bg_color)
				
				def on_leave(e):
					draw_button(bg_color)
				
				def on_click(e):
					if command:
						command()
				
				canvas.bind("<Enter>", on_enter)
				canvas.bind("<Leave>", on_leave)
				canvas.bind("<Button-1>", on_click)
				
				return container
			
			# Buttons (swapped positions, left-aligned text)
			no_btn = make_left_aligned_button(btn_frame,
			                                   "No, quisiera delimitar los videos manualmente",
			                                   on_manual,
			                                   "#015ac9",
			                                   "#FFFFFF",
			                                   380,
			                                   50)
			no_btn.pack(side=tk.LEFT, padx=(0, 15))
			
			yes_btn = make_left_aligned_button(btn_frame, 
			                                    "Sí, delimita automáticamente los videos restantes",
			                                    on_auto_delimit,
			                                    "#fcc835",
			                                    "#000000",
			                                    380,
			                                    50)
			yes_btn.pack(side=tk.LEFT)

		def perform_save_delimitaciones():
			"""Actually save the ROIs to perma_rois.json"""
			try:
				import json
				from pathlib import Path
				
				if roi_state["source_type"] == "videos":
					# For videos: save to perma_rois.json (single ROI set)
					# Use the first video's ROIs as the canonical set
					if not video_list:
						info_lbl.config(text="⚠ No hay videos cargados", fg="#FF0000")
						return
					
					first_video_bn = Path(video_list[0]).name
					rois_to_save = roi_state["rois_map"].get(first_video_bn, [])
					
					if not rois_to_save:
						info_lbl.config(text="⚠ No hay ROIs para guardar", fg="#FF0000")
						return
					
					# Save to perma_rois.json (in Main_Codigos root)
					save_path = Path(__file__).parent.parent / "perma_rois.json"
					
					with open(save_path, "w", encoding="utf-8") as f:
						json.dump(rois_to_save, f, indent=2, ensure_ascii=False)
					
					info_lbl.config(text=f"✓ Delimitaciones guardadas en {save_path.name}", fg="#5BA8C9")
				
				elif roi_state["source_type"] == "cameras":
					# For cameras: save to perma_rois_by_camera.json
					save_path = Path(__file__).parent.parent / "perma_rois_by_camera.json"
					
					# Convert camera indices to strings for JSON
					camera_rois = {}
					for cam_idx, rois in roi_state["rois_map"].items():
						if isinstance(cam_idx, int):
							# Use camera IP/URL as key if available
							if cam_idx < len(rtsp_urls_list):
								cam_data = rtsp_urls_list[cam_idx]
								if isinstance(cam_data, dict):
									key = cam_data.get("url", f"camera_{cam_idx}")
								else:
									key = cam_data if isinstance(cam_data, str) else f"camera_{cam_idx}"
							else:
								key = f"camera_{cam_idx}"
							camera_rois[key] = rois
						else:
							camera_rois[str(cam_idx)] = rois
					
					with open(save_path, "w", encoding="utf-8") as f:
						json.dump(camera_rois, f, indent=2, ensure_ascii=False)
					
					info_lbl.config(text=f"✓ Delimitaciones guardadas en {save_path.name}", fg="#5BA8C9")
				
			except Exception as e:
				info_lbl.config(text=f"⚠ Error al guardar: {str(e)}", fg="#FF0000")

		# Now connect save_all_btn after all functions are defined
		save_all_btn.configure(command=save_all_delimitaciones)

		def load_first_frame_from_camera(cam_index: int) -> bool:
			"""Captura un frame de una cámara RTSP."""
			if cam_index >= len(rtsp_urls_list):
				return False
			cam_data = rtsp_urls_list[cam_index]
			rtsp_url = cam_data.get("url") if isinstance(cam_data, dict) else cam_data
			try:
				import cv2
				cap = cv2.VideoCapture(rtsp_url)
				ok, frame = False, None
				# Intentar capturar frame con timeout
				for _ in range(30):
					ok, frame = cap.read()
					if ok and frame is not None:
						break
				cap.release()
				if not ok or frame is None:
					return False
				# Original RGB
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				orig_h, orig_w = rgb.shape[:2]
				# Determine target display size
				container_w = canvas_container.winfo_width()
				container_h = canvas_container.winfo_height()
				if not container_w or not container_h or container_w < 200 or container_h < 150:
					builder.update_idletasks()
					container_w = canvas_container.winfo_width() or 640
					container_h = canvas_container.winfo_height() or 480
				# Stretch to fill
				display_w = container_w
				display_h = container_h
				resized = cv2.resize(rgb, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
				from PIL import Image, ImageTk as _ImageTk
				img = Image.fromarray(resized)
				photo = _ImageTk.PhotoImage(img)
				roi_canvas.delete("all")
				roi_state["photo"] = photo
				roi_state["scale_x"] = display_w / float(orig_w)
				roi_state["scale_y"] = display_h / float(orig_h)
				roi_state["orig_size"] = (orig_w, orig_h)
				roi_state["canvas_img_id"] = roi_canvas.create_image(0, 0, anchor="nw", image=photo)
				roi_canvas.config(width=display_w, height=display_h)
				# Update info label for camera
				cam_name = cam_data.get("np", f"Cámara {cam_index+1}")
				info_lbl.config(text=f"Cámara {cam_index+1}/{len(rtsp_urls_list)}: {cam_name} — Haga clic para dibujar ROI")
				redraw_overlay()
				return True
			except Exception as e:
				return False

		def load_first_frame_of(index: int) -> bool:
			"""Carga el primer frame de un video o cámara según el tipo de fuente."""
			if roi_state["source_type"] == "cameras":
				return load_first_frame_from_camera(index)
			if index >= len(video_list):
				return False
			path = video_list[index]
			
			# Get the frame index to load (from video_items if available)
			frame_idx = 0
			if index < len(video_items):
				frame_idx = video_items[index].get("frame_index", 0)
			
			# Prefer FFMPEG backend if available
			try:
				cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
			except Exception:
				cap = cv2.VideoCapture(path)
			
			# Try to find a valid frame, skipping corrupted ones
			ok, frame = False, None
			frame_attempt = 0
			max_frame_attempts = 10  # Try up to 10 different positions
			
			while not ok or frame is None:
				if frame_attempt >= max_frame_attempts:
					break
				
				# Set frame position (try current frame_idx + attempt * 25)
				current_frame_pos = frame_idx + (frame_attempt * 25)
				cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
				
				# Try to read this frame (with multiple read attempts for broken H264 headers)
				for read_attempt in range(30):
					ok, frame = cap.read()
					if ok and frame is not None:
						break
				
				frame_attempt += 1
			
			cap.release()
			if not ok or frame is None:
				return False
			# Original RGB
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			orig_h, orig_w = rgb.shape[:2]
			# Determine target display size (fill available canvas_container)
			container_w = canvas_container.winfo_width()
			container_h = canvas_container.winfo_height()
			if not container_w or not container_h or container_w < 200 or container_h < 150:
				# Ensure geometry is laid out before computing size
				builder.update_idletasks()
				container_w = canvas_container.winfo_width() or 640
				container_h = canvas_container.winfo_height() or 480
			# Stretch to fill (independent scales)
			display_w = container_w
			display_h = container_h
			resized = cv2.resize(rgb, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
			from PIL import Image, ImageTk as _ImageTk
			img = Image.fromarray(resized)
			photo = _ImageTk.PhotoImage(img)
			roi_canvas.delete("all")
			roi_state["photo"] = photo
			roi_state["scale_x"] = display_w / float(orig_w)
			roi_state["scale_y"] = display_h / float(orig_h)
			roi_state["orig_size"] = (orig_w, orig_h)
			roi_state["points"] = []
			roi_state["naming_mode"] = False
			roi_state["canvas_img_id"] = roi_canvas.create_image(0, 0, image=photo, anchor="nw")
			roi_canvas.config(width=display_w, height=display_h)

			# Update info label based on source type
			if roi_state["source_type"] == "cameras":
				cam_name = rtsp_urls_list[index].get("np", f"Cámara {index+1}") if index < len(rtsp_urls_list) else f"Cámara {index+1}"
				info_lbl.config(text=f"Cámara {index+1}/{len(rtsp_urls_list)}: {cam_name} — Haga clic para dibujar ROI")
			else:
				bn = Path(path).name
				info_lbl.config(text=f"Video {index+1}/{len(video_list)}: {bn} — Haga clic para dibujar ROI")
			redraw_overlay()
			return True

		# Now redefine select_video and populate_video_list with full implementations
		def select_video(index: int):
			"""Select a video from the list and load it in the canvas"""
			# Deselect previous
			for item in video_items:
				if item["selected"]:
					item["frame"].configure(bg="#002858")
					# Update child widgets background
					for child in item["frame"].winfo_children():
						try:
							child.configure(bg="#002858")
						except Exception:
							pass
					item["selected"] = False
			
			# Select new
			if 0 <= index < len(video_items):
				video_items[index]["selected"] = True
				video_items[index]["frame"].configure(bg="#015aca")
				# Update child widgets background
				for child in video_items[index]["frame"].winfo_children():
					try:
						child.configure(bg="#015aca")
					except Exception:
						pass
				selected_video_index["value"] = index
				roi_state["source_index"] = index
				
				# Load video frame in canvas
				load_first_frame_of(index)

		def populate_video_list():
			"""Populate the left column with all loaded videos or cameras"""
			# Clear existing items
			for item in video_items:
				item["frame"].destroy()
			video_items.clear()
			
			# Populate based on source type
			if roi_state["source_type"] == "cameras":
				# Create items for each camera
				for i, cam_data in enumerate(rtsp_urls_list):
					rtsp_url = cam_data.get("url", "")
					create_video_item(rtsp_url, i)
			else:
				# Create items for each video
				for i, video_path in enumerate(video_list):
					create_video_item(video_path, i)
			
			# Update scroll region after adding all items
			video_list_frame.update_idletasks()
			update_video_list_scroll()
			
			# Auto-select first item
			if video_items:
				select_video(0)

		def set_nav_enabled_for_third():
			# Enable next only when all sources processed or user chose No
			if roi_state["source_type"] == "videos":
				all_done = roi_state.get("source_index", 0) >= len(video_list)
			elif roi_state["source_type"] == "cameras":
				all_done = roi_state.get("source_index", 0) >= len(rtsp_urls_list)
			else:
				all_done = False
			can_advance[1] = all_done or (choice_frame.winfo_ismapped() and False)
			update_nav_state()

		# Old function for legacy buttons - commented out in new design
		# def enable_close_buttons():
		# 	enabled = (len(roi_state["points"]) >= 3) and (not roi_state["naming_mode"])
		# 	state = "normal" if enabled else "disabled"
		# 	close_and_name_btn.configure(state=state)
		# 	close_and_finish_btn.configure(state=state)

		def show_builder():
			choice_frame.grid_remove()
			builder.grid(row=0, column=0, sticky="nsew")
			# Determinar tipo de fuente según modo de procesamiento
			if processing_type["value"] == "rt":
				roi_state["source_type"] = "cameras"
				roi_state["source_index"] = 0
				if load_first_frame_of(0):
					info_lbl.configure(text=f"Cámara 1 de {len(rtsp_urls_list)} - Haga click para agregar puntos")
			else:
				roi_state["source_type"] = "videos"
				roi_state["source_index"] = 0
				if load_first_frame_of(0):
					info_lbl.configure(text=f"Video 1 de {len(video_list)} - Haga click para agregar puntos")
			set_nav_enabled_for_third()

		# Legacy functions for old ROI workflow - commented out in new design
		# Will be reimplemented with new button handlers
		"""
		def close_polygon_and_prompt_name(finalize: bool):
			if len(roi_state["points"]) < 3:
				return
			roi_state["naming_mode"] = True
			roi_state["finalize_after_name"] = finalize
			redraw_overlay()
			# Show naming controls
			name_entry.delete(0, tk.END)
			name_row.grid()
			name_entry.focus_set()

		# Old button configurations - commented out in new design
		# close_and_name_btn.configure(command=lambda: close_polygon_and_prompt_name(False))
		# close_and_finish_btn.configure(command=lambda: close_polygon_and_prompt_name(True))

		def save_named_roi_and_continue():
			name = name_entry.get().strip()
			if not name:
				return
			# Map display points to original coordinates
			sx = roi_state.get("scale_x", 1.0) or 1.0
			sy = roi_state.get("scale_y", 1.0) or 1.0
			coords = []
			for (x, y) in roi_state["points"]:
				xo = int(round(x / sx))
				yo = int(round(y / sy))
				coords.extend([xo, yo])
			
			# Save to map según tipo de fuente
			src_idx = roi_state.get("source_index", 0)
			if roi_state["source_type"] == "videos" and 0 <= src_idx < len(video_list):
				bn = Path(video_list[src_idx]).name
				if bn not in roi_state["rois_map"]:
					roi_state["rois_map"][bn] = []
				roi_state["rois_map"][bn].append({"name": name, "coords": coords})
			elif roi_state["source_type"] == "cameras" and 0 <= src_idx < len(rtsp_urls_list):
				if src_idx not in roi_state["rois_map"]:
					roi_state["rois_map"][src_idx] = []
				roi_state["rois_map"][src_idx].append({"name": name, "coords": coords})
			
			# Reset naming UI
			name_row.grid_remove()
			roi_state["naming_mode"] = False
			roi_state["points"] = []
			redraw_overlay()
			
			# Decide next action
			if roi_state["finalize_after_name"]:
				# Replicate if requested
				if apply_all_var.get():
					# Copy current source's ROIs to all remaining sources
					if roi_state["source_type"] == "videos":
						bn_ref = Path(video_list[src_idx]).name
						current_rois = list(roi_state["rois_map"].get(bn_ref, []))
						for j in range(src_idx+1, len(video_list)):
							bn_j = Path(video_list[j]).name
							roi_state["rois_map"][bn_j] = list(current_rois)
						roi_state["source_index"] = len(video_list)
						info_lbl.config(text="ROIs aplicados al resto de videos.")
					elif roi_state["source_type"] == "cameras":
						current_rois = list(roi_state["rois_map"].get(src_idx, []))
						for j in range(src_idx+1, len(rtsp_urls_list)):
							roi_state["rois_map"][j] = list(current_rois)
						roi_state["source_index"] = len(rtsp_urls_list)
						info_lbl.config(text="ROIs aplicados al resto de cámaras.")
					set_nav_enabled_for_third()
					# Auto-advance to next tab when completed
					go_next()
					return
				
				# Otherwise move to next source
				roi_state["source_index"] = src_idx + 1
				max_sources = len(rtsp_urls_list) if roi_state["source_type"] == "cameras" else len(video_list)
				if roi_state["source_index"] >= max_sources:
					if roi_state["source_type"] == "cameras":
						info_lbl.config(text="ROIs completados para todas las cámaras.")
					else:
						info_lbl.config(text="ROIs completados para todos los videos.")
					set_nav_enabled_for_third()
					# Auto-advance to next tab when completed
					go_next()
					return
				load_first_frame_of(roi_state["source_index"])
				set_nav_enabled_for_third()
			else:
				# Stay on same source for additional ROIs
				redraw_overlay()
				set_nav_enabled_for_third()

		name_accept_btn.configure(command=save_named_roi_and_continue)
		"""

		# ---------------- Tab 4: Cargar modelo (.pt) ----------------
		model_tab = tabs[2]
		model_tab.configure(bg=PROC_CONTENT_BG)
		model_tab.grid_rowconfigure(0, weight=0)
		model_tab.grid_rowconfigure(1, weight=0)
		model_tab.grid_rowconfigure(2, weight=1)
		model_tab.grid_columnconfigure(0, weight=1)

		# Title row with labels
		title_row = tk.Frame(model_tab, bg=PROC_CONTENT_BG)
		title_row.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 15))
		
		title_label = tk.Label(title_row, text="Cargar Modelo desde Carpeta", 
		                       font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		title_label.pack(side=tk.LEFT)
		
		ext_label = tk.Label(title_row, text=" (.PT)", 
		                     font=("Arial", 14, "bold"), fg="#00a3e2", bg=PROC_CONTENT_BG)
		ext_label.pack(side=tk.LEFT)

		# Browse row with textbox and button
		browse_row = tk.Frame(model_tab, bg=PROC_CONTENT_BG)
		browse_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
		browse_row.columnconfigure(0, weight=1)
		
		browse_entry = tk.Entry(browse_row, font=("Arial", 11), bg="white", fg="black",
		                        insertbackground="black", relief=tk.FLAT, 
		                        highlightthickness=0, disabledbackground="white",
		                        disabledforeground="black")
		browse_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=5)
		
		browse_btn = tk.Button(browse_row, text="Buscar", font=("Arial", 11, "bold"),
		                       bg="#015aca", fg="white", activebackground="#0B5ED7",
		                       activeforeground="white", relief=tk.FLAT, padx=20, pady=8,
		                       cursor="hand2")
		browse_btn.grid(row=0, column=1, sticky="e")

		# Drag and drop area
		model_container = tk.Frame(model_tab, bg=PROC_CONTENT_BG)
		model_container.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
		model_container.grid_rowconfigure(0, weight=1)
		model_container.grid_columnconfigure(0, weight=1)
		
		model_canvas = tk.Canvas(model_container, bg="#002858", highlightthickness=0, 
		                         relief=tk.FLAT)
		model_canvas.grid(row=0, column=0, sticky="nsew")

		model_center_label = tk.Label(model_canvas, text="¡Arrastra y suelta el modelo aquí para cargarlo!", 
		                              font=("Arial", 14, "bold"), fg="#0070c0", bg="#002858")
		model_center_label_id = model_canvas.create_window(0, 0, window=model_center_label, tags="center")

		model_list_frame = tk.Frame(model_canvas, bg="#002858")
		model_listbox = tk.Listbox(model_list_frame, bg="#002858", fg=FG_COLOR, 
		                           selectbackground=GRAY_HOVER, borderwidth=0, 
		                           highlightthickness=0, height=3, font=("Arial", 10))
		model_scroll = tk.Scrollbar(model_list_frame, orient=tk.VERTICAL, command=model_listbox.yview)
		model_listbox.configure(yscrollcommand=model_scroll.set)
		model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		model_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		model_list_window = model_canvas.create_window(0, 0, window=model_list_frame, tags="list", state="hidden")

		def draw_model_rect(c: tk.Canvas, x1, y1, x2, y2, r=16, color="#0070c0"):
			c.delete("frame")
			# No drawing - contorno removido

		def layout_model_canvas(_=None):
			w = model_canvas.winfo_width()
			h = model_canvas.winfo_height()
			pad = 20
			draw_model_rect(model_canvas, pad, pad, w-pad, h-pad, r=20, color="#0070c0")
			model_canvas.coords(model_center_label_id, w/2, h/2)
			model_canvas.coords(model_list_window, pad+12, pad+12)
			model_canvas.itemconfigure(model_list_window, anchor="nw", width=max(0, w - 2*(pad+12)), height=max(0, h - 2*(pad+12)))

		model_canvas.bind("<Configure>", layout_model_canvas)

		def update_model_next():
			can_advance[3] = MODEL_FILE["value"] is not None
			update_nav_state()

		def accept_model_path(path: str):
			if not path or not Path(path).is_file() or Path(path).suffix.lower() != ".pt":
				return
			
			# Check if this is the same model already loaded to avoid duplicate navigation
			normalized_path = str(Path(path))
			if MODEL_FILE["value"] == normalized_path:
				return  # Already loaded, don't advance again
			
			MODEL_FILE["value"] = normalized_path
			model_listbox.delete(0, tk.END)
			model_listbox.insert(tk.END, MODEL_FILE["value"])
			model_canvas.itemconfigure(model_list_window, state="normal")
			model_canvas.itemconfigure(model_center_label_id, state="hidden")
			browse_entry.configure(state="normal")
			browse_entry.delete(0, tk.END)
			browse_entry.insert(0, MODEL_FILE["value"])
			browse_entry.configure(state="readonly")
			update_model_next()
			# Avanzar automáticamente al siguiente tab
			go_next()

		def validate_model_path(event=None):
			"""Validate and load model when path is typed manually"""
			path_text = browse_entry.get().strip()
			if not path_text:
				return
			
			path_obj = Path(path_text)
			
			# If path is a .pt file directly
			if path_obj.is_file() and path_obj.suffix.lower() == ".pt":
				accept_model_path(path_text)
			# If path is a directory, find first .pt file
			elif path_obj.is_dir():
				pt_files = list(path_obj.glob("*.pt"))
				if pt_files:
					accept_model_path(str(pt_files[0]))
		
		# Bind validation to browse_entry changes - triggers on any text change
		browse_entry.bind("<KeyRelease>", validate_model_path)
		browse_entry.bind("<Return>", validate_model_path)
		browse_entry.bind("<FocusOut>", validate_model_path)

		def on_model_drop(event):
			data = event.data.strip()
			paths = []
			buf = ""; in_brace = False
			for ch in data:
				if ch == "{": in_brace = True; buf = ""; continue
				if ch == "}": in_brace = False; paths.append(buf); buf = ""; continue
				if ch == " " and not in_brace:
					if buf: paths.append(buf); buf = ""; continue
				buf += ch
			if buf: paths.append(buf)
			for p in paths:
				if Path(p).suffix.lower() == ".pt" and Path(p).is_file():
					accept_model_path(p)
					break

		def on_model_click(_):
			path = filedialog.askopenfilename(title="Seleccionar modelo", filetypes=[("PyTorch Weights", "*.pt")])
			if path:
				accept_model_path(path)

		def set_model_drop_enabled(enabled: bool):
			if enabled:
				model_canvas.configure(cursor="hand2", bg="#002858")
				model_center_label.configure(fg="#0070c0", bg="#002858")
				if DND_AVAILABLE:
					model_canvas.drop_target_register(DND_FILES)
					model_canvas.dnd_bind("<<Drop>>", on_model_drop)
				model_canvas.bind("<Button-1>", on_model_click)
			else:
				model_canvas.configure(cursor="arrow")
				model_center_label.configure(fg="#A0A9B8")
				if DND_AVAILABLE:
					try:
						model_canvas.drop_target_unregister()
					except Exception:
						pass
				model_canvas.unbind("<Button-1>")

		def on_browse_model():
			path = filedialog.askopenfilename(title="Seleccionar modelo", filetypes=[("PyTorch Weights", "*.pt")])
			if path:
				accept_model_path(path)

		browse_btn.configure(command=on_browse_model)
		set_model_drop_enabled(True)

		# ---------------- Tab 5: Configuracion ----------------
		config_tab = tabs[3]
		config_tab.configure(bg=PROC_CONTENT_BG)
		config_tab.grid_rowconfigure(0, weight=1)
		config_tab.grid_columnconfigure(0, weight=1, minsize=400)
		config_tab.grid_columnconfigure(1, weight=1)

		# Main container with two columns
		config_container = tk.Frame(config_tab, bg=PROC_CONTENT_BG)
		config_container.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
		config_container.grid_rowconfigure(0, weight=1)
		config_container.grid_columnconfigure(0, weight=0, minsize=400)
		config_container.grid_columnconfigure(1, weight=1)

		# ============ LEFT COLUMN ============
		left_column = tk.Frame(config_container, bg=PROC_CONTENT_BG)
		left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
		left_column.grid_columnconfigure(0, weight=1)

		# Section 1: Modelos
		models_label = tk.Label(left_column, text="Elige los Modelos a Utilizar", 
		                        font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		models_label.grid(row=0, column=0, sticky="w", pady=(0, 15))

		security_var = tk.BooleanVar(value=False)
		security_chk = tk.Checkbutton(left_column, text="Incluir modelos de seguridad", 
		                              variable=security_var, fg="white", bg=PROC_CONTENT_BG, 
		                              selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                              activeforeground="white", font=("Arial", 11))
		security_chk.grid(row=1, column=0, sticky="w", pady=(0, 8))

		face_blur_var = tk.BooleanVar(value=False)
		face_blur_chk = tk.Checkbutton(left_column, text="Difuminar caras", 
		                               variable=face_blur_var, fg="white", bg=PROC_CONTENT_BG, 
		                               selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                               activeforeground="white", font=("Arial", 11))
		face_blur_chk.grid(row=2, column=0, sticky="w", pady=(0, 8))

		background_processing_var = tk.BooleanVar(value=False)
		background_processing_chk = tk.Checkbutton(left_column, text="Procesar en segundo plano", 
		                                           variable=background_processing_var, fg="white", bg=PROC_CONTENT_BG, 
		                                           selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                           activeforeground="white", font=("Arial", 11))
		background_processing_chk.grid(row=3, column=0, sticky="w", pady=(0, 8))

		# Model loaded checkbox (will be updated with actual model name)
		model_loaded_var = tk.BooleanVar(value=True)
		model_name_display = MODEL_FILE.get("value", "").split("\\")[-1] if MODEL_FILE.get("value") else "Sin modelo"
		model_loaded_chk = tk.Checkbutton(left_column, text=f"Modelo cargado: {model_name_display}", 
		                                  variable=model_loaded_var, fg="white", bg=PROC_CONTENT_BG, 
		                                  selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                  activeforeground="white", disabledforeground="#888888",
		                                  font=("Arial", 11))
		model_loaded_chk.grid(row=4, column=0, sticky="w", pady=(0, 25))

		# Section 2: Processing level
		level_label = tk.Label(left_column, text="Elige el nivel de procesamiento", 
		                       font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		level_label.grid(row=5, column=0, sticky="w", pady=(0, 15))

		processing_level_var = tk.IntVar(value=1)
		
		level_light_chk = tk.Radiobutton(left_column, text="Ligero", variable=processing_level_var, 
		                                 value=0, fg="white", bg=PROC_CONTENT_BG, 
		                                 selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                 activeforeground="white", font=("Arial", 11))
		level_light_chk.grid(row=6, column=0, sticky="w", pady=(0, 8))

		level_balanced_chk = tk.Radiobutton(left_column, text="Balanceado", variable=processing_level_var, 
		                                    value=1, fg="white", bg=PROC_CONTENT_BG, 
		                                    selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                    activeforeground="white", font=("Arial", 11))
		level_balanced_chk.grid(row=7, column=0, sticky="w", pady=(0, 8))

		level_heavy_chk = tk.Radiobutton(left_column, text="Pesado", variable=processing_level_var, 
		                                 value=2, fg="white", bg=PROC_CONTENT_BG, 
		                                 selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                 activeforeground="white", font=("Arial", 11))
		level_heavy_chk.grid(row=8, column=0, sticky="w", pady=(0, 8))

		level_custom_chk = tk.Radiobutton(left_column, text="Personalizable", variable=processing_level_var, 
		                                  value=3, fg="white", bg=PROC_CONTENT_BG, 
		                                  selectcolor=PROC_CONTENT_BG, activebackground=PROC_CONTENT_BG, 
		                                  activeforeground="white", font=("Arial", 11))
		level_custom_chk.grid(row=9, column=0, sticky="w", pady=(0, 25))

		# Spacer to push button to bottom
		left_column.grid_rowconfigure(10, weight=1)

		# Back button at bottom left
		back_btn = make_rounded_button(left_column, "Regresar", go_prev, "#015aca", 
		                               width=150, height=40, fg="white")
		back_btn.grid(row=11, column=0, sticky="w", pady=(20, 0))

		# ============ RIGHT COLUMN ============
		right_column = tk.Frame(config_container, bg=PROC_CONTENT_BG)
		right_column.grid(row=0, column=1, sticky="nsew")
		right_column.grid_columnconfigure(0, weight=1)
		right_column.grid_rowconfigure(4, weight=1)

		# Videos output folder
		videos_output_label = tk.Label(right_column, text="Carpeta para guardar los videos procesados", 
		                               font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		videos_output_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

		videos_row = tk.Frame(right_column, bg=PROC_CONTENT_BG)
		videos_row.grid(row=1, column=0, sticky="ew", pady=(0, 25))
		videos_row.grid_columnconfigure(0, weight=1)

		videos_entry = tk.Entry(videos_row, font=("Arial", 11), bg="white", fg="black",
		                        insertbackground="black", relief=tk.FLAT, highlightthickness=0)
		videos_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=5)
		videos_entry.insert(0, r"C:\Arnesis\videos_procesados")

		videos_browse_btn = make_rounded_button(videos_row, "Buscar", lambda: None, "#015aca", 
		                                        width=100, height=36, fg="white")
		videos_browse_btn.grid(row=0, column=1)

		# Data output folder
		data_output_label = tk.Label(right_column, text="Carpeta para guardar los datos", 
		                             font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		data_output_label.grid(row=2, column=0, sticky="w", pady=(0, 10))

		data_row = tk.Frame(right_column, bg=PROC_CONTENT_BG)
		data_row.grid(row=3, column=0, sticky="ew", pady=(0, 25))
		data_row.grid_columnconfigure(0, weight=1)

		data_entry = tk.Entry(data_row, font=("Arial", 11), bg="white", fg="black",
		                      insertbackground="black", relief=tk.FLAT, highlightthickness=0)
		data_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=5)
		data_entry.insert(0, r"C:\Arnesis\videos_procesados\datos")

		data_browse_btn = make_rounded_button(data_row, "Buscar", lambda: None, "#015aca", 
		                                      width=100, height=36, fg="white")
		data_browse_btn.grid(row=0, column=1)

		# Info display box - Customizable settings
		info_box = tk.Frame(right_column, bg="#021e44", relief=tk.FLAT)
		info_box.grid(row=4, column=0, sticky="nsew", pady=(0, 20))
		info_box.grid_columnconfigure(0, weight=1)
		info_box.grid_columnconfigure(1, weight=2)

		# Title
		title_label = tk.Label(info_box, text="Configuración de Procesamiento", 
		                       font=("Arial", 13, "bold"), fg=FG_COLOR, bg="#021e44")
		title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(20, 15))

		# Person Detection Model
		person_det_label = tk.Label(info_box, text="Modelo de Detección:", 
		                            font=("Arial", 11), fg=FG_COLOR, bg="#021e44")
		person_det_label.grid(row=1, column=0, sticky="w", padx=20, pady=5)
		
		person_det_var = tk.StringVar(value="yolo11m.pt")
		person_det_combo = ttk.Combobox(info_box, textvariable=person_det_var, 
		                                values=["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
		                                state="disabled", width=20)
		person_det_combo.grid(row=1, column=1, sticky="ew", padx=20, pady=5)

		# Half Precision
		half_label = tk.Label(info_box, text="Precisión Mixta (FP16):", 
		                     font=("Arial", 11), fg=FG_COLOR, bg="#021e44")
		half_label.grid(row=2, column=0, sticky="w", padx=20, pady=5)
		
		half_var = tk.StringVar(value="False")
		half_combo = ttk.Combobox(info_box, textvariable=half_var, 
		                          values=["True", "False"],
		                          state="disabled", width=20)
		half_combo.grid(row=2, column=1, sticky="ew", padx=20, pady=5)

		# Ergonomics Model Complexity
		ergo_label = tk.Label(info_box, text="Complejidad Ergonomía:", 
		                     font=("Arial", 11), fg=FG_COLOR, bg="#021e44")
		ergo_label.grid(row=3, column=0, sticky="w", padx=20, pady=5)
		
		ergo_var = tk.StringVar(value="1")
		ergo_combo = ttk.Combobox(info_box, textvariable=ergo_var, 
		                          values=["0", "1", "2"],
		                          state="disabled", width=20)
		ergo_combo.grid(row=3, column=1, sticky="ew", padx=20, pady=5)

		# Face Detection Model
		face_det_label = tk.Label(info_box, text="Modelo Detección Rostro:", 
		                         font=("Arial", 11), fg=FG_COLOR, bg="#021e44")
		face_det_label.grid(row=4, column=0, sticky="w", padx=20, pady=5)
		
		face_det_var = tk.StringVar(value="yolov8m-face.pt")
		face_det_combo = ttk.Combobox(info_box, textvariable=face_det_var, 
		                              values=["yolov8s-face.pt", "yolov8m-face.pt", "yolov8l-face.pt"],
		                              state="disabled", width=20)
		face_det_combo.grid(row=4, column=1, sticky="ew", padx=20, pady=5)

		# Frameskip Configuration
		frameskip_label = tk.Label(info_box, text="Salto de Frames:", 
		                          font=("Arial", 11), fg=FG_COLOR, bg="#021e44")
		frameskip_label.grid(row=5, column=0, sticky="w", padx=20, pady=(5, 20))
		
		frameskip_var = tk.StringVar(value="Saltar 20% de frames")
		frameskip_combo = ttk.Combobox(info_box, textvariable=frameskip_var, 
		                               values=["No saltar frames", "Saltar 10% de frames", 
		                                       "Saltar 20% de frames", "Saltar 30% de frames",
		                                       "Saltar 40% de frames", "Saltar 50% de frames",
		                                       "Saltar 60% de frames", "Saltar 70% de frames",
		                                       "Saltar 80% de frames", "Saltar 90% de frames"],
		                               state="disabled", width=20)
		frameskip_combo.grid(row=5, column=1, sticky="ew", padx=20, pady=(5, 20))

		# Function to update configuration based on processing level
		def update_processing_config(*args):
			level = processing_level_var.get()
			
			if level == 0:  # Ligero
				person_det_var.set("yolo11s.pt")
				half_var.set("True")
				ergo_var.set("0")
				face_det_var.set("yolov8s-face.pt")
				frameskip_var.set("Saltar 50% de frames")
				# Disable combos
				person_det_combo.configure(state="disabled")
				half_combo.configure(state="disabled")
				ergo_combo.configure(state="disabled")
				face_det_combo.configure(state="disabled")
				frameskip_combo.configure(state="disabled")
				
			elif level == 1:  # Balanceado
				person_det_var.set("yolo11m.pt")
				half_var.set("False")
				ergo_var.set("1")
				face_det_var.set("yolov8m-face.pt")
				frameskip_var.set("Saltar 20% de frames")
				# Disable combos
				person_det_combo.configure(state="disabled")
				half_combo.configure(state="disabled")
				ergo_combo.configure(state="disabled")
				face_det_combo.configure(state="disabled")
				frameskip_combo.configure(state="disabled")
				
			elif level == 2:  # Pesado
				person_det_var.set("yolo11x.pt")
				half_var.set("False")
				ergo_var.set("2")
				face_det_var.set("yolov8l-face.pt")
				frameskip_var.set("No saltar frames")
				# Disable combos
				person_det_combo.configure(state="disabled")
				half_combo.configure(state="disabled")
				ergo_combo.configure(state="disabled")
				face_det_combo.configure(state="disabled")
				frameskip_combo.configure(state="disabled")
				
			elif level == 3:  # Personalizable
				# Enable all combos for customization
				person_det_combo.configure(state="readonly")
				half_combo.configure(state="readonly")
				ergo_combo.configure(state="readonly")
				face_det_combo.configure(state="readonly")
				frameskip_combo.configure(state="readonly")
		
		# Bind radiobutton changes to update function
		processing_level_var.trace_add("write", update_processing_config)
		
		# Initialize with default (Balanceado)
		update_processing_config()


		
		def save_camera_group_config():
			"""Save current RT configuration and camera group"""
			import json
			from pathlib import Path
			
	# Create popup to get group metadata
			popup = tk.Toplevel(win)
			popup.title("Guardar Grupo de Cámaras")
			popup.geometry("500x350")
			popup.configure(bg=PROC_CONTENT_BG)
			popup.transient(win)
			popup.grab_set()
			
			# Title
			title_lbl = tk.Label(popup, text="Datos del grupo de cámaras", 
			                     font=("Arial", 14, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			title_lbl.pack(pady=(20, 10))
			
			# Form frame
			form_frame = tk.Frame(popup, bg=PROC_CONTENT_BG)
			form_frame.pack(pady=10, padx=40, fill=tk.X)
			
			# Linea
			linea_lbl = tk.Label(form_frame, text="Línea:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			linea_lbl.grid(row=0, column=0, sticky="w", pady=10)
			linea_entry = tk.Entry(form_frame, font=("Arial", 11), fg="black", bg="white")
			linea_entry.grid(row=0, column=1, sticky="ew", pady=10, padx=(10, 0))
			linea_entry.focus()
			
			# Segmento
			segmento_lbl = tk.Label(form_frame, text="Segmento:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			segmento_lbl.grid(row=1, column=0, sticky="w", pady=10)
			segmento_var = tk.StringVar(value="1")
			segmento_combo = ttk.Combobox(form_frame, textvariable=segmento_var, font=("Arial", 11), 
			                              values=[str(i) for i in range(1, 7)], state="readonly", width=18)
			segmento_combo.grid(row=1, column=1, sticky="ew", pady=10, padx=(10, 0))
			
			# Area
			area_lbl = tk.Label(form_frame, text="Área:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			area_lbl.grid(row=2, column=0, sticky="w", pady=10)
			area_entry = tk.Entry(form_frame, font=("Arial", 11), fg="black", bg="white")
			area_entry.grid(row=2, column=1, sticky="ew", pady=10, padx=(10, 0))
			
			# Planta
			planta_lbl = tk.Label(form_frame, text="Planta:", font=("Arial", 11), fg=FG_COLOR, bg=PROC_CONTENT_BG)
			planta_lbl.grid(row=3, column=0, sticky="w", pady=10)
			planta_var = tk.StringVar(value="1")
			planta_combo = ttk.Combobox(form_frame, textvariable=planta_var, font=("Arial", 11), 
			                            values=["1", "2"], state="readonly", width=18)
			planta_combo.grid(row=3, column=1, sticky="ew", pady=10, padx=(10, 0))
			
			# Configure column weights
			form_frame.columnconfigure(1, weight=1)
			
			# Buttons
			btn_frame = tk.Frame(popup, bg=PROC_CONTENT_BG)
			btn_frame.pack(pady=(20, 20))
			
			def on_save():
				linea = linea_entry.get().strip()
				segmento = segmento_var.get()
				area = area_entry.get().strip()
				planta = planta_var.get()
				
				if not linea or not area:
					messagebox.showwarning("Datos requeridos", "Por favor completa los campos Línea y Área.")
					return
				
				try:
					# Generate group name from metadata
					group_name = f"L{linea}_S{segmento}_A{area}_P{planta}"
					
					# Prepare data
					data = {
						"metadata": {
							"linea": linea,
							"segmento": segmento,
							"area": area,
							"planta": planta
						},
						"cameras": [],
						"model_path": MODEL_FILE.get("value", ""),
						"settings": {
							"security": security_var.get(),
							"face_blur": face_blur_var.get(),
							"background_processing": background_processing_var.get(),
							"processing_level": processing_level_var.get(),
							"videos_output": videos_entry.get().strip(),
							"data_output": data_entry.get().strip()
						}
					}
					
					# Save cameras with their ROIs
					for i, cam in enumerate(camera_widgets):
						cam_copy = cam.copy()
						# Add ROIs for this camera if they exist
						if i in roi_state.get("rois_map", {}):
							# Deep copy ROIs and ensure coords are JSON-serializable
							rois_copy = []
							for roi in roi_state["rois_map"][i]:
								roi_dict = {
									"name": roi.get("name", f"ROI{len(rois_copy)+1}"),
									"coords": list(roi["coords"]) if isinstance(roi["coords"], (list, tuple)) else roi["coords"].tolist()
								}
								rois_copy.append(roi_dict)
							cam_copy["rois"] = rois_copy
						data["cameras"].append(cam_copy)
				
			# Create directory if it doesn't exist
					groups_dir = Path("camera_groups")
					groups_dir.mkdir(exist_ok=True)
					
					# Save to file
					filepath = groups_dir / f"{group_name}.json"
					with open(filepath, 'w', encoding='utf-8') as f:
						json.dump(data, f, indent=2, ensure_ascii=False)
					
					popup.destroy()
					messagebox.showinfo("Éxito", f"Grupo '{group_name}' guardado correctamente en:\n{filepath}")
				
				except Exception as e:
					messagebox.showerror("Error", f"Error al guardar grupo:\n{str(e)}")
			
			def on_cancel():
				"""Close popup without saving"""
				popup.destroy()
			
			cancel_btn = make_rounded_button(btn_frame, "Cancelar", on_cancel, "#666666", 
			                                 width=140, height=40, fg="white")
			cancel_btn.pack(side=tk.LEFT, padx=(0, 10))
			
			save_btn = make_rounded_button(btn_frame, "Guardar", on_save, "#7ec331", 
			                               width=140, height=40, fg="white")
			save_btn.pack(side=tk.LEFT)
		
		save_config_btn = make_rounded_button(right_column, "Guardar configuración y grupo de cámaras", 
		                                      save_camera_group_config, "#7ec331", 
		                                      width=300, height=40, fg="white")
		save_config_btn.grid(row=5, column=0, sticky="w", pady=(10, 10))
		
		# Process button at bottom right (command set later after perform_process is defined)
		process_btn = make_rounded_button(right_column, "Comenzar Procesamiento", lambda: None, "#ffff00", 
		                                  width=250, height=45, fg="black")
		process_btn.grid(row=5, column=0, sticky="e", pady=(0, 0))

		# Browse button callbacks
		def on_browse_videos():
			p = filedialog.askdirectory(title="Seleccionar carpeta para videos procesados")
			if p:
				videos_entry.delete(0, tk.END)
				videos_entry.insert(0, p)

		def on_browse_data():
			p = filedialog.askdirectory(title="Seleccionar carpeta para datos")
			if p:
				data_entry.delete(0, tk.END)
				data_entry.insert(0, p)

		videos_browse_btn.configure(command=on_browse_videos)
		data_browse_btn.configure(command=on_browse_data)

		# Helper state & validation
		def path_exists_dir(p: str) -> bool:
			return bool(p) and Path(p).is_dir()

		def on_browse_base():
			p = filedialog.askdirectory(title="Seleccionar carpeta base para videos y CSV")
			if p:
				videos_entry.delete(0, tk.END)
				videos_entry.insert(0, p)

		def update_eta():
			pass  # Placeholder
			validate_config()

		def validate_config():
			folder_ok = path_exists_dir(videos_entry.get().strip())
			can_advance[4] = folder_ok
			process_btn.configure(state=("normal" if can_advance[4] else "disabled"))
			update_nav_state()

		processing_started = {"value": False}

		# Log queue and helper functions (needed by start_processing_ui and others)
		log_queue = []  # lines to append to log

		def _append_log_line(line: str):
			log_queue.append(line)

		def _flush_log():
			if not log_queue:
				return
			log_text.configure(state="normal")
			while log_queue:
				ln = log_queue.pop(0)
				is_error = ("ERROR" in ln) or ("Traceback" in ln) or ("Exception" in ln)
				if is_error:
					log_text.insert("end", ln + "\n", ("error",))
				else:
					log_text.insert("end", ln + "\n")
			log_text.see("end")
			log_text.configure(state="disabled")

		def start_processing_ui(config_data=None):
			try:
				# Cambiar a pestaña de procesamiento
				nonlocal current_step
				_append_log_line(f"[DEBUG] start_processing_ui llamado con config_data: {config_data is not None}")
				_append_log_line(f"[DEBUG] Cambiando a tab de procesamiento (step 4)...")
				
				current_step = 4
				update_tabs_state()
				processing_started["value"] = True
				
				_append_log_line(f"[DEBUG] Processing type: {processing_type['value']}")
				
				# Show appropriate UI based on processing type
				if processing_type["value"] == "rt":
					# RT mode: show RT frame, hide batch frame
					_append_log_line(f"[DEBUG] Mostrando UI de tiempo real...")
					batch_frame.pack_forget()
					rt_frame.pack(fill="both", expand=True)
					# Update camera status based on configured cameras
					update_rt_camera_status()
					# Hide breadcrumb tabs for RT mode
					header_frame.pack_forget()
					# Hide navigation buttons
					nav_frame.pack_forget()
				else:
					# Batch mode: show batch frame, hide RT frame
					_append_log_line(f"[DEBUG] Mostrando UI de procesamiento batch...")
					rt_frame.pack_forget()
					batch_frame.pack(fill="both", expand=True)
					# Show breadcrumb tabs for batch mode
					if not header_frame.winfo_ismapped():
						header_frame.pack(fill=tk.X, padx=20, pady=(10, 0), before=content_frame)
					# Show navigation buttons
					if not nav_frame.winfo_ismapped():
						nav_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
				
				# Diferir inicialización de UI de procesamiento sin capturar nombres
				# aún no definidos en el alcance (evita NameError por cierre temprano).
				def _invoke_begin():
					try:
						_append_log_line(f"[DEBUG] _invoke_begin ejecutándose...")
						fn = globals().get("BEGIN_PROCESSING_FN")
						_append_log_line(f"[DEBUG] BEGIN_PROCESSING_FN encontrado: {fn is not None}")
						if callable(fn):
							_append_log_line(f"[DEBUG] Llamando a BEGIN_PROCESSING_FN...")
							fn(config_data)
						else:
							error_msg = "BEGIN_PROCESSING_FN no es callable o no existe"
							_append_log_line(f"[ERROR] {error_msg}")
							messagebox.showerror(
								"Error de Configuración",
								f"{error_msg}\n\nEsto puede indicar un problema de inicialización."
							)
					except Exception as e:
						import traceback
						error_msg = f"Error en _invoke_begin:\n\n{str(e)}\n\n{traceback.format_exc()}"
						_append_log_line(f"[ERROR] {error_msg}")
						messagebox.showerror("Error _invoke_begin", error_msg)
				
				# Populate PROCESSING_CONTEXT before calling _invoke_begin
				# This must happen here because all the local functions are now defined
				PROCESSING_CONTEXT["processing_type"] = processing_type
				PROCESSING_CONTEXT["begin_rt_processing"] = begin_rt_processing
				PROCESSING_CONTEXT["launch_real_pipeline"] = launch_real_pipeline
				PROCESSING_CONTEXT["progress_var"] = progress_var
				PROCESSING_CONTEXT["update_progress_bar"] = update_progress_bar
				PROCESSING_CONTEXT["_update_processing_nav_state"] = _update_processing_nav_state
				PROCESSING_CONTEXT["_append_log_line"] = _append_log_line
				
				_append_log_line(f"[DEBUG] Programando ejecución diferida de begin_processing...")
				win.after(0, _invoke_begin)
				
			except Exception as e:
				import traceback
				error_msg = f"Error en start_processing_ui:\n\n{str(e)}\n\nDetalle técnico:\n{traceback.format_exc()}"
				messagebox.showerror(
					"Error UI Procesamiento",
					error_msg
				)
				print(f"[ERROR] start_processing_ui failed: {error_msg}")
				_append_log_line(f"[ERROR] {error_msg}")

		def perform_process():
			try:
				# Clear log and reset progress bar before starting new processing
				log_text.configure(state="normal")
				log_text.delete("1.0", "end")
				log_text.configure(state="disabled")
				progress_var.set(0)
				update_progress_bar(0)
				log_queue.clear()
				
				# if not can_advance[4]:
				# 	messagebox.showwarning(
				# 		"Validación Fallida",
				# 		"No se puede iniciar el procesamiento. Verifica que todos los campos estén correctos."
				# 	)
				# 	return
				
				# Get folder paths from new UI
				videos_output_dir = videos_entry.get().strip()
				data_output_dir = data_entry.get().strip()
				
				# Validate critical fields
				if not MODEL_FILE.get("value") and model_loaded_var.get():
					raise ValueError("No se ha seleccionado un modelo de clasificación")
				
				if not video_list and processing_type["value"] != "rt":
					raise ValueError("No se han agregado videos para procesar")
				
				if not data_output_dir:
					raise ValueError("No se ha especificado la carpeta de salida para datos CSV")
				
				# Map processing level to efficiency value
				# 0=Ligero, 1=Balanceado, 2=Pesado, 3=Personalizable
				level = processing_level_var.get()
				efficiency_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 1.0}  # Default values
				efficiency_value = efficiency_map.get(level, 1.0)
				
				# Get customizable settings
				person_det_model = person_det_var.get()
				use_half = half_var.get() == "True"
				ergo_complexity = int(ergo_var.get())
				face_model_name = face_det_var.get()
				
				# Parse frameskip setting (percentage-based)
				frameskip_text = frameskip_var.get()
				if "No saltar" in frameskip_text:
					frameskip_percentage = 0
				else:
					# Extract percentage from text like "Saltar 20% de frames"
					import re
					match = re.search(r'(\d+)%', frameskip_text)
					if match:
						frameskip_percentage = int(match.group(1))
					else:
						frameskip_percentage = 0  # Default to no skip
				
				configuration = {
					"difuminar_caras": face_blur_var.get(),
					"seguridad": security_var.get(),
					"background_processing": background_processing_var.get(),
					"eficiencia": efficiency_value,
					"peso_modelo": 1.0,  # Default weight
					"processing_level": level,
					# Customizable processing settings
					"person_det_engine": person_det_model,
					"use_half": use_half,
					"ergo_model_complexity": ergo_complexity,
					"face_model_name": face_model_name,
					"frameskip_percentage": frameskip_percentage,
					# Expected by pipeline
					"processed_dir": videos_output_dir if videos_output_dir else None,
					"csv_dir": data_output_dir,
					"classify_weights": MODEL_FILE["value"] if model_loaded_var.get() else None,
					"skip_classification": not model_loaded_var.get(),
					"input_videos": video_list,
					"output_name": (NP_ACTUAL.get("value") or "gui_run"),
					"rois_by_video": roi_state.get("rois_map"),
					"generate_output_video": bool(videos_output_dir),
					"manual_datetimes": manual_datetimes,
					# Legacy Spanish keys
					"carpeta_videos_procesados": videos_output_dir if videos_output_dir else None,
					"carpeta_csv": data_output_dir,
					# RT mode specific
					"processing_type": processing_type["value"],
					"rtsp_urls": rtsp_urls_list if processing_type["value"] == "rt" else [],
					# Group metadata
					"metadata": group_metadata.copy() if group_metadata else {},
				}
				start_processing_ui(configuration)
				
			except Exception as e:
				import traceback
				error_msg = f"Error al iniciar procesamiento:\n\n{str(e)}\n\nDetalle técnico:\n{traceback.format_exc()}"
				messagebox.showerror(
					"Error de Procesamiento",
					error_msg
				)
				print(f"[ERROR] perform_process failed: {error_msg}")

		# Connect process button command
		process_btn.configure(command=perform_process)

		# Initial state
		update_eta()
		validate_config()

		# Modify nav state to hide 'Siguiente' on config step
		_original_update_nav_state = update_nav_state
		def update_nav_state_wrapper():
			# Show/hide prev button based on current step
			if current_step == 0:
				prev_btn_container.pack_forget()
			else:
				if not prev_btn_container.winfo_ismapped():
					prev_btn_container.pack(side=tk.LEFT, padx=10)
			
			# Hide next button on config step (index 3 in new 0-indexed system)
			if current_step == 3:
				if next_btn_container.winfo_ismapped():
					next_btn_container.pack_forget()
			else:
				if not next_btn_container.winfo_ismapped():
					next_btn_container.pack(side=tk.RIGHT, padx=10)
			
			# Update process button state separately
			if current_step == 3:
				process_btn.configure(state=("normal" if can_advance[3] else "disabled"))
		update_nav_state = update_nav_state_wrapper  # override reference
		update_nav_state_wrapper()

		# ---------------- Tab 6 (index 5): Procesamiento ----------------
		processing_tab = tabs[4]
		processing_tab.configure(bg=BG_COLOR)
		# Container frame for dynamic UI (batch vs RT)
		proc_container = tk.Frame(processing_tab, bg=BG_COLOR)
		proc_container.pack(fill="both", expand=True)

		# ===== BATCH PROCESSING UI (videos mode) =====
		batch_frame = tk.Frame(proc_container, bg=BG_COLOR)
		
		# Header
		proc_header = tk.Label(batch_frame, text="Procesamiento", font=("Arial", 16, "bold"), fg=FG_COLOR, bg=BG_COLOR)
		proc_header.pack(anchor="w", padx=16, pady=(18, 8))

		# Start date label
		from datetime import datetime
		start_date_label = tk.Label(batch_frame, text="Fecha de Inicio: --/--/---- --:-- --", 
		                            font=("Arial", 12), fg=FG_COLOR, bg=BG_COLOR)
		start_date_label.pack(anchor="w", padx=16, pady=(0, 12))

		# Custom progress bar with rounded corners
		progress_frame = tk.Frame(batch_frame, bg=BG_COLOR)
		progress_frame.pack(fill="x", padx=16, pady=(0, 12))
		progress_frame.grid_columnconfigure(0, weight=1)

		# Canvas for custom rounded progress bar
		progress_canvas = tk.Canvas(progress_frame, height=40, bg=BG_COLOR, highlightthickness=0)
		progress_canvas.grid(row=0, column=0, sticky="ew", padx=(0, 12))
		
		# Percentage label
		percent_lbl = tk.Label(progress_frame, text="0%", font=("Arial", 12, "bold"), fg=FG_COLOR, bg=BG_COLOR)
		percent_lbl.grid(row=0, column=1, sticky="e")

		# Function to draw rounded rectangle
		def draw_rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
			points = [
				x1+radius, y1,
				x1+radius, y1,
				x2-radius, y1,
				x2-radius, y1,
				x2, y1,
				x2, y1+radius,
				x2, y1+radius,
				x2, y2-radius,
				x2, y2-radius,
				x2, y2,
				x2-radius, y2,
				x2-radius, y2,
				x1+radius, y2,
				x1+radius, y2,
				x1, y2,
				x1, y2-radius,
				x1, y2-radius,
				x1, y1+radius,
				x1, y1+radius,
				x1, y1
			]
			return canvas.create_polygon(points, smooth=True, **kwargs)

		# Progress bar state
		progress_bg_rect = None
		progress_fill_rect = None
		progress_var = tk.IntVar(value=0)

		def update_progress_bar(value):
			nonlocal progress_bg_rect, progress_fill_rect
			progress_canvas.delete("all")
			
			# Get canvas dimensions
			canvas_width = progress_canvas.winfo_width()
			if canvas_width <= 1:
				canvas_width = 600  # Default width
			canvas_height = 40
			
			# Draw background (white rounded rectangle)
			margin = 2
			bg_x1, bg_y1 = margin, margin
			bg_x2, bg_y2 = canvas_width - margin, canvas_height - margin
			progress_bg_rect = draw_rounded_rect(progress_canvas, bg_x1, bg_y1, bg_x2, bg_y2, 
			                                     radius=10, fill="white", outline="")
			
			# Draw progress fill (blue rounded rectangle)
			if value > 0:
				fill_width = int((canvas_width - 2*margin) * (value / 100.0))
				if fill_width > 4:  # Only draw if visible
					fill_x2 = margin + fill_width
					progress_fill_rect = draw_rounded_rect(progress_canvas, bg_x1, bg_y1, fill_x2, bg_y2,
					                                       radius=10, fill="#00b0f0", outline="")
			
			# Update percentage label
			percent_lbl.configure(text=f"{int(value)}%")

		# Bind canvas resize to redraw
		progress_canvas.bind("<Configure>", lambda e: update_progress_bar(progress_var.get()))
		
		# Initialize progress bar
		update_progress_bar(0)

		# Configuration info label (between progress bar and log)
		config_info_label = tk.Label(batch_frame, text="", font=("Arial", 10), fg="#00b0f0", bg=BG_COLOR, 
		                             anchor="w", justify="left")
		config_info_label.pack(fill="x", padx=16, pady=(8, 8))

		# Log output area
		log_frame = tk.Frame(batch_frame, bg=BG_COLOR)
		log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
		log_frame.grid_rowconfigure(0, weight=1)
		log_frame.grid_columnconfigure(0, weight=1)
		log_text = tk.Text(log_frame, bg="#0E2F66", fg="#D8E4F7", height=10, wrap="word")
		log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
		log_text.configure(yscrollcommand=log_scroll.set, state="disabled")
		# Resaltado de errores en el log
		log_text.tag_configure("error", foreground="#FF6B6B", font=("Arial", 10, "bold"))
		log_text.grid(row=0, column=0, sticky="nsew")
		log_scroll.grid(row=0, column=1, sticky="ns")
		
		# ===== RT PROCESSING UI (real-time mode) =====
		rt_frame = tk.Frame(proc_container, bg=BG_COLOR)
		
		# Custom header for RT mode (replaces breadcrumb tabs)
		rt_custom_header = tk.Frame(rt_frame, bg=BG_COLOR, height=80)
		rt_custom_header.pack(fill=tk.X, side=tk.TOP, padx=16, pady=(18, 8))
		rt_custom_header.pack_propagate(False)
		
		# Mini logo on the left
		if mini_logo_img:
			rt_logo_label = tk.Label(rt_custom_header, image=mini_logo_img, bg=BG_COLOR)
			rt_logo_label.image = mini_logo_img
			rt_logo_label.pack(side=tk.LEFT, padx=(0, 20))
		
		# Title label
		rt_title_label = tk.Label(rt_custom_header, text="Procesamiento en Tiempo Real", 
		                          font=("Arial", 16, "bold"), fg="white", bg=BG_COLOR)
		rt_title_label.pack(side=tk.LEFT, padx=(0, 30))
		
		# Stats panel
		rt_stats_panel = tk.Frame(rt_custom_header, bg="#021e44", relief=tk.FLAT, bd=0)
		rt_stats_panel.pack(side=tk.LEFT, padx=(0, 30), fill=tk.Y)
		
		rt_cpu_label = tk.Label(rt_stats_panel, text="Uso de CPU: 80%", fg="white", bg="#021e44", 
		                        font=("Arial", 10, "bold"))
		rt_cpu_label.pack(side=tk.LEFT, padx=15, pady=10)
		
		rt_gpu_label = tk.Label(rt_stats_panel, text="Uso de GPU: 90%", fg="white", bg="#021e44", 
		                        font=("Arial", 10, "bold"))
		rt_gpu_label.pack(side=tk.LEFT, padx=15, pady=10)
		
		rt_fps_header_label = tk.Label(rt_stats_panel, text="FPS Total: 40", fg="white", bg="#021e44", 
		                               font=("Arial", 10, "bold"))
		rt_fps_header_label.pack(side=tk.LEFT, padx=15, pady=10)
		
		rt_cameras_count_label = tk.Label(rt_stats_panel, text="Cámaras: 4", fg="white", bg="#021e44", 
		                                  font=("Arial", 10, "bold"))
		rt_cameras_count_label.pack(side=tk.LEFT, padx=15, pady=10)
		
		# Finalizar button
		rt_finalizar_btn = tk.Button(rt_custom_header, text="Finalizar", font=("Arial", 11, "bold"), 
		                             fg="black", bg="#ffc735", activebackground="#e6b530", 
		                             activeforeground="black", relief=tk.FLAT, padx=30, pady=10,
		                             cursor="hand2")
		rt_finalizar_btn.pack(side=tk.RIGHT, padx=(0, 0))
		
		# Main content area with two columns (anchored below the custom header)
		rt_content = tk.Frame(rt_frame, bg="", height=600)
		rt_content.pack(fill="x", expand=False, side=tk.TOP, padx=16, pady=(15, 0))
		rt_content.pack_propagate(False)
		rt_content.grid_rowconfigure(0, weight=1)
		rt_content.grid_columnconfigure(0, weight=0, minsize=300)
		rt_content.grid_columnconfigure(1, weight=1)
		
		# LEFT COLUMN: Camera list
		rt_left_column = tk.Frame(rt_content, bg=PROC_CONTENT_BG, width=350)
		rt_left_column.grid(row=0, column=0, sticky="nsew", padx=(10, 10), pady=0)
		rt_left_column.grid_propagate(False)
		
		# Camera list frame (no scrollbar)
		rt_camera_list_frame = tk.Frame(rt_left_column, bg=PROC_CONTENT_BG)
		rt_camera_list_frame.pack(fill=tk.BOTH, expand=True)
		
		# Create camera items (10 cameras max)
		rt_camera_items = []
		for i in range(10):
			cam_item = tk.Frame(rt_camera_list_frame, bg="#002858", relief=tk.FLAT, bd=1, height=200)
			cam_item.pack(fill=tk.X, pady=(0, 20), padx=8)
			cam_item.pack_propagate(False)
			
			# Top row: name and status
			cam_top_row = tk.Frame(cam_item, bg="#002858")
			cam_top_row.pack(fill=tk.X, padx=15, pady=(15, 8))
			
			cam_name_label = tk.Label(cam_top_row, text=f"Cámara {i+1}", fg="white", bg="#556272", 
			                          font=("Arial", 14, "bold"), anchor="w")
			cam_name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
			
			# Create rounded status label using Canvas
			is_active = (i % 2 == 0)  # Simulating active/inactive cameras
			status_bg = "#7ec331" if is_active else "#ec5a2d"
			status_text = "OK" if is_active else "NOK"
			
			status_canvas = tk.Canvas(cam_top_row, width=60, height=28, bg="#002858", 
			                          highlightthickness=0)
			status_canvas.pack(side=tk.RIGHT)
			
			# Draw rounded rectangle for status
			radius = 8
			status_canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
			                         fill=status_bg, outline=status_bg)
			status_canvas.create_arc(60-radius*2, 0, 60, radius*2, start=0, extent=90, 
			                         fill=status_bg, outline=status_bg)
			status_canvas.create_arc(0, 28-radius*2, radius*2, 28, start=180, extent=90, 
			                         fill=status_bg, outline=status_bg)
			status_canvas.create_arc(60-radius*2, 28-radius*2, 60, 28, start=270, extent=90, 
			                         fill=status_bg, outline=status_bg)
			status_canvas.create_rectangle(radius, 0, 60-radius, 28, fill=status_bg, outline=status_bg)
			status_canvas.create_rectangle(0, radius, 60, 28-radius, fill=status_bg, outline=status_bg)
			status_canvas.create_text(30, 14, text=status_text, font=("Arial", 11, "bold"), fill="white")
			
			# Bottom row: textbox
			cam_textbox = tk.Entry(cam_item, bg="#354b69", fg="white", font=("Arial", 13), 
			                       relief=tk.FLAT, insertbackground="white")
			cam_textbox.insert(0, "Leoni PN")
			cam_textbox.pack(fill=tk.X, padx=15, pady=(0, 15))
			
			rt_camera_items.append({
				"frame": cam_item,
				"name_label": cam_name_label,
				"status_canvas": status_canvas,
				"textbox": cam_textbox,
				"is_active": is_active
			})
		
		def update_rt_camera_status():
			"""Update RT camera list status based on configured cameras"""
			# Count configured cameras
			num_configured = len(camera_widgets)
			
			for i, rt_item in enumerate(rt_camera_items):
				if i < num_configured:
					# Camera is configured
					cam_data = camera_widgets[i]
					is_connected = cam_data.get("connected", False) and cam_data.get("status") == "OK"
					
					# Update name
					rt_item["name_label"].config(text=cam_data.get("name", f"Cámara {i+1}"))
					
					# Update textbox
					# rt_item["textbox"].delete(0, tk.END)
					# rt_item["textbox"].insert(0, cam_data.get("ip", "No configurada"))
					
					# Update status canvas
					status_bg = "#7ec331" if is_connected else "#ec5a2d"
					status_text = "OK" if is_connected else "NOK"
					rt_item["is_active"] = is_connected
					
					# Redraw status canvas
					canvas = rt_item["status_canvas"]
					canvas.delete("all")
					radius = 8
					canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(60-radius*2, 0, 60, radius*2, start=0, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(0, 28-radius*2, radius*2, 28, start=180, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(60-radius*2, 28-radius*2, 60, 28, start=270, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_rectangle(radius, 0, 60-radius, 28, fill=status_bg, outline=status_bg)
					canvas.create_rectangle(0, radius, 60, 28-radius, fill=status_bg, outline=status_bg)
					canvas.create_text(30, 14, text=status_text, font=("Arial", 11, "bold"), fill="white")
				else:
					# No camera configured for this slot
					rt_item["name_label"].config(text=f"Cámara {i+1}")
					rt_item["textbox"].delete(0, tk.END)
					rt_item["textbox"].insert(0, "No configurada")
					rt_item["is_active"] = False
					
					# Set status to NOK
					canvas = rt_item["status_canvas"]
					canvas.delete("all")
					status_bg = "#ec5a2d"
					status_text = "NOK"
					radius = 8
					canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(60-radius*2, 0, 60, radius*2, start=0, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(0, 28-radius*2, radius*2, 28, start=180, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_arc(60-radius*2, 28-radius*2, 60, 28, start=270, extent=90, 
					                  fill=status_bg, outline=status_bg)
					canvas.create_rectangle(radius, 0, 60-radius, 28, fill=status_bg, outline=status_bg)
					canvas.create_rectangle(0, radius, 60, 28-radius, fill=status_bg, outline=status_bg)
					canvas.create_text(30, 14, text=status_text, font=("Arial", 11, "bold"), fill="white")
			
			# Reconfigure video feed grid layout based on number of cameras
			# Hide all frames first
			for frame in rt_camera_frames:
				frame.grid_forget()
			
			# Clear all row/column configurations (up to 4 rows x 3 columns for 10+ cameras)
			for i in range(4):
				rt_grid_container.grid_rowconfigure(i, weight=0)
			for i in range(3):
				rt_grid_container.grid_columnconfigure(i, weight=0)
			
			# Show and position only configured cameras
			if num_configured == 1:
				# 1 camera: full space
				rt_camera_frames[0].grid(row=0, column=0, sticky="nsew")
				rt_grid_container.grid_rowconfigure(0, weight=1)
				rt_grid_container.grid_columnconfigure(0, weight=1)
			elif num_configured == 2:
				# 2 cameras: side by side
				rt_camera_frames[0].grid(row=0, column=0, sticky="nsew", padx=(0, 2))
				rt_camera_frames[1].grid(row=0, column=1, sticky="nsew", padx=(2, 0))
				rt_grid_container.grid_rowconfigure(0, weight=1)
				rt_grid_container.grid_columnconfigure(0, weight=1)
				rt_grid_container.grid_columnconfigure(1, weight=1)
			elif num_configured == 3:
				# 3 cameras: 1 on top, 2 on bottom
				rt_camera_frames[0].grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 2))
				rt_camera_frames[1].grid(row=1, column=0, sticky="nsew", padx=(0, 2), pady=(2, 0))
				rt_camera_frames[2].grid(row=1, column=1, sticky="nsew", padx=(2, 0), pady=(2, 0))
				rt_grid_container.grid_rowconfigure(0, weight=1)
				rt_grid_container.grid_rowconfigure(1, weight=1)
				rt_grid_container.grid_columnconfigure(0, weight=1)
				rt_grid_container.grid_columnconfigure(1, weight=1)
			elif num_configured == 4:
				# 4 cameras: 2x2 grid
				rt_camera_frames[0].grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))
				rt_camera_frames[1].grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=(0, 2))
				rt_camera_frames[2].grid(row=1, column=0, sticky="nsew", padx=(0, 2), pady=(2, 0))
				rt_camera_frames[3].grid(row=1, column=1, sticky="nsew", padx=(2, 0), pady=(2, 0))
				rt_grid_container.grid_rowconfigure(0, weight=1)
				rt_grid_container.grid_rowconfigure(1, weight=1)
				rt_grid_container.grid_columnconfigure(0, weight=1)
				rt_grid_container.grid_columnconfigure(1, weight=1)
			elif num_configured <= 6:
				# 5-6 cameras: 2x3 grid
				for idx in range(num_configured):
					row = idx // 3
					col = idx % 3
					padx_left = (0, 2) if col < 2 else (2, 0)
					pady_top = (0, 2) if row < 1 else (2, 0)
					rt_camera_frames[idx].grid(row=row, column=col, sticky="nsew", padx=padx_left, pady=pady_top)
				for r in range(2):
					rt_grid_container.grid_rowconfigure(r, weight=1)
				for c in range(3):
					rt_grid_container.grid_columnconfigure(c, weight=1)
			elif num_configured <= 9:
				# 7-9 cameras: 3x3 grid
				for idx in range(num_configured):
					row = idx // 3
					col = idx % 3
					padx_left = (0, 2) if col < 2 else (2, 0)
					pady_top = (0, 2) if row < 2 else (2, 0)
					rt_camera_frames[idx].grid(row=row, column=col, sticky="nsew", padx=padx_left, pady=pady_top)
				for r in range(3):
					rt_grid_container.grid_rowconfigure(r, weight=1)
				for c in range(3):
					rt_grid_container.grid_columnconfigure(c, weight=1)
			elif num_configured == 10:
				# 10 cameras: 4x3 grid (3+3+2+2)
				for idx in range(num_configured):
					row = idx // 3
					col = idx % 3
					padx_left = (0, 2) if col < 2 else (2, 0)
					pady_top = (0, 2) if row < 3 else (2, 0)
					rt_camera_frames[idx].grid(row=row, column=col, sticky="nsew", padx=padx_left, pady=pady_top)
				for r in range(4):
					rt_grid_container.grid_rowconfigure(r, weight=1)
				for c in range(3):
					rt_grid_container.grid_columnconfigure(c, weight=1)
		
		# RIGHT COLUMN: Video panels and controls
		rt_right_column = tk.Frame(rt_content, bg=PROC_CONTENT_BG)
		rt_right_column.grid(row=0, column=1, sticky="nsew", padx=(10, 10), pady=10)
		rt_right_column.grid_rowconfigure(0, weight=1)
		rt_right_column.grid_columnconfigure(0, weight=1)
		
		# Label for background processing mode
		rt_light_mode_label = tk.Label(
			rt_right_column,
			text="Procesamiento en Segundo Plano\n\nActivado para optimizar rendimiento.\nLos frames no se mostrarán pero el CSV se generará normalmente.",
			font=("Arial", 16, "bold"),
			fg="#7ec331",
			bg=PROC_CONTENT_BG,
			justify="center"
		)
		# Not packed initially, will show only in background mode
		
		# Camera feeds grid (dynamic layout based on configured cameras)
		rt_grid = tk.Frame(rt_right_column, bg=PROC_CONTENT_BG)
		rt_grid.pack(fill="both", expand=True, pady=(0, 12))
		
		# Container for grid view (dynamic layout)
		rt_grid_container = tk.Frame(rt_grid, bg=BG_COLOR)
		rt_grid_container.pack(fill="both", expand=True)
		
		# Camera label widgets (will hold processed frames)
		rt_camera_labels = []
		rt_camera_frames = []  # Keep reference to frames for management
		rt_popup_windows = [None] * 10  # Track popup windows for each camera (max 10)
		rt_popup_labels = [None] * 10  # Track labels in popup windows (max 10)
		rt_camera_active = [False] * 10  # Track which cameras are processing (max 10)
		
		# Helper function to update camera label with PIL image maintaining aspect ratio
		def update_camera_image(label_idx, pil_image):
			"""Update camera label with PIL image, scaling to fit available space."""
			if label_idx >= len(rt_camera_labels):
				return
			label = rt_camera_labels[label_idx]
			
			# Store original image
			label._current_pil_image = pil_image
			
			# Get current label size
			label.update_idletasks()
			width = label.winfo_width()
			height = label.winfo_height()
			
			if width > 1 and height > 1 and pil_image:
				# Calculate scale to fit while maintaining aspect ratio
				img_w, img_h = pil_image.size
				scale_w = width / img_w
				scale_h = height / img_h
				scale = min(scale_w, scale_h)
				new_w = int(img_w * scale)
				new_h = int(img_h * scale)
				
				if new_w > 0 and new_h > 0:
					resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
					photo = ImageTk.PhotoImage(resized)
					label.configure(image=photo, text="")
					label.image = photo  # Keep reference
		
		for i in range(10):
			# Dynamic size frame that respects aspect ratio
			cam_frame = tk.Frame(rt_grid_container, bg="#0E2F66", relief=tk.SUNKEN, bd=2)
			# Don't grid yet - will be positioned by update_rt_camera_status()
			
			# Label configured to scale images maintaining aspect ratio
			cam_label = tk.Label(cam_frame, bg="#0E2F66", text=f"Cámara {i+1}\nEsperando...", 
			                     fg="#A0A9B8", font=("Arial", 12), cursor="hand2",
			                     compound="center", anchor="center")
			cam_label.pack(fill="both", expand=True)
			
			# Bind to resize event to maintain aspect ratio
			def make_resize_handler(label_widget, cam_idx):
				def on_resize(event):
					# Only resize if label has an image and has meaningful size
					if hasattr(label_widget, '_current_pil_image') and label_widget._current_pil_image:
						if event.width > 1 and event.height > 1:
							# Use the update function to rescale
							update_camera_image(cam_idx, label_widget._current_pil_image)
				return on_resize
			
			cam_label.bind("<Configure>", make_resize_handler(cam_label, i))
			rt_camera_labels.append(cam_label)
			rt_camera_frames.append(cam_frame)
			
			# Click handler to create fullscreen popup window
			def make_click_handler(cam_idx):
				def on_click(event):
					# Check if popup already exists for this camera
					if rt_popup_windows[cam_idx] is not None:
						try:
							if rt_popup_windows[cam_idx].winfo_exists():
								return  # Popup already open
						except Exception:
							pass
					
					# Create fullscreen popup window
					popup = tk.Toplevel(win)
					popup.title(f"Cámara {cam_idx + 1}")
					popup.configure(bg="#000000")
					
					# Make fullscreen
					try:
						popup.attributes("-fullscreen", True)
					except Exception:
						try:
							popup.state("zoomed")
						except Exception:
							pass
					
					# Create label to display camera feed
					popup_label = tk.Label(popup, bg="#000000", cursor="hand2")
					popup_label.pack(fill="both", expand=True)
					
					# Store references
					rt_popup_windows[cam_idx] = popup
					rt_popup_labels[cam_idx] = popup_label
					
					# Click handler to close popup
					def close_popup(e):
						try:
							popup.destroy()
							rt_popup_windows[cam_idx] = None
							rt_popup_labels[cam_idx] = None
						except Exception:
							pass
					
					popup_label.bind("<Button-1>", close_popup)
					
					# Close popup on window close
					def on_popup_close():
						try:
							popup.destroy()
							rt_popup_windows[cam_idx] = None
							rt_popup_labels[cam_idx] = None
						except Exception:
							pass
					
					popup.protocol("WM_DELETE_WINDOW", on_popup_close)
				return on_click
			
			cam_label.bind("<Button-1>", make_click_handler(i))
		
		# RT runtime control checkboxes (synced with config tab)
		rt_controls_frame = tk.Frame(rt_right_column, bg=PROC_CONTENT_BG)
		rt_controls_frame.pack(padx=16, pady=(0, 8))
		
		# Shared state for runtime control (accessed by processing threads)
		rt_runtime_config = {
			"difuminar_caras": False,
			"seguridad": False,
			"background_processing": False
		}
		
		# Sync config tab checkboxes with rt_runtime_config
		def update_config_face_blur():
			rt_runtime_config["difuminar_caras"] = face_blur_var.get()
		
		def update_config_security():
			rt_runtime_config["seguridad"] = security_var.get()
		
		def update_config_background_processing():
			rt_runtime_config["background_processing"] = background_processing_var.get()
		
		# Configure callbacks for config tab checkboxes
		security_chk.configure(command=update_config_security)
		face_blur_chk.configure(command=update_config_face_blur)
		background_processing_chk.configure(command=update_config_background_processing)
		
		# Create checkboxes that sync with config tab and update shared state
		rt_face_blur_var = tk.BooleanVar(value=False)
		rt_security_var = tk.BooleanVar(value=False)
		rt_light_mode_var = tk.BooleanVar(value=False)
		
		def update_runtime_face_blur():
			rt_runtime_config["difuminar_caras"] = rt_face_blur_var.get()
			face_blur_var.set(rt_face_blur_var.get())  # Sync with config tab
		
		def update_runtime_security():
			rt_runtime_config["seguridad"] = rt_security_var.get()
			security_var.set(rt_security_var.get())  # Sync with config tab
		
		def update_runtime_light_mode():
			"""Toggle between light mode (background processing) and normal mode."""
			rt_runtime_config["background_processing"] = rt_light_mode_var.get()
			background_processing_var.set(rt_light_mode_var.get())  # Sync with config tab
			_set_worker_processing_state(mode=("light" if rt_light_mode_var.get() else "normal"))
			if rt_light_mode_var.get():
				# Hide grid, show light mode label
				rt_grid.pack_forget()
				rt_light_mode_label.pack(fill="both", expand=True, pady=(0, 12))
			else:
				# Hide light mode label, show grid
				rt_light_mode_label.pack_forget()
				rt_grid.pack(fill="both", expand=True, pady=(0, 12))
		
		rt_face_blur_chk = tk.Checkbutton(
			rt_controls_frame,
			text="Difuminar Caras",
			variable=rt_face_blur_var,
			command=update_runtime_face_blur,
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG,
			selectcolor=PROC_CONTENT_BG,
			activebackground=PROC_CONTENT_BG,
			activeforeground=FG_COLOR
		)
		rt_security_chk = tk.Checkbutton(
			rt_controls_frame,
			text="Seguridad",
			variable=rt_security_var,
			command=update_runtime_security,
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG,
			selectcolor=PROC_CONTENT_BG,
			activebackground=PROC_CONTENT_BG,
			activeforeground=FG_COLOR
		)
		rt_light_mode_chk = tk.Checkbutton(
			rt_controls_frame,
			text="Modo Light (Procesamiento en segundo plano)",
			variable=rt_light_mode_var,
			command=update_runtime_light_mode,
			fg=FG_COLOR,
			bg=PROC_CONTENT_BG,
			selectcolor=PROC_CONTENT_BG,
			activebackground=PROC_CONTENT_BG,
			activeforeground=FG_COLOR
		)
		
		rt_face_blur_chk.pack(side=tk.LEFT, padx=(0, 16))
		rt_security_chk.pack(side=tk.LEFT, padx=(0, 16))
		rt_light_mode_chk.pack(side=tk.LEFT)

		def _apply_light_mode_from_remote(is_light: bool):
			rt_light_mode_var.set(bool(is_light))
			update_runtime_light_mode()

		WORKER_CONTROL_STATE["callbacks"]["set_light_mode"] = (
			lambda is_light: win.after(0, lambda: _apply_light_mode_from_remote(bool(is_light)))
		)

		def sync_rt_controls_from_config():
			rt_face_blur_var.set(face_blur_var.get())
			rt_security_var.set(security_var.get())
			rt_light_mode_var.set(background_processing_var.get())
			rt_runtime_config["difuminar_caras"] = rt_face_blur_var.get()
			rt_runtime_config["seguridad"] = rt_security_var.get()
			rt_runtime_config["background_processing"] = rt_light_mode_var.get()

		sync_rt_controls_state["func"] = sync_rt_controls_from_config

		# Dashboard hyperlink (centered at bottom)
		dashboard_frame = tk.Frame(rt_right_column, bg=PROC_CONTENT_BG)
		dashboard_frame.pack(padx=16, pady=(0, 16))
		dashboard_link = tk.Label(
			dashboard_frame,
			text="Visualizar Dashboard",
			fg="#4A90E2",  # Blue color for hyperlink
			bg=PROC_CONTENT_BG,
			font=("Arial", 12, "underline"),
			cursor="hand2"
		)
		dashboard_link.pack()
		
		def open_dashboard_browser(_=None):
			"""Open dashboard URL in default browser."""
			import webbrowser
			try:
				webbrowser.open("http://10.156.236.208:8501")
			except Exception as e:
				print(f"Error opening browser: {e}")
		
		dashboard_link.bind("<Button-1>", open_dashboard_browser)
		
		# Hover effect for hyperlink
		def on_enter_link(_):
			dashboard_link.configure(fg="#6BB6FF")  # Lighter blue on hover
		
		def on_leave_link(_):
			dashboard_link.configure(fg="#4A90E2")  # Original blue
		
		dashboard_link.bind("<Enter>", on_enter_link)
		dashboard_link.bind("<Leave>", on_leave_link)
		
		# Hide both frames initially (will show appropriate one when processing starts)
		batch_frame.pack_forget()
		rt_frame.pack_forget()

		def go_to_fin():
			# Solo permite avanzar si progreso completo
			if progress_var.get() >= 100:
				# Ajustar current_step y refrescar tabs
				nonlocal current_step
				current_step = 6  # índice de pestaña Fin
				update_tabs_state()
		
		# ===== RT processing callbacks =====
		def go_to_fin_rt():
			# RT mode: allow advance to Fin tab
			nonlocal current_step
			current_step = 5
			update_tabs_state()
		
		rt_finalizar_btn.configure(command=go_to_fin_rt)
		
		# RT processing state
		rt_threads_active = {"value": False}
		rt_stop_event = threading.Event()
		streamlit_process = {"p": None}  # Store streamlit subprocess
		
		def _stop_rt_processing():
			rt_stop_event.set()
			rt_threads_active["value"] = False
			_set_worker_processing_state(rt_running=False, camera_count=0, groups=[])
			can_advance[4] = True
			update_nav_state()
			
			# Stop streamlit process if running
			if streamlit_process["p"] and streamlit_process["p"].poll() is None:
				try:
					streamlit_process["p"].terminate()
				except Exception:
					pass

		WORKER_CONTROL_STATE["callbacks"]["stop_processing"] = lambda: win.after(0, _stop_rt_processing)

		def _start_group_from_remote(group_name: str):
			"""Load a camera group by linea name and start RT processing (called from /start_group HTTP)."""
			group_name_u = str(group_name).strip().upper()
			groups_dir = Path("camera_groups")
			if not groups_dir.exists():
				return
			target_files = []
			for gfile in groups_dir.glob("*.json"):
				try:
					with open(gfile, "r", encoding="utf-8") as f:
						data = json.load(f)
					linea = str(data.get("metadata", {}).get("linea", "")).strip().upper()
					if linea == group_name_u:
						target_files.append(gfile)
				except Exception:
					pass
			if not target_files:
				return
			camera_widgets.clear()
			for item in list(camera_list_items):
				try:
					item["frame"].destroy()
				except Exception:
					pass
			camera_list_items.clear()
			roi_state["rois_map"].clear()
			camera_offset = 0
			loaded_group_metadata = []
			last_valid_model_path = ""
			last_settings = {}
			for group_file in target_files:
				with open(group_file, "r", encoding="utf-8") as f:
					data = json.load(f)
				group_metadata_item = data.get("metadata", {})
				group_linea = str(group_metadata_item.get("linea", "")).strip()
				cameras = data.get("cameras", [])
				for cam_data in cameras:
					cam_payload = cam_data.copy()
					if group_linea and not str(cam_payload.get("linea", "")).strip():
						cam_payload["linea"] = group_linea
					camera_widgets.append(cam_payload)
					idx = len(camera_widgets) - 1
					list_item = create_camera_list_item(list_container, idx, cam_payload)
					camera_list_items.append(list_item)
				for cam_index, cam_data in enumerate(cameras):
					if "rois" in cam_data:
						merged_index = camera_offset + cam_index
						roi_state["rois_map"][merged_index] = cam_data["rois"]
				camera_offset += len(cameras)
				model_path = data.get("model_path", "")
				if model_path and Path(model_path).exists():
					last_valid_model_path = model_path
				settings = data.get("settings", {})
				if settings:
					last_settings = settings
				metadata = data.get("metadata", {})
				if metadata:
					loaded_group_metadata.append(metadata)
			if camera_widgets:
				selected_camera_index[0] = 0
				load_camera_to_form(0)
				update_list_selection()
				update_camera_button_state()
			if last_valid_model_path:
				MODEL_FILE["value"] = last_valid_model_path
			if last_settings:
				security_var.set(last_settings.get("security", False))
				face_blur_var.set(last_settings.get("face_blur", False))
				background_processing_var.set(last_settings.get("background_processing", False))
				processing_level_var.set(last_settings.get("processing_level", 1))
			group_metadata.clear()
			if len(loaded_group_metadata) == 1:
				group_metadata.update(loaded_group_metadata[0])
			elif len(loaded_group_metadata) > 1:
				group_metadata.update({"groups": loaded_group_metadata})
			update_advance_state()
			win.after(150, lambda: perform_process())

		WORKER_CONTROL_STATE["callbacks"]["start_group"] = (
			lambda g: win.after(0, lambda: _start_group_from_remote(g))
		)
		
		# ===== Batch processing callbacks =====
		# (Buttons removed - cancel, rerun, go to fin)

		def _update_processing_nav_state():
			# Deshabilitar botón 'Siguiente' mientras se procesa; usar botón dedicado
			if current_step == 5:
				# Disable next button container during processing
				pass  # next_btn_container is managed by update_nav_state
			update_nav_state_wrapper()

		# Hilo y lectura de progreso desde stdout
		pipeline_proc = {"p": None}
		progress_queue = []  # simple buffer
		# log_queue moved earlier before start_processing_ui
		canceled = {"value": False}
		last_config_file = {"value": None}

		# _append_log_line and _flush_log moved earlier before start_processing_ui

		def launch_real_pipeline(cfg):
			try:
				import json, threading, time
				from datetime import datetime
				from pathlib import Path
				
				# Validate configuration
				if not cfg:
					raise ValueError("Configuración vacía recibida en launch_real_pipeline")
				
				_append_log_line(f"[DEBUG] Iniciando pipeline con {len(cfg)} parámetros de configuración")
				_append_log_line(f"[DEBUG] Modo frozen: {getattr(sys, 'frozen', False)}")
				_append_log_line(f"[DEBUG] Base root: {_base_root()}")
				
				# Update start date label
				now = datetime.now()
				date_str = now.strftime("%d/%m/%Y %I:%M %p")
				start_date_label.configure(text=f"Fecha de Inicio: {date_str}")
				
				# Update configuration info label
				classify_weights_path = cfg.get("classify_weights", "")
				if not classify_weights_path and not cfg.get("skip_classification", False):
					raise ValueError("No se especificaron pesos de clasificación en la configuración")
				
				_append_log_line(f"[DEBUG] Pesos de clasificación: {classify_weights_path}")
				
				if classify_weights_path:
					model_name = Path(classify_weights_path).stem
				else:
					model_name = "N/A"
				
				face_blur = cfg.get("difuminar_caras", False)
				security_enabled = cfg.get("seguridad", False)
				processing_level = cfg.get("processing_level", 1)
				
				level_names = {0: "Ligero", 1: "Balanceado", 2: "Pesado", 3: "Personalizable"}
				level_name = level_names.get(processing_level, "Balanceado")
				
				config_parts = [f"Modelo: {model_name}.pt"]
				if face_blur:
					config_parts.append("Difuminar rostros")
				if security_enabled:
					config_parts.append("Ergonomía")
				
				config_text = ", ".join(config_parts) + f" | Tipo de procesamiento: {level_name}"
				config_info_label.configure(text=config_text)
				
				# Save config for debugging
				try:
					cfg_dir = Path(_base_root()) / "logs"
					cfg_dir.mkdir(exist_ok=True)
					last_path = cfg_dir / "gui_last_config.json"
					_append_log_line(f"[DEBUG] Guardando config en: {last_path}")
					with open(last_path, "w", encoding="utf-8") as f:
						json.dump(cfg, f, indent=2, ensure_ascii=False)
					last_config_file["value"] = str(last_path)
					_append_log_line(f"[DEBUG] Configuración guardada exitosamente")
				except Exception as e:
					_append_log_line(f"[WARN] Error guardando configuración: {e}")
				
				# Create VideoProcessor instance
				try:
					_append_log_line(f"[DEBUG] Creando instancia de VideoProcessor...")
					processor = VideoProcessor(cfg)
					_append_log_line(f"[DEBUG] VideoProcessor creado exitosamente")
				except Exception as e:
					import traceback
					error_detail = traceback.format_exc()
					_append_log_line(f"[ERROR] Error inicializando VideoProcessor: {e}")
					_append_log_line(f"[ERROR] Traceback:\n{error_detail}")
					messagebox.showerror(
						"Error VideoProcessor",
						f"No se pudo crear el VideoProcessor:\n\n{str(e)}\n\nVer log para detalles."
					)
					return
				
				# Progress callback to update GUI
				def progress_callback(pct):
					progress_queue.append(pct)
				
				# Log callback to update console text
				def log_callback(msg):
					_append_log_line(msg)
				
				# Run processing in background thread
				def run_processing():
					try:
						_append_log_line("[GUI] Iniciando procesamiento con VideoProcessor integrado...")
						processor.process_videos(progress_callback=progress_callback, log_callback=log_callback)
						_append_log_line("[GUI] ✅ Procesamiento completado exitosamente")
						progress_queue.append(100.0)
					except Exception as e:
						_append_log_line(f"[GUI] ❌ Error en procesamiento: {e}")
						import traceback
						_append_log_line(traceback.format_exc())
				
				# Start processing thread
				try:
					_append_log_line(f"[DEBUG] Iniciando thread de procesamiento...")
					proc_thread = threading.Thread(target=run_processing, daemon=True)
					proc_thread.start()
					pipeline_proc["p"] = proc_thread  # Store thread instead of subprocess
					_append_log_line(f"[DEBUG] Thread iniciado exitosamente")
				except Exception as e:
					import traceback
					error_detail = traceback.format_exc()
					_append_log_line(f"[ERROR] Error iniciando thread: {e}")
					_append_log_line(f"[ERROR] Traceback:\n{error_detail}")
					messagebox.showerror(
						"Error Thread",
						f"No se pudo iniciar el thread de procesamiento:\n\n{str(e)}\n\nVer log para detalles."
					)
					return
				
				# Start progress update loop
				win.after(200, apply_progress_updates)
				
			except Exception as e:
				import traceback
				error_msg = f"Error crítico en launch_real_pipeline:\n\n{str(e)}\n\nDetalle técnico:\n{traceback.format_exc()}"
				messagebox.showerror(
					"Error Crítico de Pipeline",
					error_msg
				)
				print(f"[ERROR] launch_real_pipeline failed: {error_msg}")
				_append_log_line(f"[ERROR] {error_msg}")

		def apply_progress_updates():
			if current_step != 4:
				return
			_flush_log()
			while progress_queue:
				val = progress_queue.pop(0)
				progress_var.set(int(val))
				update_progress_bar(val)
				if val >= 100.0:
					can_advance[4] = True
					update_nav_state()
					return
			# Check if thread is still alive
			p = pipeline_proc.get("p")
			if p and isinstance(p, threading.Thread) and p.is_alive():
				win.after(300, apply_progress_updates)
			else:
				_flush_log()
				if canceled["value"]:
					_append_log_line("[GUI] Proceso cancelado por el usuario.")
					_flush_log()
					return
				if progress_var.get() < 100 and not progress_queue:
					_append_log_line("[GUI] Proceso finalizado sin progreso (ver líneas arriba para diagnóstico).")
					_flush_log()
					return
				can_advance[4] = True
				update_nav_state()

		# ===== RT PROCESSING IMPLEMENTATION =====
		def begin_rt_processing(config_data):
			"""Start real-time processing for configured cameras."""
			if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
				def show_error():
					rt_camera_labels[0].configure(text="Error: torch o ultralytics\nno disponible")
				win.after(0, show_error)
				return
			
			# Reset RT state
			rt_stop_event.clear()
			rt_threads_active["value"] = True
			
			# Get RT configuration
			camera_urls = rtsp_urls_list  # List of validated RTSP URLs
			num_cameras = len(camera_urls)
			active_groups = []
			seen_groups = set()
			for cam in camera_urls:
				g_raw = str((cam or {}).get("linea", "")).strip()
				if not g_raw:
					continue
				for g_part in [p.strip() for p in g_raw.replace(";", ",").split(",") if p.strip()]:
					g_norm = g_part.upper()
					if g_norm in KNOWN_CONTROL_PANEL_GROUPS and g_norm not in seen_groups:
						seen_groups.add(g_norm)
						active_groups.append(g_norm)

			if not active_groups:
				metadata = config_data.get("metadata", {})
				meta_line = str(metadata.get("linea", "")).strip()
				if meta_line:
					for g_part in [p.strip() for p in meta_line.replace(";", ",").split(",") if p.strip()]:
						g_norm = g_part.upper()
						if g_norm in KNOWN_CONTROL_PANEL_GROUPS and g_norm not in seen_groups:
							seen_groups.add(g_norm)
							active_groups.append(g_norm)

				meta_groups = metadata.get("groups", [])
				if isinstance(meta_groups, (list, tuple)):
					for meta_group in meta_groups:
						if not isinstance(meta_group, dict):
							continue
						meta_group_line = str(meta_group.get("linea", "")).strip()
						if not meta_group_line:
							continue
						for g_part in [p.strip() for p in meta_group_line.replace(";", ",").split(",") if p.strip()]:
							g_norm = g_part.upper()
							if g_norm in KNOWN_CONTROL_PANEL_GROUPS and g_norm not in seen_groups:
								seen_groups.add(g_norm)
								active_groups.append(g_norm)

			_set_worker_processing_state(
				rt_running=True,
				camera_count=num_cameras,
				mode=("light" if config_data.get("background_processing", False) else "normal"),
				groups=active_groups,
			)
			rois_map = config_data.get("rois_by_video", {})
			print(f"[DEBUG] ROIs en begin_rt_processing: {len(rois_map)} cámaras con ROIs configurados")
			for cam_idx, rois_list in rois_map.items():
				print(f"[DEBUG]   Cámara {cam_idx}: {len(rois_list)} ROIs")
			
			# Initialize runtime config from config tab values
			rt_runtime_config["difuminar_caras"] = config_data.get("difuminar_caras", False)
			rt_runtime_config["seguridad"] = config_data.get("seguridad", False)
			rt_runtime_config["background_processing"] = config_data.get("background_processing", False)
		
			# Get frameskip configuration
			frameskip_percentage = config_data.get("frameskip_percentage", 0)
		
			# Calculate dynamic AGGREGATION_FRAMES based on frameskip
			# Goal: Maintain ~1 second time windows regardless of frameskip
			# Base: 25 frames at 25 fps = 1 second (0% skip)
			# With skip, we need fewer processed frames to cover same real time
			if frameskip_percentage > 0:
				# Effective frame rate after skipping
				effective_fps = 25 * (1 - frameskip_percentage / 100)
				# Round to nearest integer, minimum 5 frames for statistical validity
				AGGREGATION_FRAMES = max(5, round(effective_fps))
				print(f"[RT] Frameskip {frameskip_percentage}% → AGGREGATION_FRAMES ajustado a {AGGREGATION_FRAMES} frames ({AGGREGATION_FRAMES/effective_fps:.2f}s)")
			else:
				AGGREGATION_FRAMES = 25
				print(f"[RT] Sin frameskip → AGGREGATION_FRAMES = {AGGREGATION_FRAMES} frames (1.0s)")
		
			
			# Set initial UI state based on background_processing mode
			if rt_runtime_config["background_processing"]:
				rt_grid.pack_forget()
				rt_light_mode_label.pack(fill="both", expand=True, pady=(0, 12))
			else:
				rt_light_mode_label.pack_forget()
				rt_grid.pack(fill="both", expand=True, pady=(0, 12))
			
			# CSV output path from configuration tab (base_dir specified by user)
			csv_output_dir = config_data.get("csv_dir", "output")
			csv_output_path = Path(csv_output_dir) / "real_time_predictions.csv"
			
			# Launch head count dashboard automatically with CSV path
			try:
				import subprocess
				dashboard_script = Path(__file__).parent / "dashboard_head_count.py"
				
				if dashboard_script.exists():
					# Kill any existing dashboard process if running
					existing = None
					try:
						existing = streamlit_process.get("p") if isinstance(streamlit_process, dict) else None
					except Exception:
						existing = None
					if existing and getattr(existing, "poll", lambda: 1)() is None:
						try:
							existing.terminate()
						except Exception:
							try:
								existing.kill()
							except Exception:
								pass
					# Launch dashboard with CSV path argument
					cmd = [
						sys.executable,
						str(dashboard_script),
						"--csv", str(csv_output_path)
					]
					p = subprocess.Popen(
						cmd,
						creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
						stdout=subprocess.DEVNULL,
						stderr=subprocess.DEVNULL
					)
					streamlit_process["p"] = p
					try:
						DASHBOARD_CHILD_PROCS.append(p)
					except Exception:
						pass
					print(f"[GUI] Head count dashboard lanzado: {' '.join(cmd)}")
			except Exception as e:
				print(f"[GUI] Error lanzando head count dashboard: {e}")

			# Launch conveyor reporter (background Flask) to show entry/exit windows
			try:
				reporter_script = Path(__file__).parent / "conveyor_reporter.py"
				if reporter_script.exists():
					# Kill any existing reporter processes we tracked
					try:
						for rp in list(REPORTER_CHILD_PROCS):
							if rp is None:
								continue
							if getattr(rp, 'poll', lambda: 1)() is None:
								try:
									rp.terminate()
								except Exception:
									try:
										rp.kill()
									except Exception:
										pass
					except Exception:
						pass
					# Determine a conveyor name from metadata to pass to reporter
					rt_meta = config_data.get("metadata", {})
					_meta_parts = [str(rt_meta.get(k, "") or "").strip() for k in ("linea", "segmento", "area", "planta")]
					_meta_parts = [p for p in _meta_parts if p]
					_conveyor_name = " - ".join(_meta_parts) if _meta_parts else ""
					# Launch reporter with same CSV path and conveyor name
					cmd2 = [sys.executable, str(reporter_script), "--csv", str(csv_output_path), "--conveyor", _conveyor_name]
					p2 = subprocess.Popen(
						cmd2,
						creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
						stdout=subprocess.DEVNULL,
						stderr=subprocess.DEVNULL
					)
					try:
						REPORTER_CHILD_PROCS.append(p2)
					except Exception:
						pass
					print(f"[GUI] Conveyor reporter lanzado: {' '.join(cmd2)}")
			except Exception as e:
				print(f"[GUI] Error lanzando conveyor reporter: {e}")
			
			# Shared state for CSV writing
			csv_lock = threading.Lock()
			predictions_buffer = []  # Buffer to accumulate predictions
			last_csv_write = {"time": time.time()}
			
			# Extract metadata from config
			rt_metadata = config_data.get("metadata", {})
			rt_linea = rt_metadata.get("linea", "")
			rt_segmento = rt_metadata.get("segmento", "")
			rt_area = rt_metadata.get("area", "")
			rt_planta = rt_metadata.get("planta", "")
			
			# Model paths
			repo_root = Path(__file__).parent.parent
			person_det_weights = repo_root / "yolo11x.pt"
			classify_weights = Path(MODEL_FILE["value"]) if (MODEL_FILE.get("value") and model_loaded_var.get()) else None
			
			# Processing parameters
			IMG_SIZE = 512
			CONF_PERSON = 0.60
			CONF_STATE = 0.50
			DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
			USE_HALF = False
			
			# Light mode optimizations (EDIT THESE VARIABLES TO TUNE PERFORMANCE)
			LIGHT_MODE_FPS_LIMIT = 10  # Maximum FPS in light mode (lower = less GPU usage)
			LIGHT_MODE_FRAMESKIP = 90  # Force frameskip percentage in light mode (0-90)
			LIGHT_MODE_RESOLUTION_SCALE = 0.5  # Resolution scale for inference in light mode (0.5 = half resolution)
			
			# Aggregation settings - NOTA: AGGREGATION_FRAMES se calcula dinámicamente en begin_rt_processing()
			# basado en el frameskip_percentage para mantener ventanas temporales consistentes de ~1 segundo
			
			# Ergonomics settings
			ERGO_MODEL_COMPLEXITY = 1
			ERGO_MAX_POSE_MODELS = 9
			ERGO_ARM_ANGLE_THRESH = 90.0
			ERGO_BACK_ANGLE_THRESH = 160.0
			
			# Helper functions (from Real-time-Proccessing-GUI.py)
			def get_New_Class(old_class: str):
				if old_class in {"Ruteo", "Enteipado", "working", "Convolute", "Insercion", "Tomar Material", "VA", "Trabajando", "trabajando"}:
					return "VA", (0, 255, 0)
				if old_class in {"NVA", "idle", "Celular"}:
					return "NVA", (0, 0, 255)
				return old_class, (192, 192, 192)
			
			def clamp_box(x1, y1, x2, y2, w, h):
				x1 = max(0, min(int(x1), w - 1))
				y1 = max(0, min(int(y1), h - 1))
				x2 = max(0, min(int(x2), w - 1))
				y2 = max(0, min(int(y2), h - 1))
				if x2 <= x1 or y2 <= y1:
					return None
				return x1, y1, x2, y2
			
			def _point_in_any_roi(rois, point):
				if not rois:
					return None
				x, y = point
				for roi in rois:
					try:
						if cv2.pointPolygonTest(roi["coords"], (int(x), int(y)), False) >= 0:
							return roi["name"]
					except Exception:
						continue
				return None
			
			def _adjust_probs_by_roi(probs: np.ndarray, class_names, roi_name: str):
				"""Ajusta probabilidades basado en tipo de ROI con simetría ±10%.
				
				Reglas:
				- Conveyor ('conv'): VA +10%, NVA -10%
				- Estación ('est'): VA +10%, NVA -10%
				- Aisle ('aisle'): NVA +10%, VA -10%
				- Otros: Sin ajuste
				"""
				if probs is None or roi_name is None:
					return probs
				roi_lower = roi_name.lower()
				is_conveyor = "conv" in roi_lower  # Acepta: conv, conveyor, convolute
				is_aisle = "aisle" in roi_lower
				is_est = "est" in roi_lower  # Acepta: est, estacion, station
				if not (is_conveyor or is_aisle or is_est):
					return probs
				adj = probs.copy().astype(float)
				for i, cname in enumerate(class_names):
					mapped, _ = get_New_Class(cname)
					if is_conveyor or is_est:
						# Conveyor y Estaciones: más probable trabajar (VA)
						if mapped == "VA":
							adj[i] = min(adj[i] + 0.10, 1.0)
						elif mapped == "NVA":
							adj[i] = max(adj[i] - 0.10, 0.0)
					elif is_aisle:
						# Pasillos: más probable idle (NVA)
						if mapped == "NVA":
							adj[i] = min(adj[i] + 0.10, 1.0)
						elif mapped == "VA":
							adj[i] = max(adj[i] - 0.10, 0.0)
				s = float(adj.sum())
				if s > 0:
					adj = adj / s
				return adj
			
			# Ergonomics classification function
			def classify_ergonomics(landmarks):
				"""Classify ergonomics based on pose landmarks (OK/NG)."""
				def get_angle(a, b, c):
					a = np.array(a, dtype=float)
					b = np.array(b, dtype=float)
					c = np.array(c, dtype=float)
					ba = a - b
					bc = c - b
					denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
					cos_angle = float(np.dot(ba, bc) / denom)
					return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
				
				def to_xyz(idx):
					lm = landmarks[idx]
					return [lm.x, lm.y, lm.z]
				
				def midpoint(p, q):
					return [(p[0]+q[0])/2.0, (p[1]+q[1])/2.0, (p[2]+q[2])/2.0]
				
				try:
					# MediaPipe pose landmark indices
					L_HIP, R_HIP = 24, 23
					L_SH, R_SH = 12, 11
					L_ELB, R_ELB = 14, 13
					L_KNE, R_KNE = 26, 25
					
					l_hip, r_hip = to_xyz(L_HIP), to_xyz(R_HIP)
					l_sh, r_sh = to_xyz(L_SH), to_xyz(R_SH)
					l_elb, r_elb = to_xyz(L_ELB), to_xyz(R_ELB)
					l_kne, r_kne = to_xyz(L_KNE), to_xyz(R_KNE)
					
					# Calculate midpoints
					mid_hip = midpoint(l_hip, r_hip)
					mid_shoulder = midpoint(l_sh, r_sh)
					mid_elbows = midpoint(l_elb, r_elb)
					mid_knee = midpoint(l_kne, r_kne)
					
					# Rule 1: Arms above shoulders
					arm_angle = get_angle(mid_hip, mid_shoulder, mid_elbows)
					if arm_angle > ERGO_ARM_ANGLE_THRESH:
						return "NG"
					
					# Rule 2: Back too straight or bent
					back_angle = get_angle(mid_knee, mid_hip, mid_shoulder)
					if back_angle < ERGO_BACK_ANGLE_THRESH:
						return "NG"
					
					return "OK"
				except Exception:
					return "OK"  # Conservative default
			
			# CSV writing function
			def write_csv_predictions():
				"""Writes predictions buffer to CSV every second."""
				if not csv_output_path:
					return
				
				try:
					# Ensure output directory exists
					csv_output_path.parent.mkdir(parents=True, exist_ok=True)
					
					with csv_lock:
						if not predictions_buffer:
							return
						
						# Check if file exists to determine if we need header
						file_exists = csv_output_path.exists()
						
					# Write predictions
					with open(csv_output_path, 'a', newline='', encoding='utf-8') as f:
						writer = csv.writer(f)
						if not file_exists:
							# Always write header with new columns: segment, area, plant (between line and side)
							writer.writerow(["date", "weekday", "camera", "np", "line", "segment", "area", "plant", "side", "station", "person_id", "HH:MM:SS", "class_perf", "class_ergo", "coord_x", "coord_y", "person_count"])
						writer.writerows(predictions_buffer)
						# Clear buffer after writing
						predictions_buffer.clear()
				except Exception:
					pass
			
			# Worker thread for each camera
			def process_camera(cam_index, cam_info, label_widget, rois):
				try:
					# Extract camera data
					rtsp_url = cam_info.get("url") if isinstance(cam_info, dict) else cam_info
					cam_np = cam_info.get("np", "") if isinstance(cam_info, dict) else ""
					cam_linea = cam_info.get("linea", "") if isinstance(cam_info, dict) else ""
					
					print(f"[DEBUG] process_camera cam_{cam_index}: Recibidos {len(rois)} ROIs")
					for roi in rois:
						print(f"[DEBUG]   ROI '{roi['name']}': {len(roi['coords'])} vértices")
					
					# Load models
					person_model = YOLO(str(person_det_weights)).to(DEVICE)
					cls_model = YOLO(str(classify_weights)).to(DEVICE) if classify_weights is not None else None
					
					# Apply light mode resolution reduction if enabled
					if rt_runtime_config["background_processing"]:
						effective_img_size = int(IMG_SIZE * LIGHT_MODE_RESOLUTION_SCALE)
					else:
						effective_img_size = IMG_SIZE
					if DEVICE.startswith("cuda") and USE_HALF:
						try:
							person_model.half()
							cls_model.half()
						except Exception:
							pass
					
					# Processing state
					va_count = 0
					nva_count = 0
					t_prev = time.time()
					fps = 0.0
					last_csv_sample = time.time()  # Track last CSV sample time
					last_head_sample = 0.0  # Track last head detection sample time
					processed_frame_counter = 0  # For frameskip logic
					
					# Aggregation state (per-person with 1-block delay)
					agg_counts = {}  # dict[tid] -> dict[label] -> count
					agg_frame_counter = 0  # count frames in current block
					display_labels = {}  # dict[tid] -> aggregated label from PREVIOUS block
					display_ready = False  # True after first block closes
					
					# Ergonomics state (if enabled)
					# Initialize pose_instances to None, will be created dynamically if needed
					pose_instances = None
					ergo_agg_counts = {}  # dict[tid] -> dict['OK'|'NG'] -> count
					ergo_agg_frame_counter = 0
					ergo_display_labels = {}  # dict[tid] -> 'OK' or 'NG' from PREVIOUS block
					ergo_display_ready = False
					
					# Head count state for person_count (updated per frame by head detection model)
					head_station_count = {}
					head_global_count = 0
					
					# Face blur: YOLO face detection model
					face_model = None
					if rt_runtime_config["difuminar_caras"] and YOLO_AVAILABLE:
						try:
							face_weights_path = _base_root() / "yolov8l-face.pt"
							if not face_weights_path.exists():
								face_weights_path = _base_root() / "yolov8m-face.pt"  # Fallback
							if face_weights_path.exists():
								face_model = YOLO(str(face_weights_path))
						except Exception as e:
							print(f"[RT] Could not load face detection model for cam {cam_index}: {e}")
							face_model = None
					
					# Head detection model for person_count (counts heads, avoids body occlusion)
					head_model = None
					try:
						head_weights_path = _base_root() / "head_model_new_trained.pt"
						if head_weights_path.exists():
							head_model = YOLO(str(head_weights_path)).to(DEVICE)
							print(f"[RT] Head detection model loaded for cam {cam_index}")
						else:
							print(f"[RT] head_model_new_trained.pt not found, using person detection for count")
					except Exception as e:
						print(f"[RT] Could not load head detection model for cam {cam_index}: {e}")
					
					# Camera metadata (extract from URL)
					camera_id = f"cam_{cam_index}"
					try:
						# Extract host from RTSP URL
						import re
						match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', rtsp_url)
						if match:
							camera_id = match.group(1).split('.')[-1]  # Last octet
					except Exception:
						pass					# Stream tracking
					stream = person_model.track(
						source=rtsp_url,
						device=DEVICE,
						imgsz=effective_img_size,
						conf=CONF_PERSON,
						tracker="bytetrack.yaml",
						stream=True,
						verbose=False,
						classes=0,
						persist=True
					)
					
					for result in stream:
						if rt_stop_event.is_set():
							break
						
						frame = result.orig_img
						if frame is None:
							continue
						
						# Apply frameskip based on percentage (same logic as video processing)
						# In light mode, force higher frameskip for GPU optimization
						effective_frameskip = LIGHT_MODE_FRAMESKIP if rt_runtime_config["background_processing"] else frameskip_percentage
						if effective_frameskip > 0:
							# Calculate if this frame should be skipped based on percentage
							# Using modulo pattern: skip M frames every N frames based on percentage
							frame_mod = (processed_frame_counter % 10)
							skip_count = int(effective_frameskip / 10)  # How many frames to skip per 10
							if frame_mod < skip_count:
								processed_frame_counter += 1
								continue
						
						processed_frame_counter += 1
						
						h, w = frame.shape[:2]
						
						# Collect persons
						people = []
						boxes = getattr(result, "boxes", None)
						if boxes is not None:
							for box in boxes:
								cls = int(box.cls[0].item()) if hasattr(box, "cls") else 0
								if cls != 0:
									continue
								tid = int(box.id[0].item()) if box.id is not None else 0
								x1, y1, x2, y2 = box.xyxy[0].tolist()
								clamped = clamp_box(x1, y1, x2, y2, w, h)
								if not clamped:
									continue
								x1, y1, x2, y2 = clamped
								
								# ROI filtering
								roi_name = None
								cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
								if rois:
									roi_name = _point_in_any_roi(rois, (cx, cy))
									if roi_name is None:
										continue
								
								roi = frame[y1:y2, x1:x2]
								if roi.size == 0:
									continue
								people.append({'tid': tid, 'bbox': (x1, y1, x2, y2), 'roi': roi, 'roi_name': roi_name})
						
						# Ergonomics: evaluate posture for each person (if enabled)
						ergo_labels = {}
						if rt_runtime_config["seguridad"] and MP_AVAILABLE and people:
							# Initialize pose_instances dynamically if needed
							if pose_instances is None:
								pose_instances = OrderedDict()
								print(f"[RT] Pose instances initialized dynamically for cam {cam_index}")
							
							for p in people:
								tid = p['tid']
								# Get or create Pose instance for this track ID (LRU cache)
								pose_model = pose_instances.get(tid)
								if pose_model is None:
									try:
										pose_model = mp_pose.Pose(
											static_image_mode=False,
											model_complexity=ERGO_MODEL_COMPLEXITY,
											smooth_landmarks=True,
											min_detection_confidence=0.5,
											min_tracking_confidence=0.9
										)
										pose_instances[tid] = pose_model
									except Exception:
										continue
								else:
									try:
										pose_instances.move_to_end(tid)
									except Exception:
										pass
								
								# Evict LRU if exceeds maximum
								try:
									while len(pose_instances) > ERGO_MAX_POSE_MODELS:
										old_tid, old_inst = pose_instances.popitem(last=False)
										try:
											old_inst.close()
										except Exception:
											pass
								except Exception:
									pass
								
								# Process pose
								(x1, y1, x2, y2) = p['bbox']
								roi_bgr = p['roi']
								rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
								pose_out = None  # Initialize to None
								try:
									pose_out = pose_model.process(rgb)
								except Exception:
									# Error processing, discard this instance
									try:
										pose_model.close()
									except Exception:
										pass
									try:
										if tid in pose_instances:
											pose_instances.pop(tid, None)
									except Exception:
										pass
									continue
								
								# Classify ergonomics
								if pose_out and getattr(pose_out, 'pose_world_landmarks', None):
									landmarks_world = pose_out.pose_world_landmarks.landmark
									status = classify_ergonomics(landmarks_world)
									if status == "OK":
										ergo_labels[tid] = ("Good posture", (0, 160, 0))
									else:
										ergo_labels[tid] = ("Not good posture", (0, 0, 160))
									# Accumulate for aggregation
									try:
										if tid not in ergo_agg_counts:
											ergo_agg_counts[tid] = {"OK": 0, "NG": 0}
										ergo_agg_counts[tid][status] = ergo_agg_counts[tid].get(status, 0) + 1
									except Exception:
										pass
								elif pose_out and getattr(pose_out, 'pose_landmarks', None):
									# Fallback: assume OK if only 2D landmarks
									status = "OK"
									ergo_labels[tid] = ("Good posture", (0, 160, 0))
									try:
										if tid not in ergo_agg_counts:
											ergo_agg_counts[tid] = {"OK": 0, "NG": 0}
										ergo_agg_counts[tid][status] = ergo_agg_counts[tid].get(status, 0) + 1
									except Exception:
										pass
								
								# Store landmarks for drawing later (after bounding boxes)
								# This prevents keypoints from being overwritten by rectangle borders
								if pose_out and getattr(pose_out, 'pose_landmarks', None):
									# Store landmarks with bbox for later drawing
									if tid not in ergo_labels:
										ergo_labels[tid] = ("Unknown", (160, 160, 160))
									# Add landmarks to ergo_labels tuple (text, color, landmarks, bbox)
									current_label = ergo_labels[tid]
									ergo_labels[tid] = (current_label[0], current_label[1], pose_out.pose_landmarks, (x1, y1, x2, y2))
						
						# Classify persons and ACCUMULATE
						if people and cls_model is not None:
							cls_results = cls_model([p['roi'] for p in people], device=DEVICE, conf=CONF_STATE, verbose=False)
							for res, p in zip(cls_results, people):
								probs = res.probs.data.cpu().numpy() if hasattr(res, "probs") else None
								if probs is not None:
									class_names = list(res.names.values())
									if p.get('roi_name'):
										probs = _adjust_probs_by_roi(probs, class_names, p['roi_name'])
									name = class_names[int(np.argmax(probs))]
								else:
									name = "Unknown"
								mapped, _ = get_New_Class(name)
								if mapped not in {"VA", "NVA"}:
									mapped = "Neutral"
								
								# Accumulate in current block
								tid = p['tid']
								if tid not in agg_counts:
									agg_counts[tid] = {}
								agg_counts[tid][mapped] = agg_counts[tid].get(mapped, 0) + 1
						
						# Advance frame counter for aggregation
						agg_frame_counter += 1
						if agg_frame_counter >= AGGREGATION_FRAMES:
							# Close block: calculate majorities for NEXT block display
							new_display = {}
							for tid, cnts in agg_counts.items():
								if not cnts:
									continue
								# Preference in ties: VA > NVA > Neutral
								pref = {"VA": 2, "NVA": 1, "Neutral": 0}
								best = max(cnts.items(), key=lambda kv: (kv[1], pref.get(kv[0], -1)))
								new_display[tid] = best[0]
							display_labels = new_display
							display_ready = True
							agg_counts.clear()
							agg_frame_counter = 0
						
						# Advance ergonomics frame counter (if enabled)
						if rt_runtime_config["seguridad"] and MP_AVAILABLE:
							ergo_agg_frame_counter += 1
							if ergo_agg_frame_counter >= AGGREGATION_FRAMES:
								# Close ergonomics block: calculate majorities
								new_ergo_display = {}
								for tid, cnts in ergo_agg_counts.items():
									ok_c = int(cnts.get("OK", 0))
									ng_c = int(cnts.get("NG", 0))
									maj = "NG" if ng_c > ok_c else "OK"
									new_ergo_display[tid] = maj
								ergo_display_labels = new_ergo_display
								ergo_display_ready = True
								ergo_agg_counts.clear()
								ergo_agg_frame_counter = 0
						
						# Draw persons using aggregated labels (from previous block)
						# Skip all rendering if in background processing mode
						if not rt_runtime_config["background_processing"]:
							# Reset counts for display
							va_count = 0
							nva_count = 0
							for p in people:
								tid = p['tid']
								# Use aggregated label from previous block
								label = display_labels.get(tid) if display_ready else None
								
								if label == "VA":
									color = (0, 255, 0)
									va_count += 1
								elif label == "NVA":
									color = (0, 0, 255)
									nva_count += 1
								else:
									color = (160, 160, 160)  # Gray for neutral/unknown
								
								# Draw on frame
								(x1, y1, x2, y2) = p['bbox']
								cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
								if label:
									label_text = f"ID:{tid} {label}"
									cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
								
								# Draw ergonomics label if enabled and available
								if rt_runtime_config["seguridad"] and tid in ergo_labels:
									ergo_data = ergo_labels[tid]
									ergo_text, ergo_color = ergo_data[0], ergo_data[1]
									cv2.putText(frame, ergo_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ergo_color, 2)
									
									# Draw keypoints on top of everything (if landmarks were detected)
									if len(ergo_data) > 2 and ergo_data[2] is not None and mp_drawing is not None and mp_drawing_styles is not None:
										try:
											# Get the person's ROI from frame with current bbox
											pose_landmarks = ergo_data[2]
											# Draw directly on the frame region
											person_roi = frame[y1:y2, x1:x2]
											mp_drawing.draw_landmarks(
												person_roi,
												pose_landmarks,
												mp_pose.POSE_CONNECTIONS,
												landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
											)
										except Exception as e:
											print(f"[RT] Error drawing keypoints for tid {tid}: {e}")
						
							# Apply face blur (if enabled) - after drawing labels
							# Load face model dynamically if enabled but not yet loaded
							if rt_runtime_config["difuminar_caras"] and YOLO_AVAILABLE:
								if face_model is None:
									try:
										face_weights_path = _base_root() / "yolov8l-face.pt"
										if not face_weights_path.exists():
											face_weights_path = _base_root() / "yolov8m-face.pt"
										if face_weights_path.exists():
											face_model = YOLO(str(face_weights_path))
											print(f"[RT] Face model loaded dynamically for cam {cam_index}")
									except Exception as e:
										print(f"[RT] Could not load face model dynamically for cam {cam_index}: {e}")
								
								if face_model is not None:
									try:
										h_frame, w_frame = frame.shape[:2]
										results_face = face_model(frame, verbose=False)
										for r in results_face:
											if getattr(r, 'boxes', None) is None:
												continue
											boxes = r.boxes.xyxy
											if boxes is None:
												continue
											for (x1_f, y1_f, x2_f, y2_f) in boxes.cpu().numpy().astype(int):
												x1_f = max(0, min(x1_f, w_frame-1))
												x2_f = max(0, min(x2_f, w_frame-1))
												y1_f = max(0, min(y1_f, h_frame-1))
												y2_f = max(0, min(y2_f, h_frame-1))
												if x2_f <= x1_f or y2_f <= y1_f:
													continue
												face_roi = frame[y1_f:y2_f, x1_f:x2_f]
												if face_roi.size == 0:
													continue
												blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
												frame[y1_f:y2_f, x1_f:x2_f] = blurred
									except Exception as e:
										print(f"[RT] Error applying face blur for cam {cam_index}: {e}")
						
						# Run head detection for person_count and bbox visualization (once per second)
						if head_model is not None and (time.time() - last_head_sample >= 1.0):
							try:
								head_results = head_model(frame, verbose=False, conf=0.3)
								_head_station_count = {}
								_head_global_count = 0
								if head_results and head_results[0].boxes is not None:
									for hbox in head_results[0].boxes:
										hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0].tolist())
										hcx = (hx1 + hx2) // 2
										hcy = (hy1 + hy2) // 2
										h_roi_name = None
										if rois:
											h_roi_name = _point_in_any_roi(rois, (hcx, hcy))
											if h_roi_name is None:
												continue
										cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 165, 0), 2)
										if rois:
											if h_roi_name:
												_head_station_count[h_roi_name] = _head_station_count.get(h_roi_name, 0) + 1
										else:
											_head_global_count += 1
								head_station_count = _head_station_count
								head_global_count = _head_global_count
								last_head_sample = time.time()
							except Exception as e:
								print(f"[RT] Head model error for cam {cam_index}: {e}")
						
						# Draw ROIs on frame
						if rois:
							for roi in rois:
								try:
									pts = roi["coords"].reshape(-1, 1, 2)
									cv2.polylines(frame, [pts], True, (0, 255, 255), 2)  # Cyan color
									# Add ROI name
									roi_name = roi.get("name", "ROI")
									# Get centroid for text placement
									M = cv2.moments(pts)
									if M["m00"] != 0:
										cx = int(M["m10"] / M["m00"])
										cy = int(M["m01"] / M["m00"])
										cv2.putText(frame, roi_name, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
								except Exception as e:
									print(f"[RT] Error drawing ROI: {e}")
						
						# Calculate FPS
						t_now = time.time()
						dt = t_now - t_prev
						if dt > 0:
							fps = 0.9 * fps + 0.1 * (1.0 / dt)
						t_prev = t_now

						# Publish latest processed frame to in-memory worker stream (no disk dependency)
						try:
							stream_id = str(cam_index)
							camera_name = str(cam_np).strip() if str(cam_np).strip() else camera_id
							camera_group = str(cam_linea).strip() if str(cam_linea).strip() else str(rt_linea).strip()
							_update_live_rt_frame(stream_id, camera_name, frame, camera_group)
						except Exception as e:
							print(f"[WORKER_FRAMES] publish_failed cam_index={cam_index}: {e}")
						
						# Display frame in GUI - ALWAYS when not in background mode
						if not rt_runtime_config["background_processing"]:
							try:
								# print(f"[RT-DEBUG] Cam {cam_index}: Updating display, frame shape: {frame.shape}, people: {len(people)}, VA: {va_count}, NVA: {nva_count}")
								# Fixed cell dimensions matching the grid
								CELL_W = 640
								CELL_H = 480
								
								# Prepare list of targets to update
								targets = [(label_widget, CELL_W, CELL_H)]  # Always update grid
								
								# Check if popup window exists for this camera
								if cam_index < len(rt_popup_labels) and rt_popup_labels[cam_index] is not None:
									try:
										if rt_popup_labels[cam_index].winfo_exists():
											# Get popup window size
											popup_w = rt_popup_windows[cam_index].winfo_width()
											popup_h = rt_popup_windows[cam_index].winfo_height()
											if popup_w > 100 and popup_h > 100:  # Valid size
												targets.append((rt_popup_labels[cam_index], popup_w, popup_h))
												# print(f"[RT-DEBUG] Cam {cam_index}: Also updating popup window")
									except Exception as e:
										# print(f"[RT-DEBUG] Cam {cam_index}: Error checking popup: {e}")
										pass
								
								def update_all_displays():
									try:
										for idx, (target_label, display_w, display_h) in enumerate(targets):
											try:
												# Check if target widget still exists
												if not target_label.winfo_exists():
													continue
												
												frame_h, frame_w = frame.shape[:2]
												aspect_frame = frame_w / float(frame_h)
												aspect_display = display_w / float(display_h)
												
												# Calculate size to fit within bounds while preserving aspect ratio
												if aspect_frame > aspect_display:
													# Frame is wider
													new_w = display_w
													new_h = int(display_w / aspect_frame)
												else:
													# Frame is taller
													new_h = display_h
													new_w = int(display_h * aspect_frame)
												
												frame_resized = cv2.resize(frame, (new_w, new_h))
												frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
												img = Image.fromarray(frame_rgb)
												imgtk = ImageTk.PhotoImage(image=img)
												
												target_label.configure(image=imgtk, text="")
												target_label.image = imgtk  # Keep reference
												# print(f"[RT-DEBUG] Cam {cam_index}: Display {idx} updated successfully")
											except Exception as e:
												# print(f"[RT-DEBUG] Cam {cam_index}: Error updating display {idx}: {e}")
												pass
									except Exception as e:
										# print(f"[RT-DEBUG] Cam {cam_index}: Error in update_all_displays: {e}")
										pass
								
								try:
									win.after(0, update_all_displays)
								except Exception as e:
									# print(f"[RT-DEBUG] Cam {cam_index}: Error scheduling display update: {e}")
									pass
							except Exception as e:
								print(f"[RT] Error preparing display for cam {cam_index}: {e}")
								pass
						else:
							# print(f"[RT-DEBUG] Cam {cam_index}: Background mode, skipping display update")
							# Apply FPS limit in light mode to reduce GPU usage
							if LIGHT_MODE_FPS_LIMIT > 0:
								time.sleep(1.0 / LIGHT_MODE_FPS_LIMIT)
						
						# Add to CSV buffer every second (using aggregated labels)
						if csv_output_path and (t_now - last_csv_sample >= 1.0) and display_ready:
							# print(f"[RT-DEBUG] Cam {cam_index}: Writing CSV sample, {len(people)} people detected")
							from datetime import datetime
							dt_now = datetime.now()
							date_str = dt_now.strftime("%Y-%m-%d")
							weekday_str = dt_now.strftime("%A")
							time_str = dt_now.strftime("%H:%M:%S")
							
							with csv_lock:
								# Calculate person_count using head detection model (if loaded), else fallback to people
								if head_model is not None:
									global_person_count = head_global_count if not rois else None
									station_person_count = head_station_count if rois else {}
								elif rois:
									station_person_count = {}
									for person in people:
										roi_n = person.get('roi_name')
										if roi_n:
											station_person_count[roi_n] = station_person_count.get(roi_n, 0) + 1
									global_person_count = None
								else:
									station_person_count = {}
									global_person_count = len(people)
								for p in people:
									tid = p['tid']
									label = display_labels.get(tid)
									csv_label = "" if cls_model is None else label
									if label or cls_model is None:  # Write row with empty class_perf when classification disabled
										# Get ergonomics label if enabled and ready
										class_ergo = ""
										if rt_runtime_config["seguridad"] and ergo_display_ready:
											class_ergo = ergo_display_labels.get(tid, "Unknown")
										
										# Calculate bounding box center coordinates for heatmap
										(x1, y1, x2, y2) = p['bbox']
										coord_x = int((x1 + x2) / 2)
										coord_y = int((y1 + y2) / 2)
										
										# person_count: same logic as batch
										roi_n = p.get('roi_name')
										if global_person_count is not None:
											person_count = global_person_count
										else:
											person_count = station_person_count.get(roi_n, 0) if roi_n else 0
										
										row = [
											date_str,
											weekday_str,
											camera_id,
											"",  # np - leave empty for now
											rt_linea,  # line from group metadata
											rt_segmento,  # segment from group metadata
											rt_area,  # area from group metadata
											rt_planta,  # plant from group metadata
											cam_np,  # side: camera name
											p.get('roi_name') or 'Unknown',
											float(tid),
											time_str,
											csv_label
										]
										# Add ergonomics column if enabled
										if rt_runtime_config["seguridad"]:
											row.append(class_ergo)
										else:
											row.append("")  # Empty class_ergo when ergonomics disabled
										# Add coordinates for heatmap/spaghetti diagram
										row.append(coord_x)
										row.append(coord_y)
										# Add person_count
										row.append(person_count)
										predictions_buffer.append(row)
							last_csv_sample = t_now
				except Exception as e:
					def show_error():
						try:
							if label_widget.winfo_exists():
								label_widget.configure(text=f"Error:\n{str(e)[:50]}")
						except Exception:
							pass
					try:
						win.after(0, show_error)
					except Exception:
						pass
			
			# Start threads for each camera
			threads = []
			for i, cam_info in enumerate(camera_urls):
				if i >= 10:  # Max 10 cameras
					break
				# Get ROIs for this camera
				cam_rois_raw = rois_map.get(i, [])
				cam_rois = []
				for roi_item in cam_rois_raw:
					coords = np.array(roi_item["coords"], dtype=np.int32).reshape(-1, 2)
					cam_rois.append({"coords": coords, "name": roi_item.get("name", f"ROI{len(cam_rois)+1}")})
				
				label_widget = rt_camera_labels[i]
				t = threading.Thread(target=process_camera, args=(i, cam_info, label_widget, cam_rois), daemon=True)
				t.start()
				threads.append(t)
			
			# Hide unused camera slots
			for i in range(num_cameras, 10):
				rt_camera_labels[i].configure(text="No configurada")
			
			# CSV writer thread
			def csv_writer_loop():
				"""Periodically writes buffered predictions to CSV."""
				while not rt_stop_event.is_set():
					write_csv_predictions()
					time.sleep(1.0)
			
			if csv_output_path:
				threading.Thread(target=csv_writer_loop, daemon=True).start()
			
			# Shared state for status updates
			rt_status = {"fps": 0.0, "va": 0, "nva": 0}
			
			# Status updater thread
			def update_rt_status():
				while not rt_stop_event.is_set():
					# Update FPS and stats
					def update_labels():
						try:
							# Check if widgets still exist before updating
							# Note: rt_fps_label and rt_stats_label are not defined in the current version
							# TODO: Add these labels to the RT UI if needed
							pass
							# if rt_fps_label.winfo_exists():
							# 	rt_fps_label.configure(text=f"FPS: {rt_status['fps']:.1f}")
							# if rt_stats_label.winfo_exists():
							# 	rt_stats_label.configure(text=f"VA: {rt_status['va']} | NVA: {rt_status['nva']}")
						except Exception:
							pass
					try:
						win.after(0, update_labels)
					except Exception:
						break
					time.sleep(1.0)
			threading.Thread(target=update_rt_status, daemon=True).start()

		# ---------------- Tab 6 (index 5): Fin ----------------
		fin_tab = tabs[5]
		fin_tab.configure(bg=PROC_CONTENT_BG)
		fin_tab.grid_rowconfigure(0, weight=1)
		fin_tab.grid_columnconfigure(0, weight=1)
		fin_summary_lbl = tk.Label(fin_tab, text="Fin del procesamiento.", font=("Arial", 16, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		fin_summary_lbl.pack(padx=24, pady=32, anchor="nw")

	def make_button_frame(parent: tk.Widget, text: str, icon_path: Path, command):
		"""Create a horizontal button with icon and text, with hover and active states."""
		# Container frame with transparent background
		container = tk.Frame(parent, bg=BG_COLOR)
		
		# Canvas for rounded rectangle background
		canvas = tk.Canvas(container, bg=BG_COLOR, highlightthickness=0, cursor="hand2")
		canvas.pack(fill=tk.BOTH, expand=True)
		
		# Store the rounded rectangle ID
		rect_id = {"value": None}
		
		def draw_rounded_rect(width, height, radius, fill_color):
			"""Draw a rounded rectangle on the canvas"""
			canvas.delete("button_bg")
			if width > 2*radius and height > 2*radius:
				# Create rounded rectangle using multiple shapes
				points = [
					radius, 0,
					width - radius, 0,
					width, radius,
					width, height - radius,
					width - radius, height,
					radius, height,
					0, height - radius,
					0, radius
				]
				rect_id["value"] = canvas.create_polygon(points, fill=fill_color, smooth=True, tags="button_bg")
		
		# Inner container for content (on top of canvas)
		inner = tk.Frame(canvas, bg=BUTTON_BG, highlightthickness=0)
		inner_window = canvas.create_window(0, 0, window=inner, anchor="nw", tags="content")
		
		# Load and display icon
		icon_img = load_icon(icon_path, 100, 100, invert=False)
		if icon_img is not None:
			icon_label = tk.Label(inner, image=icon_img, bg=BUTTON_BG)
			icon_label.image = icon_img  # Keep reference
			icon_label.pack(pady=(50, 25))
		else:
			# Fallback if icon not found
			icon_label = tk.Label(inner, text="[?]", font=("Arial", 24), fg=FG_COLOR, bg=BUTTON_BG)
			icon_label.pack(pady=(20, 15))
		
		# Text label - split text by \n for multi-line display
		text_lines = text.split('\n')
		text_labels = []  # Store all text label widgets
		bold_words = ["PROCESAMIENTO", "ENTRENAMIENTO", "DASHBOARD", "VISUALIZACIÓN"]  # Words to make bold
		for line in text_lines:
			# Check if line contains a bold word
			is_bold = any(word in line for word in bold_words)
			font_weight = "bold" if is_bold else "normal"
			line_label = tk.Label(inner, text=line, font=("Arial", 20, font_weight), fg=FG_COLOR, bg=BUTTON_BG)
			line_label.pack()
			text_labels.append(line_label)
		
		# Store original colors
		state = {"pressed": False}
		
		def update_colors(bg):
			"""Update all widget colors including rounded background"""
			inner.config(bg=bg)
			icon_label.config(bg=bg)
			# Update all text labels
			for lbl in text_labels:
				lbl.config(bg=bg)
			# Redraw rounded rectangle with new color
			w = canvas.winfo_width()
			h = canvas.winfo_height()
			if w > 1 and h > 1:
				draw_rounded_rect(w, h, 25, bg)
		
		def on_canvas_resize(event):
			"""Redraw rounded rectangle when canvas is resized"""
			w = event.width
			h = event.height
			current_color = BUTTON_ACTIVE if state["pressed"] else BUTTON_BG
			draw_rounded_rect(w, h, 25, current_color)
			# Update inner frame size and position
			canvas.coords(inner_window, 15, 15)
			canvas.itemconfig(inner_window, width=w-30, height=h-30)
		
		canvas.bind("<Configure>", on_canvas_resize)
		
		def on_enter(e):
			if not state["pressed"]:
				update_colors(BUTTON_HOVER)
		
		def on_leave(e):
			if not state["pressed"]:
				update_colors(BUTTON_BG)
		
		def on_press(e):
			state["pressed"] = True
			update_colors(BUTTON_ACTIVE)
		
		def on_release(e):
			state["pressed"] = False
			update_colors(BUTTON_HOVER if canvas.winfo_containing(e.x_root, e.y_root) == canvas else BUTTON_BG)
			command()
		
		# Bind events to all widgets
		widgets_to_bind = [canvas, inner, icon_label]
		widgets_to_bind.extend(text_labels)
		
		for widget in widgets_to_bind:
			widget.bind("<Enter>", on_enter)
			widget.bind("<Leave>", on_leave)
			widget.bind("<ButtonPress-1>", on_press)
			widget.bind("<ButtonRelease-1>", on_release)
		
		return container

	def load_icon(path: Path, max_w: int, max_h: int, invert: bool = False):
		if Image is None or ImageTk is None or not path.exists():
			return None
		try:
			img = Image.open(path).convert("RGBA")
			ratio = min(max_w / img.width, max_h / img.height, 1.0)
			if ratio < 1.0:
				img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
			if invert and ImageOps is not None:
				# Preserve alpha while inverting RGB
				r, g, b, a = img.split()
				rgb = Image.merge("RGB", (r, g, b))
				inv = ImageOps.invert(rgb)
				img = Image.merge("RGBA", (*inv.split(), a))
			return ImageTk.PhotoImage(img)
		except Exception:
			return None

	# Paths
	assets_root = _base_root() / "assets"
	ngui_root = assets_root / "NGUI"
	logo_path = ngui_root / "ArnesisLogo.png"
	rt_logo_path = ngui_root / "RTLogo.png"
	videos_logo_path = ngui_root / "VideosLogo.png"
	entrenamiento_logo_path = ngui_root / "EntrenamientoLogo.png"
	dashboard_logo_path = ngui_root / "DashboardLogo.png"
	video_path = assets_root / "Arnesis_Robot_Greeting_Animation.mp4"

	# Secret feature: Triple-click counter for video easter egg
	logo_click_state = {"count": 0, "last_time": 0}

	def open_llm_chat_window():
		"""Open LLM Chat window with drag-and-drop support."""
		chat_win = tk.Toplevel(root)
		chat_win.title("Arnesis - AI Chat Assistant")
		chat_win.geometry("800x600")
		chat_win.configure(bg=BG_COLOR)
		_center_window(chat_win, 800, 600)
		
		# Initialize/Rewrite log file for this conversation
		log_dir = Path("logs")
		log_dir.mkdir(exist_ok=True)
		log_file = log_dir / "chat_llm_queries.log"
		
		# Rewrite log file with timestamp header
		with open(log_file, 'w', encoding='utf-8') as f:
			f.write("="*80 + "\n")
			f.write(f"ARNESIS AI CHAT ASSISTANT - NUEVA CONVERSACIÓN\n")
			f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("="*80 + "\n\n")
		
		# Chat state
		chat_state = {
			"history": [],  # List of {"role": "user"/"assistant", "content": str}
			"current_file": None,
			"file_label_var": None,
			"log_file": log_file
		}
		
		# Main container
		main_frame = tk.Frame(chat_win, bg=BG_COLOR)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
		
		# Chat display area (top)
		chat_display_frame = tk.Frame(main_frame, bg=BG_COLOR)
		chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
		
		try:
			from tkinter.scrolledtext import ScrolledText
			chat_display = ScrolledText(
				chat_display_frame,
				wrap=tk.WORD,
				bg="#1e1e1e",
				fg=FG_COLOR,
				font=("Arial", 10),
				state=tk.DISABLED,
				relief=tk.FLAT,
				borderwidth=2
			)
			chat_display.pack(fill=tk.BOTH, expand=True)
		except ImportError:
			chat_display = tk.Text(
				chat_display_frame,
				wrap=tk.WORD,
				bg="#1e1e1e",
				fg=FG_COLOR,
				font=("Arial", 10),
				state=tk.DISABLED
			)
			chat_display.pack(fill=tk.BOTH, expand=True)
		
		# File info area
		file_frame = tk.Frame(main_frame, bg=BG_COLOR, height=30)
		file_frame.pack(fill=tk.X, pady=(0, 5))
		file_frame.pack_propagate(False)
		
		file_label_var = tk.StringVar(value="No hay archivo cargado")
		chat_state["file_label_var"] = file_label_var
		file_label = tk.Label(
			file_frame,
			textvariable=file_label_var,
			fg="#5BA8C9",
			bg=BG_COLOR,
			font=("Arial", 9, "italic")
		)
		file_label.pack(side=tk.LEFT)
		
		# Clear file button
		def clear_file():
			chat_state["current_file"] = None
			file_label_var.set("No hay archivo cargado")
		
		clear_btn = tk.Button(
			file_frame,
			text="✕",
			command=clear_file,
			bg=BUTTON_BG,
			fg=FG_COLOR,
			font=("Arial", 10, "bold"),
			relief=tk.FLAT,
			cursor="hand2"
		)
		clear_btn.pack(side=tk.RIGHT, padx=5)
		
		# Input area (bottom)
		input_frame = tk.Frame(main_frame, bg=BG_COLOR)
		input_frame.pack(fill=tk.X)
		
		# File upload button
		def upload_file():
			from tkinter import filedialog
			file_path = filedialog.askopenfilename(
				title="Seleccionar archivo CSV",
				filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
			)
			if file_path:
				chat_state["current_file"] = file_path
				file_label_var.set(f"Archivo: {os.path.basename(file_path)}")
		
		upload_btn = tk.Button(
			input_frame,
			text="📎",
			command=upload_file,
			bg=BUTTON_BG,
			fg=FG_COLOR,
			font=("Arial", 14),
			width=3,
			relief=tk.FLAT,
			cursor="hand2"
		)
		upload_btn.pack(side=tk.LEFT, padx=(0, 5))
		
		# Text input
		input_text = tk.Text(
			input_frame,
			height=3,
			bg="#1e1e1e",
			fg=FG_COLOR,
			font=("Arial", 10),
			relief=tk.FLAT,
			borderwidth=2
		)
		input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
		
		# Send button

		def render_markdown(text_widget, content, base_tag="assistant"):
			"""Simple Markdown renderer for Tkinter Text widget."""
			import re
			
			lines = content.split('\n')
			for line in lines:
				# Headers
				if line.startswith('# '):
					text_widget.insert(tk.END, line[2:] + '\n', (base_tag, 'header1'))
				elif line.startswith('## '):
					text_widget.insert(tk.END, line[3:] + '\n', (base_tag, 'header2'))
				elif line.startswith('### '):
					text_widget.insert(tk.END, line[4:] + '\n', (base_tag, 'header3'))
				# Bullet lists
				elif line.strip().startswith('- ') or line.strip().startswith('* '):
					text_widget.insert(tk.END, '  • ' + line.strip()[2:] + '\n', (base_tag, 'list'))
				# Numbered lists
				elif re.match(r'^\d+\.\s', line.strip()):
					text_widget.insert(tk.END, '  ' + line.strip() + '\n', (base_tag, 'list'))
				# Blockquotes
				elif line.startswith('> '):
					text_widget.insert(tk.END, '  ❝ ' + line[2:] + '\n', (base_tag, 'quote'))
				else:
					# Process inline markdown (bold, italic, code)
					if not line.strip():
						text_widget.insert(tk.END, '\n', base_tag)
						continue
					
					# Split by code blocks first
					parts = re.split(r'(`[^`]+`)', line)
					for part in parts:
						if part.startswith('`') and part.endswith('`'):
							# Code
							text_widget.insert(tk.END, part[1:-1], (base_tag, 'code'))
						else:
							# Process bold and italic
							segments = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', part)
							for seg in segments:
								if seg.startswith('**') and seg.endswith('**'):
									# Bold
									text_widget.insert(tk.END, seg[2:-2], (base_tag, 'bold'))
								elif seg.startswith('*') and seg.endswith('*') and not seg.startswith('**'):
									# Italic
									text_widget.insert(tk.END, seg[1:-1], (base_tag, 'italic'))
								else:
									text_widget.insert(tk.END, seg, base_tag)
					text_widget.insert(tk.END, '\n', base_tag)
		

		def send_message():
			user_msg = input_text.get("1.0", tk.END).strip()
			if not user_msg:
				return
			
			print("\n[DEBUG - send_message] ===================")
			print(f"[DEBUG - send_message] User message: {user_msg}")
			
			# Clear input
			input_text.delete("1.0", tk.END)
			
			# Add user message to history and display
			chat_state["history"].append({"role": "user", "content": user_msg})
			print(f"[DEBUG - send_message] History length: {len(chat_state['history'])}")
			chat_display.config(state=tk.NORMAL)
			chat_display.insert(tk.END, f"Usuario: {user_msg}\n", "user")
			
			# Show attached file if present
			file_path = chat_state.get("current_file")
			if file_path:
				file_name = os.path.basename(file_path)
				chat_display.insert(tk.END, f"[Archivo adjunto: {file_name}]\n", "file_attachment")
			
			chat_display.insert(tk.END, "\n", "user")
			chat_display.config(state=tk.DISABLED)
			chat_display.see(tk.END)
			
			# Build conversation context from history (exclude current message)
			conversation_messages = []
			if len(chat_state["history"]) > 1:
				# Get all messages except the last one (current user message)
				previous_messages = chat_state["history"][:-1]
				print(f"[DEBUG - send_message] Including {len(previous_messages)} previous messages in history")
				# Build proper message array for OpenAI format
				for msg in previous_messages:
					conversation_messages.append({
						"role": msg["role"],
						"content": msg["content"]
					})
				print(f"[DEBUG - send_message] Conversation messages: {len(conversation_messages)}")
			else:
				print("[DEBUG - send_message] First message, no history")
			
			# Call LLM
			chat_display.config(state=tk.NORMAL)
			chat_display.insert(tk.END, "Asistente: Pensando...\n\n", "assistant")
			chat_display.config(state=tk.DISABLED)
			chat_display.see(tk.END)
			chat_win.update()
			
			file_path = chat_state.get("current_file")
			print(f"[DEBUG - send_message] Calling LLM with file: {file_path}")
			
			# Log the query BEFORE sending to LLM
			log_file = chat_state.get("log_file")
			if log_file:
				try:
					with open(log_file, 'a', encoding='utf-8') as f:
						f.write("-"*80 + "\n")
						f.write(f"Query #{len(chat_state['history'])} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
						f.write("-"*80 + "\n")
						f.write(f"USER MESSAGE:\n{user_msg}\n\n")
						
						if file_path:
							f.write(f"ATTACHED FILE:\n{file_path}\n\n")
						
						if conversation_messages:
							f.write(f"CONVERSATION HISTORY ({len(conversation_messages)} messages):\n")
							for idx, msg in enumerate(conversation_messages, 1):
								f.write(f"  [{idx}] {msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}\n")
							f.write("\n")
						else:
							f.write("CONVERSATION HISTORY: (none - first message)\n\n")
				except Exception as log_err:
					print(f"[WARNING] Failed to write to log: {log_err}")
			
			response = call_llm_query(
				user_msg,
				csv_path=file_path,
				system_prompt=None,
				conversation_history=conversation_messages if conversation_messages else None
			)
			
			# Log the final query if PandasAI processed it
			if log_file and file_path and response.get("final_query"):
				try:
					with open(log_file, 'a', encoding='utf-8') as f:
						f.write("FINAL QUERY (processed by PandasAI):\n")
						final_query = response["final_query"]
						# Truncate if too long for readability
						if len(final_query) > 500:
							f.write(f"{final_query[:500]}...\n[Query truncated, total length: {len(final_query)} chars]\n\n")
						else:
							f.write(f"{final_query}\n\n")
				except Exception as log_err:
					print(f"[WARNING] Failed to log final query: {log_err}")
			
			# Update display
			print(f"[DEBUG - send_message] Response received: success={response.get('success')}")
			chat_display.config(state=tk.NORMAL)
			# Remove "Pensando..."
			chat_display.delete("end-3l", "end-2l")
			
			if response.get("success"):
				assistant_msg = response.get("result", "Sin respuesta")
				print(f"[DEBUG - send_message] Assistant message length: {len(assistant_msg)}")
				# Ensure proper UTF-8 encoding
				if isinstance(assistant_msg, bytes):
					assistant_msg = assistant_msg.decode('utf-8', errors='replace')
				chat_state["history"].append({"role": "assistant", "content": assistant_msg})
				print(f"[DEBUG - send_message] Added to history. New length: {len(chat_state['history'])}")
				
				# Log the successful response
				if log_file:
					try:
						with open(log_file, 'a', encoding='utf-8') as f:
							f.write(f"LLM RESPONSE (SUCCESS):\n{assistant_msg}\n\n")
					except Exception as log_err:
						print(f"[WARNING] Failed to log response: {log_err}")
				
				# Render with markdown support
				chat_display.insert(tk.END, "Asistente:\n", "assistant")
				render_markdown(chat_display, assistant_msg, "assistant")
				chat_display.insert(tk.END, "\n", "assistant")
			else:
				error_msg = response.get("error", "Error desconocido")
				error_type = response.get("error_type", "unknown")
				print(f"[DEBUG - send_message] ERROR: {error_msg[:200]}")
				print(f"[DEBUG - send_message] ERROR TYPE: {error_type}")
				
				# Log the error response
				if log_file:
					try:
						with open(log_file, 'a', encoding='utf-8') as f:
							f.write(f"LLM RESPONSE (ERROR):\n")
							f.write(f"Error Type: {error_type}\n")
							f.write(f"Error Message: {error_msg}\n\n")
					except Exception as log_err:
						print(f"[WARNING] Failed to log error: {log_err}")
				
				# Determine user-friendly error message
				if error_type == "connection":
					user_error_msg = "Error de Conexión (Asistente parece estar inaccesible en estos momentos)"
				elif error_type == "timeout":
					user_error_msg = "Timeout (no se recibió respuesta tras esperar 1min)"
				elif error_type == "script_not_found":
					user_error_msg = "Error de Configuración (componente del asistente no encontrado)"
				elif error_type == "empty_response":
					user_error_msg = "Sin Respuesta (el asistente no devolvió contenido)"
				elif error_type == "json_error":
					user_error_msg = "Error de Formato (respuesta inválida del asistente)"
				elif error_type == "script_error":
					user_error_msg = "Error Interno (fallo al procesar la consulta)"
				else:
					# Generic error message
					user_error_msg = "Error Desconocido (ocurrió un problema inesperado)"
				
				chat_display.insert(tk.END, f"Asistente: {user_error_msg}\n\n", "error")
			
			print("[DEBUG - send_message] ===================")
			chat_display.config(state=tk.DISABLED)
			chat_display.see(tk.END)
		
		send_btn = tk.Button(
			fg="white",
			font=("Arial", 10, "bold"),
			width=10,
			relief=tk.FLAT,
			cursor="hand2"
		)
		send_btn.pack(side=tk.RIGHT)
		
		# Bind Enter key to send
		def on_enter(event):
			if not event.state & 0x1:  # No Shift key
				send_message()
				return "break"
		
		input_text.bind("<Return>", on_enter)
		
		# Drag and drop support
		if DND_AVAILABLE:
			try:
				def on_drop(event):
					files = chat_win.tk.splitlist(event.data)
					if files:
						file_path = files[0].strip('{}"')
						if os.path.exists(file_path):
							chat_state["current_file"] = file_path
							file_label_var.set(f"Archivo: {os.path.basename(file_path)}")
				
				main_frame.drop_target_register(DND_FILES)
				main_frame.dnd_bind('<<Drop>>', on_drop)
			except Exception:
				pass
		
		# Markdown rendering helper
		# Configure text tags for base styles
		chat_display.tag_config("user", foreground="#5BA8C9")
		chat_display.tag_config("assistant", foreground=FG_COLOR)
		chat_display.tag_config("error", foreground="#ff6b6b")
		chat_display.tag_config("file_attachment", foreground="#9B9B9B", font=("Arial", 9, "italic"))
		
		# Configure markdown tags
		try:
			chat_display.tag_config("header1", font=("Arial", 14, "bold"), foreground="#5BA8C9")
			chat_display.tag_config("header2", font=("Arial", 12, "bold"), foreground="#5BA8C9")
			chat_display.tag_config("header3", font=("Arial", 11, "bold"), foreground="#5BA8C9")
			chat_display.tag_config("bold", font=("Arial", 10, "bold"))
			chat_display.tag_config("italic", font=("Arial", 10, "italic"))
			chat_display.tag_config("code", font=("Courier New", 9), background="#2d2d2d", foreground="#ffc735")
			chat_display.tag_config("list", foreground="#E6EEF9")
			chat_display.tag_config("quote", font=("Arial", 10, "italic"), foreground="#999")
		except Exception:
			pass  # Fallback if font configuration fails
		
		# Welcome message
		chat_display.config(state=tk.NORMAL)
		render_markdown(chat_display, "# ¡Bienvenido al Chat de Arnesis AI!\n\nPuedes hacer preguntas generales o subir un archivo CSV para análisis específicos.\n", "assistant")
		chat_display.config(state=tk.DISABLED)

	def play_secret_video():
		"""Secret easter egg: Play robot greeting animation video"""
		if not video_path.exists():
			return
		
		# Create video window
		video_win = tk.Toplevel(root)
		video_win.title("🤖 Arnesis Robot")
		video_win.configure(bg="#000000")
		video_win.resizable(False, False)
		
		# Remove window decorations for cleaner look
		video_win.overrideredirect(True)
		
		# Video canvas
		video_canvas = tk.Label(video_win, bg="#000000")
		video_canvas.pack()
		
		# Open video with OpenCV
		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			video_win.destroy()
			return
		
		# Get video properties
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_delay = int(1000 / fps) if fps > 0 else 33
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		
		# Set window size
		video_win.geometry(f"{width}x{height}")
		
		# Center window
		screen_width = video_win.winfo_screenwidth()
		screen_height = video_win.winfo_screenheight()
		x = (screen_width - width) // 2
		y = (screen_height - height) // 2
		video_win.geometry(f"{width}x{height}+{x}+{y}")
		
		# Keep window on top
		video_win.attributes('-topmost', True)
		
		def update_frame():
			"""Read and display next video frame"""
			ret, frame = cap.read()
			if ret:
				# Convert BGR to RGB
				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				# Convert to PIL Image
				img = Image.fromarray(frame_rgb)
				# Convert to PhotoImage
				photo = ImageTk.PhotoImage(image=img)
				# Update canvas
				video_canvas.config(image=photo)
				video_canvas.image = photo
				# Schedule next frame
				video_win.after(frame_delay, update_frame)
			else:
				# Video finished, close window
				cap.release()
				video_win.destroy()
		
		# Start playback
		update_frame()
		
		# Allow clicking to close video early
		def close_video(event=None):
			cap.release()
			video_win.destroy()
		
		video_win.bind("<Button-1>", close_video)
		video_win.bind("<Escape>", close_video)

	def on_logo_click(event):
		"""Single click: open chat, triple click: play video"""
		current_time = time.time()
		
		# Reset counter if more than 0.5 seconds since last click
		if current_time - logo_click_state["last_time"] > 0.5:
			logo_click_state["count"] = 0
		
		logo_click_state["count"] += 1
		logo_click_state["last_time"] = current_time
		
		# Check if triple-clicked
		if logo_click_state["count"] >= 3:
			logo_click_state["count"] = 0
			play_secret_video()
		elif logo_click_state["count"] == 1:
			# Open chat on first click (delayed to allow for potential triple-click)
			root.after(600, lambda: open_llm_chat_window() if logo_click_state["count"] == 1 else None)
	# ========== TOP SECTION: Logo and Tagline ==========
	top_section = tk.Frame(container, bg=BG_COLOR)
	top_section.pack(fill=tk.X, expand=True, pady=(50, 20))
	
	# Logo
	logo_img = _load_logo_image(logo_path) if logo_path else None
	if logo_img is not None:
		logo_label = tk.Label(top_section, image=logo_img, bg=BG_COLOR)
		logo_label.image = logo_img
		logo_label.pack(pady=(0, 15))
		# Bind triple-click secret feature
		logo_label.bind("<Button-1>", on_logo_click)
	else:
		logo_label = tk.Label(top_section, text="ARNESIS", font=("Arial", 32, "bold"), fg="#2DD4E8", bg=BG_COLOR)
		logo_label.pack(pady=(0, 15))
		logo_label.bind("<Button-1>", on_logo_click)
	
	# Tagline
	# tagline = tk.Label(top_section, text="AI-Powered Workforce Optimization", 
	#                    font=("Arial", 13), fg="#5BA8C9", bg=BG_COLOR)
	# tagline.pack()
	
	# ========== BOTTOM SECTION: Horizontal Buttons ==========
	bottom_section = tk.Frame(container, bg=BG_COLOR)
	bottom_section.pack(fill=tk.BOTH, expand=True, padx=60, pady=(0, 50))
	
	# Create 4 buttons horizontally
	buttons_container = tk.Frame(bottom_section, bg=BG_COLOR)
	buttons_container.pack(expand=True, fill=tk.BOTH)
	
	# Configure grid for 4 equal columns
	for i in range(4):
		buttons_container.grid_columnconfigure(i, weight=1, uniform="button")
	buttons_container.grid_rowconfigure(0, weight=1)
	
	# Button 1: Procesamiento en Tiempo Real
	def open_rt_processing():
		if mode_state.get("index", 0) == 1:
			import subprocess, os
			script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head_counter_both.py")
			subprocess.Popen(["python", script_path, "camera"])
		else:
			open_procesamiento_window(preset_type="rt", auto_scan_on_open=True)
	
	rt_button = make_button_frame(buttons_container, "PROCESAMIENTO\nEN TIEMPO REAL", 
	                              rt_logo_path, open_rt_processing)
	rt_button.grid(row=0, column=0, padx=10, pady=(10,100), sticky="nsew")
	
	# Button 2: Procesamiento de Videos
	def open_video_processing():
		if mode_state.get("index", 0) == 1:
			import subprocess, os
			script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head_counter_both.py")
			subprocess.Popen(["python", script_path, "video"])
		else:
			open_procesamiento_window(preset_type="videos")
	
	videos_button = make_button_frame(buttons_container, "PROCESAMIENTO\nDE VIDEOS", 
	                                  videos_logo_path, open_video_processing)
	videos_button.grid(row=0, column=1, padx=10, pady=(10,100), sticky="nsew")
	
	# Button 3: Entrenamiento de Modelos
	entrenamiento_button = make_button_frame(buttons_container, "ENTRENAMIENTO\nDE MODELOS", 
	                                         entrenamiento_logo_path, open_entrenamiento_window)
	entrenamiento_button.grid(row=0, column=2, padx=10, pady=(10,100), sticky="nsew")

	def open_controller_window():
		"""Custom control panel layout for controller mode."""
		CTRL_BG = "#161616"
		CTRL_SECTION_BG = "#1f1f1f"
		CTRL_LABEL = "#0684b2"
		STATUS_IDLE = "#6f6f6f"
		STATUS_OK = "#2f9e44"
		POLL_MS = 1800
		auth_token = _get_control_auth_token()

		controller_win = tk.Toplevel(root)
		controller_win.title("Control Panel")
		root.withdraw()

		def _on_close_controller():
			controller_win.destroy()
			root.deiconify()

		controller_win.protocol("WM_DELETE_WINDOW", _on_close_controller)
		try:
			controller_win.state("zoomed")
		except Exception:
			try:
				controller_win.attributes("-zoomed", True)
			except Exception:
				pass
		_center_window(controller_win, 1200, 760)
		controller_win.configure(bg=CTRL_BG)

		outer = tk.Frame(controller_win, bg=CTRL_BG)
		outer.pack(fill=tk.BOTH, expand=True)

		header = tk.Frame(outer, bg=CTRL_BG, height=72)
		header.pack(fill=tk.X, padx=22, pady=(12, 8))
		header.pack_propagate(False)

		mini_logo_path = assets_root / "NGUI" / "ArnesisMiniLogo.png"
		mini_logo_img = load_icon(mini_logo_path, 170, 70, invert=False)
		if mini_logo_img is not None:
			logo_lbl = tk.Label(header, image=mini_logo_img, bg=CTRL_BG)
			logo_lbl.image = mini_logo_img
			logo_lbl.pack(side=tk.LEFT, padx=(6, 12), pady=6)

		title_lbl = tk.Label(
			header,
			text="| Control Panel",
			font=("Arial", 22, "bold"),
			fg=CTRL_LABEL,
			bg=CTRL_BG
		)
		title_lbl.pack(side=tk.LEFT, pady=6)

		workers_window_state = {"win": None, "text": None}
		workers_runtime = {}
		refresh_state = {"running": False, "after_id": None}
		frame_windows = {}

		def _normalize_endpoint(endpoint: str) -> str:
			e = str(endpoint or "").strip()
			if not e:
				return ""
			if e.startswith("http://") or e.startswith("https://"):
				return e
			return f"http://{e}"

		def _parse_worker_specs() -> list:
			raw = str(os.environ.get("ARNESIS_CONTROLLER_WORKERS", "")).strip()
			specs = []
			if raw:
				chunks = [c.strip() for c in raw.split(";") if c.strip()]
				for chunk in chunks:
					parts = [p.strip() for p in chunk.split("|")]
					endpoint = parts[0] if len(parts) >= 1 else ""
					name = parts[1] if len(parts) >= 2 else endpoint
					groups = []
					if len(parts) >= 3 and parts[2]:
						groups = [g.strip().upper() for g in parts[2].split(",") if g.strip()]
					norm_endpoint = _normalize_endpoint(endpoint)
					if norm_endpoint:
						specs.append({"endpoint": norm_endpoint, "name": name or endpoint, "groups": groups})
			if not specs:
				default_port = str(os.environ.get("ARNESIS_WORKER_CONTROL_PORT", "8765")).strip() or "8765"
				specs = [{"endpoint": _normalize_endpoint(f"127.0.0.1:{default_port}"), "name": "local-worker", "groups": []}]
			return specs

		worker_specs = _parse_worker_specs()

		body = tk.Frame(outer, bg=CTRL_SECTION_BG)
		body.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 18))

		def _add_section(parent, section_name: str):
			section = tk.Frame(parent, bg=CTRL_SECTION_BG)
			section.pack(fill=tk.X, padx=20, pady=(18, 0))
			label = tk.Label(section, text=section_name, font=("Arial", 16, "bold"), fg=FG_COLOR, bg=CTRL_SECTION_BG)
			label.pack(anchor="w")
			content = tk.Frame(section, bg=CTRL_SECTION_BG)
			content.pack(fill=tk.X, pady=(10, 2))
			return content

		high_impact_content = _add_section(body, "Lineas de Alto Impacto")
		final_content = _add_section(body, "Final")
		_add_section(body, "WPA")

		line_cards = ["501", "502", "504", "404", "406", "204", "508", "311", "407", "507"]
		final_cards = ["312", "FOAMV2"]
		known_groups = {g.upper() for g in (line_cards + final_cards)}
		cards_by_group = {}
		live_badge_icon = None
		try:
			if Image is not None and ImageTk is not None:
				live_icon_path = assets_root / "live_icon.png"
				if live_icon_path.exists():
					icon_img = Image.open(live_icon_path).convert("RGBA")
					icon_img = icon_img.resize((9, 9), Image.LANCZOS)
					live_badge_icon = ImageTk.PhotoImage(icon_img)
		except Exception:
			live_badge_icon = None

		def _draw_rounded_rect(canvas, x1, y1, x2, y2, radius, fill):
			# Draw a smooth rounded rectangle using a polygon path.
			points = [
				x1 + radius, y1,
				x2 - radius, y1,
				x2, y1,
				x2, y1 + radius,
				x2, y2 - radius,
				x2, y2,
				x2 - radius, y2,
				x1 + radius, y2,
				x1, y2,
				x1, y2 - radius,
				x1, y1 + radius,
				x1, y1,
			]
			return canvas.create_polygon(points, smooth=True, fill=fill, outline=fill)

		def _create_group_card(parent, line_id: str, idx: int):
			card = tk.Canvas(
				parent,
				width=138,
				height=108,
				bg=CTRL_SECTION_BG,
				highlightthickness=0,
				bd=0,
			)
			card.grid(row=0, column=idx, padx=6, pady=4, sticky="nsew")
			shape_id = _draw_rounded_rect(card, 2, 2, 136, 106, 16, STATUS_IDLE)
			text_id = card.create_text(69, 54, text=line_id, font=("Arial", 18, "bold"), fill="white")
			live_box_id = card.create_rectangle(88, 6, 132, 24, outline="", fill="#ff0000", state="hidden")
			live_icon_id = None
			live_text_id = card.create_text(104, 15, text="LIVE", font=("Arial", 8, "bold"), fill="white", state="hidden")
			if live_badge_icon is not None:
				live_icon_id = card.create_image(124, 15, image=live_badge_icon, state="hidden")
			info = {
				"canvas": card,
				"shape_id": shape_id,
				"text_id": text_id,
				"live_box_id": live_box_id,
				"live_icon_id": live_icon_id,
				"live_text_id": live_text_id,
				"live_badge_icon": live_badge_icon,
				"group": str(line_id).upper(),
			}
			cards_by_group[str(line_id).upper()] = info
			return info

		def _infer_groups_from_frames(frames_payload: list) -> set:
			out = set()
			for f in frames_payload or []:
				frame_group = str((f or {}).get("group", "")).strip().upper()
				if frame_group and frame_group in known_groups:
					out.add(frame_group)
				camera = str((f or {}).get("camera", ""))
				stream_id = str((f or {}).get("stream_id", ""))
				merged = f"{camera} {stream_id}".upper()
				for group in known_groups:
					if group and group in merged:
						out.add(group)
			return out

		def _bool_tf(v) -> str:
			return "t" if bool(v) else "f"

		def _render_workers_window_text():
			win = workers_window_state.get("win")
			text_widget = workers_window_state.get("text")
			if not win or not text_widget:
				return
			if not win.winfo_exists():
				workers_window_state["win"] = None
				workers_window_state["text"] = None
				return

			text_widget.configure(state="normal")
			text_widget.delete("1.0", tk.END)
			for spec in worker_specs:
				endpoint = spec.get("endpoint", "")
				run = workers_runtime.get(endpoint, {})
				node_name = str(run.get("node_id") or spec.get("name") or "unknown")
				line1 = f"{endpoint.replace('http://', '')} | {node_name}\n"
				text_widget.insert(tk.END, line1, "bold")
				groups = run.get("groups", [])
				groups_txt = "/".join(groups) if groups else "-"
				line2 = (
					f"running: {_bool_tf(run.get('running'))} | "
					f"proccesing_rt: {_bool_tf(run.get('processing_rt'))} | "
					f"groups: {groups_txt}\n\n"
				)
				text_widget.insert(tk.END, line2)
			text_widget.configure(state="disabled")

		def _open_workers_window():
			existing = workers_window_state.get("win")
			if existing and existing.winfo_exists():
				existing.deiconify()
				existing.lift()
				_render_workers_window_text()
				return

			win = tk.Toplevel(controller_win)
			win.title("Workers conectados")
			win.configure(bg=CTRL_SECTION_BG)
			win.geometry("740x430")

			wrap = tk.Frame(win, bg=CTRL_SECTION_BG)
			wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

			text = tk.Text(
				wrap,
				bg="#121212",
				fg=FG_COLOR,
				insertbackground=FG_COLOR,
				font=("Arial", 11),
				relief=tk.FLAT,
				bd=0,
			)
			text.pack(fill=tk.BOTH, expand=True)
			text.tag_configure("bold", font=("Arial", 11, "bold"))

			workers_window_state["win"] = win
			workers_window_state["text"] = text

			def _on_close_workers():
				workers_window_state["win"] = None
				workers_window_state["text"] = None
				win.destroy()

			win.protocol("WM_DELETE_WINDOW", _on_close_workers)
			_render_workers_window_text()

		gear_btn = tk.Button(
			header,
			text="⚙",
			font=("Arial", 18, "bold"),
			fg=FG_COLOR,
			bg=CTRL_BG,
			activebackground=CTRL_BG,
			activeforeground=FG_COLOR,
			relief=tk.FLAT,
			bd=0,
			cursor="hand2",
			command=_open_workers_window,
		)
		gear_btn.pack(side=tk.RIGHT, padx=(0, 8), pady=6)

		def _open_assign_worker_popup(group: str):
			"""Show a popup listing connected workers; idle workers can be selected to run the group."""
			popup = tk.Toplevel(controller_win)
			popup.title(f"Asignar Worker - Grupo {group}")
			popup.configure(bg=CTRL_SECTION_BG)
			popup.geometry("520x420")
			popup.transient(controller_win)
			popup.grab_set()

			tk.Label(
				popup,
				text=f"Selecciona un worker para correr el grupo {group}",
				font=("Arial", 12, "bold"),
				fg=FG_COLOR,
				bg=CTRL_SECTION_BG,
			).pack(pady=(16, 6), padx=16, anchor="w")

			status_lbl = tk.Label(popup, text="", font=("Arial", 10), fg="#aaaaaa", bg=CTRL_SECTION_BG)
			status_lbl.pack(padx=16, anchor="w")

			canvas_frame = tk.Frame(popup, bg=CTRL_SECTION_BG)
			canvas_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 4))

			list_canvas = tk.Canvas(canvas_frame, bg=CTRL_SECTION_BG, highlightthickness=0)
			scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=list_canvas.yview)
			list_canvas.configure(yscrollcommand=scrollbar.set)
			scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
			list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

			list_inner = tk.Frame(list_canvas, bg=CTRL_SECTION_BG)
			inner_win_id = list_canvas.create_window((0, 0), window=list_inner, anchor="nw")

			def _on_inner_configure(event):
				list_canvas.configure(scrollregion=list_canvas.bbox("all"))
				list_canvas.itemconfigure(inner_win_id, width=list_canvas.winfo_width())

			list_inner.bind("<Configure>", _on_inner_configure)
			list_canvas.bind("<Configure>", lambda e: list_canvas.itemconfigure(inner_win_id, width=e.width))

			selected_worker = {"value": None}
			row_frames = []

			running_workers = [(ep, rt) for ep, rt in workers_runtime.items() if rt.get("running")]

			if not running_workers:
				tk.Label(
					list_inner,
					text="No hay workers conectados",
					font=("Arial", 11),
					fg="#aaaaaa",
					bg=CTRL_SECTION_BG,
				).pack(pady=20)
			else:
				for endpoint, runtime in running_workers:
					is_processing = bool(runtime.get("processing_rt"))
					name = runtime.get("node_id") or runtime.get("name") or endpoint
					groups_str = ", ".join(runtime.get("groups", [])) or "—"
					state_text = "EN PROCESO" if is_processing else "Disponible"
					state_color = "#ff6b35" if is_processing else "#2f9e44"
					row_bg = "#2a2a2a"

					row = tk.Frame(list_inner, bg=row_bg, bd=0)
					row.pack(fill=tk.X, pady=3, ipady=6)
					row_frames.append((row, endpoint, runtime, is_processing))

					tk.Label(
						row,
						text=f"{name}",
						font=("Arial", 11, "bold"),
						fg=FG_COLOR,
						bg=row_bg,
						anchor="w",
					).pack(side=tk.TOP, padx=12, pady=(6, 0), anchor="w")
					tk.Label(
						row,
						text=f"{endpoint}  |  Grupos: {groups_str}",
						font=("Arial", 9),
						fg="#aaaaaa",
						bg=row_bg,
						anchor="w",
					).pack(side=tk.TOP, padx=12, pady=(0, 6), anchor="w")
					tk.Label(
						row,
						text=state_text,
						font=("Arial", 9, "bold"),
						fg=state_color,
						bg=row_bg,
						anchor="e",
					).place(relx=1.0, rely=0.5, anchor="e", x=-12)

					if not is_processing:
						def _make_select(ep=endpoint, rt=runtime, r=row):
							def _on_select(_event=None):
								selected_worker["value"] = rt
								for rf, _, _, ip in row_frames:
									rf.configure(bg="#2a2a2a")
									for w in rf.winfo_children():
										try:
											w.configure(bg="#2a2a2a")
										except Exception:
											pass
								r.configure(bg="#0e3a6e")
								for w in r.winfo_children():
									try:
										w.configure(bg="#0e3a6e")
									except Exception:
										pass
								status_lbl.config(text=f"Seleccionado: {rt.get('node_id') or ep}")
							return _on_select
						fn = _make_select()
						row.bind("<Button-1>", fn)
						for child in row.winfo_children():
							child.bind("<Button-1>", fn)
						row.configure(cursor="hand2")

			def _on_start():
				w = selected_worker["value"]
				if not w:
					status_lbl.config(text="Selecciona un worker primero")
					return
				ep = w.get("endpoint", "")
				try:
					_http_json_request(
						f"{ep}/start_group",
						method="POST",
						payload={"group": group},
						timeout=3.0,
						auth_token=auth_token,
					)
					popup.destroy()
				except Exception as e:
					status_lbl.config(text=f"Error: {e}")

			btn_frame = tk.Frame(popup, bg=CTRL_SECTION_BG)
			btn_frame.pack(fill=tk.X, padx=16, pady=(4, 16))

			tk.Button(
				btn_frame,
				text="Iniciar",
				font=("Arial", 11, "bold"),
				fg="white",
				bg="#015aca",
				activebackground="#0147a0",
				relief=tk.FLAT,
				padx=20,
				pady=8,
				command=_on_start,
			).pack(side=tk.LEFT)
			tk.Button(
				btn_frame,
				text="Cancelar",
				font=("Arial", 11),
				fg=FG_COLOR,
				bg="#444444",
				activebackground="#555555",
				relief=tk.FLAT,
				padx=20,
				pady=8,
				command=popup.destroy,
			).pack(side=tk.LEFT, padx=(8, 0))

		def _open_group_frames_window(worker_data: dict, group: str):
			endpoint = str(worker_data.get("endpoint", "")).strip()
			if not endpoint:
				return
			win_key = f"{endpoint}|{group}"
			existing = frame_windows.get(win_key)
			if existing and existing.winfo_exists():
				existing.deiconify()
				existing.lift()
				return

			win = tk.Toplevel(controller_win)
			win.title(f"Frames - {group} ({endpoint.replace('http://', '')})")
			win.configure(bg=CTRL_SECTION_BG)
			win.geometry("900x620")

			state = {"after_id": None, "photos": {}, "widgets": {}}

			header_lbl = tk.Label(
				win,
				text=f"Grupo {group} | {endpoint}",
				font=("Arial", 12, "bold"),
				fg=FG_COLOR,
				bg=CTRL_SECTION_BG,
			)
			header_lbl.pack(anchor="w", padx=12, pady=(10, 4))

			status_lbl = tk.Label(win, text="", font=("Arial", 10), fg=FG_COLOR, bg=CTRL_SECTION_BG)
			status_lbl.pack(anchor="w", padx=12, pady=(0, 8))

			streams_container = tk.Frame(win, bg=CTRL_SECTION_BG)
			streams_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

			def _reflow_stream_widgets():
				for i, sid in enumerate(list(state["widgets"].keys())):
					w = state["widgets"].get(sid)
					if not w:
						continue
					row = i // 3
					col = i % 3
					w["frame"].grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
				for col in range(3):
					streams_container.grid_columnconfigure(col, weight=1)

			def _ensure_stream_widget(stream_id: str):
				entry = state["widgets"].get(stream_id)
				if entry:
					return entry
				card = tk.Frame(streams_container, bg="#0f0f0f", bd=1, relief=tk.SOLID)
				title = tk.Label(card, text=stream_id, font=("Arial", 9, "bold"), fg=FG_COLOR, bg="#0f0f0f")
				title.pack(anchor="w", padx=6, pady=(6, 4))
				image_lbl = tk.Label(card, bg="#0f0f0f")
				image_lbl.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
				entry = {"frame": card, "title": title, "image": image_lbl}
				state["widgets"][stream_id] = entry
				_reflow_stream_widgets()
				return entry

			def _matches_group(frame_info: dict) -> bool:
				frame_group = str((frame_info or {}).get("group", "")).strip().upper()
				if frame_group and frame_group == group.upper():
					return True
				camera = str((frame_info or {}).get("camera", ""))
				stream_id = str((frame_info or {}).get("stream_id", ""))
				merged = f"{camera} {stream_id}".upper()
				return group.upper() in merged

			def _poll_frames():
				if not win.winfo_exists():
					return
				try:
					resp = _http_json_request(f"{endpoint}/frames", timeout=1.5, auth_token=auth_token)
					frames = [f for f in resp.get("frames", []) if _matches_group(f)]
					if not frames:
						status_lbl.config(text="Sin frames para este grupo")
						state["last_had_frames"] = False
						return

					state["last_had_frames"] = True
					active_streams = []
					for frame_info in frames:
						stream_id = str(frame_info.get("stream_id", "")).strip()
						if not stream_id:
							continue
						active_streams.append(stream_id)
						widget = _ensure_stream_widget(stream_id)
						widget["title"].config(text=f"{stream_id} | {frame_info.get('camera', '')}")
						img_bytes = _http_bytes_request(f"{endpoint}/frame/{stream_id}", timeout=1.8, auth_token=auth_token)
						if not img_bytes:
							continue
						if Image is not None and ImageTk is not None:
							img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
							img.thumbnail((280, 180), Image.LANCZOS)
							photo = ImageTk.PhotoImage(img)
							state["photos"][stream_id] = photo
							widget["image"].configure(image=photo)

					to_remove = [sid for sid in list(state["widgets"].keys()) if sid not in set(active_streams)]
					for sid in to_remove:
						try:
							state["widgets"][sid]["frame"].destroy()
						except Exception:
							pass
						state["widgets"].pop(sid, None)
						state["photos"].pop(sid, None)
					if to_remove:
						_reflow_stream_widgets()

					status_lbl.config(text=f"Frames del grupo {group}: {len(active_streams)}")
				except Exception as e:
					status_lbl.config(text=f"Error obteniendo frames: {e}")
					state["last_had_frames"] = False
				finally:
					if win.winfo_exists():
						delay = 40 if state.get("last_had_frames") else 5000
						state["after_id"] = win.after(delay, _poll_frames)

			def _on_close_frames():
				after_id = state.get("after_id")
				if after_id is not None:
					try:
						win.after_cancel(after_id)
					except Exception:
						pass
				frame_windows.pop(win_key, None)
				try:
					_http_json_request(f"{endpoint}/set_mode", method="POST", payload={"mode": "light"}, timeout=1.5, auth_token=auth_token)
				except Exception:
					pass
				win.destroy()

			win.protocol("WM_DELETE_WINDOW", _on_close_frames)
			frame_windows[win_key] = win
			try:
				_http_json_request(f"{endpoint}/set_mode", method="POST", payload={"mode": "normal"}, timeout=1.5, auth_token=auth_token)
			except Exception:
				pass
			_poll_frames()

		def _pick_worker_for_group(group: str):
			group_u = str(group).upper()
			live_candidate = None
			running_candidate = None
			for _endpoint, runtime in workers_runtime.items():
				groups = set(runtime.get("groups", []))
				if group_u not in groups:
					continue
				if bool(runtime.get("running")) and bool(runtime.get("processing_rt")):
					live_candidate = runtime
					break
				if bool(runtime.get("running")) and running_candidate is None:
					running_candidate = runtime
			return live_candidate or running_candidate

		def _apply_card_state(group: str, running: bool, live: bool):
			info = cards_by_group.get(str(group).upper())
			if not info:
				return
			canvas = info["canvas"]
			color = STATUS_OK if running else STATUS_IDLE
			canvas.itemconfigure(info["shape_id"], fill=color, outline=color)
			badge_state = "normal" if live else "hidden"
			canvas.itemconfigure(info["live_box_id"], state=badge_state)
			if info.get("live_icon_id") is not None:
				canvas.itemconfigure(info["live_icon_id"], state=badge_state)
			canvas.itemconfigure(info["live_text_id"], state=badge_state)

		def _update_cards_from_runtime():
			group_state = {g: {"running": False, "live": False} for g in known_groups}
			for _endpoint, runtime in workers_runtime.items():
				if not bool(runtime.get("running")):
					continue
				for g in runtime.get("groups", []):
					if g not in group_state:
						continue
					group_state[g]["running"] = True
					if bool(runtime.get("processing_rt")):
						group_state[g]["live"] = True

			for g in known_groups:
				_apply_card_state(g, group_state[g]["running"], group_state[g]["live"])

			_render_workers_window_text()

		def _poll_workers_once():
			result = {}
			for spec in worker_specs:
				endpoint = spec.get("endpoint", "")
				name = spec.get("name", endpoint)
				groups_from_config = {str(g).upper() for g in spec.get("groups", []) if str(g).strip()}
				runtime = {
					"endpoint": endpoint,
					"name": name,
					"node_id": name,
					"running": False,
					"processing_rt": False,
					"groups": sorted(groups_from_config),
				}
				try:
					status = _http_json_request(f"{endpoint}/status", timeout=1.0, auth_token=auth_token)
					runtime["running"] = bool(status.get("ok"))
					runtime["node_id"] = str(status.get("node_id") or name)
					proc = status.get("processing", {}) if isinstance(status, dict) else {}
					runtime["processing_rt"] = bool(proc.get("rt_running"))
					raw_status_groups = proc.get("groups", [])
					status_groups = set()
					if isinstance(raw_status_groups, (list, tuple, set)):
						for g in raw_status_groups:
							g_norm = str(g or "").strip().upper()
							if g_norm:
								status_groups.add(g_norm)
					elif isinstance(raw_status_groups, str):
						g_norm = raw_status_groups.strip().upper()
						if g_norm:
							status_groups.add(g_norm)
					runtime["groups"] = sorted(set(runtime.get("groups", [])) | status_groups)
				except Exception:
					runtime["running"] = False
					runtime["processing_rt"] = False

				try:
					frames_resp = _http_json_request(f"{endpoint}/frames", timeout=1.2, auth_token=auth_token)
					frames = frames_resp.get("frames", []) if isinstance(frames_resp, dict) else []
					inferred = _infer_groups_from_frames(frames)
					all_groups = set(runtime.get("groups", [])) | inferred
					runtime["groups"] = sorted(all_groups)
				except Exception:
					pass

				result[endpoint] = runtime
			return result

		def _refresh_workers_async():
			if refresh_state["running"]:
				return
			refresh_state["running"] = True

			def _worker_refresh():
				collected = _poll_workers_once()

				def _apply_on_ui():
					workers_runtime.clear()
					workers_runtime.update(collected)
					_update_cards_from_runtime()
					refresh_state["running"] = False
					if controller_win.winfo_exists():
						refresh_state["after_id"] = controller_win.after(POLL_MS, _refresh_workers_async)

				controller_win.after(0, _apply_on_ui)

			threading.Thread(target=_worker_refresh, daemon=True).start()

		for idx, line_id in enumerate(line_cards):
			card_info = _create_group_card(high_impact_content, line_id, idx)
			def _make_card_handler(g=str(line_id).upper()):
				def _on_card_click(_e):
					live = next(
						(rt for rt in workers_runtime.values()
						 if rt.get("running") and rt.get("processing_rt")
						 and g in {str(x).upper() for x in rt.get("groups", [])}),
						None,
					)
					if live:
						_open_group_frames_window(live, g)
					else:
						_open_assign_worker_popup(g)
				return _on_card_click
			card_info["canvas"].bind("<Button-1>", _make_card_handler())

		for idx in range(len(line_cards)):
			high_impact_content.grid_columnconfigure(idx, weight=1)

		for idx, line_id in enumerate(final_cards):
			card_info = _create_group_card(final_content, line_id, idx)
			def _make_final_card_handler(g=str(line_id).upper()):
				def _on_card_click(_e):
					live = next(
						(rt for rt in workers_runtime.values()
						 if rt.get("running") and rt.get("processing_rt")
						 and g in {str(x).upper() for x in rt.get("groups", [])}),
						None,
					)
					if live:
						_open_group_frames_window(live, g)
					else:
						_open_assign_worker_popup(g)
				return _on_card_click
			card_info["canvas"].bind("<Button-1>", _make_final_card_handler())

		for idx in range(len(final_cards)):
			final_content.grid_columnconfigure(idx, weight=1)

		def _on_close_controller_extended():
			after_id = refresh_state.get("after_id")
			if after_id is not None:
				try:
					controller_win.after_cancel(after_id)
				except Exception:
					pass
			for key, fw in list(frame_windows.items()):
				try:
					if fw and fw.winfo_exists():
						fw.destroy()
				except Exception:
					pass
				frame_windows.pop(key, None)
			ww = workers_window_state.get("win")
			if ww is not None:
				try:
					if ww.winfo_exists():
						ww.destroy()
				except Exception:
					pass
			_on_close_controller()

		controller_win.protocol("WM_DELETE_WINDOW", _on_close_controller_extended)
		_refresh_workers_async()
	
	# Button 4: Dashboard
	def open_dashboard():
		# New color scheme for dashboard window (same as processing)
		PROC_BG = "#01326a"  # Main background
		PROC_CONTENT_BG = "#02234e"  # Content area background
		PROC_TAB_ACTIVE = "#5BA8C9"  # Active tab
		PROC_TAB_PREVIOUS = "#ffc735"  # Previous tab
		PROC_TAB_FUTURE = "#E6EEF9"  # Future tab
		PROC_BTN_NORMAL = "#015aca"
		PROC_BTN_CONFIRM = "#ffc735"
		
		# Load assets root early for header logo
		assets_root = _base_root() / "assets"
		
		dashboard_win = tk.Toplevel(root)
		dashboard_win.title("Dashboard")
		# Hide main window while dashboard window is open
		root.withdraw()
		
		# Store streamlit process
		streamlit_process = {"p": None}
		
		def _on_close_dashboard():
			# Kill streamlit process if running
			if streamlit_process["p"] and streamlit_process["p"].poll() is None:
				try:
					streamlit_process["p"].terminate()
				except Exception:
					pass
			dashboard_win.destroy()
			root.deiconify()
		
		dashboard_win.protocol("WM_DELETE_WINDOW", _on_close_dashboard)
		
		# Start maximized
		try:
			dashboard_win.state("zoomed")
		except Exception:
			try:
				dashboard_win.attributes("-zoomed", True)
			except Exception:
				pass
		
		# Keep a fallback geometry if zoom not supported
		_center_window(dashboard_win, 900, 600)
		dashboard_win.configure(bg=PROC_BG)
		
		wrapper = tk.Frame(dashboard_win, bg=PROC_BG)
		wrapper.pack(fill=tk.BOTH, expand=True)
		
		# ========== TOP HEADER: Logo + Breadcrumb ==========
		header_frame = tk.Frame(wrapper, bg=PROC_BG, height=80)
		header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
		header_frame.pack_propagate(False)
		
		# Mini logo on the left
		mini_logo_path = assets_root / "NGUI" / "ArnesisMiniLogo.png"
		menorque_path = assets_root / "NGUI" / "menorque.png"
		mini_logo_img = load_icon(mini_logo_path, 200, 100, invert=False)
		if mini_logo_img:
			logo_label = tk.Label(header_frame, image=mini_logo_img, bg=PROC_BG)
			logo_label.image = mini_logo_img
			logo_label.pack(side=tk.LEFT, padx=10)
		
		# Breadcrumb tabs container
		breadcrumb_frame = tk.Frame(header_frame, bg=PROC_BG)
		breadcrumb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		# Tab titles
		tab_titles = ["Cargar CSV", "Preview", "Heatmap"]
		
		# Create tab labels with separators
		breadcrumb_labels = []
		separator_imgs = []
		menorque_img = load_icon(menorque_path, 12, 12, invert=False) if menorque_path.exists() else None
		
		for i, title in enumerate(tab_titles):
			lbl = tk.Label(breadcrumb_frame, text=title, font=("Arial", 10, "bold"), 
			               bg=PROC_BG, fg=PROC_TAB_FUTURE, cursor="hand2")
			lbl.pack(side=tk.LEFT, padx=(5, 5))
			breadcrumb_labels.append(lbl)
			
			if i < len(tab_titles) - 1:
				if menorque_img:
					sep = tk.Label(breadcrumb_frame, image=menorque_img, bg=PROC_BG)
					sep.image = menorque_img
					sep.pack(side=tk.LEFT, padx=(0, 5))
					separator_imgs.append(sep)
					
		
		# ========== CONTENT AREA ==========
		content_frame = tk.Frame(wrapper, bg=PROC_CONTENT_BG)
		content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
		
		# Create tabs
		tabs = []
		for _ in tab_titles:
			tab = tk.Frame(content_frame, bg=PROC_CONTENT_BG)
			tabs.append(tab)
		
		# Workflow state
		current_step = 0
		can_advance = [False, False, False]
		
		# State for loaded CSV files
		csv_files = []
		loaded_csv_path = {"value": None}
		
		# Navigation buttons frame
		nav = tk.Frame(wrapper, bg=PROC_BG)
		nav.pack(fill=tk.X, padx=20, pady=(0, 10))
		
		def update_breadcrumb():
			for i, lbl in enumerate(breadcrumb_labels):
				if i < current_step:
					lbl.configure(fg=PROC_TAB_PREVIOUS)
				elif i == current_step:
					lbl.configure(fg=PROC_TAB_ACTIVE)
				else:
					lbl.configure(fg=PROC_TAB_FUTURE)
		
		def show_current_tab():
			for t in tabs:
				t.pack_forget()
			tabs[current_step].pack(fill=tk.BOTH, expand=True)
			update_breadcrumb()
		
		def update_nav_state():
			# Update button states
			if current_step == 0:
				prev_btn_canvas.configure(state="disabled")
			else:
				prev_btn_canvas.configure(state="normal")
			
			if current_step >= len(tab_titles) - 1 or not can_advance[current_step]:
				next_btn_canvas.configure(state="disabled")
			else:
				next_btn_canvas.configure(state="normal")
		
		def go_prev():
			nonlocal current_step
			if current_step > 0:
				current_step -= 1
				show_current_tab()
				update_nav_state()
		
		def go_next():
			nonlocal current_step
			if current_step < len(tab_titles) - 1 and can_advance[current_step]:
				current_step += 1
				show_current_tab()
				update_nav_state()
		
		# Helper for rounded buttons
		def make_rounded_button(parent, text, command, bg_color, width=120, height=40, fg=None):
			container = tk.Frame(parent, bg=parent.cget("bg"))
			canvas = tk.Canvas(container, width=width, height=height, bg=parent.cget("bg"), 
			                   highlightthickness=0, cursor="hand2")
			canvas.pack()
			
			state = {"enabled": True, "bg": bg_color, "fg": fg or "white"}
			
			def draw():
				canvas.delete("all")
				color = "#555" if not state["enabled"] else state["bg"]
				canvas.create_rounded_rectangle(2, 2, width-2, height-2, radius=8, fill=color, outline="")
				canvas.create_text(width//2, height//2, text=text, fill=state["fg"], 
				                   font=("Arial", 10, "bold"))
			
			def on_enter(e):
				if state["enabled"]:
					canvas.configure(cursor="hand2")
			
			canvas.bind("<Enter>", on_enter)
			canvas.bind("<Button-1>", lambda e: command() if state["enabled"] else None)
			
			def configure(state_val=None, **kwargs):
				if state_val:
					state["enabled"] = (state_val != "disabled")
					draw()
			
			canvas.configure = configure
			
			# Helper for rounded rectangles
			def create_rounded_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
				points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1, 
				          x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
				          x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2, x1, y2,
				          x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
				return canvas.create_polygon(points, smooth=True, **kwargs)
			
			canvas.create_rounded_rectangle = create_rounded_rectangle
			draw()
			container.winfo_children = lambda: [canvas]
			return container
		
		# Create navigation buttons
		prev_btn_container = make_rounded_button(nav, "Anterior", go_prev, PROC_BTN_NORMAL, width=120, height=40)
		prev_btn_canvas = prev_btn_container.winfo_children()[0]
		prev_btn_container.pack(side=tk.LEFT, padx=10)
		
		next_btn_container = make_rounded_button(nav, "Siguiente", go_next, PROC_BTN_CONFIRM, width=120, height=40)
		next_btn_canvas = next_btn_container.winfo_children()[0]
		next_btn_container.pack(side=tk.RIGHT, padx=10)
		
		# ============================================================
		# TAB 0: Cargar CSV
		# ============================================================
		csv_tab = tabs[0]
		csv_tab.grid_rowconfigure(0, weight=1)
		csv_tab.grid_columnconfigure(0, weight=1)
		
		# Drop area for CSV files
		csv_drop_container = tk.Frame(csv_tab, bg="#002e66", bd=3, relief=tk.FLAT, 
		                              highlightthickness=3, highlightbackground="#002e66")
		csv_drop_container.grid(row=0, column=0, sticky="nsew", padx=40, pady=40)
		csv_drop_container.grid_rowconfigure(0, weight=1)
		csv_drop_container.grid_columnconfigure(0, weight=1)
		
		csv_canvas = tk.Canvas(csv_drop_container, bg="#002959", highlightthickness=0)
		csv_canvas.grid(row=0, column=0, sticky="nsew")
		
		csv_center_label = tk.Label(
			csv_canvas, 
			text="¡Arrastra y suelta el csv aquí para cargar\nlos datos y generar un dashboard con recomendaciones!", 
			font=("Arial", 16, "bold"), 
			fg="#00a6e5", 
			bg="#002959",
			justify=tk.CENTER
		)
		csv_center_label_window = csv_canvas.create_window(0, 0, window=csv_center_label, tags="center")
		
		# Listbox for loaded files
		csv_list_frame = tk.Frame(csv_canvas, bg="#002959")
		csv_listbox = tk.Listbox(csv_list_frame, bg="#002959", fg=FG_COLOR, 
		                         selectbackground=GRAY_HOVER, borderwidth=0, highlightthickness=0)
		csv_scrollbar = tk.Scrollbar(csv_list_frame, orient=tk.VERTICAL, command=csv_listbox.yview)
		csv_listbox.configure(yscrollcommand=csv_scrollbar.set)
		csv_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		csv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		csv_list_window = csv_canvas.create_window(0, 0, window=csv_list_frame, tags="list", state="hidden")
		
		def layout_csv_canvas(_=None):
			w, h = csv_canvas.winfo_width(), csv_canvas.winfo_height()
			csv_canvas.coords("center", w // 2, h // 2)
			csv_canvas.coords("list", w // 2, h // 2)
			csv_canvas.itemconfig("list", width=w - 40, height=h - 40)
		
		csv_canvas.bind("<Configure>", layout_csv_canvas)
		
		def parse_csv_files(s: str) -> list[str]:
			"""Parse dropped file paths from drag-and-drop event data."""
			paths = []
			if not s:
				return paths
			parts = s.strip().split()
			for p in parts:
				p = p.strip()
				if not p:
					continue
				if p.startswith("{") and p.endswith("}"):
					p = p[1:-1]
				p = os.path.normpath(p)
				if os.path.exists(p):
					paths.append(p)
			return paths
		
		def filter_csv_files(paths: list[str]) -> list[str]:
			"""Filter for .csv files only."""
			return [p for p in paths if p.lower().endswith(".csv")]
		
		def refresh_csv_listbox():
			"""Update the listbox with loaded CSV files."""
			csv_listbox.delete(0, tk.END)
			for f in csv_files:
				csv_listbox.insert(tk.END, os.path.basename(f))
			
			if csv_files:
				csv_canvas.itemconfig("center", state="hidden")
				csv_canvas.itemconfig("list", state="normal")
			else:
				csv_canvas.itemconfig("center", state="normal")
				csv_canvas.itemconfig("list", state="hidden")
		
		def launch_streamlit_dashboard(csv_path):
			"""Launch Streamlit dashboard with the given CSV."""
			try:
				import subprocess
				repo_root = Path(__file__).parent.parent
				dashboard_script = repo_root / "Post-Proccessing" / "realtime_dashboard.py"
				
				if dashboard_script.exists():
					# Kill any existing streamlit process
					if streamlit_process["p"] and streamlit_process["p"].poll() is None:
						try:
							streamlit_process["p"].terminate()
						except Exception:
							pass
					
					# Launch streamlit with CSV path argument
					cmd = [
						"streamlit", "run",
						str(dashboard_script),
						"--",
						str(csv_path)
					]
					streamlit_process["p"] = subprocess.Popen(
						cmd,
						creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
						stdout=subprocess.DEVNULL,
						stderr=subprocess.DEVNULL
					)
					print(f"[Dashboard] Streamlit lanzado: {' '.join(cmd)}")
			except Exception as e:
				print(f"[Dashboard] Error lanzando streamlit: {e}")
		
		# COMMENTED OUT: Old PandasAI analysis display - now using heatmap visualization
		# def display_analysis_results(results: dict):
		# 	"""Display PandasAI analysis results in the text widget."""
		# 	analysis_text.configure(state=tk.NORMAL)
		# 	analysis_text.delete(1.0, tk.END)
		# 	
		# 	if not results["success"]:
		# 		# Show error
		# 		analysis_text.insert(tk.END, "⚠️ Error en el Análisis\n\n", "title")
		# 		analysis_text.insert(tk.END, f"{results.get('error', 'Error desconocido')}\n", "error")
		# 	else:
		# 		# Show successful analysis results
		# 		analysis_text.insert(tk.END, "✅ Análisis Completado\n\n", "title")
		# 		
		# 		analyses = results.get("analyses", {})
		# 		
		# 		# 1. Productivity by ROI
		# 		if "productivity" in analyses:
		# 			analysis_text.insert(tk.END, "📊 1. Productividad por ROI\n", "subtitle")
		# 			analysis_text.insert(tk.END, f"{analyses['productivity']}\n\n", "content")
		# 		
		# 		# 2. Temporal patterns
		# 		if "temporal" in analyses:
		# 			analysis_text.insert(tk.END, "⏰ 2. Patrones Temporales\n", "subtitle")
		# 			analysis_text.insert(tk.END, f"{analyses['temporal']}\n\n", "content")
		# 		
		# 		# 3. Ergonomics correlation
		# 		if "ergonomics" in analyses:
		# 			analysis_text.insert(tk.END, "🧍 3. Ergonomía\n", "subtitle")
		# 			analysis_text.insert(tk.END, f"{analyses['ergonomics']}\n\n", "content")
		# 		
		# 		# 4. Recommendations
		# 		if "recommendations" in analyses:
		# 			analysis_text.insert(tk.END, "💡 4. Recomendaciones\n", "subtitle")
		# 			analysis_text.insert(tk.END, f"{analyses['recommendations']}\n", "content")
		# 	
		# 	analysis_text.configure(state=tk.DISABLED)
		
		# def run_analysis_async(csv_path):
		# 	"""Run PandasAI analysis in a separate thread to avoid blocking UI."""
		# 	def worker():
		# 		try:
		# 			print("\n" + "="*60)
		# 			print("[PandasAI] Iniciando análisis inteligente...")
		# 			print(f"[PandasAI] CSV: {os.path.basename(csv_path)}")
		# 			print("="*60)
		# 			
		# 			results = run_complete_analysis(csv_path)
		# 			
		# 			print("\n" + "="*60)
		# 			if results["success"]:
		# 				print("[PandasAI] ✅ Análisis completado exitosamente")
		# 			else:
		# 				print(f"[PandasAI] ⚠️ Error: {results.get('error', 'desconocido')}")
		# 			print("="*60 + "\n")
		# 			
		# 			# Update UI in main thread
		# 			dashboard_win.after(0, lambda: display_analysis_results(results))
		# 		except Exception as e:
		# 			print(f"\n[PandasAI] Error inesperado: {e}")
		# 			import traceback
		# 			traceback.print_exc()
		# 			
		# 			error_results = {
		# 				"success": False,
		# 				"error": f"Error inesperado: {e}"
		# 			}
		# 			dashboard_win.after(0, lambda: display_analysis_results(error_results))
		# 	
		# 	thread = threading.Thread(target=worker, daemon=True)
		# 	thread.start()
		
		def accept_csv_files(paths: list[str]):
			"""Add CSV files to the list and launch dashboard."""
			valid = filter_csv_files(paths)
			if valid:
				csv_path = valid[0]  # Take first CSV
				if csv_path not in csv_files:
					csv_files.append(csv_path)
				loaded_csv_path["value"] = csv_path
				refresh_csv_listbox()
				
				# Launch Streamlit dashboard
				launch_streamlit_dashboard(csv_path)
				
				# Run PandasAI analysis (will handle errors if environment not configured)
				# run_analysis_async(csv_path)
				
				# Enable advance and go to next tab
				can_advance[0] = True
				update_nav_state()
				go_next()
		
		def on_csv_drop(event):
			paths = parse_csv_files(event.data)
			accept_csv_files(paths)
		
		def on_csv_click(_):
			selected = filedialog.askopenfilenames(
				title="Seleccionar archivos CSV",
				filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
			)
			if selected:
				accept_csv_files(list(selected))
		
		# Enable drag-and-drop
		if DND_AVAILABLE:
			csv_canvas.drop_target_register(DND_FILES)
			csv_canvas.dnd_bind("<<Drop>>", on_csv_drop)
		
		csv_canvas.bind("<Button-1>", on_csv_click)
		
		# ============================================================
		# TAB 1: Preview
		# ============================================================
		preview_tab = tabs[1]
		preview_tab.grid_rowconfigure(0, weight=0)
		preview_tab.grid_rowconfigure(1, weight=1)
		preview_tab.grid_rowconfigure(2, weight=0)
		preview_tab.grid_rowconfigure(3, weight=0)  # Analysis results row
		preview_tab.grid_columnconfigure(0, weight=1)
		preview_tab.grid_columnconfigure(1, weight=1)
		
		# LEFT SIDE: CSV Preview
		# Title
		preview_title = tk.Label(preview_tab, text="Preview del CSV", 
		                         font=("Arial", 18, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		preview_title.grid(row=0, column=0, pady=(20, 10), sticky="w", padx=40)
		
		# Preview area with scrollbars
		preview_frame = tk.Frame(preview_tab, bg=PROC_CONTENT_BG)
		preview_frame.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=(0, 20))
		preview_frame.grid_rowconfigure(0, weight=1)
		preview_frame.grid_columnconfigure(0, weight=1)
		
		# Configure ttk style for Treeview
		style = ttk.Style()
		style.theme_use("clam")
		style.configure("Preview.Treeview",
		                background="#002858",
		                foreground=FG_COLOR,
		                fieldbackground="#002858",
		                borderwidth=2,
		                relief=tk.SOLID)
		style.configure("Preview.Treeview.Heading",
		                background="#01326a",
		                foreground=FG_COLOR,
		                font=("Arial", 10, "bold"))
		style.map("Preview.Treeview",
		          background=[("selected", "#043c86")],
		          foreground=[("selected", FG_COLOR)])
		
		# Treeview (table) for preview
		preview_tree = ttk.Treeview(preview_frame, style="Preview.Treeview", height=15, show="headings")
		preview_tree.grid(row=0, column=0, sticky="nsew")
		
		# Scrollbars
		preview_vscroll = tk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_tree.yview)
		preview_vscroll.grid(row=0, column=1, sticky="ns")
		preview_tree.configure(yscrollcommand=preview_vscroll.set)
		
		preview_hscroll = tk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=preview_tree.xview)
		preview_hscroll.grid(row=1, column=0, sticky="ew")
		preview_tree.configure(xscrollcommand=preview_hscroll.set)
		
		# Info label below table
		preview_info = tk.Label(preview_tab, text="", font=("Arial", 10), 
		                        fg="#5BA8C9", bg=PROC_CONTENT_BG)
		preview_info.grid(row=2, column=0, pady=(10, 10), sticky="w", padx=40)
		
		video_title = tk.Label(preview_tab, text="Mas visualizaciones", 
		                       font=("Arial", 18, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		video_title.grid(row=0, column=1, pady=(20, 10), sticky="w", padx=40)
		
		# Drop area for videos
		video_drop_container = tk.Frame(preview_tab, bg="#002e66", bd=3, relief=tk.FLAT, 
		                                highlightthickness=3, highlightbackground="#002e66")
		video_drop_container.grid(row=1, column=1, sticky="nsew", padx=(20, 40), pady=(0, 20))
		video_drop_container.grid_rowconfigure(0, weight=1)
		video_drop_container.grid_columnconfigure(0, weight=1)
		
		video_canvas = tk.Canvas(video_drop_container, bg="#002959", highlightthickness=0)
		video_canvas.grid(row=0, column=0, sticky="nsew")
		
		video_center_label = tk.Label(
			video_canvas, 
			text="¡Arrastra y suelta un video\ncorrespondiente de los datos dados\npara generar nuevos visualizaciones de datos!",
			font=("Arial", 16, "bold"), 
			fg="#00a6e5", 
			bg="#002959",
			justify=tk.CENTER
		)
		video_center_label_window = video_canvas.create_window(0, 0, window=video_center_label, tags="center")
		
		# Listbox for loaded videos
		video_list_frame = tk.Frame(video_canvas, bg="#002959")
		video_listbox = tk.Listbox(video_list_frame, bg="#002959", fg=FG_COLOR, 
		                           selectbackground=GRAY_HOVER, borderwidth=0, highlightthickness=0)
		video_scrollbar = tk.Scrollbar(video_list_frame, orient=tk.VERTICAL, command=video_listbox.yview)
		video_listbox.configure(yscrollcommand=video_scrollbar.set)
		video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		video_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		video_list_window = video_canvas.create_window(0, 0, window=video_list_frame, tags="list", state="hidden")
		
		def layout_video_canvas(_=None):
			w, h = video_canvas.winfo_width(), video_canvas.winfo_height()
			video_canvas.coords("center", w // 2, h // 2)
			video_canvas.coords("list", w // 2, h // 2)
			video_canvas.itemconfig("list", width=w - 40, height=h - 40)
		
		video_canvas.bind("<Configure>", layout_video_canvas)
		
		# State for loaded video files
		video_files = []
		
		def parse_video_files(s: str) -> list[str]:
			"""Parse dropped file paths from drag-and-drop event data."""
			paths = []
			if not s:
				return paths
			
			# Handle paths with braces and spaces correctly
			buf = ""
			in_brace = False
			for ch in s:
				if ch == "{":
					in_brace = True
					buf = ""
				elif ch == "}":
					in_brace = False
					if buf:
						paths.append(buf)
					buf = ""
				elif ch == " " and not in_brace:
					if buf:
						paths.append(buf)
						buf = ""
				else:
					buf += ch
			if buf:
				paths.append(buf)
			
			# Normalize and validate paths
			validated_paths = []
			for p in paths:
				p = os.path.normpath(p)
				if os.path.exists(p):
					validated_paths.append(p)
			
			return validated_paths
		
		def filter_video_files(paths: list[str]) -> list[str]:
			"""Filter for video files only."""
			video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
			return [p for p in paths if os.path.splitext(p.lower())[1] in video_exts]
		
		def refresh_video_listbox():
			"""Update the listbox with loaded video files."""
			video_listbox.delete(0, tk.END)
			for f in video_files:
				video_listbox.insert(tk.END, os.path.basename(f))
			
			if video_files:
				video_canvas.itemconfig("center", state="hidden")
				video_canvas.itemconfig("list", state="normal")
			else:
				video_canvas.itemconfig("center", state="normal")
				video_canvas.itemconfig("list", state="hidden")
		
		def create_heatmap_from_coords(frame, coords, frame_width, frame_height, sigma=20):
			"""Create density heatmap from coordinates"""
			# Create empty heatmap
			heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
			
			# Add Gaussian blur at each coordinate
			for x, y in coords:
				# Ensure coordinates are within bounds
				if 0 <= x < frame_width and 0 <= y < frame_height:
					heatmap[y, x] += 1
			
			# Apply Gaussian blur for smooth heatmap
			if sigma > 0:
				heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
			
			# Normalize to 0-255
			if heatmap.max() > 0:
				heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
			else:
				heatmap = heatmap.astype(np.uint8)
			
			# Apply colormap (COLORMAP_JET: blue=low, red=high)
			heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
			
			# Convert BGR to RGB
			heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
			
			# Blend with original frame (70% heatmap, 30% original)
			# Make black areas (no data) transparent
			mask = heatmap > 0
			result = frame.copy()
			result[mask] = cv2.addWeighted(
				frame[mask], 0.3,
				heatmap_colored[mask], 0.7,
				0
			)
			
			return result
		
		def accept_video_files(paths: list[str]):
			valid = filter_video_files(paths)
			for p in valid:
				if p not in video_files:
					video_files.append(p)
			refresh_video_listbox()

			# Enable Heatmap tab and switch to it
			if valid:
				can_advance[1] = True  # Enable Heatmap tab
				update_nav_state()
				go_next()  # Switch to Heatmap tab
		def on_video_drop(event):
			paths = parse_video_files(event.data)
			accept_video_files(paths)
		
		def on_video_click(_):
			selected = filedialog.askopenfilenames(
				title="Seleccionar archivos de video",
				filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv *.webm"), ("All files", "*.*")]
			)
			if selected:
				accept_video_files(list(selected))
		
		# Enable drag-and-drop for videos
		if DND_AVAILABLE:
			video_drop_container.drop_target_register(DND_FILES)
			video_drop_container.dnd_bind("<<Drop>>", on_video_drop)
			video_canvas.drop_target_register(DND_FILES)
			video_canvas.dnd_bind("<<Drop>>", on_video_drop)
			video_center_label.drop_target_register(DND_FILES)
			video_center_label.dnd_bind("<<Drop>>", on_video_drop)
		
		video_canvas.bind("<Button-1>", on_video_click)
		
		def load_csv_preview():
			"""Load and display CSV preview in table format."""
			# Clear existing data
			for item in preview_tree.get_children():
				preview_tree.delete(item)
			preview_info.configure(text="")
			
			if loaded_csv_path["value"]:
				try:
					import csv
					with open(loaded_csv_path["value"], "r", encoding="utf-8") as f:
						reader = csv.reader(f)
						rows = list(reader)
						
						if not rows:
							preview_info.configure(text="CSV vacío")
							return
						
						# Set up columns from header
						headers = rows[0]
						preview_tree["columns"] = headers
						
						# Configure columns
						for col in headers:
							preview_tree.heading(col, text=col)
							preview_tree.column(col, width=150, minwidth=100, anchor="w")
						
						# Insert data rows (first 100 rows after header)
						data_rows = len(rows) - 1  # Total data rows (excluding header)
						max_rows = min(100, data_rows)
						for i in range(1, max_rows + 1):
							if i < len(rows):
								preview_tree.insert("", "end", values=rows[i])
						
						# Update info label
						if data_rows > 100:
							preview_info.configure(text=f"Mostrando las primeras 100 filas de {data_rows} totales ({data_rows - 100} filas más)")
						else:
							preview_info.configure(text=f"Mostrando todas las {data_rows} filas")
							
				except Exception as e:
					# Show error in a single column table
					preview_tree["columns"] = ["Error"]
					preview_tree.heading("Error", text="Error")
					preview_tree.column("Error", width=500, anchor="w")
					preview_tree.insert("", "end", values=[f"Error cargando CSV: {e}"])
					preview_info.configure(text="")
		

		# ============================================================
		# TAB 2: Heatmap
		# ============================================================
		heatmap_tab = tabs[2]
		heatmap_tab.configure(bg=PROC_CONTENT_BG)
		heatmap_tab.grid_rowconfigure(0, weight=0)  # Title row
		heatmap_tab.grid_rowconfigure(1, weight=0)  # Time controls row
		heatmap_tab.grid_rowconfigure(2, weight=0)  # Animation controls row
		heatmap_tab.grid_rowconfigure(3, weight=1)  # Canvas row
		heatmap_tab.grid_rowconfigure(4, weight=0)  # Stats row
		heatmap_tab.grid_columnconfigure(0, weight=1)
		
		# Title
		heatmap_title = tk.Label(heatmap_tab, text="Visualización de Heatmap - Densidad de Movimiento", 
		                         font=("Arial", 18, "bold"), fg=FG_COLOR, bg=PROC_CONTENT_BG)
		heatmap_title.grid(row=0, column=0, pady=(20, 10), sticky="w", padx=40)
		
		# Time period control frame
		time_frame = tk.LabelFrame(heatmap_tab, text="Período de Tiempo", bg=PROC_CONTENT_BG, 
		                           fg=FG_COLOR, font=("Arial", 10, "bold"), padx=10, pady=10)
		time_frame.grid(row=1, column=0, pady=10, sticky="ew", padx=40)
		
		# Start time
		tk.Label(time_frame, text="Hora Inicio (HH:MM:SS):", background=PROC_CONTENT_BG, 
		         foreground=FG_COLOR).grid(row=0, column=0, padx=5, sticky=tk.W)
		start_time_var = tk.StringVar(value="00:00:00")
		start_time_entry = tk.Entry(time_frame, textvariable=start_time_var, width=12, 
		                             font=("Arial", 11), bg="#FFFFFF", fg="#000000")
		start_time_entry.grid(row=0, column=1, padx=5)
		
		# End time
		tk.Label(time_frame, text="Hora Fin (HH:MM:SS):", background=PROC_CONTENT_BG, 
		         foreground=FG_COLOR).grid(row=0, column=2, padx=5, sticky=tk.W)
		end_time_var = tk.StringVar(value="23:59:59")
		end_time_entry = tk.Entry(time_frame, textvariable=end_time_var, width=12, 
		                           font=("Arial", 11), bg="#FFFFFF", fg="#000000")
		end_time_entry.grid(row=0, column=3, padx=5)
		
		# Update button
		update_heatmap_btn = tk.Button(time_frame, text="Actualizar Heatmap", 
		                                bg=PROC_BTN_NORMAL, fg="white", font=("Arial", 10, "bold"),
		                                command=lambda: update_heatmap_display())
		update_heatmap_btn.grid(row=0, column=4, padx=10)
		
		# Reset button
		reset_heatmap_btn = tk.Button(time_frame, text="Resetear", 
		                               bg=PROC_BTN_NORMAL, fg="white", font=("Arial", 10, "bold"),
		                               command=lambda: reset_time_range())
		reset_heatmap_btn.grid(row=0, column=5, padx=5)
		
		# Text filter (Row 1)
		tk.Label(time_frame, text="Filtro de Texto:", background=PROC_CONTENT_BG, 
		         foreground=FG_COLOR).grid(row=1, column=0, padx=5, pady=(10, 0), sticky=tk.W)
		filter_text_var = tk.StringVar(value="")
		filter_text_entry = tk.Entry(time_frame, textvariable=filter_text_var, width=30, 
		                              font=("Arial", 11), bg="#FFFFFF", fg="#000000")
		filter_text_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=(10, 0), sticky=tk.W)
		
		tk.Label(time_frame, text="(Dejar vacío para no filtrar)", background=PROC_CONTENT_BG, 
		         foreground="#5BA8C9", font=("Arial", 8)).grid(row=1, column=4, columnspan=2, padx=5, pady=(10, 0), sticky=tk.W)
		
		# Animation controls frame
		anim_frame = tk.LabelFrame(heatmap_tab, text="Controles de Animación", bg=PROC_CONTENT_BG, 
		                           fg=FG_COLOR, font=("Arial", 10, "bold"), padx=10, pady=10)
		anim_frame.grid(row=2, column=0, pady=10, sticky="ew", padx=40)
		
		# Play/Pause button
		play_btn = tk.Button(anim_frame, text="▶ Iniciar Animación", width=20,
		                      bg=PROC_BTN_CONFIRM, fg="white", font=("Arial", 10, "bold"),
		                      command=lambda: toggle_animation())
		play_btn.grid(row=0, column=0, padx=5)
		
		# Stop button
		stop_btn = tk.Button(anim_frame, text="⬛ Detener", width=15,
		                      bg=PROC_BTN_NORMAL, fg="white", font=("Arial", 10, "bold"),
		                      command=lambda: stop_animation())
		stop_btn.grid(row=0, column=1, padx=5)
		
		# Interval control
		tk.Label(anim_frame, text="Intervalo (ms):", background=PROC_CONTENT_BG, 
		         foreground=FG_COLOR).grid(row=0, column=2, padx=5, sticky=tk.W)
		interval_var = tk.DoubleVar(value=1)
		interval_spinbox = tk.Spinbox(anim_frame, from_=0.001, to=10000, increment=0.001, 
		                               textvariable=interval_var, width=8, font=("Arial", 10),
		                               bg="#FFFFFF", fg="#000000")
		interval_spinbox.grid(row=0, column=3, padx=5)
		
		# Auto-replay checkbox
		auto_replay_var = tk.BooleanVar(value=True)
		auto_replay_check = tk.Checkbutton(anim_frame, text="Auto-repetir", variable=auto_replay_var,
		                                     bg=PROC_CONTENT_BG, fg=FG_COLOR, selectcolor="#002858",
		                                     font=("Arial", 10))
		auto_replay_check.grid(row=0, column=4, padx=10)
		
		# Animation progress label
		anim_progress_label = tk.Label(anim_frame, text="Listo", font=("Arial", 9), 
		                                bg=PROC_CONTENT_BG, fg="#5BA8C9")
		anim_progress_label.grid(row=0, column=5, padx=15)
		
		# Export to GIF button
		export_gif_btn = tk.Button(anim_frame, text="💾 Exportar GIF", width=15,
		                            bg="#015aca", fg="white", font=("Arial", 10, "bold"),
		                            command=lambda: export_animation_to_gif())
		export_gif_btn.grid(row=0, column=6, padx=10)
		
		# Canvas for heatmap display
		heatmap_canvas_frame = tk.Frame(heatmap_tab, bg=PROC_CONTENT_BG)
		heatmap_canvas_frame.grid(row=3, column=0, sticky="nsew", padx=40, pady=(0, 10))
		heatmap_canvas_frame.grid_rowconfigure(0, weight=1)
		heatmap_canvas_frame.grid_columnconfigure(0, weight=1)
		
		heatmap_display_canvas = tk.Canvas(heatmap_canvas_frame, bg="#002858", 
		                                    relief=tk.SOLID, borderwidth=2, highlightthickness=0)
		heatmap_display_canvas.grid(row=0, column=0, sticky="nsew")
		
		# Info label for heatmap
		heatmap_display_info = tk.Label(heatmap_display_canvas, 
		                                 text="Arrastra un video en el tab Preview para visualizar el heatmap",
		                                 font=("Arial", 12), fg="#5BA8C9", bg="#002858", justify=tk.CENTER)
		heatmap_display_canvas.create_window(0, 0, window=heatmap_display_info, tags="info")
		
		# Storage for heatmap photo and data
		heatmap_data = {
			"photo": None,
			"all_coords": [],
			"min_time": "00:00:00",
			"max_time": "23:59:59",
			"frame": None,
			"frame_width": 0,
			"frame_height": 0,
			"is_animating": False,
			"animation_timer": None,
			"current_animation_time": None,
			"last_filter_text": None
		}
		
		def layout_heatmap_display_canvas(_=None):
			w, h = heatmap_display_canvas.winfo_width(), heatmap_display_canvas.winfo_height()
			heatmap_display_canvas.coords("info", w // 2, h // 2)
			if heatmap_display_canvas.find_withtag("image"):
				heatmap_display_canvas.coords("image", w // 2, h // 2)
		
		heatmap_display_canvas.bind("<Configure>", layout_heatmap_display_canvas)
		
		# Stats label
		heatmap_stats_label = tk.Label(heatmap_tab, text="", font=("Arial", 9), 
		                                bg=PROC_CONTENT_BG, fg="#5BA8C9")
		heatmap_stats_label.grid(row=4, column=0, pady=5, sticky="w", padx=40)
		
		# Heatmap functions
		def filter_coords_by_time(start_time, end_time):
			"""Filter coordinates by time range"""
			filtered_coords = []
			for item in heatmap_data["all_coords"]:
				if start_time <= item['time'] <= end_time:
					filtered_coords.append(item['coord'])
			return filtered_coords
		
		def filter_coords_by_sliding_window(current_time, window_seconds=60):
			"""Filter coordinates within a sliding time window (for animation fade effect)"""
			from datetime import datetime, timedelta
			filtered_coords = []
			
			try:
				current_dt = datetime.strptime(current_time, "%H:%M:%S")
				window_start_dt = current_dt - timedelta(seconds=window_seconds)
				window_start = window_start_dt.strftime("%H:%M:%S")
				
				# Filter coords within the window
				for item in heatmap_data["all_coords"]:
					if window_start <= item['time'] <= current_time:
						filtered_coords.append(item['coord'])
			except Exception as e:
				print(f"[ERROR] filter_coords_by_sliding_window: {e}")
			
			return filtered_coords
		
		def update_heatmap_display():
			"""Update the displayed heatmap"""
			try:
				# Check if we have required data
				if not video_files:
					heatmap_display_info.configure(text="No hay video cargado\nArrastra un video en el tab Preview")
					return
				
				if not loaded_csv_path["value"]:
					heatmap_display_info.configure(text="No hay CSV cargado\nCarga un CSV primero")
					return
				
				# Check if filter text has changed
				current_filter = filter_text_var.get().strip()
				if heatmap_data["last_filter_text"] != current_filter:
					# Filter changed, reload data
					heatmap_data["all_coords"] = []
					heatmap_data["last_filter_text"] = current_filter
				
				# Load data if not already loaded
				if not heatmap_data["all_coords"]:
					load_heatmap_data()
				
				# If no data after loading, show frame without data
				if not heatmap_data["all_coords"]:
					if heatmap_data["frame"] is not None:
						# Show empty frame
						display_frame = heatmap_data["frame"].copy()
						from PIL import Image, ImageTk
						pil_image = Image.fromarray(display_frame)
						
						# Scale image to fit canvas
						heatmap_display_canvas.update_idletasks()
						max_width = heatmap_display_canvas.winfo_width() - 40
						max_height = heatmap_display_canvas.winfo_height() - 40
						
						width_scale = max_width / heatmap_data["frame_width"]
						height_scale = max_height / heatmap_data["frame_height"]
						scale = min(width_scale, height_scale, 1.0)
						
						if scale < 1.0:
							new_width = int(heatmap_data["frame_width"] * scale)
							new_height = int(heatmap_data["frame_height"] * scale)
							pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
						
						photo = ImageTk.PhotoImage(pil_image)
						heatmap_data["photo"] = photo
						
						heatmap_display_canvas.delete("all")
						heatmap_display_canvas.create_image(
							heatmap_display_canvas.winfo_width() // 2,
							heatmap_display_canvas.winfo_height() // 2,
							image=photo, tags="image"
						)
						
						heatmap_stats_label.configure(text="No hay datos para mostrar con el filtro actual")
					return
				
				# Get time range from entries
				start_time = start_time_var.get()
				end_time = end_time_var.get()
				
				# Validate time format
				try:
					from datetime import datetime
					datetime.strptime(start_time, "%H:%M:%S")
					datetime.strptime(end_time, "%H:%M:%S")
				except ValueError:
					heatmap_stats_label.configure(text="Formato de hora inválido! Usa HH:MM:SS")
					return
				
				# Filter coordinates by time range
				filtered_coords = filter_coords_by_time(start_time, end_time)
				
				# If no coordinates after time filtering, show frame without data
				if len(filtered_coords) == 0:
					if heatmap_data["frame"] is not None:
						# Show empty frame
						display_frame = heatmap_data["frame"].copy()
						from PIL import Image, ImageTk
						pil_image = Image.fromarray(display_frame)
						
						# Scale image to fit canvas
						heatmap_display_canvas.update_idletasks()
						max_width = heatmap_display_canvas.winfo_width() - 40
						max_height = heatmap_display_canvas.winfo_height() - 40
						
						width_scale = max_width / heatmap_data["frame_width"]
						height_scale = max_height / heatmap_data["frame_height"]
						scale = min(width_scale, height_scale, 1.0)
						
						if scale < 1.0:
							new_width = int(heatmap_data["frame_width"] * scale)
							new_height = int(heatmap_data["frame_height"] * scale)
							pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
						
						photo = ImageTk.PhotoImage(pil_image)
						heatmap_data["photo"] = photo
						
						heatmap_display_canvas.delete("all")
						heatmap_display_canvas.create_image(
							heatmap_display_canvas.winfo_width() // 2,
							heatmap_display_canvas.winfo_height() // 2,
							image=photo, tags="image"
						)
						
						heatmap_stats_label.configure(text=f"No hay datos en el período: {start_time} - {end_time}")
					return
				
				# Create heatmap with filtered coordinates
				display_frame = heatmap_data["frame"].copy()
				heatmap_result = create_heatmap_from_coords(display_frame, filtered_coords, 
				                                             heatmap_data["frame_width"], 
				                                             heatmap_data["frame_height"])
				
				# Convert to PIL Image
				from PIL import Image, ImageTk
				pil_image = Image.fromarray(heatmap_result)
				
				# Scale image to fit canvas
				heatmap_display_canvas.update_idletasks()
				max_width = heatmap_display_canvas.winfo_width() - 40
				max_height = heatmap_display_canvas.winfo_height() - 40
				
				# Calculate scaling
				width_scale = max_width / heatmap_data["frame_width"]
				height_scale = max_height / heatmap_data["frame_height"]
				scale = min(width_scale, height_scale, 1.0)
				
				if scale < 1.0:
					new_width = int(heatmap_data["frame_width"] * scale)
					new_height = int(heatmap_data["frame_height"] * scale)
					pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
				
				# Convert to PhotoImage
				photo = ImageTk.PhotoImage(pil_image)
				heatmap_data["photo"] = photo
				
				# Display on canvas
				heatmap_display_canvas.delete("all")
				heatmap_display_canvas.create_image(
					heatmap_display_canvas.winfo_width() // 2,
					heatmap_display_canvas.winfo_height() // 2,
					image=photo, tags="image"
				)
				
				# Update stats
				filter_info = f" (Filtro: '{current_filter}')" if current_filter else ""
				stats_text = f"Período: {start_time} - {end_time} | Coordenadas: {len(filtered_coords)}{filter_info}"
				heatmap_stats_label.configure(text=stats_text)
				
			except Exception as e:
				import traceback
				error_msg = f"Error actualizando heatmap: {str(e)}"
				print(f"[ERROR] {error_msg}")
				print(traceback.format_exc())
				heatmap_stats_label.configure(text=error_msg)
		
		def load_heatmap_data():
			"""Load CSV data and video frame for heatmap"""
			try:
				if not video_files or not loaded_csv_path["value"]:
					return
				
				video_path = video_files[0]
				
				# Load CSV data
				import pandas as pd
				df = pd.read_csv(loaded_csv_path["value"])
				
				# Check required columns
				required_cols = ["coord_x", "coord_y"]
				for col in required_cols:
					if col not in df.columns:
						heatmap_display_info.configure(text=f"Error: Columna '{col}' no encontrada")
						return
				
				# Filter by text if filter is not empty
				filter_text = filter_text_var.get().strip()
				if filter_text and "video_name" in df.columns:
					# Filter rows where video_name contains the filter text
					df = df[df["video_name"].str.contains(filter_text, case=False, na=False)]
					
					if len(df) == 0:
						# No data matching filter, but continue to load video frame
						heatmap_data["all_coords"] = []
						# Load video frame anyway
						cap = cv2.VideoCapture(video_path)
						if cap.isOpened():
							heatmap_data["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
							heatmap_data["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
							cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
							ret, frame = cap.read()
							cap.release()
							if ret:
								heatmap_data["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						return
				
				# Remove rows with NaN
				df = df.dropna(subset=["coord_x", "coord_y"])
				
				if len(df) == 0:
					# No valid coordinates, but continue to load video frame
					heatmap_data["all_coords"] = []
					# Load video frame anyway
					cap = cv2.VideoCapture(video_path)
					if cap.isOpened():
						heatmap_data["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						heatmap_data["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
						ret, frame = cap.read()
						cap.release()
						if ret:
							heatmap_data["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					return
				
				# Store all coordinates with timestamps if available
				heatmap_data["all_coords"] = []
				if "HH:MM:SS" in df.columns:
					for _, row in df.iterrows():
						try:
							cx = int(row["coord_x"])
							cy = int(row["coord_y"])
						except (ValueError, TypeError):
							continue
						heatmap_data["all_coords"].append({
							'time': row["HH:MM:SS"],
							'coord': (cx, cy)
						})
					# Find time range
					heatmap_data["min_time"] = df["HH:MM:SS"].min()
					heatmap_data["max_time"] = df["HH:MM:SS"].max()
					start_time_var.set(heatmap_data["min_time"])
					end_time_var.set(heatmap_data["max_time"])
				else:
					# No timestamp, use all coordinates
					for _, row in df.iterrows():
						try:
							cx = int(row["coord_x"])
							cy = int(row["coord_y"])
						except (ValueError, TypeError):
							continue
						heatmap_data["all_coords"].append({
							'time': "00:00:00",
							'coord': (cx, cy)
						})
				
				# Load video frame
				cap = cv2.VideoCapture(video_path)
				if not cap.isOpened():
					heatmap_display_info.configure(text="Error abriendo video")
					return
				
				heatmap_data["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				heatmap_data["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				
				# Get first frame
				cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
				ret, frame = cap.read()
				cap.release()
				
				if ret:
					heatmap_data["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				else:
					heatmap_data["frame"] = np.zeros((heatmap_data["frame_height"], 
					                                     heatmap_data["frame_width"], 3), dtype=np.uint8)
				
				# Update title
				heatmap_title.configure(
					text=f"Visualización de Heatmap - {len(heatmap_data['all_coords'])} coordenadas de '{os.path.basename(video_path)}'"
				)
				
			except Exception as e:
				import traceback
				print(f"[ERROR] Loading heatmap data: {e}")
				print(traceback.format_exc())
				heatmap_display_info.configure(text=f"Error cargando datos: {str(e)}")
		
		def reset_time_range():
			"""Reset time range to full dataset"""
			start_time_var.set(heatmap_data["min_time"])
			end_time_var.set(heatmap_data["max_time"])
			update_heatmap_display()
		
		def toggle_animation():
			"""Toggle animation play/pause"""
			if heatmap_data["is_animating"]:
				pause_animation()
			else:
				start_animation()
		
		def start_animation():
			"""Start the heatmap animation"""
			start_time = start_time_var.get()
			end_time = end_time_var.get()
			
			# Validate time format
			try:
				from datetime import datetime
				datetime.strptime(start_time, "%H:%M:%S")
				datetime.strptime(end_time, "%H:%M:%S")
			except ValueError:
				heatmap_stats_label.configure(text="Formato de hora inválido!")
				return
			
			# Initialize animation
			heatmap_data["is_animating"] = True
			heatmap_data["current_animation_time"] = start_time
			play_btn.configure(text="⏸ Pausar Animación")
			anim_progress_label.configure(text=f"Animando: {start_time}")
			
			# Start animation loop
			animate_step()
		
		def pause_animation():
			"""Pause the animation"""
			heatmap_data["is_animating"] = False
			if heatmap_data["animation_timer"]:
				dashboard_win.after_cancel(heatmap_data["animation_timer"])
				heatmap_data["animation_timer"] = None
			play_btn.configure(text="▶ Iniciar Animación")
			anim_progress_label.configure(text="Pausado")
		
		def stop_animation():
			"""Stop and reset the animation"""
			heatmap_data["is_animating"] = False
			if heatmap_data["animation_timer"]:
				dashboard_win.after_cancel(heatmap_data["animation_timer"])
				heatmap_data["animation_timer"] = None
			heatmap_data["current_animation_time"] = None
			play_btn.configure(text="▶ Iniciar Animación")
			anim_progress_label.configure(text="Detenido")
			
			# Reset to full range display
			update_heatmap_display()
		
		def export_animation_to_gif():
			"""Export heatmap animation to GIF file"""
			print("\n[DEBUG GIF EXPORT] ========== INICIO ==========")
			start_time = start_time_var.get()
			end_time = end_time_var.get()
			print(f"[DEBUG GIF EXPORT] Rango de tiempo: {start_time} - {end_time}")
			
			# Validate time format
			try:
				from datetime import datetime, timedelta
				start_dt = datetime.strptime(start_time, "%H:%M:%S")
				end_dt = datetime.strptime(end_time, "%H:%M:%S")
				print(f"[DEBUG GIF EXPORT] ✓ Formato de tiempo validado")
			except ValueError as e:
				print(f"[DEBUG GIF EXPORT] ✗ Error de formato de tiempo: {e}")
				heatmap_stats_label.configure(text="Formato de hora inválido para exportar!")
				return
			
			# Calculate total seconds
			total_seconds = int((end_dt - start_dt).total_seconds())
			print(f"[DEBUG GIF EXPORT] Duración total: {total_seconds} segundos")
			if total_seconds <= 0:
				print(f"[DEBUG GIF EXPORT] ✗ Rango de tiempo inválido")
				heatmap_stats_label.configure(text="Rango de tiempo inválido para exportar!")
				return
			
			# Check if we have data
			coords_count = len(heatmap_data["all_coords"]) if heatmap_data["all_coords"] else 0
			has_frame = heatmap_data["frame"] is not None
			print(f"[DEBUG GIF EXPORT] Coordenadas disponibles: {coords_count}")
			print(f"[DEBUG GIF EXPORT] Frame disponible: {has_frame}")
			if not heatmap_data["all_coords"] or heatmap_data["frame"] is None:
				print(f"[DEBUG GIF EXPORT] ✗ No hay datos para exportar")
				heatmap_stats_label.configure(text="No hay datos para exportar!")
				return
			
			# Ask user for save location
			from tkinter import filedialog
			print(f"[DEBUG GIF EXPORT] Abriendo diálogo para guardar archivo...")
			output_path = filedialog.asksaveasfilename(
				defaultextension=".gif",
				filetypes=[("GIF files", "*.gif"), ("All files", "*.*")],
				initialfile=f"heatmap_{start_time.replace(':', '')}_to_{end_time.replace(':', '')}.gif"
			)
			
			print(f"[DEBUG GIF EXPORT] Ruta seleccionada: {output_path}")
			if not output_path:
				print(f"[DEBUG GIF EXPORT] ✗ Usuario canceló selección de archivo")
				return
			
			# Disable export button during export
			print(f"[DEBUG GIF EXPORT] Deshabilitando botón de exportación...")
			export_gif_btn.config(state="disabled", text="Exportando...")
			heatmap_stats_label.configure(text="Generando frames de GIF...")
			dashboard_win.update()
			
			# Generate frames
			from PIL import Image, ImageEnhance
			frames = []
			current_time = start_time
			frame_count = 0
			iteration_count = 0
			max_iterations = 10000  # Safety limit to prevent infinite loops
			
			# Determine skip based on total duration to keep GIF reasonable
			if total_seconds > 300:
				skip_seconds = 10  # For long videos, sample every 10 seconds
			elif total_seconds > 60:
				skip_seconds = 5   # For medium videos, sample every 5 seconds
			else:
				skip_seconds = 1   # For short videos, sample every second
			
			print(f"[DEBUG GIF EXPORT] Salto de segundos: {skip_seconds}")
			print(f"[DEBUG GIF EXPORT] Iniciando generación de frames...")
			print(f"[DEBUG GIF EXPORT] Tiempo inicial: {current_time}, Tiempo final: {end_time}")
			
			# Convert to datetime for reliable comparison
			current_dt = datetime.strptime(current_time, "%H:%M:%S")
			end_dt = datetime.strptime(end_time, "%H:%M:%S")
			
			while current_dt <= end_dt and iteration_count < max_iterations:
				iteration_count += 1
				current_time = current_dt.strftime("%H:%M:%S")
				
				if iteration_count % 50 == 0:
					print(f"[DEBUG GIF EXPORT] Iteración {iteration_count}, tiempo actual: {current_time}")
				
				# Get coordinates using sliding window (same as animation)
				filtered_coords = filter_coords_by_sliding_window(current_time)
				
				if iteration_count == 1:
					print(f"[DEBUG GIF EXPORT] Primera iteración: {len(filtered_coords)} coordenadas filtradas")
				
				# Always create a frame (with or without heatmap)
				display_frame = heatmap_data["frame"].copy()
				
				if len(filtered_coords) > 0:
					# Create heatmap with coordinates
					heatmap_result = create_heatmap_from_coords(
						display_frame, filtered_coords,
						heatmap_data["frame_width"],
						heatmap_data["frame_height"]
					)
				else:
					# No coordinates, use plain frame (fade effect)
					heatmap_result = display_frame
				
				# Add timestamp label in bottom-left corner
				time_text = f"Hora: {current_time}"
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = heatmap_data["frame_width"] / 800  # Scale based on frame width
				font_thickness = max(1, int(heatmap_data["frame_width"] / 400))
				text_size = cv2.getTextSize(time_text, font, font_scale, font_thickness)[0]
				text_x = 10
				text_y = heatmap_data["frame_height"] - 10
				
				# Draw text background (semi-transparent black rectangle)
				padding = 5
				cv2.rectangle(heatmap_result, 
				              (text_x - padding, text_y - text_size[1] - padding),
				              (text_x + text_size[0] + padding, text_y + padding),
				              (0, 0, 0), -1)
				
				# Draw text in white with black outline for visibility
				cv2.putText(heatmap_result, time_text, (text_x, text_y), font, 
				            font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)  # Black outline
				cv2.putText(heatmap_result, time_text, (text_x, text_y), font, 
				            font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)  # White text
				
				# Convert to PIL Image
				pil_image = Image.fromarray(heatmap_result)
				frames.append(pil_image)
				frame_count += 1
				
				if frame_count % 5 == 0:
					print(f"[DEBUG GIF EXPORT] Frames generados: {frame_count} (tiempo actual: {current_time}, coords: {len(filtered_coords)})")
					heatmap_stats_label.configure(text=f"Generados {frame_count} frames...")
					dashboard_win.update_idletasks()  # Use update_idletasks() instead of update()
				
				# Advance time
				try:
					current_dt = current_dt + timedelta(seconds=skip_seconds)
				except Exception as e:
					print(f"[DEBUG GIF EXPORT] Error avanzando tiempo: {e}")
					break
			
			if iteration_count >= max_iterations:
				print(f"[DEBUG GIF EXPORT] ⚠️ Alcanzado límite máximo de iteraciones: {max_iterations}")
			
			print(f"[DEBUG GIF EXPORT] Total de iteraciones: {iteration_count}")
			print(f"[DEBUG GIF EXPORT] Total de frames generados: {len(frames)}")
			if len(frames) == 0:
				print(f"[DEBUG GIF EXPORT] ✗ No se generaron frames!")
				heatmap_stats_label.configure(text="No se generaron frames!")
				export_gif_btn.config(state="normal", text="💾 Exportar GIF")
				return
			
			# Save as GIF
			print(f"[DEBUG GIF EXPORT] Iniciando guardado de GIF...")
			heatmap_stats_label.configure(text=f"Guardando GIF con {len(frames)} frames...")
			dashboard_win.update()
			
			try:
				# Calculate duration per frame (in milliseconds)
				duration = max(50, 1000 // len(frames))  # At least 50ms per frame
				print(f"[DEBUG GIF EXPORT] Duración por frame: {duration}ms")
				
				# Convert frames to RGB mode and enhance colors for better vibrancy
				print(f"[DEBUG GIF EXPORT] Convirtiendo frames a RGB y aplicando mejoras...")
				rgb_frames = []
				for i, frame in enumerate(frames):
					if frame.mode != 'RGB':
						frame = frame.convert('RGB')
					
					# Enhance brightness (1.0 = original, >1.0 = brighter)
					enhancer = ImageEnhance.Brightness(frame)
					frame = enhancer.enhance(1.15)
					
					# Enhance contrast (1.0 = original, >1.0 = more contrast)
					enhancer = ImageEnhance.Contrast(frame)
					frame = enhancer.enhance(1.15)
					
					# Enhance color saturation (1.0 = original, >1.0 = more saturated)
					enhancer = ImageEnhance.Color(frame)
					frame = enhancer.enhance(1.6)
					
					rgb_frames.append(frame)
					
					if (i + 1) % 20 == 0:
						print(f"[DEBUG GIF EXPORT] Mejorados {i + 1}/{len(frames)} frames")
				
				print(f"[DEBUG GIF EXPORT] Guardando archivo GIF en: {output_path}")
				# Save with optimization disabled to preserve color quality
				rgb_frames[0].save(
					output_path,
					save_all=True,
					append_images=rgb_frames[1:],
					duration=duration,
					loop=0,  # Infinite loop
					optimize=False  # Disable optimization to preserve colors
				)
				
				print(f"[DEBUG GIF EXPORT] ✓ GIF guardado exitosamente!")
				print(f"[DEBUG GIF EXPORT] Ruta: {output_path}")
				print(f"[DEBUG GIF EXPORT] Frames: {len(frames)}")
				print(f"[DEBUG GIF EXPORT] Tamaño del archivo: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
				heatmap_stats_label.configure(text=f"✅ GIF exportado: {os.path.basename(output_path)} ({len(frames)} frames)")
			except Exception as e:
				print(f"[DEBUG GIF EXPORT] ✗ ERROR al guardar GIF: {e}")
				heatmap_stats_label.configure(text=f"❌ Error guardando GIF: {str(e)}")
				import traceback
				print(f"[ERROR] Exporting GIF: {e}")
				print(traceback.format_exc())
			
			# Re-enable export button
			print(f"[DEBUG GIF EXPORT] Rehabilitando botón de exportación...")
			print(f"[DEBUG GIF EXPORT] ========== FIN ==========\n")
			export_gif_btn.config(state="normal", text="💾 Exportar GIF")
		
		def animate_step():
			"""Execute one animation step"""
			if not heatmap_data["is_animating"]:
				return
			
			start_time = start_time_var.get()
			end_time = end_time_var.get()
			
			# Get coordinates within sliding window (fades out old data)
			# Window size: 30 seconds (adjustable)
			filtered_coords = filter_coords_by_sliding_window(heatmap_data["current_animation_time"])
			
			if heatmap_data["frame"] is not None:
				# Create and display heatmap (or plain frame if no coords)
				display_frame = heatmap_data["frame"].copy()
				
				if len(filtered_coords) > 0:
					# Create heatmap with available coordinates
					heatmap_result = create_heatmap_from_coords(display_frame, filtered_coords,
					                                             heatmap_data["frame_width"],
					                                             heatmap_data["frame_height"])
				else:
					# No coordinates in window, show plain frame (faded effect)
					heatmap_result = display_frame
				
				# Add timestamp label in bottom-left corner
				time_text = f"Hora: {heatmap_data['current_animation_time']}"
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = heatmap_data["frame_width"] / 800  # Scale based on frame width
				font_thickness = max(1, int(heatmap_data["frame_width"] / 400))
				text_size = cv2.getTextSize(time_text, font, font_scale, font_thickness)[0]
				text_x = 10
				text_y = heatmap_data["frame_height"] - 10
				
				# Draw text background (semi-transparent black rectangle)
				padding = 5
				cv2.rectangle(heatmap_result, 
				              (text_x - padding, text_y - text_size[1] - padding),
				              (text_x + text_size[0] + padding, text_y + padding),
				              (0, 0, 0), -1)
				
				# Draw text in white with black outline for visibility
				cv2.putText(heatmap_result, time_text, (text_x, text_y), font, 
				            font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)  # Black outline
				cv2.putText(heatmap_result, time_text, (text_x, text_y), font, 
				            font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)  # White text
				
				from PIL import Image, ImageTk
				pil_image = Image.fromarray(heatmap_result)
				
				# Scale image if needed
				heatmap_display_canvas.update_idletasks()
				max_width = heatmap_display_canvas.winfo_width() - 40
				max_height = heatmap_display_canvas.winfo_height() - 40
				
				width_scale = max_width / heatmap_data["frame_width"]
				height_scale = max_height / heatmap_data["frame_height"]
				scale = min(width_scale, height_scale, 1.0)
				
				if scale < 1.0:
					new_width = int(heatmap_data["frame_width"] * scale)
					new_height = int(heatmap_data["frame_height"] * scale)
					pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
				
				photo = ImageTk.PhotoImage(pil_image)
				heatmap_data["photo"] = photo
				
				heatmap_display_canvas.delete("all")
				heatmap_display_canvas.create_image(
					heatmap_display_canvas.winfo_width() // 2,
					heatmap_display_canvas.winfo_height() // 2,
					image=photo, tags="image"
				)
				
				# Update stats
				stats_text = f"Animando: {start_time} → {heatmap_data['current_animation_time']} | Coordenadas: {len(filtered_coords)}"
				heatmap_stats_label.configure(text=stats_text)
				anim_progress_label.configure(text=f"Actual: {heatmap_data['current_animation_time']}")
				
				# Force GUI update
				dashboard_win.update_idletasks()
			
			# Check if reached end
			if heatmap_data["current_animation_time"] >= end_time:
				if auto_replay_var.get():
					# Restart animation
					heatmap_data["current_animation_time"] = start_time
				else:
					# Stop animation
					heatmap_data["is_animating"] = False
					play_btn.configure(text="▶ Iniciar Animación")
					anim_progress_label.configure(text="¡Completo!")
					return
			
			# Advance time based on animation speed
			# For very fast speeds, skip multiple seconds to avoid rendering bottleneck
			interval = interval_var.get()
			if interval <= 0.001:
				# Ultra fast: skip 1000 seconds per tick
				skip_seconds = 1000
			elif interval <= 0.01:
				# Very fast: skip 100 seconds per tick
				skip_seconds = 100
			elif interval <= 0.1:
				# Fast: skip 10 seconds per tick
				skip_seconds = 10
			else:
				# Normal: advance 1 second per tick
				skip_seconds = 1
			
			try:
				from datetime import datetime, timedelta
				current_dt = datetime.strptime(heatmap_data["current_animation_time"], "%H:%M:%S")
				next_dt = current_dt + timedelta(seconds=skip_seconds)
				heatmap_data["current_animation_time"] = next_dt.strftime("%H:%M:%S")
			except ValueError:
				stop_animation()
				return
			
			# Schedule next step
			interval_ms = int(interval_var.get())
			heatmap_data["animation_timer"] = dashboard_win.after(interval_ms, animate_step)
		
		# Load preview when tab is shown
		def on_tab_change():
			if current_step == 1:
				load_csv_preview()
			elif current_step == 2:
				# Load heatmap when switching to heatmap tab
				if video_files and loaded_csv_path["value"]:
					update_heatmap_display()
		# Override show_current_tab to trigger preview load
		original_show = show_current_tab
		def show_current_tab():
			original_show()
			on_tab_change()
		
		# Initialize
		update_nav_state()
		show_current_tab()
	
	dashboard_button = make_button_frame(
		buttons_container,
		"VISUALIZACIÓN\nDE DATOS",
		dashboard_logo_path,
		(open_controller_window if resolved_role == "controller" else open_dashboard)
	)
	dashboard_button.grid(row=0, column=3, padx=10, pady=(10,100), sticky="nsew")

	# ========== MODE SLIDER (top-left corner) ==========
	ARNESIS_MODES = [
		{
			"name": "Performance",
			"bg": "#02234e",
			"btn_bg": "#021e44",
			"btn_hover": "#043c86",
			"btn_active": "#065ed4",
			"pill": "#015bcb",
		},
		{
			"name": "Discipline",
			"bg": "#0a2414",
			"btn_bg": "#77bd86",
			"btn_hover": "#9acba8",
			"btn_active": "#5aa070",
			"pill": "#77bd86",
		},
		{
			"name": "Quality",
			"bg": "#2d0a0a",
			"btn_bg": "#ff6961",
			"btn_hover": "#ff8880",
			"btn_active": "#d94840",
			"pill": "#ff6961",
		},
	]

	mode_state = {"index": 0, "animating": False}

	# Slider geometry
	_PILL_X0, _PILL_Y0, _PILL_X1, _PILL_Y1 = 10, 8, 160, 34
	_PILL_R = 13
	_CIRCLE_POSITIONS = [23, 85, 147]   # cx for each of the 3 modes
	_CIRCLE_R = 11

	def _hex_to_rgb(h):
		h = h.lstrip("#")
		return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

	def _lerp_color(c1, c2, t):
		r1, g1, b1 = _hex_to_rgb(c1)
		r2, g2, b2 = _hex_to_rgb(c2)
		return "#{:02x}{:02x}{:02x}".format(
			int(r1 + (r2 - r1) * t),
			int(g1 + (g2 - g1) * t),
			int(b1 + (b2 - b1) * t),
		)

	slider_canvas = tk.Canvas(
		container, width=170, height=55,
		bg=BG_COLOR, highlightthickness=0, cursor="hand2"
	)
	slider_canvas.place(x=15, y=12)

	def _draw_slider(circle_x, pill_color, label_text):
		slider_canvas.delete("all")
		x0, y0, x1, y1 = _PILL_X0, _PILL_Y0, _PILL_X1, _PILL_Y1
		r = _PILL_R
		# Pill shape
		slider_canvas.create_arc(x0, y0, x0+2*r, y0+2*r, start=90, extent=90, fill=pill_color, outline=pill_color)
		slider_canvas.create_arc(x1-2*r, y0, x1, y0+2*r, start=0, extent=90, fill=pill_color, outline=pill_color)
		slider_canvas.create_arc(x0, y1-2*r, x0+2*r, y1, start=180, extent=90, fill=pill_color, outline=pill_color)
		slider_canvas.create_arc(x1-2*r, y1-2*r, x1, y1, start=270, extent=90, fill=pill_color, outline=pill_color)
		slider_canvas.create_rectangle(x0+r, y0, x1-r, y1, fill=pill_color, outline=pill_color)
		slider_canvas.create_rectangle(x0, y0+r, x1, y1-r, fill=pill_color, outline=pill_color)
		# Position dots
		cy = (_PILL_Y0 + _PILL_Y1) // 2
		dot_color = _lerp_color(pill_color, "#ffffff", 0.45)
		for px in _CIRCLE_POSITIONS:
			slider_canvas.create_oval(px-4, cy-4, px+4, cy+4, fill=dot_color, outline=dot_color)
		# Moving circle (knob)
		slider_canvas.create_oval(
			circle_x - _CIRCLE_R, cy - _CIRCLE_R,
			circle_x + _CIRCLE_R, cy + _CIRCLE_R,
			fill="white", outline=""
		)
		# Mode label below pill
		slider_canvas.create_text(85, 48, text=label_text, font=("Arial", 8, "bold"), fill="#cccccc")

	# Draw initial state
	_draw_slider(_CIRCLE_POSITIONS[0], ARNESIS_MODES[0]["pill"], ARNESIS_MODES[0]["name"])

	def _walk_recolor(widget, old_bg, new_bg, old_btn, new_btn):
		"""Recursively walk widget tree, remapping background colors."""
		try:
			bg = widget.cget("background").lower()
			if bg == old_bg:
				widget.config(background=new_bg)
			elif bg == old_btn:
				widget.config(background=new_btn)
		except Exception:
			pass
		# Update canvas polygon fill items tagged "button_bg"
		try:
			for item in widget.find_withtag("button_bg"):
				fill = widget.itemcget(item, "fill").lower()
				if fill == old_btn:
					widget.itemconfig(item, fill=new_btn)
				elif fill == old_bg:
					widget.itemconfig(item, fill=new_bg)
		except Exception:
			pass
		for child in widget.winfo_children():
			_walk_recolor(child, old_bg, new_bg, old_btn, new_btn)

	def _do_mode_animation(old_idx, new_idx, step, total_steps,
	                        cur_bg, cur_btn, cur_pill, cur_cx):
		t = (step + 1) / total_steps
		t = min(1.0, t)
		# Smooth ease in-out
		t_ease = t * t * (3.0 - 2.0 * t)

		old_m = ARNESIS_MODES[old_idx]
		new_m = ARNESIS_MODES[new_idx]

		next_bg   = _lerp_color(old_m["bg"],     new_m["bg"],     t)
		next_btn  = _lerp_color(old_m["btn_bg"],  new_m["btn_bg"],  t)
		next_pill = _lerp_color(old_m["pill"],    new_m["pill"],   t)

		old_cx = _CIRCLE_POSITIONS[old_idx]
		new_cx = _CIRCLE_POSITIONS[new_idx]
		next_cx = int(old_cx + (new_cx - old_cx) * t_ease)

		# Recolor widget tree
		_walk_recolor(root, cur_bg.lower(), next_bg, cur_btn.lower(), next_btn)
		slider_canvas.config(bg=next_bg)

		# Slider label flips at midpoint
		label = new_m["name"] if t >= 0.5 else old_m["name"]
		_draw_slider(next_cx, next_pill, label)

		if step + 1 < total_steps:
			root.after(16, lambda: _do_mode_animation(
				old_idx, new_idx, step + 1, total_steps,
				next_bg, next_btn, next_pill, next_cx
			))
		else:
			mode_state["animating"] = False
			global BG_COLOR, BUTTON_BG, BUTTON_HOVER, BUTTON_ACTIVE
			BG_COLOR    = new_m["bg"]
			BUTTON_BG   = new_m["btn_bg"]
			BUTTON_HOVER = new_m["btn_hover"]
			BUTTON_ACTIVE = new_m["btn_active"]
			_draw_slider(_CIRCLE_POSITIONS[new_idx], new_m["pill"], new_m["name"])

	def _advance_mode(event=None):
		if mode_state["animating"]:
			return
		old_idx = mode_state["index"]
		new_idx = (old_idx + 1) % len(ARNESIS_MODES)
		mode_state["index"] = new_idx
		mode_state["animating"] = True
		old_m = ARNESIS_MODES[old_idx]
		_do_mode_animation(
			old_idx, new_idx,
			step=0, total_steps=18,
			cur_bg=old_m["bg"],
			cur_btn=old_m["btn_bg"],
			cur_pill=old_m["pill"],
			cur_cx=_CIRCLE_POSITIONS[old_idx],
		)

	slider_canvas.bind("<Button-1>", _advance_mode)
	# Hover effect: slight brightness on pill
	def _slider_hover(e):
		if not mode_state["animating"]:
			m = ARNESIS_MODES[mode_state["index"]]
			lighter = _lerp_color(m["pill"], "#ffffff", 0.15)
			_draw_slider(_CIRCLE_POSITIONS[mode_state["index"]], lighter, m["name"])

	def _slider_leave(e):
		if not mode_state["animating"]:
			m = ARNESIS_MODES[mode_state["index"]]
			_draw_slider(_CIRCLE_POSITIONS[mode_state["index"]], m["pill"], m["name"])

	slider_canvas.bind("<Enter>", _slider_hover)
	slider_canvas.bind("<Leave>", _slider_leave)

	# In controller mode, open central dashboard automatically on startup.
	if resolved_role == "controller":
		root.after(200, open_controller_window)

	root.mainloop()


# ============================================================================
# VideoProcessor Class - Consolidated from Proccess_Classifier_GUI.py
# ============================================================================
class VideoProcessor:
	"""
	Unified video processing pipeline for detection + classification + ROI tracking.
	Encapsulates all logic from Proccess_Classifier_GUI.py as a reusable class.
	"""
	
	def __init__(self, config: dict):
		"""Initialize processor with configuration from GUI."""
		# Configuration from GUI
		self.classify_weights = config.get("classify_weights")
		self.person_det_engine = config.get("person_det_engine", "yolo11x.pt")
		self.input_videos = config.get("input_videos", [])
		self.csv_dir = config.get("csv_dir", "output_semanas_procesadas")
		self.processed_dir = config.get("processed_dir")
		self.output_name = config.get("output_name", "Arnesis")
		self.rois_by_video = config.get("rois_by_video")
		self.device = config.get("device", "cuda")
		self.generate_video = config.get("generate_output_video", True)
		
		# Processing parameters
		self.conf_person = config.get("conf_person", 0.60)
		self.conf_state = config.get("conf_state", 0.50)
		self.sample_rate = config.get("sample_rate", 1)
		self.frameskip_percentage = config.get("frameskip_percentage", 0)  # Percentage of frames to skip (0-90)
		self.aggregation_frames = config.get("aggregation_frames", 125)
		self.frame_rate_real = config.get("frame_rate_real", 25.0)
		self.pre_size = tuple(config.get("pre_size", [1280, 720]))
		self.output_width = config.get("output_width", 1280)
		self.output_height = config.get("output_height", 720)
		self.output_fps = config.get("output_fps", 25)
		self.use_half = config.get("use_half", False)
		
		# Face blur and ergonomics settings
		self.difuminar_caras = config.get("difuminar_caras", False)
		self.seguridad = config.get("seguridad", False)
		self.face_model_name = config.get("face_model_name", "yolov8l-face.pt")
		self.ergo_complexity = config.get("ergo_complexity", 2)
		
		# ROI settings
		self.use_rois = config.get("use_rois", False)
		self.use_predetermined_rois = config.get("use_predetermined_rois", True)
		self.use_perma_rois = config.get("use_perma_rois", False)
		self.perma_rois_path = config.get("perma_rois_path", "perma_rois.json")
		
		# Predetermined ROIs
		self.pred_rois = config.get("pred_rois", [
			{"coords": [642,718,1277,717,1279,548,819,100,625,284], "name": "CONV 1"},
			{"coords": [819,100,625,284,619,150,714,35], "name": "CONV 2"},
			{"coords": [619,150,714,35,663,1,611,59], "name": "CONV 3"},
			{"coords": [611,59,619,150,495,86,541,21], "name": "AISLE 3"},
			{"coords": [619,150,625,284,436,209,495,86], "name": "AISLE 2"},
			{"coords": [625,284,642,718,280,717,436,209], "name": "AISLE 1"}
		])
		
		# Watermark settings
		self.stamp_path = config.get("stamp_path", "assets/confidential.png")
		self.stamp_opacity = config.get("stamp_opacity", 0.35)
		
		# Manual datetimes for videos without datetime in filename
		self.manual_datetimes = config.get("manual_datetimes", {})
		
		# Models (loaded on demand)
		self.person_model = None
		self.cls_model = None
		self.face_model = None
		self.pose_instances = None  # For MediaPipe pose tracking
		self.stamp_rgb = None
		self.alpha_mask = None
		self.colors = {"Neutral": (192, 192, 192)}
		
		# Progress tracking
		self.progress_callback = None
		self.log_callback = None
		self.total_to_process = 0
		self.processed_count = 0
		
		# Base root resolution
		self.root = self._base_root()
		self.classify_model_name = None
		
		# Real seconds per block calculation
		self.real_seconds_per_block = self.aggregation_frames * self.sample_rate / self.frame_rate_real
	
	def _base_root(self) -> Path:
		"""Determine base root considering PyInstaller frozen mode."""
		if getattr(sys, 'frozen', False):
			# When frozen, use the directory containing the .exe
			return Path(os.path.dirname(sys.executable))
		return Path(__file__).resolve().parent.parent
	
	def _resolve_path(self, p: str) -> str:
		"""Resolve relative paths against ROOT."""
		if not p:
			return p
		return p if os.path.isabs(p) else str(self.root / p)
	
	def _resolve_yolo_model(self, model_name: str) -> str:
		"""Resolve YOLO model path, allowing auto-download if not found.
		
		When running as .exe (frozen), looks for models in the .exe directory.
		If model doesn't exist, YOLO will auto-download it to that location.
		
		Args:
			model_name: Model filename (e.g., 'yolo11x.pt') or absolute path
			
		Returns:
			Absolute path to model file (may not exist yet if needs download)
		"""
		# If absolute path provided, use as-is
		if os.path.isabs(model_name):
			return model_name
		
		# Check if it's a standard YOLO model name (e.g., yolo11x.pt, yolov8n.pt)
		# These are models that YOLO can auto-download
		yolo_pretrained_patterns = [
			'yolo11', 'yolov8', 'yolov5', 'yolov9', 'yolov10',
			'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
			'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
		]
		
		is_pretrained = any(pattern in model_name.lower() for pattern in yolo_pretrained_patterns)
		
		if is_pretrained:
			# For pretrained models, look in .exe directory when frozen
			if getattr(sys, 'frozen', False):
				# In frozen mode, use .exe directory
				exe_dir = Path(os.path.dirname(sys.executable))
				model_path = exe_dir / model_name
				self._log(f"[INFO] Buscando modelo en directorio .exe: {model_path}")
				if not model_path.exists():
					self._log(f"[INFO] Modelo no encontrado, YOLO lo descargará automáticamente")
				return str(model_path)
			else:
				# In development mode, use project root
				model_path = self.root / model_name
				if model_path.exists():
					return str(model_path)
				# If not in root, let YOLO handle it (will download to cache)
				return model_name
		else:
			# For custom models (like bifurcacionV4_best.pt), use standard resolution
			return self._resolve_path(model_name)
	
	def _select_device(self, preferred: str) -> str:
		"""Select computation device (CUDA or CPU)."""
		try:
			if preferred and preferred.lower().startswith("cuda"):
				if TORCH_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available():
					return "cuda"
		except Exception:
			pass
		return "cpu"
	
	def _load_watermark(self):
		"""Load and prepare watermark overlay."""
		stamp_path_resolved = self._resolve_path(self.stamp_path)
		stamp = cv2.imread(stamp_path_resolved, cv2.IMREAD_UNCHANGED)
		if stamp is None:
			self._log(f"[WARN] Marca de agua no encontrada en: {stamp_path_resolved}. Continuando sin overlay.")
			self.stamp_rgb = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
			self.alpha_mask = np.zeros((self.output_height, self.output_width), dtype=float)
		else:
			if stamp.shape[2] == 4:
				b, g, r, a = cv2.split(stamp)
				stamp_rgb = cv2.merge((b, g, r))
				alpha_mask = (a.astype(float) / 255.0) * self.stamp_opacity
			else:
				stamp_rgb = stamp
				alpha_mask = np.full(stamp.shape[:2], self.stamp_opacity, dtype=float)
			self.stamp_rgb = cv2.resize(stamp_rgb, (self.output_width, self.output_height))
			self.alpha_mask = cv2.resize(alpha_mask, (self.output_width, self.output_height))
	
	def _log(self, message):
		"""Log message to callback or stdout."""
		if self.log_callback:
			self.log_callback(message)
		else:
			print(message, flush=True)
	
	def _load_models(self):
		"""Load YOLO models for person detection and classification."""
		if not YOLO_AVAILABLE:
			raise RuntimeError("YOLO no está disponible. Instala ultralytics.")
		
		# Resolve paths - use special resolver for YOLO models
		person_engine = self._resolve_yolo_model(self.person_det_engine)
		# Load models - YOLO will auto-download person_engine if needed
		self._log(f"[INFO] Cargando modelo de detección: {os.path.basename(person_engine)}")
		self.person_model = YOLO(person_engine)  # Will download if file doesn't exist
		if self.classify_weights:
			classify_weights = self._resolve_path(self.classify_weights)  # Custom models use standard path
			if not os.path.isfile(classify_weights):
				raise FileNotFoundError(f"Pesos de clasificación no encontrados: {classify_weights}")
			self._log(f"[INFO] Cargando modelo de clasificación: {os.path.basename(classify_weights)}")
			self.cls_model = YOLO(classify_weights)
		else:
			self._log("[INFO] Clasificación deshabilitada (sin modelo de clasificación)")
			self.cls_model = None
		
		# Move to device
		self.device = self._select_device(self.device)
		
		try:
			self.person_model = self.person_model.to(self.device)
			if self.cls_model is not None:
				self.cls_model = self.cls_model.to(self.device)
			self._log(f"[INFO] Usando dispositivo: {self.device}")
		except AssertionError:
			self._log("[WARN] CUDA no disponible; usando CPU")
			self.device = "cpu"
			self.person_model = self.person_model.to("cpu")
			if self.cls_model is not None:
				self.cls_model = self.cls_model.to("cpu")
		except Exception as e:
			self._log(f"[WARN] Falla al mover a {self.device}: {e}. Usando CPU")
			self.device = "cpu"
			self.person_model = self.person_model.to("cpu")
			if self.cls_model is not None:
				self.cls_model = self.cls_model.to("cpu")
		
		# Apply half precision if requested and on CUDA
		if self.device.startswith("cuda") and self.use_half:
			try:
				self.person_model.half()
				if self.cls_model is not None:
					self.cls_model.half()
			except Exception:
				pass
		
		# Generate color mapping for classes
		if self.cls_model is not None:
			names = self.cls_model.names
			random.seed(0)
			for class_name in names.values():
				if class_name not in self.colors:
					self.colors[class_name] = tuple(random.randint(0, 255) for _ in range(3))
		
		# Store model name
		self.classify_model_name = os.path.basename(classify_weights).replace('.pt', '') if self.classify_weights else None
		
		# Load face detection model if face blur is enabled
		if self.difuminar_caras:
			self._log("[INFO] Face blur habilitado. Cargando modelo de detección de caras...")
			try:
				face_weights = self._resolve_yolo_model(self.face_model_name)
				if not os.path.isfile(face_weights):
					# Try alternative face model
					face_weights = self._resolve_yolo_model("yolov8m-face.pt")
				
				if os.path.isfile(face_weights):
					self.face_model = YOLO(face_weights)
					self.face_model = self.face_model.to(self.device)
					self._log(f"[INFO] Modelo de caras cargado: {os.path.basename(face_weights)}")
				else:
					self._log("[WARN] No se encontró modelo de detección de caras. Face blur deshabilitado.")
					self.difuminar_caras = False
			except Exception as e:
				self._log(f"[WARN] Error al cargar modelo de caras: {e}. Face blur deshabilitado.")
				self.difuminar_caras = False
		
		# Initialize MediaPipe Pose for ergonomics if enabled
		if self.seguridad:
			self._log("[INFO] Ergonomía habilitada. Inicializando MediaPipe Pose...")
			try:
				if mp_pose is not None:
					from collections import OrderedDict
					self.pose_instances = OrderedDict()
					self._log(f"[INFO] MediaPipe Pose inicializado con complejidad {self.ergo_complexity}")
				else:
					self._log("[WARN] MediaPipe no disponible. Ergonomía deshabilitada.")
					self.seguridad = False
			except Exception as e:
				self._log(f"[WARN] Error al inicializar MediaPipe: {e}. Ergonomía deshabilitada.")
				self.seguridad = False
		
		# Emit early progress
		if self.progress_callback:
			self.progress_callback(1.0)
		else:
			print("GUI_PROGRESS 1.00", flush=True)
	
	@staticmethod
	def get_new_class(old_class):
		"""Map original class to VA/NVA/Neutral."""
		if old_class in ["Ruteo", "Enteipado", "working", "Convolute", "Insercion", "Tomar Material"]:
			return "VA", (0, 255, 0)
		elif old_class in ["NVA", "idle", "Celular"]:
			return "NVA", (0, 0, 255)
		else:
			return old_class, (192, 192, 192)
	
	def _get_all_videos(self, paths):
		"""Get all video files from paths (files or directories)."""
		video_exts = ('.mp4', '.avi', '.mov', '.mkv')
		all_videos = []
		for path in paths:
			resolved = self._resolve_path(path)
			if os.path.isfile(resolved) and resolved.lower().endswith(video_exts):
				all_videos.append(resolved)
			elif os.path.isdir(resolved):
				for root, _, files in os.walk(resolved):
					for f in files:
						if f.lower().endswith(video_exts):
							all_videos.append(os.path.join(root, f))
		return all_videos
	
	def _load_rois_for_video(self, basename, first_frame):
		"""Load ROIs for a specific video."""
		# Priority: ROIs from GUI by video
		if self.rois_by_video and basename in self.rois_by_video:
			rois_cfg = self.rois_by_video.get(basename, [])
			rois = []
			for item in rois_cfg:
				coords = np.array(item.get('coords', []), np.int32).reshape(-1, 2)
				rois.append({'coords': coords, 'name': item.get('name', 'ROI')})
			return rois
		elif self.use_rois and self.use_perma_rois:
			perma_path = self._resolve_path(self.perma_rois_path)
			if os.path.exists(perma_path):
				return self._load_perma_rois(perma_path)
			else:
				rois = self._select_rois(first_frame)
				self._save_perma_rois(rois, perma_path)
				return rois
		elif self.use_rois and self.use_predetermined_rois:
			return [
				{'coords': np.array(pr['coords'], np.int32).reshape(-1, 2), 'name': pr['name']}
				for pr in self.pred_rois
			]
		elif self.use_rois:
			return self._select_rois(first_frame)
		else:
			return []
	
	def _save_perma_rois(self, rois, path):
		"""Save ROIs to JSON file."""
		data = [{'coords': roi['coords'].reshape(-1).tolist(), 'name': roi['name']} for roi in rois]
		with open(path, 'w') as f:
			json.dump(data, f, indent=2)
		self._log(f"ROIs guardadas en: {path}")
	
	def _load_perma_rois(self, path):
		"""Load ROIs from JSON file."""
		with open(path, 'r') as f:
			data = json.load(f)
		rois = []
		for item in data:
			coords = np.array(item['coords'], dtype=np.int32).reshape(-1, 2)
			rois.append({'coords': coords, 'name': item['name']})
		self._log(f"Cargadas {len(rois)} ROIs de: {path}")
		return rois
	
	def _select_rois(self, frame):
		"""Manual ROI selection with mouse."""
		rois, current = [], []
		display = frame.copy()
		
		def mouse_cb(ev, x, y, flags, param):
			nonlocal current, display
			if ev == cv2.EVENT_LBUTTONDOWN:
				current.append((x, y))
				display = frame.copy()
				for roi in rois:
					cv2.polylines(display, [roi['coords'].reshape(-1, 1, 2)], True, (255, 255, 0), 2)
				for pt in current:
					cv2.circle(display, pt, 4, (0, 255, 255), -1)
				if len(current) > 1:
					pts = np.array(current, np.int32).reshape(-1, 1, 2)
					cv2.polylines(display, [pts], False, (0, 255, 255), 1)
		
		cv2.namedWindow("Define ROIs")
		cv2.setMouseCallback("Define ROIs", mouse_cb)
		self._log("Define tus ROIs (a=cerrar, q=guardar y salir)")
		while True:
			cv2.imshow("Define ROIs", display)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('a') and len(current) >= 3:
				name = input(f"Nombre ROI#{len(rois)+1}: ") or f"ROI{len(rois)+1}"
				rois.append({'coords': np.array(current, np.int32), 'name': name})
				self._log(f"ROI '{name}' guardado")
				current, display = [], frame.copy()
			elif k == ord('q'):
				if len(current) >= 3:
					name = input(f"Nombre ROI#{len(rois)+1}: ") or f"ROI{len(rois)+1}"
					rois.append({'coords': np.array(current, np.int32), 'name': name})
					self._log(f"ROI '{name}' guardado")
				break
		cv2.destroyWindow("Define ROIs")
		return rois
	
	def _apply_face_blur(self, frame):
		"""Apply face blur to detected faces in frame."""
		if not self.difuminar_caras or self.face_model is None:
			return
		
		try:
			h_frame, w_frame = frame.shape[:2]
			results_face = self.face_model(frame, verbose=False)
			for r in results_face:
				if getattr(r, 'boxes', None) is None:
					continue
				boxes = r.boxes.xyxy
				if boxes is None:
					continue
				for (x1_f, y1_f, x2_f, y2_f) in boxes.cpu().numpy().astype(int):
					x1_f = max(0, min(x1_f, w_frame-1))
					x2_f = max(0, min(x2_f, w_frame-1))
					y1_f = max(0, min(y1_f, h_frame-1))
					y2_f = max(0, min(y2_f, h_frame-1))
					if x2_f <= x1_f or y2_f <= y1_f:
						continue
					face_roi = frame[y1_f:y2_f, x1_f:x2_f]
					if face_roi.size == 0:
						continue
					blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
					frame[y1_f:y2_f, x1_f:x2_f] = blurred
		except Exception as e:
			self._log(f"[WARN] Error aplicando face blur: {e}")
	
	def _process_ergonomics(self, frame, person_bbox, track_id):
		"""Process ergonomics for a person using MediaPipe Pose.
		
		Returns:
			tuple: (pose_status, color) where pose_status is "OK" or "NG"
		"""
		if not self.seguridad or mp_pose is None:
			return None, None
		
		try:
			x1, y1, x2, y2 = person_bbox
			
			# Get or create pose instance for this track
			if track_id not in self.pose_instances:
				if len(self.pose_instances) >= 10:
					# Remove oldest instance (FIFO)
					_, old_pose = self.pose_instances.popitem(last=False)
					if hasattr(old_pose, 'close'):
						old_pose.close()
					del old_pose
				self.pose_instances[track_id] = mp_pose.Pose(
					static_image_mode=False,
					model_complexity=self.ergo_complexity,
					smooth_landmarks=True,
					min_detection_confidence=0.5,
					min_tracking_confidence=0.9
				)
			
			pose_instance = self.pose_instances[track_id]
			person_img = frame[y1:y2, x1:x2]
			rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
			result_pose = pose_instance.process(rgb_person)
			
			if result_pose.pose_world_landmarks:
				landmarks = result_pose.pose_world_landmarks.landmark
				pose_status = self._classify_ergonomics(landmarks)
				color = (0, 255, 0) if pose_status == "OK" else (0, 0, 255)
				return pose_status, color
			
		except Exception as e:
			self._log(f"[WARN] Error procesando ergonomía para track {track_id}: {e}")
		
		return None, None
	
	def _classify_ergonomics(self, landmarks):
		"""Classify ergonomics posture based on MediaPipe landmarks.
		
		Args:
			landmarks: MediaPipe pose_world_landmarks.landmark list
			
		Returns:
			str: "OK" or "NG"
		"""
		def get_angle(a, b, c):
			a = np.array(a)
			b = np.array(b)
			c = np.array(c)
			ba = a - b
			bc = c - b
			cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
			angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
			return np.degrees(angle)
		
		# Back angle: angle between hip, shoulder, and nose
		left_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
		left_hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
		nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
		back_angle = get_angle(left_hip, left_shoulder, nose)
		
		# Arms: wrist height relative to shoulders (Y-axis 3D)
		left_wrist_y = landmarks[15].y
		right_wrist_y = landmarks[16].y
		left_shoulder_y = landmarks[11].y
		right_shoulder_y = landmarks[12].y
		
		# Criteria
		if back_angle < 40:
			return "NG"
		if left_wrist_y < left_shoulder_y - 0.05 or right_wrist_y < right_shoulder_y - 0.05:
			return "NG"
		return "OK"
	
	def _writer_thread_func(self, q, temp_path, output_fps, size):
		"""Video writer thread."""
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		out = cv2.VideoWriter(temp_path, fourcc, output_fps, size)
		while True:
			frame = q.get()
			if frame is None:
				break
			out.write(frame)
		out.release()
	
	def _flush_segment(self, writer_q, buffer, counts, rois, start_time, segment_idx, 
	                    video_name, csv_rows, video_date, video_weekday, time_str, ergonomics_results=None):
		"""Flush segment and generate CSV rows.
		
		Args:
			ergonomics_results: dict mapping track_id to (pose_status, pose_color) tuples
		"""
		if ergonomics_results is None:
			ergonomics_results = {}
		
		# Determine majority class per person
		final_state = {}
		for tid, cnts in counts.items():
			if not cnts:
				final_state[tid] = 'Neutral'
			else:
				final_state[tid] = max(cnts, key=cnts.get)
		
		added_before = len(csv_rows)
		
		# Determine which tracks entered each ROI
		roi_members = {r['name']: set() for r in rois}
		tid_to_roi = {}
		if rois:
			for _, dets in buffer:
				for d in dets:
					tid = d['track_id']
					x1, y1, x2, y2 = d['coords']
					cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
					for roi in rois:
						if cv2.pointPolygonTest(roi['coords'], (cx, cy), False) >= 0:
							roi_members[roi['name']].add(tid)
							tid_to_roi[tid] = roi['name']
			
			# Calculate person count per station in the LAST frame of buffer
			station_person_count = {r['name']: 0 for r in rois}
			if buffer:  # Ensure buffer is not empty
				_, last_frame_dets = buffer[-1]  # Get last frame detections
				for d in last_frame_dets:
					tid = d['track_id']
					x1, y1, x2, y2 = d['coords']
					cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
					for roi in rois:
						if cv2.pointPolygonTest(roi['coords'], (cx, cy), False) >= 0:
							station_person_count[roi['name']] += 1
							break  # Count each person only once per station
			
			# Generate CSV rows per ROI
			for station, tids in roi_members.items():
				for tid in tids:
					cls, color = self.get_new_class(final_state.get(tid, 'Neutral'))
					csv_cls = "" if self.cls_model is None else cls
					# Get last known position of this track for coordinates
					coord_x, coord_y = 0, 0
					for _, dets in buffer:
						for d in dets:
							if d['track_id'] == tid:
								x1, y1, x2, y2 = d['coords']
								coord_x = int((x1 + x2) / 2)
								coord_y = int((y1 + y2) / 2)
					# Get person count for this station
					person_count = station_person_count.get(station, 0)
					csv_rows.append([video_name, video_date, video_weekday, station, tid, time_str, csv_cls, coord_x, coord_y, person_count])
		
			# Add "sin_registro" rows for ROIs without any persons
			for station in station_person_count:
				if station_person_count[station] == 0:
					# No persons in this station, add a "sin_registro" row
					csv_rows.append([video_name, video_date, video_weekday, station, "sin_registro", time_str, "sin_registro", "sin_registro", "sin_registro", 0])
		else:
			# No ROIs: use GLOBAL station
		# Calculate person count in GLOBAL station (last frame of buffer)
			global_person_count = 0
			if buffer:  # Ensure buffer is not empty
				_, last_frame_dets = buffer[-1]  # Get last frame detections
				global_person_count = len(last_frame_dets)
		
			for tid in final_state.keys():
				cls, color = self.get_new_class(final_state.get(tid, 'Neutral'))
				csv_cls = "" if self.cls_model is None else cls
				# Get last known position of this track for coordinates
				coord_x, coord_y = 0, 0
				for _, dets in buffer:
					for d in dets:
						if d['track_id'] == tid:
							x1, y1, x2, y2 = d['coords']
							coord_x = int((x1 + x2) / 2)
							coord_y = int((y1 + y2) / 2)
				csv_rows.append([video_name, video_date, video_weekday, "GLOBAL", tid, time_str, csv_cls, coord_x, coord_y, global_person_count])
		for idx, (frame, dets) in enumerate(buffer):
			# Draw ROIs
			for roi in rois:
				cv2.polylines(frame, [roi['coords'].reshape(-1, 1, 2)], True, (255, 255, 0), 2)
			
			# Draw timestamp on LAST frame (for debugging)
			if idx == len(buffer) - 1:
				timestamp_text = f"Timestamp: {time_str}"
				cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
			
			# Draw detections
			for d in dets:
				tid = d['track_id']
				x1, y1, x2, y2 = d['coords']
				cls, color = self.get_new_class(final_state.get(tid, 'Neutral'))
				roi_name = tid_to_roi.get(tid, "")
				
				label = f"ID:{tid} {cls}"
				if roi_name:
					label += f" [{roi_name}]"
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
				cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				
				# Draw ergonomics label if enabled and available
				if self.seguridad and tid in ergonomics_results:
					ergo_status, ergo_color = ergonomics_results[tid]
					ergo_text = f"Postura: {ergo_status}"
					cv2.putText(frame, ergo_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ergo_color, 2)
			
			# Apply watermark
			out = cv2.resize(frame, (self.output_width, self.output_height))
			for ch in range(3):
				out[:, :, ch] = (
					self.alpha_mask * self.stamp_rgb[:, :, ch]
					+ (1 - self.alpha_mask) * out[:, :, ch]
				).astype(np.uint8)
			
			if writer_q is not None:
				writer_q.put(out)
		
		# Debug info
		added_after = len(csv_rows)
		added_count = added_after - added_before
		if added_count > 0:
			self._log(f"[CSV] +{added_count} filas agregadas en seg {segment_idx} ({video_name})")
		else:
			self._log(f"[CSV] 0 filas agregadas en seg {segment_idx} ({video_name})")
		
		buffer.clear()
		counts.clear()
	
	def process_videos(self, progress_callback=None, log_callback=None):
		"""Main processing loop for all videos."""
		self.progress_callback = progress_callback
		self.log_callback = log_callback
		
		# Print banner
		self._log(f"\n{'='*60}")
		self._log(f"🚀 INICIANDO PROCESAMIENTO DE VIDEOS")
		self._log(f"{'='*60}")
		
		# Load models and watermark
		self._load_models()
		self._load_watermark()
		
		# Get all videos to process
		all_videos = self._get_all_videos(self.input_videos)
		if not all_videos:
			raise RuntimeError("No se encontraron videos para procesar en las rutas indicadas.")
		
		self.total_to_process = len(all_videos)
		self._log(f"[INFO] Videos encontrados: {self.total_to_process}")
		
		if self.total_to_process == 0:
			if progress_callback:
				progress_callback(100.0)
			else:
				self._log("GUI_PROGRESS 100.00")
			return
		
		# Setup CSV
		os.makedirs(self.csv_dir, exist_ok=True)
		csv_path = os.path.join(self.csv_dir, f"{self.output_name}.csv")
		csv_header = ["video_name", "date", "weekday", "station", "person_id", "HH:MM:SS", "class", "coord_x", "coord_y", "person_count"]
		
		if not os.path.exists(csv_path):
			with open(csv_path, 'w', newline='') as f:
				w = csv.writer(f)
				w.writerow(csv_header)
		
		# Process each video
		self.processed_count = 0
		for input_video in all_videos:
			csv_rows = []
			basename = os.path.basename(input_video)
			
			# Extract date from filename or use manual datetime
			if input_video in self.manual_datetimes:
				# Use manually provided datetime
				start_dt = self.manual_datetimes[input_video]
				video_name = os.path.splitext(basename)[0]
				video_date = start_dt.strftime("%Y-%m-%d")
				video_weekday = start_dt.strftime("%A")
				initial_time = start_dt
				use_video_time = True
				self._log(f"[INFO] Using manual datetime for video: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
			else:
				# Try to extract from filename
				parts = basename.split('_')
				try:
					try:
						start_dt = datetime.strptime(parts[-4], "%Y%m%d%H%M%S")
					except Exception:
						try:
							start_dt = datetime.strptime(basename, "%Y%m%d%H%M%S")
						except Exception:
							start_dt = datetime.strptime(parts[-3], "%Y%m%d%H%M%S")
					video_name = os.path.splitext(basename)[0]
					video_date = start_dt.strftime("%Y-%m-%d")
					video_weekday = start_dt.strftime("%A")
					initial_time = start_dt
					use_video_time = True
				except Exception:
					now = datetime.now()
					video_name = os.path.splitext(basename)[0]
					video_date = now.strftime("%Y-%m-%d")
					video_weekday = now.strftime("%A")
					initial_time = datetime.strptime("00:00:00", "%H:%M:%S")
					use_video_time = False
			
			# Setup output paths
			ts = datetime.now().strftime("%Y%m%d_%H%M%S")
			out_dir = self.processed_dir if self.processed_dir else f"videos_procesados/{self.output_name}"
			os.makedirs(out_dir, exist_ok=True)
			temp_path = os.path.join(out_dir, f"{video_name}_{ts}_proc.mp4")
			final_path = os.path.join(out_dir, f"{video_name}_T_{ts}_M_{self.classify_model_name}.mp4")
			
			# Load first frame and ROIs - try multiple positions if first frame is corrupted
			cap0 = cv2.VideoCapture(input_video)
			ret, first = False, None
			frame_attempt = 0
			max_attempts = 10  # Try up to 10 positions (0, 25, 50, ..., 225)
			
			while not ret or first is None:
				if frame_attempt >= max_attempts:
					break
				
				# Set frame position
				if frame_attempt > 0:
					cap0.set(cv2.CAP_PROP_POS_FRAMES, frame_attempt * 25)
				
				ret, first = cap0.read()
				frame_attempt += 1
			
			cap0.release()
			
			if not ret or first is None:
				self._log(f"[ERROR] No se pudo leer ningún frame válido del video (intentos: {frame_attempt}): {input_video}")
				continue
			
			# Track how many frames to skip at the beginning if corrupted frames detected
			skip_initial_frames = 0
			if frame_attempt > 1:
				skip_initial_frames = (frame_attempt - 1) * 25
				self._log(f"[INFO] Frame inicial corrupto. Frame válido encontrado en posición {skip_initial_frames}. Se saltarán los primeros {skip_initial_frames} frames en el procesamiento.")
			
			first = cv2.resize(first, self.pre_size)
			
			# Get total frames for progress
			cap_info = cv2.VideoCapture(input_video)
			total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
			cap_info.release()
			
			# Load ROIs
			rois = self._load_rois_for_video(basename, first)
			
			self._log(f"[INFO] ▶ Procesando video: {basename} | ROIs: {len(rois)} | Frames totales: {total_frames}")
			
			# Start video writer if needed
			if self.generate_video:
				write_q = Queue(maxsize=50)
				thr = threading.Thread(
					target=self._writer_thread_func,
					args=(write_q, temp_path, self.output_fps, (self.output_width, self.output_height)),
					daemon=True
				)
				thr.start()
			else:
				write_q = None
				thr = None
			
			# Process video
			frame_buffer, track_counts = [], {}
			ergonomics_results = {}  # Store ergonomics per track: {tid: (status, color)}
			frame_counter, segment_idx = 0, 0
			
			# Open video capture and skip to valid frame if needed
			cap_process = cv2.VideoCapture(input_video)
			if skip_initial_frames > 0:
				cap_process.set(cv2.CAP_PROP_POS_FRAMES, skip_initial_frames)
				self._log(f"[INFO] Iniciando procesamiento desde frame {skip_initial_frames}")
			
			last_progress_print = -1
			processed_frame_counter = 0  # Counter for frames actually processed (after skip)
			
			# Process frames manually to avoid corrupted frames
			while True:
				ret, frame = cap_process.read()
				if not ret or frame is None:
					break
				
				frame_counter += 1
				
				# Skip if frame is corrupted or invalid
				if not hasattr(frame, 'shape') or frame.size == 0:
					continue
				
				
				if frame_counter % self.sample_rate != 0:
					continue
				
				# Apply frameskip based on percentage
				if self.frameskip_percentage > 0:
					# Calculate if this frame should be skipped based on percentage
					# Using modulo pattern: skip M frames every N frames based on percentage
					frame_mod = (processed_frame_counter % 10)
					skip_count = int(self.frameskip_percentage / 10)  # How many frames to skip per 10
					if frame_mod < skip_count:
						processed_frame_counter += 1
						continue
				
				processed_frame_counter += 1
				
				# Run detection and tracking on this frame
				results = self.person_model.track(
					source=frame,
					device=self.device,
					imgsz=self.pre_size,
					conf=self.conf_person,
					tracker="bytetrack.yaml",
					persist=True,
					verbose=False,
					classes=0
				)
				
				# Get first result (single frame)
				if not results or len(results) == 0:
					continue
				result = results[0]
				
				# Prepare crops for batch classification
				crops, infos = [], []
				for box in result.boxes:
					# Only process person class (class 0)
					if box.cls.cpu().item() != 0:
						continue
					
					tid = box.id.cpu().item() if box.id is not None else 0
					x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
					
					# ROI filtering: Skip persons whose center is outside all ROIs
					# Apply filtering whenever ROIs are defined (from GUI or config)
					if rois:
						cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
						inside_any_roi = any(
							cv2.pointPolygonTest(r['coords'], (cx, cy), False) >= 0
							for r in rois
						)
						if not inside_any_roi:
							# Person is outside all ROIs, skip completely
							continue
					
					# Extract person crop for classification
					roi = frame[y1:y2, x1:x2]
					if roi.size:
						crops.append(roi)
						infos.append((tid, (x1, y1, x2, y2)))
					else:
						# Empty crop, but track exists - add to buffer with neutral state
						if tid not in track_counts:
							track_counts[tid] = {}
						frame_buffer.append((frame, [{'track_id': tid, 'coords': (x1, y1, x2, y2)}]))
				
				# Batch classification
				detections = []
				if crops and self.cls_model is None:
					# Classification disabled: assign Neutral to all detections
					for tid, coords in infos:
						if tid not in track_counts:
							track_counts[tid] = {}
						track_counts[tid]["Neutral"] = track_counts[tid].get("Neutral", 0) + 1
						detections.append({'track_id': tid, 'coords': coords})
				if crops and self.cls_model is not None:
					results = self.cls_model(crops, device=self.device, conf=self.conf_state, verbose=False)
					for res, (tid, coords) in zip(results, infos):
						# Get probabilities
						probs = res.probs.data.cpu().numpy()
						class_names = list(res.names.values())
						
						# ROI-based probability adjustment
						x1, y1, x2, y2 = coords
						cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
						roi_name = None
						for roi in rois:
							if cv2.pointPolygonTest(roi['coords'], (cx, cy), False) >= 0:
								roi_name = roi['name']
								break
						
						# Get top-2 classes
						top2_indices = probs.argsort()[-2:][::-1]
						top2_class_names = [class_names[i] for i in top2_indices]
						
						# Find working and idle indices
						idx_working = next((i for i, n in enumerate(top2_class_names) 
						                   if n.lower() in ["va", "ruteo", "enteipado", "working", "convolute", "insercion", "tomar material"]), None)
						idx_idle = next((i for i, n in enumerate(top2_class_names) 
						                if n.lower() in ["nva", "idle", "celular"]), None)
						
						# Adjust based on ROI
						if idx_working is not None and idx_idle is not None:
							if roi_name and roi_name.split(" ")[0].strip().lower() == 'conv':
								probs[top2_indices[idx_working]] = min(probs[top2_indices[idx_working]] + 0.10, 1.0)
								probs[top2_indices[idx_idle]] = max(probs[top2_indices[idx_idle]] - 0.10, 0.0)
							else:
								probs[top2_indices[idx_idle]] = min(probs[top2_indices[idx_idle]] + 0.10, 1.0)
								probs[top2_indices[idx_working]] = max(probs[top2_indices[idx_working]] - 0.10, 0.0)
							total = probs.sum()
							if total > 0:
								probs = probs / total
						
						# Select final class
						cls_idx = int(np.argmax(probs))
						cls_name = class_names[cls_idx]
						if tid not in track_counts:
							track_counts[tid] = {}
						track_counts[tid][cls_name] = track_counts[tid].get(cls_name, 0) + 1
						detections.append({'track_id': tid, 'coords': coords})
						
						# Process ergonomics if enabled
						if self.seguridad:
							pose_status, pose_color = self._process_ergonomics(frame, coords, tid)
							if pose_status:
								ergonomics_results[tid] = (pose_status, pose_color)
				
				# Apply face blur to the frame if enabled (before adding to buffer)
				if self.difuminar_caras:
					self._apply_face_blur(frame)
				
				frame_buffer.append((frame, detections))
				
				# Flush segment
				if len(frame_buffer) >= self.aggregation_frames:
					# Calculate timestamp based on actual video frame position (considering skipped frames)
					actual_frame_position = skip_initial_frames + frame_counter
					elapsed_seconds = actual_frame_position / self.frame_rate_real
					if use_video_time:
						seg_time = start_dt + timedelta(seconds=elapsed_seconds)
						time_str = seg_time.strftime("%H:%M:%S")
					else:
						time_str = (datetime.strptime("00:00:00", "%H:%M:%S") + 
						           timedelta(seconds=elapsed_seconds)).strftime("%H:%M:%S")
					
					self._flush_segment(write_q if self.generate_video else None, frame_buffer, track_counts, rois,
					                   initial_time, segment_idx, video_name, csv_rows, video_date, video_weekday, time_str, ergonomics_results)
					segment_idx += 1
				# Progress update
				if total_frames > 0 and self.total_to_process > 0:
					actual_frame_position = skip_initial_frames + frame_counter
					partial = (actual_frame_position / max(total_frames, 1))
					overall = (self.processed_count + partial) / self.total_to_process * 100.0
					current_int = int(overall)
					if current_int != last_progress_print:
						last_progress_print = current_int
						if progress_callback:
							progress_callback(overall)
						else:
							self._log(f"GUI_PROGRESS {overall:.2f}")
					
					# Log frame progress every 100 frames
					if frame_counter % 100 == 0:
						self._log(f"[PROGRESO] {basename}: Frames {actual_frame_position}/{total_frames} ({(actual_frame_position/total_frames*100):.1f}%)")
			
			# Close video capture
			cap_process.release()
			
			# Flush final segment
			if frame_buffer:
				# Calculate timestamp based on actual video frame position (considering skipped frames)
				actual_frame_position = skip_initial_frames + frame_counter
				elapsed_seconds = actual_frame_position / self.frame_rate_real
				if use_video_time:
					seg_time = start_dt + timedelta(seconds=elapsed_seconds)
					time_str = seg_time.strftime("%H:%M:%S")
				else:
					time_str = (datetime.strptime("00:00:00", "%H:%M:%S") + 
				           timedelta(seconds=elapsed_seconds)).strftime("%H:%M:%S")
				self._flush_segment(write_q if self.generate_video else None, frame_buffer, track_counts, rois,
			                   initial_time, segment_idx, video_name, csv_rows, video_date, video_weekday, time_str, ergonomics_results)
			
			# Close video writer
			if self.generate_video and write_q is not None:
				write_q.put(None)
				thr.join()
				os.rename(temp_path, final_path)
				self._log(f"✅ Video procesado: {video_name} → {final_path}")
			
			# Update progress
			self.processed_count += 1
			if self.total_to_process > 0:
				pct = (self.processed_count / self.total_to_process) * 100.0
				if progress_callback:
					progress_callback(pct)
				else:
					self._log(f"GUI_PROGRESS {pct:.2f}")
			
			# Write CSV rows
			if csv_rows:
				with open(csv_path, 'a', newline='') as f:
					w = csv.writer(f)
					w.writerows(csv_rows)
				self._log(f"[CSV] ✓ Escribió {len(csv_rows)} filas para {video_name} → {csv_path}")
			else:
				self._log(f"[CSV] ⚠ Sin filas para {video_name}. Verifique ROIs y detecciones.")
		
		self._log(f"\n{'='*60}")
		self._log(f"✅ PROCESAMIENTO COMPLETADO")
		self._log(f"{'='*60}")
		self._log(f"Total de videos procesados: {self.total_to_process}")
		self._log(f"CSV generado: {csv_path}")
		self._log(f"{'='*60}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--role", choices=["worker", "controller"], default=None)
	args, _unknown = parser.parse_known_args()
	main(node_role=args.role)
