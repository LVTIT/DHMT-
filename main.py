"""
main.py - FastAPI backend for Human Motion Capture.

Run:
    uvicorn main:app --reload

The backend reads webcam/video frames, extracts MediaPipe Pose landmarks,
draws a 2D OpenCV overlay, classifies gestures, logs predictions to CSV, and
streams compact 3D joint data to the browser through WebSocket /ws/pose.
"""

from __future__ import annotations

import asyncio
import csv
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gesture_classifier import GestureClassifier, GestureResult
from kinematics import compute_all_angles, compute_bone_lengths, compute_fk_summary, demo_ik_targets
from motion_filter import LandmarkSmoother
from pose_extractor import PoseExtractor
from utils import encode_jpeg_bytes, landmarks_delta, parse_source
from visualizer_2d import Visualizer2D


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
UPLOAD_DIR = ROOT_DIR / "uploads"
LOG_DIR = ROOT_DIR / "logs"

CAMERA_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]
CAMERA_BACKEND_BY_NAME = {name: backend_id for name, backend_id in CAMERA_BACKENDS}
PROCESS_MAX_WIDTH = 960
PROCESS_MAX_HEIGHT = 720
OVERLAY_JPEG_QUALITY = 72
MJPEG_TARGET_FPS = 15.0


def configure_capture_low_latency(cap: cv2.VideoCapture) -> None:
    """Ask OpenCV to keep only fresh webcam frames when the backend supports it."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)


def frame_stats(frame: np.ndarray) -> Dict[str, float]:
    """Return simple brightness statistics for camera diagnostics."""
    if frame is None or frame.size == 0:
        return {"mean": 0.0, "std": 0.0}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "mean": round(float(np.mean(gray)), 2),
        "std": round(float(np.std(gray)), 2),
    }


def parse_aspect_ratio(aspect: str, frame: np.ndarray) -> Optional[float]:
    """Return target width/height ratio, or None for native frame ratio."""
    aspect_text = str(aspect or "auto").strip().lower()
    if aspect_text in {"auto", "native"}:
        return None
    if aspect_text in {"16:9", "landscape"}:
        return 16.0 / 9.0
    if aspect_text in {"9:16", "portrait"}:
        return 9.0 / 16.0
    if ":" in aspect_text:
        left, right = aspect_text.split(":", 1)
        try:
            return float(left) / float(right)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    return w / max(1, h)


def rotate_frame(frame: np.ndarray, rotate: int) -> np.ndarray:
    """Rotate a frame by 0, 90, 180, or 270 degrees clockwise."""
    angle = int(rotate or 0) % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resize_with_aspect(frame: np.ndarray, aspect: str, fit: str) -> np.ndarray:
    """
    Convert a frame to the requested aspect while preserving useful content.

    contain = letterbox/pillarbox without cropping.
    cover = center crop to fill the target aspect.
    """
    target_ratio = parse_aspect_ratio(aspect, frame)
    if target_ratio is None or frame is None or frame.size == 0:
        return frame

    fit_mode = str(fit or "contain").strip().lower()
    h, w = frame.shape[:2]
    source_ratio = w / max(1, h)

    if abs(source_ratio - target_ratio) < 0.01:
        return frame

    if fit_mode == "cover":
        if source_ratio > target_ratio:
            new_w = max(1, int(round(h * target_ratio)))
            x0 = max(0, (w - new_w) // 2)
            return frame[:, x0:x0 + new_w]
        new_h = max(1, int(round(w / target_ratio)))
        y0 = max(0, (h - new_h) // 2)
        return frame[y0:y0 + new_h, :]

    # contain mode keeps the full iPhone frame visible for easier setup.
    if source_ratio > target_ratio:
        new_h = max(1, int(round(w / target_ratio)))
        pad_total = max(0, new_h - h)
        top = pad_total // 2
        bottom = pad_total - top
        return cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    new_w = max(1, int(round(h * target_ratio)))
    pad_total = max(0, new_w - w)
    left = pad_total // 2
    right = pad_total - left
    return cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize_to_realtime_bounds(frame: np.ndarray) -> np.ndarray:
    """Downscale very large camera frames so MediaPipe and MJPEG stay realtime."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(PROCESS_MAX_WIDTH / max(1, w), PROCESS_MAX_HEIGHT / max(1, h), 1.0)
    if scale >= 0.999:
        return frame
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def prepare_camera_frame(frame: np.ndarray, config: Dict[str, Union[str, int]]) -> np.ndarray:
    """Apply runtime camera rotation and aspect-ratio framing."""
    rotated = rotate_frame(frame, int(config.get("rotate", 0)))
    framed = resize_with_aspect(
        rotated,
        str(config.get("aspect", "auto")),
        str(config.get("fit", "contain")),
    )
    return resize_to_realtime_bounds(framed)


def is_numeric_camera_source(source: Union[int, str]) -> bool:
    """Return True for local webcam/virtual-camera sources."""
    parsed_source, _ = parse_forced_camera_backend(source)
    return isinstance(parsed_source, int)


def drop_one_stale_camera_frame(cap: cv2.VideoCapture, source: Union[int, str]) -> None:
    """Discard one buffered camera frame so slow processing does not add visible lag."""
    if not is_numeric_camera_source(source):
        return
    try:
        cap.grab()
    except Exception:
        return


def parse_forced_camera_backend(source: Union[int, str]) -> tuple[Union[int, str], Optional[str]]:
    """
    Parse camera source strings like "1@MSMF" or "1@DSHOW".

    URL sources are left untouched, so http://... still works normally.
    """
    if isinstance(source, str) and "@" in source:
        index_text, backend_text = source.split("@", 1)
        backend_name = backend_text.strip().upper()
        if index_text.strip().isdigit() and backend_name in CAMERA_BACKEND_BY_NAME:
            return int(index_text.strip()), backend_name
    return source, None


def is_seekable_file_or_url_source(source: Union[int, str]) -> bool:
    """Return True for string video files/URLs, not forced camera strings."""
    parsed_source, forced_backend = parse_forced_camera_backend(source)
    return isinstance(parsed_source, str) and forced_backend is None


def open_video_source(source: Union[int, str]) -> tuple[Optional[cv2.VideoCapture], str]:
    """
    Open a webcam, virtual camera, URL, or video file.

    Windows virtual cameras such as Camo often work best through DirectShow,
    so numeric camera sources try multiple OpenCV backends before failing.
    """
    source, forced_backend = parse_forced_camera_backend(source)
    if isinstance(source, int):
        backends = (
            [(forced_backend, CAMERA_BACKEND_BY_NAME[forced_backend])]
            if forced_backend
            else CAMERA_BACKENDS
        )
        for backend_name, backend_id in backends:
            cap = cv2.VideoCapture(source, backend_id)
            if cap.isOpened():
                configure_capture_low_latency(cap)
                return cap, backend_name
            cap.release()
        return None, "NONE"

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap, "URL_OR_FILE"
    cap.release()
    return None, "NONE"


def scan_camera_sources(max_index: int = 8, all_backends: bool = False) -> List[Dict]:
    """Probe webcam/virtual-camera indices and return readable sources."""
    cameras = []
    for index in range(max_index + 1):
        for backend_name, backend_id in CAMERA_BACKENDS:
            cap = cv2.VideoCapture(index, backend_id)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if ok and frame is not None and frame.size > 0:
                stats = frame_stats(frame)
                cameras.append({
                    "index": index,
                    "backend": backend_name,
                    "source": f"{index}@{backend_name}",
                    "width": width or int(frame.shape[1]),
                    "height": height or int(frame.shape[0]),
                    "fps": round(fps, 2),
                    "frame_mean": stats["mean"],
                    "frame_std": stats["std"],
                })
                if not all_backends:
                    break
    return cameras


class PauseRequest(BaseModel):
    """Pause/resume request body."""

    paused: bool


class SourceRequest(BaseModel):
    """Switch source request body."""

    source: str = "0"


class CameraConfigRequest(BaseModel):
    """Camera framing/orientation request body."""

    aspect: str = "auto"
    fit: str = "contain"
    rotate: int = 0


class WebSocketManager:
    """Manage connected browser clients and broadcast JSON pose payloads."""

    def __init__(self) -> None:
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._clients:
                self._clients.remove(websocket)

    async def broadcast(self, payload: Dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        disconnected = []
        for websocket in clients:
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                disconnected.append(websocket)
        for websocket in disconnected:
            await self.disconnect(websocket)

    @property
    def client_count(self) -> int:
        return len(self._clients)


class GestureCSVLogger:
    """Append recognized pose labels to logs/gesture_log.csv."""

    def __init__(
        self,
        path: Path,
        min_interval_s: float = 5.0,
        no_person_interval_s: float = 15.0,
    ) -> None:
        self.path = path
        self.min_interval_s = min_interval_s
        self.no_person_interval_s = no_person_interval_s
        self._lock = threading.Lock()
        self._last_label: Optional[str] = None
        self._last_write = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "detected_pose", "confidence"])

    def log(self, result: GestureResult) -> None:
        """Write label changes and periodic updates to CSV."""
        now = time.time()
        if result.label == self._last_label:
            interval = self.no_person_interval_s if result.label == "NO_PERSON" else self.min_interval_s
            if now - self._last_write < interval:
                return
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    result.label,
                    f"{result.confidence:.4f}",
                ])
        self._last_label = result.label
        self._last_write = now


class PoseRuntime:
    """Background capture and pose-processing loop used by the FastAPI app."""

    def __init__(
        self,
        manager: WebSocketManager,
        source: Union[int, str] = 0,
        model_path: Optional[str] = None,
        delta_threshold: float = 0.012,
    ) -> None:
        self.manager = manager
        self.model_path = model_path
        self.delta_threshold = delta_threshold

        self._source: Union[int, str] = source
        self._source_revision = 0
        self._camera_config: Dict[str, Union[str, int]] = {
            "aspect": "9:16",
            "fit": "contain",
            "rotate": 0,
        }
        self._source_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._paused = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._latest_payload = self._empty_payload("Starting camera")
        self._latest_jpeg = self._placeholder_jpeg("Starting camera")
        self._last_sent_landmarks = None
        self._last_sent_label: Optional[str] = None
        self._last_sent_detected: Optional[bool] = None
        self._last_sent_at = 0.0

        self._logger = GestureCSVLogger(LOG_DIR / "gesture_log.csv")

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the processing thread."""
        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="pose-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the processing thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def set_paused(self, paused: bool) -> Dict:
        """Pause or resume frame processing."""
        with self._state_lock:
            self._paused = paused
            self._latest_payload["metadata"]["paused"] = paused
        self._broadcast_latest()
        return self.state()

    def set_source(self, source: Union[int, str]) -> Dict:
        """Request the worker thread to switch video source."""
        parsed = parse_source(source)
        with self._source_lock:
            self._source = parsed
            self._source_revision += 1
        with self._state_lock:
            self._paused = False
            self._latest_payload = self._empty_payload(f"Switching source to {parsed}")
            self._latest_jpeg = self._placeholder_jpeg(f"Switching source to {parsed}")
        self._broadcast_latest()
        return self.state()

    def set_camera_config(self, config: CameraConfigRequest) -> Dict:
        """Update camera orientation/framing without reopening the source."""
        aspect = config.aspect.strip().lower()
        if aspect not in {"auto", "native", "16:9", "9:16", "landscape", "portrait"}:
            raise ValueError("aspect must be auto, 16:9, or 9:16")
        fit = config.fit.strip().lower()
        if fit not in {"contain", "cover"}:
            raise ValueError("fit must be contain or cover")
        rotate = int(config.rotate) % 360
        if rotate not in {0, 90, 180, 270}:
            raise ValueError("rotate must be 0, 90, 180, or 270")

        with self._source_lock:
            self._camera_config = {
                "aspect": aspect,
                "fit": fit,
                "rotate": rotate,
            }
        self._broadcast_latest()
        return self.state()

    def state(self) -> Dict:
        """Return current backend state for UI controls."""
        with self._source_lock:
            source = self._source
            camera_config = dict(self._camera_config)
        with self._state_lock:
            payload = dict(self._latest_payload)
            paused = self._paused
        return {
            "source": str(source),
            "camera_config": camera_config,
            "paused": paused,
            "client_count": self.manager.client_count,
            "latest": payload,
            "log_path": str((LOG_DIR / "gesture_log.csv").relative_to(ROOT_DIR)),
        }

    def latest_payload(self) -> Dict:
        with self._state_lock:
            return dict(self._latest_payload)

    def latest_jpeg(self) -> bytes:
        with self._state_lock:
            return bytes(self._latest_jpeg)

    def _run(self) -> None:
        """Worker loop: read frames, process pose, broadcast changed skeletons."""
        extractor: Optional[PoseExtractor] = None
        cap: Optional[cv2.VideoCapture] = None
        current_revision = -1
        classifier = GestureClassifier()
        visualizer = Visualizer2D()
        smoother = LandmarkSmoother(alpha=0.55)
        source_backend = "NONE"

        try:
            extractor = PoseExtractor(model_path=self.model_path)
            while not self._stop_event.is_set():
                with self._source_lock:
                    requested_source = self._source
                    requested_revision = self._source_revision
                    camera_config = dict(self._camera_config)

                if cap is None or requested_revision != current_revision:
                    if cap is not None:
                        cap.release()
                    cap, source_backend = open_video_source(requested_source)
                    current_revision = requested_revision
                    smoother.reset()
                    if cap is None or not cap.isOpened():
                        message = f"Cannot open source: {requested_source}"
                        self._publish_no_person(message)
                        time.sleep(1.0)
                        cap = None
                        continue

                if self._is_paused():
                    time.sleep(0.05)
                    continue

                try:
                    drop_one_stale_camera_frame(cap, requested_source)
                    ok, frame = cap.read()
                except Exception as exc:
                    self._publish_no_person(
                        f"Camera read error on {requested_source} via {source_backend}: {exc}"
                    )
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue

                if not ok:
                    if is_seekable_file_or_url_source(requested_source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    self._publish_no_person(
                        f"Opened source {requested_source} via {source_backend}, but no frames were readable"
                    )
                    cap.release()
                    cap = None
                    time.sleep(0.25)
                    continue

                try:
                    started = time.perf_counter()
                    raw_h, raw_w = frame.shape[:2]
                    frame = prepare_camera_frame(frame, camera_config)
                    proc_h, proc_w = frame.shape[:2]
                    current_frame_stats = frame_stats(frame)
                    pose_frame = extractor.extract(frame)
                    detection = pose_frame.to_detection()
                    detection = smoother.apply(detection) if detection is not None else None

                    angles: Dict[str, float] = {}
                    metadata: Dict = {}
                    if detection is not None:
                        angles = compute_all_angles(detection["landmarks_3d"], detection["visibility"], vis_threshold=0.3)
                        metadata = {
                            "bone_lengths": compute_bone_lengths(detection["landmarks_3d"]),
                            "fk": compute_fk_summary(detection["landmarks_3d"]),
                            "ik_demo": demo_ik_targets(detection["landmarks_3d"]),
                        }

                    gesture = classifier.classify(detection, angles)
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    fps = 1.0 / elapsed
                    metadata.update({
                        "fps": fps,
                        "source": str(requested_source),
                        "source_backend": source_backend,
                        "camera_aspect": str(camera_config.get("aspect", "auto")),
                        "camera_fit": str(camera_config.get("fit", "contain")),
                        "camera_rotate": int(camera_config.get("rotate", 0)),
                        "raw_width": raw_w,
                        "raw_height": raw_h,
                        "frame_width": proc_w,
                        "frame_height": proc_h,
                        "frame_ratio": round(proc_w / max(1, proc_h), 4),
                        "max_frame_width": PROCESS_MAX_WIDTH,
                        "max_frame_height": PROCESS_MAX_HEIGHT,
                        "latency_mode": "low",
                        "frame_mean": current_frame_stats["mean"],
                        "frame_std": current_frame_stats["std"],
                        "paused": False,
                        "client_count": self.manager.client_count,
                    })

                    overlay = frame.copy()
                    visualizer.draw(overlay, detection, angles, gesture=gesture, vis_threshold=0.35)
                    jpeg = encode_jpeg_bytes(overlay, quality=OVERLAY_JPEG_QUALITY)

                    payload = self._build_payload(detection, angles, gesture, metadata)
                    self._logger.log(gesture)
                    self._set_latest(payload, jpeg)
                    if self._should_broadcast(payload):
                        self._broadcast(payload)

                    # Keep CPU usage tame while still feeling realtime.
                    time.sleep(max(0.0, (1.0 / 30.0) - (time.perf_counter() - started)))
                except Exception as exc:
                    self._publish_no_person(
                        f"Frame processing error on {requested_source} via {source_backend}: {exc}"
                    )
                    time.sleep(0.2)
                    continue
        except Exception as exc:
            self._publish_no_person(f"Runtime error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if extractor is not None:
                extractor.close()

    def _is_paused(self) -> bool:
        with self._state_lock:
            return self._paused

    def _set_latest(self, payload: Dict, jpeg: bytes) -> None:
        with self._state_lock:
            self._latest_payload = payload
            if jpeg:
                self._latest_jpeg = jpeg

    def _build_payload(
        self,
        detection: Optional[Dict],
        angles: Dict[str, float],
        gesture: GestureResult,
        metadata: Dict,
    ) -> Dict:
        if detection is None:
            return {
                "detected": False,
                "landmarks": None,
                "joints": [],
                "visibility": [],
                "angles": {},
                "gesture": gesture.to_dict(),
                "metadata": metadata,
                "message": "No person detected",
                "timestamp_ms": time.time() * 1000.0,
            }
        return {
            "detected": True,
            "landmarks": [list(map(float, point)) for point in detection["landmarks_3d"]],
            "joints": detection.get("joints", []),
            "visibility": [float(value) for value in detection["visibility"]],
            "angles": {name: float(value) for name, value in angles.items()},
            "gesture": gesture.to_dict(),
            "metadata": metadata,
            "message": "OK",
            "timestamp_ms": time.time() * 1000.0,
        }

    def _should_broadcast(self, payload: Dict) -> bool:
        now = time.perf_counter()
        label = payload.get("gesture", {}).get("label")
        detected = bool(payload.get("detected"))
        if now - self._last_sent_at > 1.0:
            changed = True
        elif detected != self._last_sent_detected or label != self._last_sent_label:
            changed = True
        elif detected:
            changed = landmarks_delta(
                payload.get("landmarks"),
                self._last_sent_landmarks,
                payload.get("visibility"),
            ) >= self.delta_threshold
        else:
            changed = False

        if not changed:
            return False

        self._last_sent_at = now
        self._last_sent_label = label
        self._last_sent_detected = detected
        self._last_sent_landmarks = payload.get("landmarks")
        return True

    def _broadcast_latest(self) -> None:
        self._broadcast(self.latest_payload())

    def _broadcast(self, payload: Dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.manager.broadcast(payload), self._loop)

    def _publish_no_person(self, message: str) -> None:
        gesture = GestureResult("NO_PERSON", 0.0, {"reason": 0.0})
        with self._source_lock:
            source = self._source
        payload = self._empty_payload(message)
        payload["metadata"]["source"] = str(source)
        self._logger.log(gesture)
        self._set_latest(payload, self._placeholder_jpeg(message))
        self._broadcast(payload)

    def _empty_payload(self, message: str) -> Dict:
        return {
            "detected": False,
            "landmarks": None,
            "joints": [],
            "visibility": [],
            "angles": {},
            "gesture": GestureResult("NO_PERSON", 0.0, {"reason": 0.0}).to_dict(),
            "metadata": {
                "fps": 0.0,
                "source": str(self._source),
                "paused": self._paused,
                "client_count": self.manager.client_count,
            },
            "message": message,
            "timestamp_ms": time.time() * 1000.0,
        }

    @staticmethod
    def _placeholder_jpeg(message: str) -> bytes:
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        frame[:] = (18, 19, 22)
        cv2.putText(frame, "Human Motion Capture", (36, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 230), 2)
        cv2.putText(frame, message[:56], (36, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 210), 2)
        cv2.putText(frame, "No person detected", (36, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 110, 255), 2)
        return encode_jpeg_bytes(frame)


manager = WebSocketManager()
runtime = PoseRuntime(manager)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start(asyncio.get_running_loop())
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="Human Motion Capture", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Serve the Three.js viewer."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/state")
async def api_state() -> JSONResponse:
    """Return backend state and latest pose payload."""
    return JSONResponse(runtime.state())


@app.get("/api/cameras")
async def api_cameras(max_index: int = 8, all_backends: bool = False) -> JSONResponse:
    """Scan local webcam/virtual-camera indices."""
    return JSONResponse({
        "cameras": scan_camera_sources(
            max_index=max(0, min(max_index, 12)),
            all_backends=all_backends,
        )
    })


@app.post("/api/pause")
async def api_pause(request: PauseRequest) -> JSONResponse:
    """Pause or resume the capture loop."""
    return JSONResponse(runtime.set_paused(request.paused))


@app.post("/api/source")
async def api_source(request: SourceRequest) -> JSONResponse:
    """Switch to a camera index or a server-local video path."""
    return JSONResponse(runtime.set_source(request.source))


@app.post("/api/camera-config")
async def api_camera_config(request: CameraConfigRequest) -> JSONResponse:
    """Update camera aspect/fit/rotation settings."""
    try:
        return JSONResponse(runtime.set_camera_config(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a video file and immediately use it as the input source."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{int(time.time())}_{Path(file.filename).name}"
    content = await file.read()
    target.write_bytes(content)
    state = runtime.set_source(str(target))
    return JSONResponse({"uploaded": str(target.relative_to(ROOT_DIR)), "state": state})


def mjpeg_generator():
    """Yield the latest OpenCV overlay as an MJPEG stream."""
    boundary = b"--frame\r\n"
    while True:
        jpeg = runtime.latest_jpeg()
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / MJPEG_TARGET_FPS)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    """Stream the 2D OpenCV overlay to the browser."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot.jpg")
async def snapshot() -> Response:
    """Return the latest OpenCV overlay frame as a single JPEG."""
    return Response(content=runtime.latest_jpeg(), media_type="image/jpeg")


@app.websocket("/ws/pose")
@app.websocket("/ws")
async def websocket_pose(websocket: WebSocket) -> None:
    """WebSocket endpoint for realtime 3D skeleton data."""
    await manager.connect(websocket)
    await websocket.send_text(json.dumps(runtime.latest_payload()))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
