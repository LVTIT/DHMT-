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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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


class PauseRequest(BaseModel):
    """Pause/resume request body."""

    paused: bool


class SourceRequest(BaseModel):
    """Switch source request body."""

    source: str = "0"


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

    def __init__(self, path: Path, min_interval_s: float = 0.45) -> None:
        self.path = path
        self.min_interval_s = min_interval_s
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
        if result.label == self._last_label and now - self._last_write < self.min_interval_s:
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
            self._latest_payload = self._empty_payload(f"Switching source to {parsed}")
            self._latest_jpeg = self._placeholder_jpeg(f"Switching source to {parsed}")
        self._broadcast_latest()
        return self.state()

    def state(self) -> Dict:
        """Return current backend state for UI controls."""
        with self._source_lock:
            source = self._source
        with self._state_lock:
            payload = dict(self._latest_payload)
            paused = self._paused
        return {
            "source": str(source),
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

        try:
            extractor = PoseExtractor(model_path=self.model_path)
            while not self._stop_event.is_set():
                with self._source_lock:
                    requested_source = self._source
                    requested_revision = self._source_revision

                if cap is None or requested_revision != current_revision:
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(requested_source)
                    current_revision = requested_revision
                    smoother.reset()
                    if not cap.isOpened():
                        message = f"Cannot open source: {requested_source}"
                        self._publish_no_person(message)
                        time.sleep(1.0)
                        cap = None
                        continue

                if self._is_paused():
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok:
                    if isinstance(requested_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    cap.release()
                    cap = None
                    time.sleep(0.25)
                    continue

                started = time.perf_counter()
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
                    "paused": False,
                    "client_count": self.manager.client_count,
                })

                overlay = frame.copy()
                visualizer.draw(overlay, detection, angles, gesture=gesture, vis_threshold=0.35)
                jpeg = encode_jpeg_bytes(overlay)

                payload = self._build_payload(detection, angles, gesture, metadata)
                self._logger.log(gesture)
                self._set_latest(payload, jpeg)
                if self._should_broadcast(payload):
                    self._broadcast(payload)

                # Keep CPU usage tame while still feeling realtime.
                time.sleep(max(0.0, (1.0 / 30.0) - (time.perf_counter() - started)))
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


@app.post("/api/pause")
async def api_pause(request: PauseRequest) -> JSONResponse:
    """Pause or resume the capture loop."""
    return JSONResponse(runtime.set_paused(request.paused))


@app.post("/api/source")
async def api_source(request: SourceRequest) -> JSONResponse:
    """Switch to a camera index or a server-local video path."""
    return JSONResponse(runtime.set_source(request.source))


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
        time.sleep(1.0 / 20.0)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    """Stream the 2D OpenCV overlay to the browser."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
