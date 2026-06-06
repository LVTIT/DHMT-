"""
server.py - FastAPI WebSocket server for 3D pose visualization.

Runs a local web server that:
1. Serves the Three.js viewer (static/index.html)
2. Provides a WebSocket endpoint that streams pose data in real-time

The main pipeline pushes detection data to this server,
and the browser client renders the 3D skeleton.
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn


# ======================================================================
# WebSocket connection manager
# ======================================================================
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts pose data."""

    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        with self._lock:
            self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client."""
        with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def broadcast(self, data: dict) -> None:
        """Send pose data to all connected clients."""
        message = json.dumps(data)
        disconnected = []
        with self._lock:
            connections = list(self._connections)
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._connections)


# ======================================================================
# FastAPI app
# ======================================================================
manager = ConnectionManager()
app = FastAPI(title="Motion Capture 3D Viewer")

# Serve static files (index.html)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the Three.js viewer page."""
    return FileResponse(str(_static_dir / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pose data streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client sends pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ======================================================================
# Server runner (threaded, non-blocking)
# ======================================================================
class PoseServer:
    """
    Wraps the FastAPI app in a background thread.

    Usage:
        server = PoseServer(port=8765)
        server.start()       # blocks until server is ready
        server.send_pose(detection, angles)
        server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self._host = host
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[uvicorn.Server] = None
        self._ready = threading.Event()

    def start(self) -> None:
        """Start the server in a background thread. Blocks until ready."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait up to 5 seconds for server to become ready
        self._ready.wait(timeout=5.0)
        print(f"[3D] Server started at http://localhost:{self._port}", flush=True)
        print(f"[3D] Open browser to see 3D skeleton viewer", flush=True)

    def _run(self) -> None:
        """Internal: run uvicorn in its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        config = uvicorn.Config(
            app, host=self._host, port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        # Signal readiness once the server starts
        original_startup = self._server.startup

        async def startup_with_signal(*args, **kwargs):
            result = await original_startup(*args, **kwargs)
            self._ready.set()
            return result

        self._server.startup = startup_with_signal
        self._loop.run_until_complete(self._server.serve())

    def stop(self) -> None:
        """Stop the server gracefully."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def send_pose(
        self,
        detection: Optional[Dict],
        angles: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Send pose data to all connected browser clients.

        Parameters
        ----------
        detection : dict | None
            Detection result with landmarks_3d and visibility.
        angles : dict | None
            Joint angles from compute_all_angles().
        """
        if self._loop is None or not self._ready.is_set():
            return
        if manager.client_count == 0:
            return

        if detection is None:
            data = {"landmarks": None, "angles": None, "metadata": metadata or {}}
        else:
            data = {
                "landmarks": [list(pt) for pt in detection["landmarks_3d"]],
                "angles": angles or {},
                "visibility": detection.get("visibility", []),
                "metadata": metadata or {},
            }

        # Schedule broadcast in the server's event loop
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), self._loop)
