"""
FastAPI dashboard server.

Endpoints
---------
GET  /health      Liveness probe → {"status": "ok"}
GET  /api/state   Latest LaneState snapshot for initial page load
WS   /ws          Streams signal_granted events as JSON to all connected clients

Static files
------------
dashboard-ui/dist is mounted at "/" when it exists (i.e. after ``npm run build``).
In development, Vite's proxy routes /api and /ws to :8000 so the built folder is
not needed — Vite's dev server at :5173 serves the React source directly.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.dashboard.broadcaster import broadcaster
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Captures the running event loop so the synchronous pipeline can schedule
    WebSocket broadcasts onto it via broadcaster.broadcast_sync().
    """
    broadcaster.set_loop(asyncio.get_running_loop())
    log.info("dashboard_server_started", port=8000)
    yield
    log.info("dashboard_server_stopped")


app = FastAPI(
    title="ATMS Dashboard",
    lifespan=_lifespan,
    docs_url=None,   # no Swagger UI in production
    redoc_url=None,
)


@app.get("/health")
async def health() -> dict:
    """
    Liveness probe.

    Returns:
        {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/api/state")
async def get_state() -> dict:
    """
    Return the latest lane-state snapshot for initial page load.

    The pipeline writes this after every signal decision; the frontend fetches
    it once on mount to avoid waiting for the first WebSocket message.

    Returns:
        Most recent signal_granted payload, or a placeholder with empty lanes.
    """
    state = broadcaster.get_state()
    if state is None:
        return {"event": "initial", "winner": None, "timestamp": None, "lanes": []}
    return state


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    WebSocket endpoint — keeps the connection alive and registered.

    The broadcaster pushes events via broadcast(); this handler only needs to
    drain incoming frames so FastAPI detects client disconnects.

    Args:
        ws: The incoming WebSocket connection.
    """
    await ws.accept()
    await broadcaster.connect(ws)
    try:
        while True:
            # Drain any client messages (ping/keepalive); we don't act on them.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws_unexpected_error")
    finally:
        await broadcaster.disconnect(ws)


# ── Static files ──────────────────────────────────────────────────────────────
# Mount the React build output when it exists. The mount must come last so
# /health, /api/*, and /ws are matched before the catch-all StaticFiles handler.
_dist = Path(__file__).resolve().parent.parent.parent / "dashboard-ui" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
    log.debug("dashboard_static_mounted", path=str(_dist))
