"""
WebSocket broadcaster for the ATMS dashboard.

A single global ``broadcaster`` instance is shared between:
- ``src/dashboard/server.py`` — registers/deregisters WebSocket clients
- ``src/pipeline/manager.py`` — calls broadcast_sync() after each signal decision

The pipeline runs in synchronous code on the main thread; uvicorn's WebSocket
server runs in an asyncio event loop on a daemon thread.  ``broadcast_sync``
bridges the gap using ``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


class DashboardBroadcaster:
    """
    Manages active WebSocket connections and fans out JSON events.

    Thread-safety: ``_latest_state`` assignment is atomic under CPython's GIL,
    so no explicit lock is needed for the dashboard's read-one-writer use.
    The WebSocket set is only mutated inside the asyncio event loop.
    """

    def __init__(self) -> None:
        self._connections: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest_state: dict[str, Any] | None = None

    # ── Called from the asyncio / FastAPI side ──────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Store the running event loop so broadcast_sync() can schedule work into it.

        Called from the FastAPI lifespan handler (inside uvicorn's event loop).

        Args:
            loop: The asyncio event loop running the FastAPI server.
        """
        self._loop = loop
        log.debug("broadcaster_loop_set")

    async def connect(self, ws: Any) -> None:
        """
        Register a new WebSocket client.

        Args:
            ws: An accepted FastAPI WebSocket instance.
        """
        self._connections.add(ws)
        log.debug("ws_client_connected", total=len(self._connections))

    async def disconnect(self, ws: Any) -> None:
        """
        Remove a WebSocket client on disconnect.

        Args:
            ws: The WebSocket instance to remove.
        """
        self._connections.discard(ws)
        log.debug("ws_client_disconnected", total=len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """
        Send JSON data to every connected client, pruning dead connections.

        Args:
            data: Payload dict serialised as JSON for each client.
        """
        dead: set[Any] = set()
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        if dead:
            self._connections -= dead
            log.debug("ws_dead_connections_pruned", count=len(dead))

    # ── Called from the synchronous pipeline side ───────────────────────────

    def update_state(self, state: dict[str, Any]) -> None:
        """
        Store the latest signal-decision payload for the /api/state endpoint.

        Args:
            state: The most recent signal_granted payload dict.
        """
        self._latest_state = state

    def get_state(self) -> dict[str, Any] | None:
        """
        Return the last stored state snapshot, or None if no decision yet.

        Returns:
            Most recent signal_granted payload, or None.
        """
        return self._latest_state

    def broadcast_sync(self, data: dict[str, Any]) -> None:
        """
        Schedule a broadcast from synchronous (pipeline) code.

        Uses run_coroutine_threadsafe so the async broadcast() coroutine runs
        on uvicorn's event loop, not the caller's thread.  No-ops silently if
        the loop isn't set yet (i.e. no dashboard clients have connected).

        Args:
            data: Payload dict to broadcast as JSON.
        """
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)


# Single shared instance — imported by server.py and manager.py
broadcaster = DashboardBroadcaster()
