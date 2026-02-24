"""Embedded WebUI server manager."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..amemorix.auth import BearerAuthMiddleware
from ..amemorix.common.logging import get_logger
from ..app_context import ScopeRuntimeManager
from .routes_compat import MemorixServer

logger = get_logger("A_Memorix.EmbeddedWebUI")


@dataclass(slots=True)
class WebUIServerState:
    scope_key: str = ""
    host: str = ""
    port: int = 0
    url: str = ""


class EmbeddedWebUIServer:
    def __init__(self, runtime_manager: ScopeRuntimeManager, plugin_config: Dict[str, Any]):
        self.runtime_manager = runtime_manager
        self.plugin_config = plugin_config or {}
        self._server: Optional[MemorixServer] = None
        self.state = WebUIServerState()

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.plugin_config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @staticmethod
    def _is_port_available(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return True
            except OSError:
                return False

    def _pick_port(self, host: str, start_port: int, tries: int) -> int:
        for idx in range(max(1, int(tries))):
            candidate = int(start_port) + idx
            if self._is_port_available(host, candidate):
                if idx > 0:
                    logger.warning(
                        "webui default port busy, fallback selected: host=%s from=%s to=%s",
                        host,
                        start_port,
                        candidate,
                    )
                return candidate
        raise RuntimeError(f"no available port from {start_port} after {tries} tries")

    async def start(self, *, scope_key: str) -> WebUIServerState:
        enabled = bool(self._cfg("webui.enabled", True))
        if not enabled:
            logger.info("webui disabled by config")
            return self.state

        if self._server is not None:
            logger.info("webui already running: url=%s scope=%s", self.state.url, self.state.scope_key)
            return self.state

        runtime = await self.runtime_manager.get_runtime(scope_key)

        host = str(self._cfg("webui.host", "0.0.0.0") or "0.0.0.0")
        start_port = int(self._cfg("webui.port", 8092) or 8092)
        max_tries = int(self._cfg("webui.port_fallback_max_tries", 20) or 20)
        port = self._pick_port(host, start_port, max_tries)

        server = MemorixServer(plugin_instance=runtime.context, host=host, port=port)
        server.app.state.context = runtime.context

        auth_enabled = bool(self._cfg("webui.auth.enabled", False))
        if auth_enabled:
            server.app.add_middleware(BearerAuthMiddleware)

        server.start()
        self._server = server
        self.state = WebUIServerState(
            scope_key=scope_key,
            host=host,
            port=port,
            url=f"http://{host}:{port}",
        )
        logger.info(
            "webui started: url=%s scope=%s auth_enabled=%s",
            self.state.url,
            self.state.scope_key,
            auth_enabled,
        )
        return self.state

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.stop()
        finally:
            logger.info("webui stopped: url=%s scope=%s", self.state.url, self.state.scope_key)
            self._server = None
            self.state = WebUIServerState()
