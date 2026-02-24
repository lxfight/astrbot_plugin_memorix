"""Scope runtime management for memorix plugin."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .amemorix.bootstrap import build_context
from .amemorix.settings import DEFAULT_CONFIG, AppSettings
from .amemorix.task_manager import TaskManager


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class LocalEmbeddingAdapter:
    """Deterministic local embedding fallback for offline mode."""

    def __init__(self, dimension: int):
        self.dimension = int(max(1, dimension))

    async def _detect_dimension(self) -> int:
        return self.dimension

    async def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._embed_text(texts)

        vectors = [self._embed_text(str(text)) for text in list(texts)]
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def _embed_text(self, text: str) -> np.ndarray:
        payload = str(text or "")
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


@dataclass(slots=True)
class ScopeRuntime:
    scope_key: str
    settings: AppSettings
    context: Any
    task_manager: TaskManager


class ScopeRuntimeManager:
    def __init__(self, *, plugin_name: str, plugin_config: Dict[str, Any]):
        self.plugin_name = plugin_name
        self.plugin_config = dict(plugin_config or {})
        self._runtimes: Dict[str, ScopeRuntime] = {}
        self._lock = asyncio.Lock()

    def _scope_dir(self, scope_key: str) -> Path:
        data_root_raw = get_astrbot_data_path()
        data_root = Path(data_root_raw) if data_root_raw else (Path.cwd() / "data")
        base = data_root / "plugin_data" / self.plugin_name / "scopes"
        base.mkdir(parents=True, exist_ok=True)
        safe_scope = str(scope_key or "default").replace("/", "_").replace("\\", "_")
        return base / safe_scope

    def _build_scope_config(self, scope_key: str) -> Dict[str, Any]:
        cfg = _deep_merge(DEFAULT_CONFIG, self.plugin_config)
        cfg.setdefault("storage", {})
        cfg["storage"]["data_dir"] = str(self._scope_dir(scope_key))

        webui_cfg = cfg.get("webui", {}) if isinstance(cfg.get("webui"), dict) else {}
        auth_cfg = webui_cfg.get("auth", {}) if isinstance(webui_cfg.get("auth"), dict) else {}
        cfg.setdefault("auth", {})
        cfg["auth"]["enabled"] = bool(auth_cfg.get("enabled", False))
        cfg["auth"]["write_tokens"] = list(auth_cfg.get("write_tokens", []))
        cfg["auth"]["read_tokens"] = list(auth_cfg.get("read_tokens", []))
        cfg["auth"]["protect_read_endpoints"] = bool(auth_cfg.get("protect_read_endpoints", False))

        return cfg

    def _patch_local_embedding(self, runtime: ScopeRuntime) -> None:
        dimension = int(runtime.settings.get("embedding.dimension", 1024) or 1024)
        adapter = LocalEmbeddingAdapter(dimension=dimension)
        runtime.context.embedding_manager = adapter
        runtime.context.retriever.embedding_manager = adapter
        if hasattr(runtime.context, "person_profile_service"):
            runtime.context.person_profile_service.embedding_manager = adapter

    async def get_runtime(self, scope_key: str) -> ScopeRuntime:
        key = str(scope_key or "default")
        existing = self._runtimes.get(key)
        if existing is not None:
            return existing

        async with self._lock:
            existing = self._runtimes.get(key)
            if existing is not None:
                return existing

            cfg = self._build_scope_config(key)
            settings = AppSettings(config=cfg)
            ctx = build_context(settings)
            task_manager = TaskManager(ctx)
            runtime = ScopeRuntime(scope_key=key, settings=settings, context=ctx, task_manager=task_manager)

            embedding_enabled = bool(settings.get("embedding.enabled", False))
            if not embedding_enabled:
                self._patch_local_embedding(runtime)

            await task_manager.start()
            self._runtimes[key] = runtime
            return runtime

    def get_known_scopes(self) -> list[str]:
        return list(self._runtimes.keys())

    async def close_all(self) -> None:
        async with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()

        for runtime in runtimes:
            try:
                await runtime.task_manager.stop()
            except Exception:
                pass
            try:
                await runtime.context.close()
            except Exception:
                pass
