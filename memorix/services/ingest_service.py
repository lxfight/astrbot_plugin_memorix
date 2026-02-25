"""Ingest orchestration service."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from astrbot.api import logger

from ..amemorix.services.import_service import ImportService
from ..app_context import ScopeRuntimeManager
from ..core.utils.time_parser import normalize_time_meta


class IngestService:
    def __init__(self, runtime_manager: ScopeRuntimeManager, plugin_config: Dict[str, Any]):
        self.runtime_manager = runtime_manager
        self.plugin_config = plugin_config or {}
        self._summary_trigger_state: Dict[str, Dict[str, float]] = {}

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.plugin_config if isinstance(self.plugin_config, dict) else {}
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    async def _maybe_enqueue_summary(
        self,
        *,
        scope_key: str,
        session_id: str,
        role: str,
        runtime: Any,
    ) -> None:
        if str(role or "").strip().lower() != "assistant":
            return
        if not bool(self._cfg("summarization.enabled", True)):
            return
        if not bool(self._cfg("summarization.auto_trigger_enabled", True)):
            return

        every_n = max(1, int(self._cfg("summarization.auto_trigger_every_n_messages", 20) or 20))
        cooldown_minutes = float(self._cfg("summarization.auto_trigger_cooldown_minutes", 15.0) or 15.0)
        min_messages = max(2, int(self._cfg("summarization.auto_trigger_min_messages", every_n) or every_n))
        context_length = max(1, int(self._cfg("summarization.context_length", 50) or 50))

        conn = getattr(runtime.context.metadata_store, "_conn", None)
        if conn is None:
            return
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(1) FROM transcript_messages WHERE session_id = ?", (str(session_id),))
        row = cursor.fetchone()
        msg_count = int(row[0] or 0) if row else 0
        if msg_count < min_messages:
            return

        milestone = msg_count // every_n
        if milestone <= 0:
            return

        state_key = f"{scope_key}:{session_id}"
        state = self._summary_trigger_state.get(state_key, {})
        last_milestone = int(state.get("milestone", 0) or 0)
        now_ts = time.time()
        last_trigger = float(state.get("triggered_at", 0.0) or 0.0)
        if milestone <= last_milestone:
            return
        if (now_ts - last_trigger) < max(0.0, cooldown_minutes * 60.0):
            return

        payload = {
            "session_id": session_id,
            "messages": [],
            "source": f"chat_summary:{session_id}",
            "context_length": context_length,
            "persist_messages": False,
        }
        await runtime.task_manager.enqueue_summary_task(payload)
        self._summary_trigger_state[state_key] = {"milestone": float(milestone), "triggered_at": now_ts}
        logger.info(
            "auto summary task enqueued: scope=%s session=%s msg_count=%s milestone=%s",
            scope_key,
            session_id,
            msg_count,
            milestone,
        )

    async def ingest_message(
        self,
        *,
        scope_key: str,
        session_id: str,
        role: str,
        content: str,
        source: str,
        user_id: str = "",
        group_id: str = "",
        platform: str = "",
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(content or "").strip()
        if not text and bool(self.plugin_config.get("ingest", {}).get("skip_empty_text", True)):
            return {"success": True, "skipped": True, "reason": "empty"}

        runtime = await self.runtime_manager.get_runtime(scope_key)
        ctx = runtime.context
        session = str(session_id or "").strip() or f"scope:{scope_key}"

        ctx.metadata_store.upsert_transcript_session(
            session_id=session,
            source=source,
            metadata={
                "scope_key": scope_key,
                "user_id": str(user_id or "").strip(),
                "group_id": str(group_id or "").strip(),
                "platform": str(platform or "").strip(),
            },
        )
        ctx.metadata_store.append_transcript_messages(
            session_id=session,
            messages=[{"role": str(role or "user"), "content": text}],
        )

        importer = ImportService(ctx)
        try:
            result = await importer.import_paragraph(
                content=text,
                source=source,
                time_meta=normalize_time_meta(time_meta or {}),
            )
            try:
                await self._maybe_enqueue_summary(
                    scope_key=scope_key,
                    session_id=session,
                    role=role,
                    runtime=runtime,
                )
            except Exception as summary_exc:
                logger.warning(
                    "auto summary enqueue failed: scope=%s session=%s err=%s",
                    scope_key,
                    session,
                    summary_exc,
                    exc_info=True,
                )
            return {"success": True, "result": result, "skipped": False}
        except Exception:
            paragraph_hash = ctx.metadata_store.add_paragraph(
                content=text,
                source=source,
                time_meta=normalize_time_meta(time_meta or {}),
            )
            try:
                await self._maybe_enqueue_summary(
                    scope_key=scope_key,
                    session_id=session,
                    role=role,
                    runtime=runtime,
                )
            except Exception as summary_exc:
                logger.warning(
                    "auto summary enqueue failed after fallback import: scope=%s session=%s err=%s",
                    scope_key,
                    session,
                    summary_exc,
                    exc_info=True,
                )
            return {
                "success": True,
                "result": {"mode": "paragraph", "hash": paragraph_hash, "embedding_skipped": True},
                "skipped": False,
            }
