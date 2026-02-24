"""Ingest orchestration service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..amemorix.services.import_service import ImportService
from ..app_context import ScopeRuntimeManager
from ..core.utils.time_parser import normalize_time_meta


class IngestService:
    def __init__(self, runtime_manager: ScopeRuntimeManager, plugin_config: Dict[str, Any]):
        self.runtime_manager = runtime_manager
        self.plugin_config = plugin_config or {}

    async def ingest_message(
        self,
        *,
        scope_key: str,
        session_id: str,
        role: str,
        content: str,
        source: str,
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
            metadata={"scope_key": scope_key},
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
            return {"success": True, "result": result, "skipped": False}
        except Exception:
            paragraph_hash = ctx.metadata_store.add_paragraph(
                content=text,
                source=source,
                time_meta=normalize_time_meta(time_meta or {}),
            )
            return {
                "success": True,
                "result": {"mode": "paragraph", "hash": paragraph_hash, "embedding_skipped": True},
                "skipped": False,
            }

