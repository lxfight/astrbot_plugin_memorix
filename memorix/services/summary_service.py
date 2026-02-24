"""Summary wrapper service."""

from __future__ import annotations

from typing import Any, Dict

from ..amemorix.services.summary_service import SummaryService as BaseSummaryService
from ..app_context import ScopeRuntimeManager


class SummaryService:
    def __init__(self, runtime_manager: ScopeRuntimeManager):
        self.runtime_manager = runtime_manager

    async def summarize_session(
        self,
        *,
        scope_key: str,
        session_id: str,
        source: str,
        context_length: int = 50,
    ) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        messages = runtime.context.metadata_store.get_transcript_messages(session_id=session_id, limit=max(1, int(context_length)))
        service = BaseSummaryService(runtime.context)
        return await service.import_from_transcript(
            session_id=session_id,
            messages=messages,
            source=source,
            context_length=context_length,
        )

