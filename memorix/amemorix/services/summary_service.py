"""Summary import service wrappers."""

from __future__ import annotations

from typing import Any, Dict, List

from amemorix.context import AppContext
from amemorix.llm_client import LLMClient
from amemorix.settings import resolve_openapi_endpoint_config
from astrbot.api import logger
from core.utils.summary_importer import SummaryImporter
from providers.astrbot_provider_bridge import AstrBotLLMClient


class SummaryService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        provider_bridge = getattr(self.ctx, "provider_bridge", None)
        if provider_bridge is not None and getattr(provider_bridge, "enabled", False):
            retries = int(self.ctx.get_config("embedding.retry.max_attempts", 3) or 3)
            self.llm_client = AstrBotLLMClient(provider_bridge=provider_bridge, max_retries=max(1, retries))
            logger.info("summary service uses AstrBot native chat provider")
        else:
            endpoint_cfg = resolve_openapi_endpoint_config(self.ctx.config, section="embedding")
            self.llm_client = LLMClient(
                base_url=str(endpoint_cfg.get("base_url", "")),
                api_key=str(endpoint_cfg.get("api_key", "")),
                model=str(
                    endpoint_cfg.get("chat_model", "")
                    or endpoint_cfg.get("model", "")
                    or "gpt-4o-mini"
                ),
                timeout_seconds=float(endpoint_cfg.get("timeout_seconds", 60) or 60),
                max_retries=int(endpoint_cfg.get("max_retries", 3) or 3),
            )
            logger.warning("summary service fallback to OpenAI-compatible client (provider bridge unavailable)")
        self.importer = SummaryImporter(
            vector_store=self.ctx.vector_store,
            graph_store=self.ctx.graph_store,
            metadata_store=self.ctx.metadata_store,
            embedding_manager=self.ctx.embedding_manager,
            plugin_config=self.ctx.config,
            llm_client=self.llm_client,
        )

    async def import_from_transcript(
        self,
        *,
        session_id: str,
        messages: List[Dict[str, Any]],
        source: str = "",
        context_length: int = 50,
        persist_messages: bool = False,
    ) -> Dict[str, Any]:
        ok, msg = await self.importer.import_from_transcript(
            session_id=session_id,
            messages=messages,
            source=source,
            context_length=context_length,
            persist_messages=persist_messages,
        )
        return {"success": ok, "message": msg}
