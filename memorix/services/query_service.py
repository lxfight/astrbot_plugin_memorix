"""Query service wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..amemorix.services.query_service import QueryService as BaseQueryService
from ..app_context import ScopeRuntimeManager


class QueryService:
    def __init__(self, runtime_manager: ScopeRuntimeManager):
        self.runtime_manager = runtime_manager

    async def search(self, *, scope_key: str, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        return await BaseQueryService(runtime.context).search(query=query, top_k=top_k)

    async def time_search(
        self,
        *,
        scope_key: str,
        query: str = "",
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        return await BaseQueryService(runtime.context).time_search(
            query=query,
            time_from=time_from,
            time_to=time_to,
            person=person,
            source=source,
            top_k=top_k,
        )

    async def stats(self, *, scope_key: str) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        return await BaseQueryService(runtime.context).stats()

