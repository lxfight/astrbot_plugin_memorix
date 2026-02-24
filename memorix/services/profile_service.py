"""Person profile wrapper service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..amemorix.services.person_profile_service import PersonProfileApiService
from ..app_context import ScopeRuntimeManager


class ProfileService:
    def __init__(self, runtime_manager: ScopeRuntimeManager):
        self.runtime_manager = runtime_manager

    async def upsert_registry_from_event(
        self,
        *,
        scope_key: str,
        platform: str,
        sender_id: str,
        sender_name: str,
    ) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        person_id = f"{platform}:{sender_id}"
        return runtime.context.metadata_store.upsert_person_registry(
            person_id=person_id,
            person_name=sender_name,
            nickname=sender_name,
            user_id=sender_id,
            platform=platform,
            metadata={"scope_key": scope_key},
        )

    async def query(
        self,
        *,
        scope_key: str,
        person_id: str = "",
        person_keyword: str = "",
        top_k: int = 12,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        runtime = await self.runtime_manager.get_runtime(scope_key)
        service = PersonProfileApiService(runtime.context)
        return await service.query(
            person_id=person_id,
            person_keyword=person_keyword,
            top_k=top_k,
            force_refresh=force_refresh,
            source_note="astrbot:profile_query",
        )

    async def set_override(self, *, scope_key: str, person_id: str, override_text: str, updated_by: str = "astrbot"):
        runtime = await self.runtime_manager.get_runtime(scope_key)
        service = PersonProfileApiService(runtime.context)
        return await service.set_override(person_id=person_id, override_text=override_text, updated_by=updated_by)

    async def delete_override(self, *, scope_key: str, person_id: str):
        runtime = await self.runtime_manager.get_runtime(scope_key)
        service = PersonProfileApiService(runtime.context)
        return await service.delete_override(person_id=person_id)

    async def upsert_registry(self, *, scope_key: str, payload: Dict[str, Any]):
        runtime = await self.runtime_manager.get_runtime(scope_key)
        service = PersonProfileApiService(runtime.context)
        return await service.upsert_registry(payload)

    async def list_registry(
        self,
        *,
        scope_key: str,
        keyword: str = "",
        page: int = 1,
        page_size: Optional[int] = None,
    ):
        runtime = await self.runtime_manager.get_runtime(scope_key)
        service = PersonProfileApiService(runtime.context)
        return await service.list_registry(keyword=keyword, page=page, page_size=page_size)

