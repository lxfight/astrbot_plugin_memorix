"""NapCat/OneBot compatibility helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


class NapcatGateway:
    """Best-effort gateway; only active for aiocqhttp runtime events."""

    @staticmethod
    async def get_msg_by_id(event, message_id: str) -> Optional[Dict[str, Any]]:
        msg_id = str(message_id or "").strip()
        if not msg_id:
            return None
        if str(getattr(event, "get_platform_name", lambda: "")() or "") != "aiocqhttp":
            return None

        client = getattr(getattr(event, "bot", None), "api", None)
        if client is None:
            return None

        try:
            result = await client.call_action("get_msg", message_id=msg_id)
            if isinstance(result, dict):
                return result
        except Exception:
            return None
        return None

