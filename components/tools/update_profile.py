from __future__ import annotations

import logging
from typing import Any

from langbot_plugin.api.definition.components.tool.tool import Tool
from langbot_plugin.api.entities.builtin.provider import session as provider_session

logger = logging.getLogger(__name__)


class UpdateProfile(Tool):

    async def call(
        self,
        params: dict[str, Any],
        session: provider_session.Session,
        query_id: int,
    ) -> str:
        store = self.plugin.memory_store

        field = params.get("field", "")
        action = params.get("action", "")
        value = params.get("value", "")

        if not all([field, action, value]):
            return "Error: field, action, and value are all required."

        if field not in ("name", "traits", "preferences", "notes"):
            return f"Error: invalid field '{field}'."

        if action not in ("set", "add", "remove"):
            return f"Error: invalid action '{action}'."

        user_key = await store.resolve_user_key(session)

        profile = await store.update_profile_field(
            user_key=user_key,
            field=field,
            action=action,
            value=value,
        )

        logger.info(
            "Updated profile %s.%s (%s) for %s",
            field,
            action,
            value[:40],
            user_key,
        )

        return "Profile updated.\n" + store.format_profile_text(profile)
