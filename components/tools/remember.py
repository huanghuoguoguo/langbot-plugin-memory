from __future__ import annotations

import logging
from typing import Any

from langbot_plugin.api.definition.components.tool.tool import Tool
from langbot_plugin.api.entities.builtin.provider import session as provider_session

logger = logging.getLogger(__name__)


class Remember(Tool):

    async def call(
        self,
        params: dict[str, Any],
        session: provider_session.Session,
        query_id: int,
    ) -> str:
        store = self.plugin.memory_store
        session_key, user_key, kb_id, isolation, config = (
            await store.resolve_user_context(session)
        )

        if not kb_id:
            return "Error: no memory knowledge base configured. Create one first."

        embedding_model_uuid = config.get("embedding_model_uuid", "")
        if not embedding_model_uuid:
            return "Error: no embedding model configured in knowledge base."

        content = params.get("content", "")
        if not content:
            return "Error: content is required."

        tags = params.get("tags", [])
        importance = params.get("importance", 2)

        episode = await store.add_episode(
            collection_id=kb_id,
            embedding_model_uuid=embedding_model_uuid,
            user_key=user_key,
            content=content,
            tags=tags,
            importance=importance,
        )

        logger.info(
            "Remembered [%s] for %s: %s",
            episode["id"],
            user_key,
            content[:80],
        )

        return f"Remembered: {content}"
