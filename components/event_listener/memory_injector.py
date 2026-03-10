from __future__ import annotations

import logging

from langbot_plugin.api.definition.components.common.event_listener import EventListener
from langbot_plugin.api.entities import events, context
from langbot_plugin.api.entities.builtin.provider.message import Message

logger = logging.getLogger(__name__)


class MemoryInjector(EventListener):
    """L1 profile injector.

    Injects user core profile into system prompt via PromptPreProcessing.
    L2 episodic memory is handled separately by KnowledgeEngine.retrieve().
    """

    def __init__(self):
        super().__init__()

        @self.handler(events.PromptPreProcessing)
        async def on_prompt_preprocess(event_ctx: context.EventContext):
            try:
                await self._inject_profile(event_ctx)
            except Exception:
                logger.exception("Failed to inject profile")

    async def _inject_profile(self, event_ctx: context.EventContext) -> None:
        store = self.plugin.memory_store
        session_name: str = event_ctx.event.session_name

        kb = await store.get_kb_config()
        isolation = kb[1].get("isolation", "session") if kb else "session"

        user_key = store.get_user_key(session_name, isolation)
        profile = await store.load_profile(user_key)
        profile_block = store.format_profile_prompt(profile)

        # Build injection parts
        blocks: list[str] = []
        if profile_block.strip():
            blocks.append(profile_block)

        # Inject current speaker identity so LLM knows who is talking
        query = event_ctx.event.query
        if query:
            sender_name = query.variables.get("sender_name", "")
            sender_id = query.sender_id
            if sender_name:
                blocks.append(f"## Current Speaker\n- Name: {sender_name}\n- ID: {sender_id}")
            elif sender_id:
                blocks.append(f"## Current Speaker\n- ID: {sender_id}")

        if not blocks:
            return

        injection = "# Long-term Memory\n\n" + "\n\n".join(blocks)

        event_ctx.event.default_prompt.append(
            Message(role="system", content=injection)
        )
