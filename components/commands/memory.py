from __future__ import annotations

import logging
from typing import AsyncGenerator

from langbot_plugin.api.definition.components.command.command import Command
from langbot_plugin.api.entities.builtin.command.context import (
    ExecuteContext,
    CommandReturn,
)

logger = logging.getLogger(__name__)


class Memory(Command):

    def __init__(self):
        super().__init__()

        @self.subcommand(
            name="",
            help="Show memory overview",
            usage="!memory",
            aliases=[],
        )
        async def root(
            self, context: ExecuteContext
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            session_key, user_key, kb_id, isolation, _ = (
                await store.resolve_user_context(context.session)
            )

            profile = await store.load_profile(user_key)

            lines = [f"[Memory] mode: {isolation}"]
            lines.append(f"Session: {session_key}")
            lines.append(f"Key: {user_key}")

            has_profile = store.has_profile_data(profile)
            if has_profile:
                lines.append(
                    f"Profile: name={profile.get('name', '-')}, "
                    f"{len(profile.get('traits', []))} traits, "
                    f"{len(profile.get('preferences', []))} prefs"
                )
            else:
                lines.append("Profile: (empty)")

            if kb_id:
                lines.append(f"L2 (Episodic): KB={kb_id[:12]}... active")
            else:
                lines.append("L2 (Episodic): no KB configured")

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="profile",
            help="Show session profile",
            usage="!memory profile",
            aliases=["p"],
        )
        async def profile_cmd(
            self, context: ExecuteContext
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            user_key = await store.resolve_user_key(context.session)

            profile = await store.load_profile(user_key)
            lines = ["[Profile]"]
            lines.append(f"Name: {profile.get('name') or '(not set)'}")
            lines.append(
                f"Traits: {', '.join(profile.get('traits', [])) or '(none)'}"
            )
            lines.append(
                f"Preferences: {', '.join(profile.get('preferences', [])) or '(none)'}"
            )
            lines.append(f"Notes: {profile.get('notes') or '(none)'}")

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="search",
            help="Search episodic memories",
            usage="!memory search <query>",
            aliases=["s"],
        )
        async def search_cmd(
            self, context: ExecuteContext
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            _, user_key, kb_id, _, config = (
                await store.resolve_user_context(context.session)
            )

            if not kb_id:
                yield CommandReturn(
                    text="[Memory] No knowledge base configured. "
                    "Cannot search episodic memories."
                )
                return

            embedding_model_uuid = config.get("embedding_model_uuid", "")

            if not context.crt_params:
                yield CommandReturn(text="Usage: !memory search <query>")
                return

            query = " ".join(context.crt_params)

            episodes = await store.search_episodes(
                collection_id=kb_id,
                embedding_model_uuid=embedding_model_uuid,
                query=query,
                user_key=user_key,
                top_k=10,
            )

            if not episodes:
                yield CommandReturn(text="[Memory] No episodes found.")
                return

            lines = [f"[Memory] Found {len(episodes)} episode(s):"]
            for ep in episodes:
                ts = ep["timestamp"][:10] if ep.get("timestamp") else "?"
                imp = ep.get("importance", 2)
                tags = ", ".join(ep.get("tags", []))
                tag_str = f" [{tags}]" if tags else ""
                lines.append(f"  {ts} (imp:{imp}){tag_str} {ep['content']}")

            yield CommandReturn(text="\n".join(lines))
