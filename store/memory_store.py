from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _default_profile() -> dict[str, Any]:
    return {
        "name": "",
        "traits": [],
        "preferences": [],
        "notes": "",
        "updated_at": "",
    }


class MemoryStore:
    """Dual-layer memory store.

    L1 (Core Profile): Binary Storage (JSON) - read/write via plugin storage API.
    L2 (Episodic Memory): vector DB - read/write via plugin vector API.
    """

    _PROFILE_FIELDS = ("name", "traits", "preferences", "notes")

    def __init__(
        self,
        plugin: Any,
        max_profile_traits: int = 20,
        max_profile_preferences: int = 10,
    ):
        self.plugin = plugin
        self.max_profile_traits = max_profile_traits
        self.max_profile_preferences = max_profile_preferences
        self._kb_config_cache: dict[str, dict[str, Any]] | None = None
        # L1 profile cache: user_key -> (monotonic_timestamp, profile_dict)
        self._profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._PROFILE_CACHE_TTL = 30  # seconds

    # ======================== common helpers ========================

    @staticmethod
    def has_profile_data(profile: dict[str, Any]) -> bool:
        return any(
            profile.get(f) for f in ("name", "traits", "preferences", "notes")
        )

    @staticmethod
    def format_profile_text(profile: dict[str, Any]) -> str:
        """Compact profile text for tool return values."""
        parts = []
        if profile.get("name"):
            parts.append(f"Name: {profile['name']}")
        if profile.get("traits"):
            parts.append(f"Traits: {', '.join(profile['traits'])}")
        if profile.get("preferences"):
            parts.append(f"Preferences: {', '.join(profile['preferences'])}")
        if profile.get("notes"):
            parts.append(f"Notes: {profile['notes']}")
        return "\n".join(parts)

    # ======================== key helpers ========================

    @staticmethod
    def get_session_key(launcher_type_value: str, launcher_id: Any) -> str:
        # This key does not include platform type (qq/wechat/...). Under the
        # current deployment assumption, memory plugin instances are effectively
        # scoped to a specific bot/runtime environment, and bots are expected to
        # be separated by platform, so cross-platform collisions are not
        # considered a practical issue here.
        return f"{launcher_type_value}_{launcher_id}"

    @staticmethod
    def get_user_key(session_key: str, isolation: str) -> str:
        if isolation == "session":
            return session_key
        return "global"

    async def resolve_user_context(
        self, session: Any
    ) -> tuple[str, str, str | None, str, dict[str, Any]]:
        """Derive session_key, user_key, kb_id, isolation from a session.

        Returns (session_key, user_key, kb_id_or_None, isolation, kb_config).
        kb_id is None and kb_config is {} when no KB is configured.
        """
        kb_id, config = None, {}
        kb = await self.get_kb_config()
        if kb:
            kb_id, config = kb
        isolation = config.get("isolation", "session")
        session_key = self.get_session_key(
            session.launcher_type.value, session.launcher_id
        )
        user_key = self.get_user_key(session_key, isolation)
        return session_key, user_key, kb_id, isolation, config

    async def resolve_user_key(self, session: Any) -> str:
        """Derive user_key from a session object (lightweight, no kb_id/config)."""
        kb = await self.get_kb_config()
        isolation = kb[1].get("isolation", "session") if kb else "session"
        session_key = self.get_session_key(
            session.launcher_type.value, session.launcher_id
        )
        return self.get_user_key(session_key, isolation)

    # ======================== KB config persistence ========================

    _KB_CONFIGS_KEY = "kb_configs"

    async def save_kb_config(self, kb_id: str, config: dict[str, Any]) -> None:
        configs = await self._read_json(self._KB_CONFIGS_KEY) or {}
        configs[kb_id] = config
        await self._write_json(self._KB_CONFIGS_KEY, configs)
        self._kb_config_cache = configs

    async def remove_kb_config(self, kb_id: str) -> None:
        configs = await self._read_json(self._KB_CONFIGS_KEY) or {}
        configs.pop(kb_id, None)
        await self._write_json(self._KB_CONFIGS_KEY, configs)
        self._kb_config_cache = configs

    async def get_kb_configs(self) -> dict[str, dict[str, Any]]:
        if self._kb_config_cache is not None:
            return self._kb_config_cache
        self._kb_config_cache = await self._read_json(self._KB_CONFIGS_KEY) or {}
        return self._kb_config_cache

    async def get_kb_config(self) -> tuple[str, dict[str, Any]] | None:
        """Return (kb_id, config) for the single registered KB, or None."""
        configs = await self.get_kb_configs()
        if configs:
            kb_id = next(iter(configs))
            return kb_id, configs[kb_id]
        return None

    # ======================== L1: profile (Binary Storage) ========================

    def _profile_key(self, user_key: str) -> str:
        return f"p:{user_key}"

    async def _read_json(self, key: str) -> Any:
        try:
            data = await self.plugin.get_plugin_storage(key)
        except Exception:
            logger.debug("storage key %s not found", key)
            return None
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("storage key %s has corrupted data: %s", key, e)
            return None

    async def _write_json(self, key: str, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        await self.plugin.set_plugin_storage(key, data)

    async def load_profile(self, user_key: str) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._profile_cache.get(user_key)
        if cached and now - cached[0] < self._PROFILE_CACHE_TTL:
            return cached[1]
        key = self._profile_key(user_key)
        profile = await self._read_json(key)
        profile = profile if profile else _default_profile()
        self._profile_cache[user_key] = (now, profile)
        return profile

    async def save_profile(self, user_key: str, profile: dict[str, Any]) -> None:
        profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        key = self._profile_key(user_key)
        await self._write_json(key, profile)
        self._profile_cache[user_key] = (time.monotonic(), profile)

    async def update_profile_field(
        self, user_key: str, field: str, action: str, value: str
    ) -> dict[str, Any]:
        profile = await self.load_profile(user_key)

        if field == "name":
            profile["name"] = value
        elif field in ("traits", "preferences"):
            items: list[str] = profile.get(field, [])
            if action == "add":
                if value not in items:
                    items.append(value)
                    max_len = self.max_profile_traits if field == "traits" else self.max_profile_preferences
                    items = items[-max_len:]
                profile[field] = items
            elif action == "remove":
                profile[field] = [i for i in items if i != value]
            elif action == "set":
                profile[field] = [value]
        elif field == "notes":
            if action == "set":
                profile["notes"] = value
            elif action == "add":
                existing = profile.get("notes", "")
                profile["notes"] = f"{existing}; {value}" if existing else value
            elif action == "remove":
                profile["notes"] = ""

        await self.save_profile(user_key, profile)
        return profile

    async def clear_profile(self, user_key: str) -> None:
        await self.save_profile(user_key, _default_profile())

    # ======================== L2: episodes (ChromaDB vector) ========================

    async def add_episode(
        self,
        collection_id: str,
        embedding_model_uuid: str,
        user_key: str,
        content: str,
        tags: list[str] | None = None,
        importance: int = 2,
        source: str = "agent",
        sender_id: str = "",
        sender_name: str = "",
    ) -> dict[str, Any]:
        """Store an episodic memory into vector DB."""
        episode_id = uuid.uuid4().hex[:12]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        importance = max(1, min(5, importance))
        tags = tags or []

        metadata = {
            "content": content,
            "tags": ",".join(tags),
            "importance": str(importance),
            "timestamp": timestamp,
            "user_key": user_key,
            "source": source,
            "sender_id": sender_id,
            "sender_name": sender_name,
        }

        vectors = await self.plugin.invoke_embedding(embedding_model_uuid, [content])

        await self.plugin.vector_upsert(
            collection_id=collection_id,
            vectors=vectors,
            ids=[episode_id],
            metadata=[metadata],
            documents=[content],
        )

        return {
            "id": episode_id,
            "content": content,
            "tags": tags,
            "importance": importance,
            "timestamp": timestamp,
        }

    async def search_episodes(
        self,
        collection_id: str,
        embedding_model_uuid: str,
        query: str,
        user_key: str | None = None,
        top_k: int = 5,
        sender_id: str = "",
        sender_name: str = "",
        time_after: str = "",
        time_before: str = "",
        importance_min: int | None = None,
        source: str = "",
    ) -> list[dict[str, Any]]:
        """Search episodic memories via vector similarity."""
        if not query.strip():
            return []

        vectors = await self.plugin.invoke_embedding(embedding_model_uuid, [query])
        query_vector = vectors[0]

        filters = {}
        if user_key:
            filters["user_key"] = user_key
        if sender_id:
            filters["sender_id"] = sender_id
        if sender_name:
            filters["sender_name"] = sender_name
        if source:
            filters["source"] = source
        if time_after or time_before:
            time_filter: dict[str, str] = {}
            if time_after:
                time_filter["$gte"] = time_after
            if time_before:
                time_filter["$lte"] = time_before
            filters["timestamp"] = time_filter
        if importance_min is not None:
            filters["importance"] = {"$gte": str(importance_min)}

        results = await self.plugin.vector_search(
            collection_id=collection_id,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters if filters else None,
        )

        episodes = []
        for r in results:
            meta = r.get("metadata", {})
            episodes.append({
                "id": r.get("id", ""),
                "content": meta.get("content", ""),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "importance": int(meta.get("importance", "2")),
                "timestamp": meta.get("timestamp", ""),
                "sender_id": meta.get("sender_id", ""),
                "sender_name": meta.get("sender_name", ""),
                "source": meta.get("source", ""),
                "score": r.get("score"),
            })
        return episodes

    async def delete_episodes_by_user(
        self, collection_id: str, user_key: str
    ) -> int:
        """Delete all episodes for a user_key."""
        return await self.plugin.vector_delete(
            collection_id=collection_id,
            filters={"user_key": user_key},
        )

    # ======================== formatting ========================

    @staticmethod
    def format_profile_prompt(profile: dict[str, Any]) -> str:
        if not MemoryStore.has_profile_data(profile):
            return ""

        parts: list[str] = []
        parts.append("## Memory (Profile)")

        if profile.get("name"):
            parts.append(f"- Name: {profile['name']}")
        if profile.get("traits"):
            parts.append(f"- Traits: {', '.join(profile['traits'])}")
        if profile.get("preferences"):
            parts.append(f"- Preferences: {', '.join(profile['preferences'])}")
        if profile.get("notes"):
            parts.append(f"- Notes: {profile['notes']}")
        if profile.get("updated_at"):
            parts.append(f"- Last updated: {profile['updated_at']}")

        return "\n".join(parts)
