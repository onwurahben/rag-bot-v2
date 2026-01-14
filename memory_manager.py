from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import uuid
import json
import os

PERSISTENT_MEMORY_FILE = "persistent_chat_memory.json"
SESSION_STORE = {}  # in-memory runtime chat histories


def load_persistent_store():
    """Load persistent JSON memory store safely."""
    if not os.path.exists(PERSISTENT_MEMORY_FILE):
        return {}

    try:
        with open(PERSISTENT_MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception:
        # corrupted JSON file → reset
        return {}


def save_persistent_store(data):
    """Write persistent JSON memory store."""
    with open(PERSISTENT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_session_history(session_id: str, persistent=False):
    """
    Returns ChatMessageHistory for session.
    Supports both runtime-only and persistent modes.
    """
    global SESSION_STORE

    if persistent:
        data = load_persistent_store()
        if session_id not in data:
            data[session_id] = []
            save_persistent_store(data)

        # hydrate runtime memory if not already loaded
        if session_id not in SESSION_STORE:
            history = ChatMessageHistory()
            for item in data[session_id]:
                history.add_user_message(item.get("user", ""))
                history.add_ai_message(item.get("bot", ""))
            SESSION_STORE[session_id] = history

    # non-persistent → runtime-only memory
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()

    return SESSION_STORE[session_id]


def wrap_chain_with_memory(chain, session_id=None, persistent=False):
    """
    Wraps LCEL chain with RunnableWithMessageHistory.
    Returns runnable and session_id.
    """
    if session_id is None:
        session_id = f"session-{uuid.uuid4()}"

    def _get_history(_session_id):
        return get_session_history(_session_id, persistent=persistent)

    runnable = RunnableWithMessageHistory(
        chain,
        _get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return runnable, session_id


def save_persistent_memory(session_id, user_msg, bot_msg):
    """Adds messages to persistent JSON safely."""
    data = load_persistent_store()

    if session_id not in data:
        data[session_id] = []

    data[session_id].append({"user": user_msg, "bot": bot_msg})
    save_persistent_store(data)
