"""
session_manager.py — In-memory conversation session management for DocuChat.

Each session is identified by a caller-supplied session_id string. The manager
stores the last N turns of conversation history (user + assistant message pairs)
and formats them into the message list expected by the OpenAI chat API.

Design decisions:
- Pure in-memory dict: simple, zero dependencies, fast.
- Thread-safe via threading.Lock: FastAPI runs async but uses a thread pool for
  sync code; a lock ensures correctness if endpoints are called concurrently.
- Oldest turns are automatically evicted when max_history_turns is exceeded.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from config import settings


@dataclass
class Turn:
    """
    A single conversational exchange: one user message and one assistant reply.
    """
    user: str
    assistant: str


@dataclass
class Session:
    """
    Conversation state for a single session_id.

    Attributes:
        history: A bounded deque of Turn objects. When it is full, appending
                 a new turn automatically discards the oldest one (FIFO).
    """
    history: Deque[Turn] = field(
        default_factory=lambda: deque(maxlen=settings.max_history_turns)
    )

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Append a completed exchange to the session history.

        Args:
            user_message: The question or input from the user.
            assistant_message: The answer produced by the assistant.
        """
        self.history.append(Turn(user=user_message, assistant=assistant_message))

    def to_message_list(self) -> list[dict]:
        """
        Serialize history into the OpenAI messages array format.

        Returns:
            A list of dicts with 'role' and 'content' keys, interleaving
            'user' and 'assistant' roles in chronological order.
        """
        messages: list[dict] = []
        for turn in self.history:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def clear(self) -> None:
        """Remove all turns from this session's history."""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        """Number of completed turns currently stored."""
        return len(self.history)


class SessionManager:
    """
    Global registry of active chat sessions.

    Thread-safe: all public methods acquire a shared lock before mutating
    or reading the sessions dict.
    """

    def __init__(self) -> None:
        """Initialise an empty session store."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, session_id: str) -> Session:
        """
        Return the Session for session_id, creating it if it doesn't exist yet.

        Args:
            session_id: Caller-supplied identifier (UUID, username, etc.).

        Returns:
            The Session object for this session_id.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session()
            return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Record a completed user↔assistant exchange for the given session.

        Creates the session automatically if it doesn't exist.

        Args:
            session_id: Target session identifier.
            user_message: The user's question/input.
            assistant_message: The assistant's reply.
        """
        session = self.get_or_create(session_id)
        with self._lock:
            session.add_turn(user_message, assistant_message)

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return the OpenAI-formatted message list for an existing session.

        Args:
            session_id: Target session identifier.

        Returns:
            List of {"role": ..., "content": ...} dicts, oldest first.
            Returns an empty list if the session doesn't exist.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return session.to_message_list()

    def clear_session(self, session_id: str) -> None:
        """
        Wipe the history for a single session without removing the session object.

        Args:
            session_id: Target session identifier. No-op if it doesn't exist.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.clear()

    def clear_all(self) -> None:
        """
        Destroy all sessions. Called by the DELETE /reset endpoint.
        """
        with self._lock:
            self._sessions.clear()

    @property
    def session_count(self) -> int:
        """Number of active sessions currently in memory."""
        with self._lock:
            return len(self._sessions)

    @property
    def total_turns(self) -> int:
        """Sum of all turns across every active session (diagnostic use)."""
        with self._lock:
            return sum(s.turn_count for s in self._sessions.values())


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of instantiating
# ---------------------------------------------------------------------------
session_manager = SessionManager()
