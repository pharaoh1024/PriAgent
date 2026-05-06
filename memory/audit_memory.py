"""
Cross-session audit memory.

Improvement over the paper: the original PriAgent treats each app in isolation.
This module adds a persistent memory layer that:
1. Records confirmed violations with their data types and app categories.
2. Summarises recurring patterns across sessions for contextual priming.
3. Provides a "pattern hint" to the SemanticVerifier, nudging it toward
   violation types that have appeared frequently — improving recall.

Implementation uses LangChain's ConversationSummaryBufferMemory for the
in-session context window, backed by a simple JSON file for persistence.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

import tiktoken
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import ANTHROPIC_API_KEY, LLM_MODEL, MEMORY_DB_PATH


class _SummaryBufferMemory:
    """
    Lightweight ConversationSummaryBuffer replacement compatible with LangChain 1.x.
    Keeps the last N messages verbatim; when the token budget is exceeded it
    calls the LLM once to compress the oldest half into a summary and retains
    only the summary + the recent tail.
    """

    def __init__(self, llm: ChatAnthropic, max_token_limit: int = 2000) -> None:
        self._llm = llm
        self._max_tokens = max_token_limit
        self._messages: List[BaseMessage] = []
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self) -> int:
        return sum(len(self._enc.encode(m.content)) for m in self._messages)

    def _summarise(self) -> None:
        half = len(self._messages) // 2
        to_compress = self._messages[:half]
        self._messages = self._messages[half:]
        text = "\n".join(f"{m.type}: {m.content}" for m in to_compress)
        summary = self._llm.invoke(
            f"Summarise this audit dialogue in ≤3 sentences:\n{text}"
        ).content
        self._messages.insert(0, AIMessage(content=f"[Summary of prior steps]: {summary}"))

    def save_context(self, human_text: str, ai_text: str) -> None:
        self._messages.append(HumanMessage(content=human_text))
        self._messages.append(AIMessage(content=ai_text))
        while self._count_tokens() > self._max_tokens and len(self._messages) > 2:
            self._summarise()

    def load_as_string(self) -> str:
        if not self._messages:
            return "No prior steps in this session."
        parts = []
        for m in self._messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            parts.append(f"{role}: {m.content}")
        return "\n".join(parts)


class AuditMemory:
    """
    Two-layer memory:
    - *session_memory*: custom SummaryBufferMemory for the current analysis
      session (auto-compresses when token budget is exceeded).
    - *persistent_db*: JSON file recording confirmed violations across all sessions,
      used to surface pattern hints in future runs.
    """

    def __init__(self, max_token_limit: int = 2000) -> None:
        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        self.session_memory = _SummaryBufferMemory(llm=llm, max_token_limit=max_token_limit)
        self._db_path = MEMORY_DB_PATH
        self._db: Dict[str, Any] = self._load_db()

    # ── persistence helpers ─────────────────────────────────────────────

    def _load_db(self) -> Dict[str, Any]:
        if self._db_path.exists():
            try:
                return json.loads(self._db_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"sessions": [], "violation_patterns": {}}

    def _save_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.write_text(
            json.dumps(self._db, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── session-level API ────────────────────────────────────────────────

    def record_agent_step(self, agent_name: str, summary: str) -> None:
        """Add a single agent's result to the rolling session context."""
        self.session_memory.save_context(
            f"[{agent_name}] completed",
            summary,
        )

    def get_session_context(self) -> str:
        """Return the (possibly summarised) session history as a string for prompt injection."""
        return self.session_memory.load_as_string()

    # ── cross-session persistence ────────────────────────────────────────

    def persist_session(
        self,
        app_name: str,
        session_id: str,
        violations: List[Dict],
    ) -> None:
        """Store confirmed violations from the current session into the JSON DB."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "app_name": app_name,
            "violations": violations,
        }
        self._db["sessions"].append(entry)

        # Update pattern counters
        patterns = self._db.setdefault("violation_patterns", {})
        for v in violations:
            for dtype in v.get("data_types_involved", []):
                patterns[dtype] = patterns.get(dtype, 0) + 1

        self._save_db()

    def get_pattern_hint(self, top_n: int = 3) -> str:
        """
        Return a short text hint summarising the most common violation types
        seen across all previous sessions.  Injected into the SemanticVerifier
        prompt to bias attention toward historically frequent violations.
        """
        patterns = self._db.get("violation_patterns", {})
        if not patterns:
            return "No prior audit patterns available."
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        top = sorted_patterns[:top_n]
        lines = [f"  - {dtype}: {count} confirmed violations" for dtype, count in top]
        return "Most frequent violation types from prior audits:\n" + "\n".join(lines)

    def total_sessions_audited(self) -> int:
        return len(self._db.get("sessions", []))
