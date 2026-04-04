"""
Cognitive Sidecar Middleware — Makes memory always-on for any LLM integration.

Instead of requiring the LLM to "decide" when to use memory (tool calls),
this middleware automatically:
1. Pre-flight: Recalls relevant memories BEFORE every LLM call
2. Post-flight: Learns from EVERY conversation exchange AFTER the LLM responds

Usage:
    from memgraph_sdk import MemgraphClient
    from memgraph_sdk.middleware import CognitiveSidecar

    client = MemgraphClient(api_key="mg_...")
    sidecar = CognitiveSidecar(client, user_id="user_123")

    # Wrap your LLM call
    messages = [{"role": "user", "content": "Help me optimize this query"}]

    # Pre-flight: Auto-recall memories and inject into messages
    enriched = sidecar.pre_flight(messages)

    # Your LLM call (OpenAI, Anthropic, etc.)
    response = openai.chat.completions.create(messages=enriched, ...)

    # Post-flight: Auto-learn from the exchange
    full_exchange = enriched + [{"role": "assistant", "content": response.choices[0].message.content}]
    sidecar.post_flight(full_exchange)

    # Or use the all-in-one wrapper:
    enriched, post_flight_fn = sidecar.wrap(messages)
    response = your_llm_call(enriched)
    post_flight_fn(response_text)
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CognitiveSidecar:
    """
    Always-on memory middleware for any LLM integration.

    The sidecar sits between your application and the LLM, automatically:
    - Injecting relevant memories as system context (pre-flight)
    - Extracting learnable signals from every exchange (post-flight)

    This means the LLM ALWAYS has relevant context without needing tool calls,
    and EVERY conversation teaches the memory system something new.
    """

    def __init__(
        self,
        client,  # MemgraphClient instance
        user_id: str,
        agent_id: str = "sidecar",
        thread_id: Optional[str] = None,
        token_budget: int = 4000,
        include_profile: bool = True,
        include_prospective: bool = True,
        auto_learn: bool = True,
    ):
        """
        Initialize the Cognitive Sidecar.

        Args:
            client: MemgraphClient instance (configured with API key)
            user_id: The user ID for memory scoping
            agent_id: Agent identifier (default: "sidecar")
            thread_id: Optional thread ID for conversation scoping
            token_budget: Max tokens for memory context injection
            include_profile: Whether to include user profile in context
            include_prospective: Whether to auto-surface deadlines/reminders
            auto_learn: Whether to automatically learn from exchanges
        """
        self.client = client
        self.user_id = user_id
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.token_budget = token_budget
        self.include_profile = include_profile
        self.include_prospective = include_prospective
        self.auto_learn = auto_learn
        self._last_memory_context = ""
        self._last_memories_found = 0

    def pre_flight(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Auto-recall: Fetch relevant memories and inject them into the message list.

        Takes the current messages and returns an enriched version with
        memory context injected as the first system message.

        Args:
            messages: List of {"role": ..., "content": ...} dicts

        Returns:
            Enriched messages with memory context prepended as system message
        """
        # Find the latest user message
        latest_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_user_msg = msg.get("content", "")
                break

        if not latest_user_msg:
            return messages

        try:
            result = self._call_pre_flight(latest_user_msg)
            self._last_memory_context = result.get("memory_context", "")
            self._last_memories_found = result.get("memories_found", 0)

            if not self._last_memory_context:
                return messages

            # Build memory system message
            memory_system_msg = {
                "role": "system",
                "content": self._build_memory_injection(result),
            }

            # Inject as the first system message
            enriched = [memory_system_msg] + list(messages)
            logger.info(
                "Pre-flight: injected %d memories for user %s",
                self._last_memories_found, self.user_id,
            )
            return enriched

        except Exception as e:
            logger.warning("Pre-flight failed (continuing without memory): %s", e)
            return messages

    def post_flight(self, messages: List[Dict[str, str]]) -> None:
        """
        Auto-learn: Extract learnable signals from the conversation exchange.

        Call this AFTER the LLM has responded. Runs in a background thread
        so it doesn't block your application.

        Args:
            messages: Full exchange including user messages and assistant responses
        """
        if not self.auto_learn:
            return

        if not messages or len(messages) < 2:
            return

        # Run in background thread to avoid blocking
        thread = threading.Thread(
            target=self._call_post_flight,
            args=(messages,),
            daemon=True,
        )
        thread.start()
        logger.info("Post-flight: background learning triggered for user %s", self.user_id)

    def wrap(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Callable[[str], None]]:
        """
        All-in-one wrapper: pre-flight + returns a post-flight callback.

        Usage:
            enriched, done = sidecar.wrap(messages)
            response = llm.chat(enriched)
            done(response.content)  # Triggers background learning

        Args:
            messages: Current conversation messages

        Returns:
            Tuple of (enriched_messages, post_flight_callback)
        """
        enriched = self.pre_flight(messages)

        def post_flight_callback(assistant_response: str):
            full_exchange = list(enriched) + [
                {"role": "assistant", "content": assistant_response}
            ]
            self.post_flight(full_exchange)

        return enriched, post_flight_callback

    def process(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Combined pre-flight + post-flight in one call using /v1/sidecar/process.

        Enriches the current messages with memory context AND triggers
        background learning from completed exchanges in the history.

        Args:
            messages: Full conversation messages

        Returns:
            Enriched messages with memory context
        """
        try:
            result = self._call_process(messages)
            self._last_memory_context = result.get("memory_context", "")
            self._last_memories_found = result.get("memories_found", 0)

            if not self._last_memory_context:
                return messages

            memory_system_msg = {
                "role": "system",
                "content": self._build_memory_injection(result),
            }

            enriched = [memory_system_msg] + list(messages)
            logger.info(
                "Process: injected %d memories, learning_triggered=%s",
                self._last_memories_found, result.get("learning_triggered", False),
            )
            return enriched

        except Exception as e:
            logger.warning("Process failed (continuing without memory): %s", e)
            return messages

    @property
    def last_context(self) -> str:
        """Get the last memory context that was injected."""
        return self._last_memory_context

    @property
    def memories_found(self) -> int:
        """Get the count of memories found in the last pre-flight."""
        return self._last_memories_found

    # ── Private helpers ──────────────────────────────────

    def _call_pre_flight(self, message: str) -> Dict[str, Any]:
        """Call the /v1/sidecar/pre-flight endpoint."""
        payload = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "message": message,
            "token_budget": self.token_budget,
            "include_profile": self.include_profile,
            "include_prospective": self.include_prospective,
        }
        if self.thread_id:
            payload["thread_id"] = self.thread_id

        return self.client.sidecar_pre_flight(
            message=payload.get("message", ""),
            user_id=self.user_id,
            thread_id=self.thread_id,
            agent_id=self.agent_id,
            token_budget=self.token_budget,
        )

    def _call_post_flight(self, messages: List[Dict[str, str]]) -> None:
        """Call the /v1/sidecar/post-flight endpoint (sync, for background thread)."""
        try:
            self.client.sidecar_post_flight(
                messages=messages,
                user_id=self.user_id,
                agent_id=self.agent_id,
                thread_id=self.thread_id,
            )
        except Exception as e:
            logger.warning("Post-flight API call failed: %s", e)

    def _call_process(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the /v1/sidecar/process endpoint."""
        return self.client.sidecar_process(
            messages=messages,
            user_id=self.user_id,
            agent_id=self.agent_id,
            thread_id=self.thread_id,
        )

    def _build_memory_injection(self, result: Dict[str, Any]) -> str:
        """Build the system message content for memory injection."""
        parts = [
            "=== MEMGRAPH MEMORY CONTEXT (Auto-injected) ===",
            "The following memories are relevant to this conversation.",
            "USE them naturally — reference the user's name, preferences, and past context.",
            "Do NOT say 'based on my memory' — just use the information as if you know it.",
            "",
        ]

        memory_context = result.get("memory_context", "")
        if memory_context:
            parts.append(memory_context)

        # Add profile summary if available
        profile = result.get("profile")
        if profile:
            tenets = profile.get("tenets", [])
            if tenets:
                parts.append("")
                parts.append("RULES YOU MUST FOLLOW:")
                for t in tenets[:5]:
                    parts.append(f"  - {t.get('key', '')}: {t.get('value', '')}")

        # Add prospective items
        prospective = result.get("prospective_items", [])
        if prospective:
            parts.append("")
            parts.append("PROACTIVE REMINDERS (mention these if relevant):")
            for p in prospective:
                parts.append(f"  - {p.get('content', '')}")

        parts.append("")
        parts.append("=== END MEMORY CONTEXT ===")

        return "\n".join(parts)
