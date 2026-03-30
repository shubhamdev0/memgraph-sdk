"""
Memgraph adapter for the OpenAI Agents SDK.

Provides zero-config memory integration for agents built with `openai-agents`.
Memory is always-on: context is injected before every LLM call, decisions and
tool usage are captured automatically, and outcomes are recorded at run end.

Usage (minimal):
    from memgraph_sdk import MemgraphClient
    from memgraph_sdk.openai_agents import MemgraphAgentHooks, memgraph_instructions

    client = MemgraphClient(api_key="mg_...")

    agent = Agent(
        name="SupportAgent",
        instructions=memgraph_instructions(client, user_id="u_123",
                                           base="You are a helpful support agent."),
        hooks=MemgraphAgentHooks(client, user_id="u_123"),
    )

Usage (run-level hooks for multi-agent):
    from memgraph_sdk.openai_agents import MemgraphRunHooks

    result = await Runner.run(
        agent,
        "Help me book a flight",
        hooks=MemgraphRunHooks(client, user_id="u_123"),
    )

Usage (memory tools — let the agent explicitly search/store):
    from memgraph_sdk.openai_agents import memgraph_tools

    agent = Agent(
        name="ResearchAgent",
        tools=memgraph_tools(client, user_id="u_123"),
    )
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dynamic Instructions (memory-enriched system prompt)
# ──────────────────────────────────────────────────────────────────────

def memgraph_instructions(
    client,
    user_id: str,
    *,
    base: str = "You are a helpful assistant.",
    agent_id: str = "agent",
    token_budget: int = 4000,
) -> Callable:
    """
    Returns a dynamic instructions function that auto-injects memory context.

    The OpenAI Agents SDK calls `instructions(context, agent)` before every
    LLM call. This function fetches relevant memories and prepends them to
    the base system prompt.

    Args:
        client: MemgraphClient instance
        user_id: User ID for memory scoping
        base: Base system prompt
        agent_id: Agent identifier
        token_budget: Max tokens for memory context

    Returns:
        A callable compatible with Agent(instructions=...)
    """
    def _dynamic_instructions(context, agent):
        try:
            result = client.sidecar_pre_flight(
                message=_extract_latest_input(context),
                user_id=user_id,
                agent_id=agent_id,
                token_budget=token_budget,
            )
            memory_context = result.get("memory_context", "")
            if memory_context:
                return f"{memory_context}\n\n---\n\n{base}"
        except Exception as e:
            logger.debug("Memgraph pre-flight failed (continuing without memory): %s", e)

        return base

    return _dynamic_instructions


# ──────────────────────────────────────────────────────────────────────
# Agent-Level Hooks (per-agent decision capture)
# ──────────────────────────────────────────────────────────────────────

class MemgraphAgentHooks:
    """
    AgentHooks implementation that automatically captures decisions.

    Attaches to a single agent and records:
    - Tool calls (on_tool_start / on_tool_end)
    - Reasoning steps (on_llm_end)
    - Decision outcome (on_end)

    Usage:
        agent = Agent(
            name="BookingAgent",
            hooks=MemgraphAgentHooks(client, user_id="u_123"),
        )
    """

    def __init__(
        self,
        client,
        user_id: str,
        agent_id: str = "agent",
        auto_learn: bool = True,
    ):
        self.client = client
        self.user_id = user_id
        self.agent_id = agent_id
        self.auto_learn = auto_learn

        # Per-run state (reset on_start)
        self._tools: List[Dict] = []
        self._steps: List[Dict] = []
        self._goal: str = ""
        self._start_time: float = 0

    async def on_start(self, context, agent):
        """Called when this agent starts processing."""
        self._tools = []
        self._steps = []
        self._goal = _extract_latest_input(context)
        self._start_time = time.monotonic()

    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked."""
        pass  # We capture on tool_end with the result

    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool completes."""
        self._tools.append({
            "tool_name": getattr(tool, "name", str(tool)),
            "tool_output": _safe_truncate(str(result)),
        })

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        """Called before an LLM call."""
        pass

    async def on_llm_end(self, context, agent, response):
        """Called after an LLM responds."""
        # Extract assistant output as a reasoning step
        output_text = _extract_response_text(response)
        if output_text:
            self._steps.append({
                "step": len(self._steps) + 1,
                "description": _safe_truncate(output_text, 500),
            })

    async def on_handoff(self, context, agent, source):
        """Called when another agent hands off to this one."""
        self._steps.append({
            "step": len(self._steps) + 1,
            "description": f"Handoff from {getattr(source, 'name', 'unknown')}",
        })

    async def on_end(self, context, agent, output):
        """Called when this agent finishes. Records the decision."""
        if not self._goal:
            return

        try:
            elapsed = time.monotonic() - self._start_time
            outcome_text = _safe_truncate(str(output), 1000) if output else ""

            self.client.record_decision(
                goal=_safe_truncate(self._goal, 500),
                reasoning_steps=self._steps,
                tools_used=self._tools,
                confidence=None,
                outcome="SUCCESS",
                outcome_assessment=outcome_text,
                agent_id=self.agent_id,
                user_id=self.user_id,
                create_snapshot=True,
            )
            logger.info(
                "Memgraph: decision captured (goal='%s', tools=%d, steps=%d, %.1fs)",
                self._goal[:50], len(self._tools), len(self._steps), elapsed,
            )
        except Exception as e:
            logger.debug("Memgraph: failed to record decision: %s", e)

        # Post-flight: auto-learn from the exchange
        if self.auto_learn:
            self._trigger_post_flight(output)

    def _trigger_post_flight(self, output):
        """Trigger background learning from the exchange."""
        try:
            messages = []
            if self._goal:
                messages.append({"role": "user", "content": self._goal})
            if output:
                messages.append({"role": "assistant", "content": str(output)})
            if len(messages) >= 2:
                self.client.sidecar_post_flight(
                    messages=messages,
                    user_id=self.user_id,
                )
        except Exception as e:
            logger.debug("Memgraph: post-flight failed: %s", e)


# ──────────────────────────────────────────────────────────────────────
# Run-Level Hooks (multi-agent orchestration)
# ──────────────────────────────────────────────────────────────────────

class MemgraphRunHooks:
    """
    RunHooks implementation for multi-agent runs.

    Captures the entire run as a single decision, tracking which agents
    participated, which tools were used, and the final outcome.

    Usage:
        result = await Runner.run(
            agent, "user query",
            hooks=MemgraphRunHooks(client, user_id="u_123"),
        )
    """

    def __init__(
        self,
        client,
        user_id: str,
        agent_id: str = "orchestrator",
        auto_learn: bool = True,
    ):
        self.client = client
        self.user_id = user_id
        self.agent_id = agent_id
        self.auto_learn = auto_learn

        # Per-run state
        self._tools: List[Dict] = []
        self._steps: List[Dict] = []
        self._agents_used: List[str] = []
        self._goal: str = ""
        self._start_time: float = 0

    async def on_agent_start(self, context, agent):
        """Called when any agent in the run starts."""
        agent_name = getattr(agent, "name", "unknown")
        self._agents_used.append(agent_name)

        if not self._goal:
            self._goal = _extract_latest_input(context)
            self._start_time = time.monotonic()

        self._steps.append({
            "step": len(self._steps) + 1,
            "description": f"Agent '{agent_name}' started",
        })

    async def on_agent_end(self, context, agent, output):
        """Called when any agent in the run completes."""
        agent_name = getattr(agent, "name", "unknown")
        output_summary = _safe_truncate(str(output), 200) if output else ""
        self._steps.append({
            "step": len(self._steps) + 1,
            "description": f"Agent '{agent_name}' finished: {output_summary}",
        })

    async def on_tool_start(self, context, agent, tool):
        pass

    async def on_tool_end(self, context, agent, tool, result):
        agent_name = getattr(agent, "name", "unknown")
        tool_name = getattr(tool, "name", str(tool))
        self._tools.append({
            "tool_name": f"{agent_name}/{tool_name}",
            "tool_output": _safe_truncate(str(result)),
        })

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        pass

    async def on_llm_end(self, context, agent, response):
        pass

    async def on_handoff(self, context, from_agent, to_agent):
        from_name = getattr(from_agent, "name", "unknown")
        to_name = getattr(to_agent, "name", "unknown")
        self._steps.append({
            "step": len(self._steps) + 1,
            "description": f"Handoff: {from_name} → {to_name}",
        })

    # RunHooks doesn't have on_end, but we provide a manual finalize method
    def finalize(self, output: Any = None, outcome: str = "SUCCESS"):
        """
        Call after Runner.run() completes to record the full decision.

        Usage:
            result = await Runner.run(agent, "query", hooks=hooks)
            hooks.finalize(result.final_output)
        """
        if not self._goal:
            return

        try:
            elapsed = time.monotonic() - self._start_time
            outcome_text = _safe_truncate(str(output), 1000) if output else ""

            self.client.record_decision(
                goal=_safe_truncate(self._goal, 500),
                reasoning_steps=self._steps,
                tools_used=self._tools,
                confidence=None,
                outcome=outcome,
                outcome_assessment=outcome_text,
                agent_id=self.agent_id,
                user_id=self.user_id,
                create_snapshot=True,
            )
            logger.info(
                "Memgraph: run decision captured (goal='%s', agents=%s, tools=%d, %.1fs)",
                self._goal[:50], self._agents_used, len(self._tools), elapsed,
            )
        except Exception as e:
            logger.debug("Memgraph: failed to record run decision: %s", e)

        if self.auto_learn:
            try:
                messages = []
                if self._goal:
                    messages.append({"role": "user", "content": self._goal})
                if output:
                    messages.append({"role": "assistant", "content": str(output)})
                if len(messages) >= 2:
                    self.client.sidecar_post_flight(messages=messages, user_id=self.user_id)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────
# Memory Tools (explicit memory access for agents)
# ──────────────────────────────────────────────────────────────────────

def memgraph_tools(client, user_id: str, agent_id: str = "agent") -> list:
    """
    Returns a list of function_tool-decorated tools that give the agent
    explicit access to Memgraph memory operations.

    Tools provided:
    - search_memory: Search for relevant memories
    - store_memory: Store a new memory/belief

    Usage:
        agent = Agent(
            name="ResearchAgent",
            tools=memgraph_tools(client, user_id="u_123"),
        )
    """
    try:
        from agents import function_tool
    except ImportError:
        raise ImportError(
            "openai-agents is required for memgraph_tools. "
            "Install it with: pip install openai-agents"
        )

    @function_tool(name_override="search_memory", description_override=(
        "Search the user's long-term memory for relevant information. "
        "Use this to recall past conversations, preferences, facts, and decisions."
    ))
    def search_memory(query: str) -> str:
        """Search memory for relevant context."""
        try:
            result = client.search(query=query, user_id=user_id, agent_id=agent_id)
            items = result.get("retrieved_items", [])
            if not items:
                return "No relevant memories found."

            lines = []
            for item in items[:10]:
                content = item.get("content", "")
                score = item.get("score", 0)
                if score > 0.5:
                    lines.append(f"- {content}")

            return "\n".join(lines) if lines else "No relevant memories found."
        except Exception as e:
            logger.debug("search_memory failed: %s", e)
            return "Memory search temporarily unavailable."

    @function_tool(name_override="store_memory", description_override=(
        "Store an important fact, preference, or decision in the user's long-term memory. "
        "Use this when you learn something worth remembering about the user."
    ))
    def store_memory(text: str, category: str = "general") -> str:
        """Store a memory."""
        try:
            client.remember(text=text, user_id=user_id, category=category)
            return f"Remembered: {text}"
        except Exception as e:
            logger.debug("store_memory failed: %s", e)
            return "Failed to store memory."

    return [search_memory, store_memory]


# ──────────────────────────────────────────────────────────────────────
# Convenience: All-in-one agent factory
# ──────────────────────────────────────────────────────────────────────

def create_memgraph_agent(
    client,
    user_id: str,
    *,
    name: str = "Agent",
    instructions: str = "You are a helpful assistant with long-term memory.",
    agent_id: str = "agent",
    tools: list = None,
    include_memory_tools: bool = False,
    token_budget: int = 4000,
    **agent_kwargs,
):
    """
    Create an OpenAI Agent with Memgraph memory fully wired in.

    This is the simplest way to add memory to an agent:

        from memgraph_sdk import MemgraphClient
        from memgraph_sdk.openai_agents import create_memgraph_agent

        client = MemgraphClient(api_key="mg_...")
        agent = create_memgraph_agent(client, user_id="u_123", name="SupportBot")
        result = await Runner.run(agent, "Hello!")

    Args:
        client: MemgraphClient instance
        user_id: User ID for memory scoping
        name: Agent name
        instructions: Base system prompt (memory context is auto-prepended)
        agent_id: Agent identifier for decision tracking
        tools: Additional tools for the agent
        include_memory_tools: Whether to add search_memory/store_memory tools
        token_budget: Max tokens for memory context injection
        **agent_kwargs: Additional kwargs passed to Agent()
    """
    try:
        from agents import Agent
    except ImportError:
        raise ImportError(
            "openai-agents is required. Install with: pip install openai-agents"
        )

    all_tools = list(tools or [])
    if include_memory_tools:
        all_tools.extend(memgraph_tools(client, user_id, agent_id))

    return Agent(
        name=name,
        instructions=memgraph_instructions(
            client, user_id,
            base=instructions,
            agent_id=agent_id,
            token_budget=token_budget,
        ),
        hooks=MemgraphAgentHooks(client, user_id, agent_id),
        tools=all_tools,
        **agent_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_latest_input(context) -> str:
    """Extract the latest user input from RunContextWrapper or AgentHookContext."""
    try:
        # AgentHookContext has .turn_input
        turn_input = getattr(context, "turn_input", None)
        if turn_input:
            for item in reversed(turn_input):
                # Can be dict or object with .content
                if isinstance(item, dict):
                    if item.get("role") == "user":
                        content = item.get("content", "")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            # Multi-part content
                            texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                            return " ".join(texts)
                elif hasattr(item, "role") and getattr(item, "role", None) == "user":
                    content = getattr(item, "content", "")
                    if isinstance(content, str):
                        return content
    except Exception:
        pass
    return ""


def _extract_response_text(response) -> str:
    """Extract text from a ModelResponse object."""
    try:
        # ModelResponse has .output list
        output = getattr(response, "output", None)
        if output:
            for item in output:
                # ResponseOutputMessage has .content list
                content = getattr(item, "content", None)
                if content and isinstance(content, list):
                    for part in content:
                        text = getattr(part, "text", None)
                        if text:
                            return text
                # Fallback: string content
                if isinstance(content, str):
                    return content
    except Exception:
        pass
    return ""


def _safe_truncate(text: str, max_len: int = 2000) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"
