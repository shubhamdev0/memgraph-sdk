"""
MemgraphMemory — Automatic decision capture middleware for AI agent frameworks.

Provides zero-effort memory instrumentation for agent frameworks like:
- OpenAI Agents SDK
- LangGraph
- AutoGen
- Custom agent loops

Automatically captures:
- Reasoning steps (LLM calls)
- Tool usage
- Context snapshots
- Decision outcomes
- Belief updates from decision feedback

Usage:
    from memgraph_sdk import MemgraphClient
    from memgraph_sdk.agent_memory import MemgraphMemory

    client = MemgraphClient(api_key="mg_...")
    memory = MemgraphMemory(client, user_id="user_123")

    # Wrap your agent loop
    with memory.capture(goal="Help user book a flight") as decision:
        # Your agent code — tool calls, LLM calls, etc.
        result = agent.run(query)

        # Record tool usage
        decision.tool("search_flights", input={"dest": "NYC"}, output=flights)
        decision.step("Selected cheapest flight", confidence=0.9)

        # Set outcome
        decision.succeed("Flight booked successfully")

    # Or use the decorator
    @memory.track(goal="Process refund")
    def handle_refund(request):
        ...
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionCapture:
    """Active decision capture context — records reasoning as it happens."""

    goal: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    episode_id: Optional[str] = None

    # Accumulated reasoning trace
    _steps: list = field(default_factory=list)
    _tools: list = field(default_factory=list)
    _beliefs_used: list = field(default_factory=list)
    _confidence: Optional[float] = None
    _outcome: Optional[str] = None
    _outcome_assessment: Optional[str] = None
    _start_time: float = field(default_factory=time.time)

    def step(
        self,
        description: str,
        tool: Optional[str] = None,
        input: Any = None,
        output: Any = None,
        confidence: Optional[float] = None,
    ):
        """Record a reasoning step."""
        self._steps.append({
            "step": len(self._steps) + 1,
            "description": description,
            "tool": tool,
            "input": _safe_serialize(input),
            "output": _safe_serialize(output),
            "confidence": confidence,
        })

    def tool(
        self,
        tool_name: str,
        input: Any = None,
        output: Any = None,
    ):
        """Record a tool usage."""
        self._tools.append({
            "tool_name": tool_name,
            "tool_input": _safe_serialize(input),
            "tool_output": _safe_serialize(output),
        })

    def use_belief(self, belief_id: str):
        """Record that a belief was consulted during this decision."""
        self._beliefs_used.append(belief_id)

    def use_beliefs(self, belief_ids: List[str]):
        """Record multiple beliefs consulted."""
        self._beliefs_used.extend(belief_ids)

    def succeed(self, assessment: str = ""):
        """Mark this decision as SUCCESS."""
        self._outcome = "SUCCESS"
        self._outcome_assessment = assessment

    def fail(self, assessment: str = ""):
        """Mark this decision as FAILURE."""
        self._outcome = "FAILURE"
        self._outcome_assessment = assessment

    def partial(self, assessment: str = ""):
        """Mark this decision as PARTIAL success."""
        self._outcome = "PARTIAL"
        self._outcome_assessment = assessment

    def set_confidence(self, confidence: float):
        """Set overall decision confidence."""
        self._confidence = confidence

    def to_payload(self) -> Dict[str, Any]:
        """Convert to API payload."""
        payload: Dict[str, Any] = {
            "goal": self.goal,
            "reasoning_steps": self._steps,
            "tools_used": self._tools,
            "beliefs_used": self._beliefs_used,
            "create_snapshot": True,
        }
        if self.agent_id:
            payload["agent_id"] = self.agent_id
        if self.user_id:
            payload["user_id"] = self.user_id
        if self.episode_id:
            payload["episode_id"] = self.episode_id
        if self._confidence is not None:
            payload["confidence"] = self._confidence
        if self._outcome:
            payload["outcome"] = self._outcome
        if self._outcome_assessment:
            payload["outcome_assessment"] = self._outcome_assessment
        return payload


class MemgraphMemory:
    """
    Automatic decision capture middleware for AI agent frameworks.

    Provides:
    1. Context manager for decision capture: `with memory.capture(goal=...) as d:`
    2. Decorator for function-level tracking: `@memory.track(goal=...)`
    3. Manual recording: `memory.record(goal=..., outcome=...)`
    4. Context graph retrieval: `memory.get_context(query)`
    5. Decision feedback loop: outcomes automatically update belief confidence
    """

    def __init__(
        self,
        client,  # MemgraphClient
        user_id: Optional[str] = None,
        agent_id: str = "agent",
        auto_feedback: bool = True,
    ):
        """
        Args:
            client: MemgraphClient instance
            user_id: Default user ID for scoping
            agent_id: Agent identifier
            auto_feedback: Whether decision outcomes automatically update belief confidence
        """
        self.client = client
        self.user_id = user_id
        self.agent_id = agent_id
        self.auto_feedback = auto_feedback

    @contextmanager
    def capture(
        self,
        goal: str,
        user_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ):
        """
        Context manager for capturing a decision.

        Usage:
            with memory.capture(goal="Book a flight") as decision:
                decision.tool("search_api", input={...}, output={...})
                decision.step("Selected cheapest option", confidence=0.9)
                decision.succeed("Flight booked")
        """
        decision = DecisionCapture(
            goal=goal,
            agent_id=self.agent_id,
            user_id=user_id or self.user_id,
            episode_id=episode_id,
        )

        try:
            yield decision
        except Exception as e:
            decision.fail(f"Exception: {str(e)}")
            raise
        finally:
            # Store decision in background
            self._store_decision_async(decision)

    def track(self, goal: str = None):
        """
        Decorator for function-level decision tracking.

        Usage:
            @memory.track(goal="Process refund")
            def handle_refund(request):
                ...
                return result
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                actual_goal = goal or f"{func.__name__}"
                with self.capture(goal=actual_goal) as decision:
                    try:
                        result = func(*args, **kwargs)
                        if decision._outcome is None:
                            decision.succeed()
                        return result
                    except Exception:
                        raise  # capture context manager handles fail
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        return decorator

    def get_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_decisions: bool = True,
    ) -> Dict[str, Any]:
        """Get context graph for a query (v2 6-stage pipeline)."""
        return self.client.get_context_graph(
            query=query,
            user_id=user_id or self.user_id,
            agent_id=self.agent_id,
            include_decisions=include_decisions,
        )

    def record(
        self,
        goal: str,
        outcome: str = "UNKNOWN",
        reasoning_steps: Optional[List[Dict]] = None,
        tools_used: Optional[List[Dict]] = None,
        beliefs_used: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        outcome_assessment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually record a decision."""
        return self.client.record_decision(
            goal=goal,
            reasoning_steps=reasoning_steps,
            tools_used=tools_used,
            beliefs_used=beliefs_used,
            confidence=confidence,
            outcome=outcome,
            outcome_assessment=outcome_assessment,
            agent_id=self.agent_id,
            user_id=self.user_id,
            create_snapshot=True,
        )

    def _store_decision_async(self, decision: DecisionCapture):
        """Store a decision in a background thread."""
        def _store():
            try:
                payload = decision.to_payload()
                result = self.client.record_decision(**payload)
                logger.info(
                    "Decision captured: goal='%s' outcome=%s decision_id=%s",
                    decision.goal[:50], decision._outcome, result.get("id"),
                )

                # Decision feedback loop: if outcome is FAILURE and beliefs were used,
                # flag those beliefs for confidence review
                if self.auto_feedback and decision._outcome == "FAILURE" and decision._beliefs_used:
                    self._apply_decision_feedback(decision)

            except Exception as e:
                logger.warning("Failed to store decision: %s", e)

        thread = threading.Thread(target=_store, daemon=True)
        thread.start()

    def _apply_decision_feedback(self, decision: DecisionCapture):
        """
        Decision feedback loop: when a decision FAILS, reduce confidence
        of beliefs that were consulted. This creates self-learning agents.

        Decision → Outcome → Belief Update
        """
        try:
            # For failed decisions, the beliefs used may have been wrong
            # We don't want to aggressively downgrade, just flag for review
            logger.info(
                "Decision feedback: FAILURE with %d beliefs used, flagging for review",
                len(decision._beliefs_used),
            )
            # The actual confidence update happens server-side via the contradiction
            # detection engine in the dreaming worker. We just record the decision
            # with outcome=FAILURE and the beliefs_used, which the analytics
            # endpoint can surface.
        except Exception as e:
            logger.warning("Decision feedback loop failed: %s", e)


def _safe_serialize(obj: Any) -> Any:
    """Safely serialize an object for JSON storage. Truncate large outputs."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, str) and len(obj) > 2000:
            return obj[:2000] + "...[truncated]"
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj[:50]]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in list(obj.items())[:50]}
    return str(obj)[:500]
