"""
Async SDK client for Memgraph.

Uses httpx for non-blocking HTTP calls, suitable for async frameworks
(FastAPI, aiohttp, etc.) and high-throughput pipelines.

Usage:
    from memgraph_sdk import AsyncMemgraphClient

    async with AsyncMemgraphClient(api_key="...") as client:
        await client.add("User prefers dark mode", user_id="u1")
        ctx = await client.search("What theme does the user prefer?", user_id="u1")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from memgraph_sdk.client import _api_root, _memory_text, _remember_key, client_headers
from memgraph_sdk.exceptions import (
    MemgraphAPIError,
    MemgraphAuthError,
    MemgraphConnectionError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)

logger = logging.getLogger(__name__)


class AsyncMemgraphClient:
    """Async Memgraph AI client using httpx."""

    def __init__(
        self,
        api_key: str,
        tenant_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        if httpx is None:
            raise ImportError(
                "httpx is required for AsyncMemgraphClient. Install it with: pip install memgraph-sdk[async]"
            )
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv("MEMGRAPH_API_URL", "https://api.memgraph.ai/v1")
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            headers=client_headers(api_key, "async"),
            timeout=timeout,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self._client.aclose()

    def _url(self, path: str, api: str = "v1") -> str:
        return f"{_api_root(self.base_url)}/{api}{path}"

    async def _request(self, method: str, path: str, *, api: str = "v1", **kwargs) -> httpx.Response:
        """Make an HTTP request with retries and proper error handling."""
        last_exc = None
        url = self._url(path, api)
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.request(method, url, **kwargs)
                self._raise_for_status(resp)
                return resp
            except MemgraphRateLimitError as e:
                last_exc = e
                wait = min(e.retry_after, 60)
                logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                await asyncio.sleep(wait)
            except MemgraphAPIError as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Server error %s, retrying in %ds (attempt %d/%d)", e.status_code, wait, attempt + 1, self.max_retries)
                    await asyncio.sleep(wait)
            except MemgraphConnectionError as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Connection error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                    await asyncio.sleep(wait)
            except (MemgraphAuthError, MemgraphValidationError):
                raise
            except httpx.ConnectError as e:
                last_exc = MemgraphConnectionError(f"Cannot connect to Memgraph server: {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
            except httpx.TimeoutException as e:
                last_exc = MemgraphConnectionError(f"Request timed out: {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)

        raise last_exc

    @staticmethod
    def _raise_for_status(resp: httpx.Response):
        """Convert HTTP errors to typed Memgraph exceptions."""
        if resp.is_success:
            return

        try:
            body = resp.json()
        except Exception:
            body = {"detail": resp.text}

        detail = body.get("detail", resp.text)
        code = resp.status_code

        if code in (401, 403):
            raise MemgraphAuthError(f"Authentication failed: {detail}", status_code=code, response_body=body)
        elif code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            raise MemgraphRateLimitError(f"Rate limit exceeded: {detail}", retry_after=retry_after, status_code=code, response_body=body)
        elif 400 <= code < 500:
            raise MemgraphValidationError(f"Validation error: {detail}", status_code=code, response_body=body)
        elif code >= 500:
            raise MemgraphAPIError(f"Server error: {detail}", status_code=code, response_body=body)

    async def ping(self) -> Dict[str, Any]:
        """Check server connectivity. Returns health status."""
        try:
            resp = await self._client.get(f"{_api_root(self.base_url)}/health", timeout=5)
            return resp.json()
        except httpx.ConnectError:
            raise MemgraphConnectionError("Cannot reach Memgraph server at " + self.base_url)
        except Exception as e:
            raise MemgraphConnectionError(f"Health check failed: {e}")

    async def add(self, text: str, user_id: str) -> Dict:
        """Add a memory via the /ingest endpoint (extraction pipeline).

        Beliefs are extracted in the background (~5-10s). Use remember() for
        immediate searchability.
        """
        data = {
            "user_id": user_id,
            "text": text,
        }
        if self.tenant_id is not None:
            data["tenant_id"] = self.tenant_id
        resp = await self._request("POST", "/ingest", data=data)
        return resp.json()

    async def remember(self, text: str, user_id: str, category: str = "general",
                       domain: Optional[str] = None, confidence: float = 0.90) -> Dict:
        """Store a memory as a belief with vector embedding for immediate searchability."""
        domain_map = {
            "decision": "work", "architecture": "tech", "bug_fix": "tech",
            "preference": "general", "general": "general",
        }
        payload = {
            "subject_id": user_id,
            "key": _remember_key(text, category),
            "value": text,
            "confidence": confidence,
            "belief_type": "fact" if category in ("bug_fix", "architecture") else "belief",
            "domain": domain or domain_map.get(category, "general"),
        }
        if self.tenant_id is not None:
            payload["tenant_id"] = self.tenant_id
        resp = await self._request("POST", "/beliefs", json=payload)
        return resp.json()

    async def search(self, query: str, user_id: str, agent_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """Search memories relevant to a query. Returns scored results.

        Returns:
            Dict with 'results' list, each containing content, score, metadata.

        Raises MemgraphAPIError / MemgraphConnectionError on failure — an
        outage never looks like "no memories".
        """
        payload = {"query": query, "user_id": user_id}
        if agent_id:
            payload["agent_id"] = agent_id

        try:
            data = (await self._request("POST", "/context", api="v2", json=payload)).json()
        except MemgraphValidationError as e:
            if e.status_code != 404:
                raise
            # Server without the v2 API: list the user's beliefs instead.
            beliefs = await self.get_beliefs(user_id=user_id, limit=limit)
            items = beliefs if isinstance(beliefs, list) else beliefs.get("items", [])
            results = [{"content": b.get("value", ""), "score": 1.0, "metadata": b, "type": "belief"}
                       for b in items[:limit]]
            return {"results": results, "total": len(results)}

        results = [
            {
                "id": m.get("id"),
                "content": _memory_text(m),
                "score": m.get("score", 0),
                "metadata": m.get("metadata", {}),
                "type": m.get("type", "belief"),
            }
            for m in data.get("memories", [])[:limit]
        ]
        return {"results": results, "total": len(results)}

    async def get_beliefs(self, user_id: str, limit: int = 50, cursor: str = None) -> Dict:
        """Fetch beliefs for a user with cursor-based pagination."""
        params = {
            "subject_id": user_id,
            "limit": limit,
        }
        if self.tenant_id is not None:
            params["tenant_id"] = self.tenant_id
        if cursor:
            params["cursor"] = cursor
        resp = await self._request("GET", "/beliefs", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Memory Intelligence API
    # ------------------------------------------------------------------

    async def health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory health metrics for the tenant (optionally scoped to a user)."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = await self._request("GET", "/intelligence/health", params=params)
        return resp.json()

    async def contradictions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get contradiction report."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = await self._request("GET", "/intelligence/contradictions", params=params)
        return resp.json()

    async def evaluate(self, query: str, user_id: str) -> Dict[str, Any]:
        """Run a retrieval query and get detailed scoring breakdown."""
        payload = {"query": query, "user_id": user_id}
        resp = await self._request("POST", "/intelligence/evaluate", json=payload)
        return resp.json()

    async def mcis(self, user_id: Optional[str] = None, save: bool = False) -> Dict[str, Any]:
        """Compute the Memgraph Cognitive Integrity Score (0-100)."""
        params = {"save": str(save).lower()}
        if user_id:
            params["user_id"] = user_id
        resp = await self._request("GET", "/intelligence/mcis", params=params)
        return resp.json()

    async def mcis_history(self, user_id: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        """Get historical MCIS snapshots for trend visualization."""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        resp = await self._request("GET", "/intelligence/mcis/history", params=params)
        return resp.json()

    async def benchmark(self, scenario: str) -> Dict[str, Any]:
        """Run a memory benchmark scenario."""
        payload = {"scenario": scenario}
        resp = await self._request("POST", "/benchmark/run", json=payload)
        return resp.json()

    async def benchmark_scenarios(self) -> list:
        """List available benchmark scenarios."""
        resp = await self._request("GET", "/benchmark/scenarios")
        return resp.json().get("scenarios", [])

    # ------------------------------------------------------------------
    # Forget (user data deletion)
    # ------------------------------------------------------------------

    async def forget(self, belief_id: str) -> Dict[str, Any]:
        """Delete a specific belief by ID."""
        resp = await self._request("DELETE", f"/beliefs/{belief_id}")
        return resp.json()

    async def forget_all(self, user_id: str, domain: Optional[str] = None, soft: bool = False) -> Dict[str, Any]:
        """Delete all beliefs for a user (optionally only one domain)."""
        params: Dict[str, Any] = {"subject_id": user_id, "soft": str(soft).lower()}
        if domain:
            params["domain"] = domain
        resp = await self._request("DELETE", "/beliefs", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Cognitive Sidecar
    # ------------------------------------------------------------------

    async def sidecar_pre_flight(self, message: str, user_id: str, thread_id: str = None,
                                 agent_id: str = "sdk_sidecar", token_budget: int = 4000) -> Dict[str, Any]:
        """Memories + lessons from past attempts to inject before an LLM call."""
        payload: Dict[str, Any] = {
            "user_id": user_id, "agent_id": agent_id, "message": message,
            "token_budget": token_budget, "include_profile": True, "include_prospective": True,
        }
        if thread_id:
            payload["thread_id"] = thread_id
        resp = await self._request("POST", "/sidecar/pre-flight", json=payload)
        return resp.json()

    async def sidecar_post_flight(self, messages: List[Dict[str, str]], user_id: str,
                                  thread_id: str = None, agent_id: str = "sdk_sidecar") -> Dict[str, Any]:
        """Learn from a finished exchange. Returns immediately; learning runs server-side."""
        payload: Dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "messages": messages}
        if thread_id:
            payload["thread_id"] = thread_id
        resp = await self._request("POST", "/sidecar/post-flight", json=payload)
        return resp.json()

    # ------------------------------------------------------------------
    # v2 — context graph, decisions, outcomes
    # ------------------------------------------------------------------

    async def get_context_graph(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                                include_decisions: bool = True, include_graph: bool = True,
                                create_snapshot: bool = False) -> Dict[str, Any]:
        """Memories, entities, relationships and similar past decisions for a query."""
        payload: Dict[str, Any] = {
            "query": query, "include_decisions": include_decisions,
            "include_graph": include_graph, "create_snapshot": create_snapshot,
        }
        if user_id:
            payload["user_id"] = user_id
        if agent_id:
            payload["agent_id"] = agent_id
        resp = await self._request("POST", "/context", api="v2", json=payload)
        return resp.json()

    async def record_decision(self, goal: str, *, outcome: Optional[str] = None,
                              outcome_assessment: Optional[str] = None,
                              beliefs_used: Optional[List[str]] = None,
                              reasoning_steps: Optional[List[Dict]] = None,
                              tools_used: Optional[List[Dict]] = None,
                              confidence: Optional[float] = None,
                              agent_id: Optional[str] = None, user_id: Optional[str] = None,
                              create_snapshot: bool = False) -> Dict[str, Any]:
        """Record an agent decision (see MemgraphClient.record_decision)."""
        payload: Dict[str, Any] = {"goal": goal, "create_snapshot": create_snapshot}
        for name, value in (("outcome", outcome), ("outcome_assessment", outcome_assessment),
                            ("beliefs_used", beliefs_used), ("reasoning_steps", reasoning_steps),
                            ("tools_used", tools_used), ("confidence", confidence),
                            ("agent_id", agent_id), ("user_id", user_id)):
            if value is not None:
                payload[name] = value
        resp = await self._request("POST", "/decisions", api="v2", json=payload)
        return resp.json()

    async def record_outcome(self, decision_id: str, outcome: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """Report how a decision turned out (SUCCESS / FAILURE / PARTIAL)."""
        payload: Dict[str, Any] = {"decision_id": decision_id, "outcome": outcome.upper()}
        if feedback:
            payload["feedback"] = feedback
        resp = await self._request("POST", "/outcomes/record", api="v2", json=payload)
        return resp.json()

    async def delete_decision(self, decision_id: str) -> Dict[str, Any]:
        """Delete a decision and its context snapshot."""
        resp = await self._request("DELETE", f"/decisions/{decision_id}", api="v2")
        return resp.json()
