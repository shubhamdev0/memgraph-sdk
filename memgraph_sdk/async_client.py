"""
Async SDK client for Memgraph.

Uses httpx for non-blocking HTTP calls, suitable for async frameworks
(FastAPI, aiohttp, etc.) and high-throughput pipelines.

Usage:
    from memgraph_sdk import AsyncMemgraphClient

    async with AsyncMemgraphClient(api_key="...", tenant_id="...") as client:
        await client.add("User prefers dark mode", user_id="u1")
        ctx = await client.search("What theme does the user prefer?", user_id="u1")
"""

import os
from typing import Dict, Any, Optional

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for AsyncMemgraphClient. Install it with: pip install httpx"
    )


class AsyncMemgraphClient:
    def __init__(self, api_key: str, tenant_id: str, base_url: str = None, timeout: float = 30.0):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-KEY": api_key},
            timeout=timeout,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self._client.aclose()

    async def add(self, text: str, user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """Add a memory via the /ingest endpoint."""
        data = {
            "tenant_id": self.tenant_id,
            "user_id": user_id,
            "text": text,
        }
        resp = await self._client.post("/ingest", data=data)
        resp.raise_for_status()
        return resp.json()

    async def search(self, query: str, user_id: str, agent_id: str = "sdk_client") -> Dict[str, Any]:
        """Retrieve relevant context for a query via /context endpoint."""
        payload = {
            "task": query,
            "user_id": user_id,
            "tenant_id": self.tenant_id,
            "agent_id": agent_id,
        }
        resp = await self._client.post("/context", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_context(self, query: str, user_id: str) -> Dict[str, Any]:
        """Alias for search()."""
        return await self.search(query, user_id)

    async def log_event(
        self,
        event_type: str,
        content: Dict[str, Any],
        user_id: str = "default_user",
        agent_id: str = "sdk_client",
        thread_id: Optional[str] = None,
    ) -> Dict:
        """Log a raw event to the /events endpoint."""
        payload = {
            "tenant_id": self.tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "event_type": event_type,
            "content": content,
        }
        if thread_id:
            payload["thread_id"] = thread_id

        resp = await self._client.post("/events", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_beliefs(self, user_id: str, limit: int = 50) -> list:
        """Fetch beliefs for a user via /beliefs endpoint."""
        resp = await self._client.get(
            "/beliefs",
            params={"tenant_id": self.tenant_id, "subject_id": user_id, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Memory Intelligence API
    # ------------------------------------------------------------------

    async def health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory health metrics for the tenant (optionally scoped to a user)."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = await self._client.get("/intelligence/health", params=params)
        resp.raise_for_status()
        return resp.json()

    async def contradictions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get contradiction report — all detected contradictions with resolution history."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = await self._client.get("/intelligence/contradictions", params=params)
        resp.raise_for_status()
        return resp.json()

    async def evaluate(self, query: str, user_id: str) -> Dict[str, Any]:
        """Run a retrieval query and get detailed scoring breakdown."""
        payload = {"query": query, "user_id": user_id}
        resp = await self._client.post("/intelligence/evaluate", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def mcis(self, user_id: Optional[str] = None, save: bool = False) -> Dict[str, Any]:
        """Compute the Memgraph Cognitive Integrity Score (0-100)."""
        params = {"save": str(save).lower()}
        if user_id:
            params["user_id"] = user_id
        resp = await self._client.get("/intelligence/mcis", params=params)
        resp.raise_for_status()
        return resp.json()

    async def mcis_history(self, user_id: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        """Get historical MCIS snapshots for trend visualization."""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        resp = await self._client.get("/intelligence/mcis/history", params=params)
        resp.raise_for_status()
        return resp.json()

    async def benchmark(self, scenario: str) -> Dict[str, Any]:
        """Run a memory benchmark scenario (e.g. 'contradiction_storm', 'retrieval_accuracy')."""
        payload = {"scenario": scenario}
        resp = await self._client.post("/benchmark/run", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def benchmark_scenarios(self) -> list:
        """List available benchmark scenarios."""
        resp = await self._client.get("/benchmark/scenarios")
        resp.raise_for_status()
        return resp.json().get("scenarios", [])
