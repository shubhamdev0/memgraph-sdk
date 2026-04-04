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
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    httpx = None

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

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with retries and proper error handling."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.request(method, path, **kwargs)
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
            base = self.base_url.rsplit("/v1", 1)[0]
            resp = await self._client.get(f"{base}/health", timeout=5)
            return resp.json()
        except httpx.ConnectError:
            raise MemgraphConnectionError("Cannot reach Memgraph server at " + self.base_url)
        except Exception as e:
            raise MemgraphConnectionError(f"Health check failed: {e}")

    async def add(self, text: str, user_id: str) -> Dict:
        """Add a memory via the /ingest endpoint (event pipeline).

        Note: Events go through async processing. Use remember() for immediate searchability.
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
        import hashlib
        short = text[:60].strip().lower()
        clean = "".join(c if c.isalnum() or c == " " else "" for c in short)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
        belief_key = f"{category}_{'_'.join(clean.split()[:5])}_{text_hash}"

        payload = {
            "subject_id": user_id,
            "key": belief_key,
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

        Uses the v2 context endpoint which returns JSON with scored memories.

        Returns:
            Dict with 'results' list, each containing content, score, metadata.
        """
        # Build v2 URL from base URL
        base = self.base_url.rstrip("/")
        if "/v1" in base:
            v2_url = base.replace("/v1", "/v2") + "/context"
        elif "/v2" in base:
            v2_url = base + "/context"
        else:
            v2_url = base + "/v2/context"

        payload = {"query": query, "user_id": user_id}
        if agent_id:
            payload["agent_id"] = agent_id

        try:
            resp = await self._client.post(v2_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            memories = data.get("memories", [])
            results = []
            for m in memories[:limit]:
                results.append({
                    "content": m.get("content", ""),
                    "score": m.get("score", 0),
                    "metadata": m.get("metadata", {}),
                    "type": m.get("type", "belief"),
                })
            return {"results": results, "total": len(results)}

        except Exception:
            try:
                beliefs = await self.get_beliefs(user_id=user_id, limit=limit)
                items = beliefs if isinstance(beliefs, list) else beliefs.get("items", [])
                results = [
                    {"content": f"{b.get('key','')}: {b.get('value','')}", "score": 1.0, "metadata": b}
                    for b in items[:limit]
                ]
                return {"results": results, "total": len(results)}
            except Exception:
                return {"results": [], "total": 0}


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
