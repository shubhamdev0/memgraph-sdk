import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from memgraph_sdk.exceptions import (
    MemgraphAPIError,
    MemgraphAuthError,
    MemgraphConnectionError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)

logger = logging.getLogger(__name__)


class MemgraphClient:
    """Memgraph AI Python SDK client.

    Args:
        api_key: Your Memgraph API key (starts with mg_)
        tenant_id: Optional tenant ID (resolved from API key if omitted)
        base_url: API URL (default: https://api.memgraph.ai/v1)
        timeout: Request timeout in seconds
        max_retries: Max retry attempts for transient failures
    """

    def __init__(
        self,
        api_key: str,
        tenant_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv("MEMGRAPH_API_URL", "https://api.memgraph.ai/v1")
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {"X-API-KEY": api_key}
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Make an HTTP request with retries and proper error handling."""
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base_url}{path}"

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.request(method, url, **kwargs)
                self._raise_for_status(resp)
                return resp
            except MemgraphRateLimitError as e:
                last_exc = e
                wait = min(e.retry_after, 60)
                logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                time.sleep(wait)
            except MemgraphAPIError as e:
                last_exc = e
                # Only retry transient server errors (500, 502, 503, 504)
                if e.status_code not in (500, 502, 503, 504):
                    raise
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Server error %s, retrying in %ds (attempt %d/%d)", e.status_code, wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
            except MemgraphConnectionError as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Connection error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
            except (MemgraphAuthError, MemgraphValidationError):
                raise  # Don't retry auth or validation errors
            except requests.ConnectionError as e:
                last_exc = MemgraphConnectionError(f"Cannot connect to Memgraph server: {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Connection error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
            except requests.Timeout as e:
                last_exc = MemgraphConnectionError(f"Request timed out: {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Timeout, retrying in %ds (attempt %d/%d)", wait, attempt + 1, self.max_retries)
                    time.sleep(wait)

        raise last_exc

    @staticmethod
    def _raise_for_status(resp: requests.Response):
        """Convert HTTP errors to typed Memgraph exceptions."""
        if resp.ok:
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

    def ping(self) -> Dict[str, Any]:
        """Check server connectivity. Returns health status."""
        try:
            resp = self._session.get(f"{self.base_url.rsplit('/v1', 1)[0]}/health", timeout=5)
            return resp.json()
        except requests.ConnectionError:
            raise MemgraphConnectionError("Cannot reach Memgraph server at " + self.base_url)
        except Exception as e:
            raise MemgraphConnectionError(f"Health check failed: {e}")

    def add(self, text: str, user_id: str) -> Dict:
        """Add a memory via the /ingest endpoint (event pipeline).

        Note: Events go through async processing and may take time to become searchable.
        For immediate searchability, use remember() instead.
        """
        data = {
            "user_id": user_id,
            "text": text,
        }
        if self.tenant_id is not None:
            data["tenant_id"] = self.tenant_id
        resp = self._request("POST", "/ingest", data=data)
        return resp.json()

    def remember(self, text: str, user_id: str, category: str = "general",
                 domain: Optional[str] = None, confidence: float = 0.90) -> Dict:
        """Store a memory as a belief with vector embedding for immediate searchability."""
        domain_map = {
            "decision": "work", "architecture": "tech", "bug_fix": "tech",
            "preference": "general", "general": "general",
        }
        import hashlib
        short = text[:60].strip().lower()
        clean = "".join(c if c.isalnum() or c == " " else "" for c in short)
        # Add short hash suffix to avoid key collisions for similar text
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
        resp = self._request("POST", "/beliefs", json=payload)
        return resp.json()

    def search(self, query: str, user_id: str, agent_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """Search memories relevant to a query. Returns scored results.

        Uses the v2 context endpoint which returns JSON with scored memories,
        including semantic similarity, recency, confidence, and keyword signals.

        Args:
            query: What to search for (natural language)
            user_id: Which user's memories to search
            agent_id: Optional agent ID for scoped retrieval
            limit: Max results to return (default 10)

        Returns:
            Dict with 'results' list, each containing:
              - content: The memory text
              - score: Relevance score (0-1)
              - metadata: key, domain, belief_type
        """
        # Use v2 context endpoint (returns proper JSON with scored memories)
        # Build the v2 URL from the base URL
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
            resp = self._session.post(v2_url, json=payload, timeout=self.timeout)
            self._raise_for_status(resp)
            data = resp.json()

            # Normalize: v2/context returns {memories: [...]}
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

        except (MemgraphAuthError, MemgraphValidationError):
            raise  # Auth/validation errors should not be silenced
        except (MemgraphConnectionError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Connection issues: fallback to get_beliefs
            try:
                beliefs = self.get_beliefs(user_id=user_id, limit=limit)
                items = beliefs if isinstance(beliefs, list) else beliefs.get("items", [])
                results = [
                    {"content": f"{b.get('key','')}: {b.get('value','')}", "score": 1.0, "metadata": b}
                    for b in items[:limit]
                ]
                return {"results": results, "total": len(results)}
            except Exception:
                return {"results": [], "total": 0}
        except Exception:
            # Unknown error: return empty (v2 endpoint may not exist on older servers)
            return {"results": [], "total": 0}

    def get_beliefs(self, user_id: str, limit: int = 50, cursor: str = None) -> Dict:
        """Fetch beliefs for a user with cursor-based pagination."""
        params = {
            "subject_id": user_id,
            "limit": limit,
        }
        if self.tenant_id is not None:
            params["tenant_id"] = self.tenant_id
        if cursor:
            params["cursor"] = cursor
        resp = self._request("GET", "/beliefs", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Memory Intelligence API
    # ------------------------------------------------------------------

    def health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory health metrics for the tenant (optionally scoped to a user)."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = self._request("GET", "/intelligence/health", params=params)
        return resp.json()

    def contradictions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get contradiction report."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = self._request("GET", "/intelligence/contradictions", params=params)
        return resp.json()

    def evaluate(self, query: str, user_id: str) -> Dict[str, Any]:
        """Run a retrieval query and get detailed scoring breakdown."""
        payload = {"query": query, "user_id": user_id}
        resp = self._request("POST", "/intelligence/evaluate", json=payload)
        return resp.json()

    def mcis(self, user_id: Optional[str] = None, save: bool = False) -> Dict[str, Any]:
        """Compute the Memgraph Cognitive Integrity Score (0-100)."""
        params = {"save": str(save).lower()}
        if user_id:
            params["user_id"] = user_id
        resp = self._request("GET", "/intelligence/mcis", params=params)
        return resp.json()

    def mcis_history(self, user_id: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        """Get historical MCIS snapshots for trend visualization."""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        resp = self._request("GET", "/intelligence/mcis/history", params=params)
        return resp.json()

    def benchmark(self, scenario: str) -> Dict[str, Any]:
        """Run a memory benchmark scenario."""
        payload = {"scenario": scenario}
        resp = self._request("POST", "/benchmark/run", json=payload)
        return resp.json()

    def benchmark_scenarios(self) -> List[Dict]:
        """List available benchmark scenarios."""
        resp = self._request("GET", "/benchmark/scenarios")
        return resp.json().get("scenarios", [])

    # ------------------------------------------------------------------
    # Temporal Query API
    # ------------------------------------------------------------------

    def belief_history(self, user_id: str, key: str, as_of: str = None,
                       include_inactive: bool = True) -> List[Dict]:
        """Get version history for a specific belief key.

        Args:
            user_id: Subject/user ID
            key: The belief key to look up history for
            as_of: Optional ISO timestamp for point-in-time snapshot
            include_inactive: Whether to include superseded versions (default True)

        Returns: List of belief versions ordered by created_at ascending
        """
        params: Dict[str, Any] = {"include_inactive": str(include_inactive).lower()}
        if as_of:
            params["as_of"] = as_of
        resp = self._request("GET", f"/beliefs/history/{user_id}/{key}", params=params)
        return resp.json()

    def belief_timeline(self, user_id: str, domain: str = None,
                        since: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get chronological timeline of all belief changes for a user.

        Args:
            user_id: Subject/user ID
            domain: Optional domain filter
            since: Optional ISO timestamp — only show changes after this time
            limit: Max results (default 50)

        Returns: {"subject_id": ..., "count": ..., "timeline": [...]}
        """
        params: Dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        if since:
            params["since"] = since
        resp = self._request("GET", f"/beliefs/timeline/{user_id}", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Memory Delete / Forget API
    # ------------------------------------------------------------------

    def forget(self, belief_id: str) -> Dict[str, Any]:
        """Delete a specific belief by ID.

        Returns: {"status": "deleted", "id": "..."}
        """
        resp = self._request("DELETE", f"/beliefs/{belief_id}")
        return resp.json()

    def forget_all(self, user_id: str, domain: Optional[str] = None,
                   soft: bool = False) -> Dict[str, Any]:
        """Bulk delete all beliefs for a user (optionally scoped to domain).

        Args:
            user_id: Subject/user ID to delete beliefs for
            domain: Optional domain filter (e.g. 'work', 'tech', 'personal')
            soft: If True, soft-delete (set inactive) instead of permanent delete

        Returns: {"status": "deleted", "deleted": count, "domain": ...}
        """
        params: Dict[str, Any] = {"subject_id": user_id, "soft": str(soft).lower()}
        if domain:
            params["domain"] = domain
        resp = self._request("DELETE", "/beliefs", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Cognitive Sidecar API
    # ------------------------------------------------------------------

    def sidecar_pre_flight(self, message: str, user_id: str, thread_id: str = None,
                           agent_id: str = "sdk_sidecar",
                           token_budget: int = 4000) -> Dict[str, Any]:
        """Auto-recall: Fetch relevant memories before an LLM call.

        Returns memory context ready to inject as a system message.
        """
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "message": message,
            "token_budget": token_budget,
            "include_profile": True,
            "include_prospective": True,
        }
        if thread_id:
            payload["thread_id"] = thread_id
        resp = self._request("POST", "/sidecar/pre-flight", json=payload)
        return resp.json()

    def sidecar_post_flight(self, messages: List[Dict[str, str]], user_id: str,
                            thread_id: str = None,
                            agent_id: str = "sdk_sidecar") -> Dict[str, Any]:
        """Auto-learn: Extract learnable signals from a conversation exchange.

        Learning happens in background — this returns immediately.
        """
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": messages,
        }
        if thread_id:
            payload["thread_id"] = thread_id
        resp = self._request("POST", "/sidecar/post-flight", json=payload)
        return resp.json()

    def sidecar_process(self, messages: List[Dict[str, str]], user_id: str,
                        thread_id: str = None, agent_id: str = "sdk_sidecar",
                        token_budget: int = 4000) -> Dict[str, Any]:
        """Combined: Auto-recall for current message + auto-learn from previous exchanges.

        This is the recommended single-call method for always-on memory.
        """
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": messages,
            "token_budget": token_budget,
            "include_profile": True,
            "include_prospective": True,
        }
        if thread_id:
            payload["thread_id"] = thread_id
        resp = self._request("POST", "/sidecar/process", json=payload)
        return resp.json()

    # ==================================================================
    # v2 — Cognitive Infrastructure API
    # ==================================================================

    def _v2_request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Make a request to the v2 API. Reuses _request() retry logic."""
        # Temporarily swap base_url to v2
        original_base = self.base_url
        self.base_url = original_base.replace("/v1", "/v2")
        try:
            return self._request(method, path, **kwargs)
        finally:
            self.base_url = original_base

    # ------------------------------------------------------------------
    # Context Graph (Core Interface)
    # ------------------------------------------------------------------

    def get_context_graph(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_decisions: bool = True,
        include_graph: bool = True,
        create_snapshot: bool = False,
    ) -> Dict[str, Any]:
        """6-stage context graph retrieval — the core Memgraph v2 interface.

        Returns structured cognitive context including:
        - Scored memories (beliefs, episodes, documents)
        - Resolved entities
        - Graph relationships
        - Related decisions
        - Optional context snapshot for decision tracking
        """
        payload: Dict[str, Any] = {
            "query": query,
            "include_decisions": include_decisions,
            "include_graph": include_graph,
            "create_snapshot": create_snapshot,
        }
        if user_id:
            payload["user_id"] = user_id
        if agent_id:
            payload["agent_id"] = agent_id
        resp = self._v2_request("POST", "/context", json=payload)
        return resp.json()

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def create_entity(
        self,
        entity_type: str,
        name: Optional[str] = None,
        properties: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new entity (person, organization, product, etc.)."""
        payload: Dict[str, Any] = {"entity_type": entity_type}
        if name:
            payload["name"] = name
        if properties:
            payload["properties"] = properties
        if aliases:
            payload["aliases"] = aliases
        resp = self._v2_request("POST", "/entities", json=payload)
        return resp.json()

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get an entity by ID."""
        resp = self._v2_request("GET", f"/entities/{entity_id}")
        return resp.json()

    def list_entities(self, entity_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """List entities, optionally filtered by type."""
        params: Dict[str, Any] = {"limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        resp = self._v2_request("GET", "/entities", params=params)
        return resp.json()

    def search_entities(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search entities by name (trigram + semantic)."""
        resp = self._v2_request("POST", "/entities/search", json={"query": query, "max_results": max_results})
        return resp.json()

    def delete_entity(self, entity_id: str) -> Dict[str, Any]:
        """Delete an entity and its relationships."""
        resp = self._v2_request("DELETE", f"/entities/{entity_id}")
        return resp.json()

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def create_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        evidence_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a temporal relationship between two entities."""
        payload: Dict[str, Any] = {
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "relation_type": relation_type,
            "confidence": confidence,
        }
        if valid_from:
            payload["valid_from"] = valid_from
        if valid_to:
            payload["valid_to"] = valid_to
        if evidence_sources:
            payload["evidence_sources"] = evidence_sources
        resp = self._v2_request("POST", "/relationships", json=payload)
        return resp.json()

    def list_relationships(
        self, entity_id: Optional[str] = None, relation_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """List relationships, optionally filtered."""
        params: Dict[str, Any] = {"limit": limit}
        if entity_id:
            params["entity_id"] = entity_id
        if relation_type:
            params["relation_type"] = relation_type
        resp = self._v2_request("GET", "/relationships", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def record_decision(
        self,
        goal: str,
        reasoning_steps: Optional[List[Dict]] = None,
        tools_used: Optional[List[Dict]] = None,
        beliefs_used: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        outcome: Optional[str] = None,
        outcome_assessment: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        create_snapshot: bool = False,
    ) -> Dict[str, Any]:
        """Record an AI agent decision with full reasoning traces."""
        payload: Dict[str, Any] = {"goal": goal, "create_snapshot": create_snapshot}
        if reasoning_steps:
            payload["reasoning_steps"] = reasoning_steps
        if tools_used:
            payload["tools_used"] = tools_used
        if beliefs_used:
            payload["beliefs_used"] = beliefs_used
        if confidence is not None:
            payload["confidence"] = confidence
        if outcome:
            payload["outcome"] = outcome
        if outcome_assessment:
            payload["outcome_assessment"] = outcome_assessment
        if agent_id:
            payload["agent_id"] = agent_id
        if user_id:
            payload["user_id"] = user_id
        if episode_id:
            payload["episode_id"] = episode_id
        resp = self._v2_request("POST", "/decisions", json=payload)
        return resp.json()

    def get_decision(self, decision_id: str) -> Dict[str, Any]:
        """Get a decision by ID."""
        resp = self._v2_request("GET", f"/decisions/{decision_id}")
        return resp.json()

    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """Explain a decision: reasoning, context snapshot, beliefs used."""
        resp = self._v2_request("GET", f"/decisions/{decision_id}/explain")
        return resp.json()

    def list_decisions(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None,
        outcome: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """List decisions, optionally filtered."""
        params: Dict[str, Any] = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        if user_id:
            params["user_id"] = user_id
        if outcome:
            params["outcome"] = outcome
        resp = self._v2_request("GET", "/decisions", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    def traverse_graph(
        self, entity_ids: List[str], max_depth: int = 2, temporal_filter: bool = True
    ) -> Dict[str, Any]:
        """Traverse the entity relationship graph from seed entities."""
        payload = {
            "entity_ids": entity_ids,
            "max_depth": max_depth,
            "temporal_filter": temporal_filter,
        }
        resp = self._v2_request("POST", "/graph/traverse", json=payload)
        return resp.json()

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def list_contradictions(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """List detected contradictions."""
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        resp = self._v2_request("GET", "/contradictions", params=params)
        return resp.json()

    def resolve_contradiction(
        self, contradiction_id: str, resolution_status: str = "resolved", resolved_by: str = "manual"
    ) -> Dict[str, Any]:
        """Resolve or dismiss a contradiction."""
        payload = {"resolution_status": resolution_status, "resolved_by": resolved_by}
        resp = self._v2_request("PATCH", f"/contradictions/{contradiction_id}", json=payload)
        return resp.json()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def analytics(self) -> Dict[str, Any]:
        """Get decision observability and memory analytics."""
        resp = self._v2_request("GET", "/analytics")
        return resp.json()
