import os
import requests
from typing import List, Dict, Any, Optional

class MemgraphClient:
    def __init__(self, api_key: str, tenant_id: str, base_url: str = None):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.base_url = base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.headers = {
            "X-API-KEY": api_key,
            # "Content-Type" is handled automatically by requests for Form data or JSON
        }

    def add(self, text: str, user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add a memory via the /ingest endpoint.
        Uses multipart/form-data as required by the backend.
        """
        data = {
            "tenant_id": self.tenant_id,
            "user_id": user_id,
            "text": text,
        }
        # If we had metadata/files, we would add them here.
        # Verify if metadata is supported by ingest_omni (it currently isn't in the Form params shown in ingest.py, 
        # but the logic creates an event with content_payload which we might want to extend later).
        
        resp = requests.post(f"{self.base_url}/ingest", data=data, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, user_id: str, agent_id: str = "sdk_client") -> Dict[str, Any]:
        """
        Retrieve relevant context for a query via /context endpoint.
        """
        payload = {
            "task": query, # Server expects 'task', not 'query' based on E2E
            "user_id": user_id,
            "tenant_id": self.tenant_id,
            "agent_id": agent_id
        }
        resp = requests.post(f"{self.base_url}/context", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    # Deprecated/Legacy methods mapped to new ones for compatibility
    def log_event(self, event_type: str, content: Dict[str, Any], metadata: Optional[Dict] = None) -> Dict:
        # Fallback to ingest if content has text
        if "text" in content:
            return self.add(content["text"], "legacy_user")
        raise NotImplementedError("Generic log_event not fully supported yet.")

    def get_context(self, query: str, user_id: str) -> Dict[str, Any]:
        return self.search(query, user_id)

    # ------------------------------------------------------------------
    # Memory Intelligence API
    # ------------------------------------------------------------------

    def health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory health metrics for the tenant (optionally scoped to a user)."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = requests.get(f"{self.base_url}/intelligence/health", params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def contradictions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get contradiction report — all detected contradictions with resolution history."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = requests.get(f"{self.base_url}/intelligence/contradictions", params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def evaluate(self, query: str, user_id: str) -> Dict[str, Any]:
        """Run a retrieval query and get detailed scoring breakdown."""
        payload = {"query": query, "user_id": user_id}
        resp = requests.post(f"{self.base_url}/intelligence/evaluate", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def mcis(self, user_id: Optional[str] = None, save: bool = False) -> Dict[str, Any]:
        """Compute the Memgraph Cognitive Integrity Score (0-100)."""
        params = {"save": str(save).lower()}
        if user_id:
            params["user_id"] = user_id
        resp = requests.get(f"{self.base_url}/intelligence/mcis", params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def mcis_history(self, user_id: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        """Get historical MCIS snapshots for trend visualization."""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        resp = requests.get(f"{self.base_url}/intelligence/mcis/history", params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def benchmark(self, scenario: str) -> Dict[str, Any]:
        """Run a memory benchmark scenario (e.g. 'contradiction_storm', 'retrieval_accuracy')."""
        payload = {"scenario": scenario}
        resp = requests.post(f"{self.base_url}/benchmark/run", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def benchmark_scenarios(self) -> List[Dict]:
        """List available benchmark scenarios."""
        resp = requests.get(f"{self.base_url}/benchmark/scenarios", headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("scenarios", [])
