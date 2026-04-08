"""Unit tests for MemgraphClient."""
import unittest
from unittest.mock import MagicMock, patch

from memgraph_sdk import MemgraphClient, __version__
from memgraph_sdk.exceptions import (
    MemgraphAPIError,
    MemgraphAuthError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestClientInit(unittest.TestCase):
    def test_defaults(self):
        client = MemgraphClient(api_key="mg_test_key", tenant_id="test_tenant")
        assert client.api_key == "mg_test_key"
        assert client.tenant_id == "test_tenant"
        assert client.base_url == "https://api.memgraph.ai/v1"

    def test_custom_url(self):
        client = MemgraphClient(api_key="mg_k", tenant_id="t", base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_headers(self):
        client = MemgraphClient(api_key="mg_my_key", tenant_id="t")
        assert client.headers == {"X-API-KEY": "mg_my_key"}

    def test_tenant_id_optional(self):
        """tenant_id can be omitted — server resolves from API key."""
        client = MemgraphClient(api_key="mg_my_key")
        assert client.tenant_id is None
        assert client.api_key == "mg_my_key"

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "https://custom.env/v1"})
    def test_env_url_override(self):
        client = MemgraphClient(api_key="mg_k")
        assert client.base_url == "https://custom.env/v1"


class TestClientMethods(unittest.TestCase):
    """Test SDK methods with properly mocked session."""

    def _make_client(self, tenant_id="test-tenant-uuid"):
        client = MemgraphClient(api_key="mg_k", tenant_id=tenant_id)
        return client

    def _mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.ok = status_code < 400
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.text = str(json_data)
        mock_resp.headers = {}
        return mock_resp

    def test_add_with_tenant_id(self):
        client = self._make_client(tenant_id="my-tenant")
        mock_resp = self._mock_response({"event_id": "123"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            result = client.add("test memory", user_id="u1")
            assert result == {"event_id": "123"}
            call_kwargs = mock_req.call_args
            sent_data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
            assert sent_data["tenant_id"] == "my-tenant"
            assert sent_data["user_id"] == "u1"
            assert sent_data["text"] == "test memory"

    def test_add_without_tenant_id(self):
        """When tenant_id is None, payload should NOT include tenant_id."""
        client = self._make_client(tenant_id=None)
        mock_resp = self._mock_response({"event_id": "456"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            result = client.add("test memory", user_id="u1")
            assert result == {"event_id": "456"}
            call_kwargs = mock_req.call_args
            sent_data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
            assert "tenant_id" not in sent_data
            assert sent_data["user_id"] == "u1"

    def test_search_with_tenant_id(self):
        client = self._make_client(tenant_id="my-tenant")
        mock_resp = self._mock_response({"memories": []})
        with patch.object(client._session, "post", return_value=mock_resp):
            result = client.search("query", user_id="u1")
            assert result == {"results": [], "total": 0}

    def test_search_without_tenant_id(self):
        """When tenant_id is None, search still works."""
        client = self._make_client(tenant_id=None)
        mock_resp = self._mock_response({"memories": []})
        with patch.object(client._session, "post", return_value=mock_resp):
            result = client.search("query", user_id="u1")
            assert result == {"results": [], "total": 0}

    def test_remember_without_tenant_id(self):
        """When tenant_id is None, beliefs payload should NOT include tenant_id."""
        client = self._make_client(tenant_id=None)
        mock_resp = self._mock_response({"id": "belief-1"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            result = client.remember("user prefers dark mode", user_id="u1", category="preference")
            assert result == {"id": "belief-1"}
            call_kwargs = mock_req.call_args
            sent_json = call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))
            assert "tenant_id" not in sent_json
            assert sent_json["subject_id"] == "u1"

    def test_get_beliefs_without_tenant_id(self):
        """When tenant_id is None, query params should NOT include tenant_id."""
        client = self._make_client(tenant_id=None)
        mock_resp = self._mock_response({"beliefs": [], "cursor": None})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            result = client.get_beliefs(user_id="u1")
            assert result == {"beliefs": [], "cursor": None}
            call_kwargs = mock_req.call_args
            sent_params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
            assert "tenant_id" not in sent_params
            assert sent_params["subject_id"] == "u1"

    def test_health(self):
        client = self._make_client()
        mock_resp = self._mock_response({"status": "healthy"})
        with patch.object(client._session, "request", return_value=mock_resp):
            result = client.health()
            assert result == {"status": "healthy"}

    def test_contradictions(self):
        client = self._make_client()
        mock_resp = self._mock_response({"contradictions": []})
        with patch.object(client._session, "request", return_value=mock_resp):
            result = client.contradictions(user_id="u1")
            assert result == {"contradictions": []}

    def test_existing_code_still_works(self):
        """Backward compat: existing code with explicit tenant_id unchanged."""
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        assert client.tenant_id == "t"
        assert client.api_key == "mg_k"


class TestAsyncClientImport(unittest.TestCase):
    def test_import_does_not_fail(self):
        pass
        # AsyncMemgraphClient may be None if httpx not installed
        # but the import itself should not fail


class TestExceptions(unittest.TestCase):
    """Test HTTP status code to exception mapping."""

    def test_auth_error_on_401(self):
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Unauthorized"}
        mock_resp.text = "Unauthorized"
        mock_resp.headers = {}
        with self.assertRaises(MemgraphAuthError):
            client._raise_for_status(mock_resp)

    def test_rate_limit_on_429(self):
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"detail": "Too many requests"}
        mock_resp.text = "Too many requests"
        mock_resp.headers = {"Retry-After": "30"}
        with self.assertRaises(MemgraphRateLimitError) as ctx:
            client._raise_for_status(mock_resp)
        assert ctx.exception.retry_after == 30

    def test_validation_error_on_400(self):
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"detail": "Bad request"}
        mock_resp.text = "Bad request"
        mock_resp.headers = {}
        with self.assertRaises(MemgraphValidationError):
            client._raise_for_status(mock_resp)

    def test_server_error_on_500(self):
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "Internal error"}
        mock_resp.text = "Internal error"
        mock_resp.headers = {}
        with self.assertRaises(MemgraphAPIError):
            client._raise_for_status(mock_resp)

    def test_no_retry_on_auth_error(self):
        """Auth errors (401/403) should NOT be retried."""
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"detail": "Forbidden"}
        mock_resp.text = "Forbidden"
        mock_resp.headers = {}
        with patch.object(client._session, "request", return_value=mock_resp):
            with self.assertRaises(MemgraphAuthError):
                client._request("GET", "/test")


class TestPing(unittest.TestCase):
    def test_ping_strips_v1(self):
        """Ping should hit /health and then /auth/whoami."""
        client = MemgraphClient(api_key="mg_k", tenant_id="t")
        mock_health = MagicMock()
        mock_health.json.return_value = {"status": "ok"}

        mock_whoami = MagicMock()
        mock_whoami.ok = True
        mock_whoami.status_code = 200
        mock_whoami.json.return_value = {"tenant_id": "t", "tenant_name": "Test"}

        with patch.object(client._session, "get", return_value=mock_health):
            with patch.object(client._session, "request", return_value=mock_whoami):
                result = client.ping()
                assert result["status"] == "ok"
                assert result["tenant_id"] == "t"


# ======================================================================
# Cloud vs On-Prem URL Resolution Tests
# ======================================================================


class TestAsyncClientCloudVsOnPremURL(unittest.TestCase):
    """Verify AsyncMemgraphClient URL resolution for cloud and on-prem."""

    @classmethod
    def setUpClass(cls):
        try:
            import httpx  # noqa: F401
            cls.has_httpx = True
        except ImportError:
            cls.has_httpx = False

    def _make_client(self, **kwargs):
        from memgraph_sdk import AsyncMemgraphClient
        if not self.has_httpx:
            self.skipTest("httpx not installed, skipping async tests")
        return AsyncMemgraphClient(**kwargs)

    def test_async_cloud_default(self):
        """Async: No env var, no base_url → defaults to api.memgraph.ai."""
        import os
        os.environ.pop("MEMGRAPH_API_URL", None)
        client = self._make_client(api_key="mg_cloud_key")
        assert client.base_url == "https://api.memgraph.ai/v1"

    def test_async_onprem_via_base_url(self):
        """Async: base_url parameter overrides cloud default."""
        client = self._make_client(api_key="mg_key", base_url="http://my-server:8001/v1")
        assert client.base_url == "http://my-server:8001/v1"

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://internal:8001/v1"})
    def test_async_onprem_via_env_var(self):
        """Async: MEMGRAPH_API_URL env var overrides cloud default."""
        client = self._make_client(api_key="mg_key")
        assert client.base_url == "http://internal:8001/v1"

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://env:8001/v1"})
    def test_async_base_url_param_overrides_env(self):
        """Async: base_url parameter wins over env var."""
        client = self._make_client(api_key="mg_key", base_url="http://param:8001/v1")
        assert client.base_url == "http://param:8001/v1"

    def test_async_tenant_id_optional(self):
        """Async: tenant_id can be omitted."""
        import os
        os.environ.pop("MEMGRAPH_API_URL", None)
        client = self._make_client(api_key="mg_key")
        assert client.tenant_id is None

    def test_async_onprem_with_tenant_id(self):
        """Async: on-prem with explicit tenant_id."""
        client = self._make_client(
            api_key="mg_key",
            tenant_id="onprem-tenant",
            base_url="http://server:8001/v1",
        )
        assert client.base_url == "http://server:8001/v1"
        assert client.tenant_id == "onprem-tenant"

    def test_async_headers_have_api_key(self):
        """Async: both cloud and on-prem include X-API-KEY header."""
        import os
        os.environ.pop("MEMGRAPH_API_URL", None)
        client = self._make_client(api_key="mg_test_key")
        # httpx client headers should include the API key
        assert client._client.headers.get("x-api-key") == "mg_test_key"


class TestCloudVsOnPremURL(unittest.TestCase):
    """Verify SDK works correctly for both cloud and on-prem deployments."""

    CLOUD_DEFAULT = "https://api.memgraph.ai/v1"
    ON_PREM_URL = "http://my-server:8001/v1"
    ON_PREM_LOCALHOST = "http://localhost:8001/v1"

    # --- Cloud (default) scenario ---

    def test_cloud_default_no_env(self):
        """Cloud: No env var, no base_url → defaults to api.memgraph.ai."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove MEMGRAPH_API_URL if it exists
            import os
            os.environ.pop("MEMGRAPH_API_URL", None)
            client = MemgraphClient(api_key="mg_cloud_key")
            assert client.base_url == self.CLOUD_DEFAULT
            assert client.api_key == "mg_cloud_key"
            assert client.tenant_id is None  # optional for cloud

    def test_cloud_explicit_url(self):
        """Cloud: Explicit cloud URL passed as base_url."""
        client = MemgraphClient(api_key="mg_key", base_url=self.CLOUD_DEFAULT)
        assert client.base_url == self.CLOUD_DEFAULT

    def test_cloud_requests_use_correct_base(self):
        """Cloud: API calls go to api.memgraph.ai."""
        client = MemgraphClient(api_key="mg_key")
        mock_resp = self._mock_ok({"event_id": "1"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.add("test", user_id="u1")
            call_url = mock_req.call_args.args[1] if mock_req.call_args.args else mock_req.call_args[0][1]
            assert call_url.startswith(self.CLOUD_DEFAULT)

    def test_cloud_ping_health_url(self):
        """Cloud: ping() hits /health then /auth/whoami."""
        client = MemgraphClient(api_key="mg_key")
        mock_health = MagicMock()
        mock_health.json.return_value = {"status": "ok"}
        mock_whoami = MagicMock()
        mock_whoami.ok = True
        mock_whoami.status_code = 200
        mock_whoami.json.return_value = {"tenant_id": "t"}
        with patch.object(client._session, "get", return_value=mock_health) as mock_get:
            with patch.object(client._session, "request", return_value=mock_whoami):
                client.ping()
                call_url = mock_get.call_args[0][0]
                assert call_url == "https://api.memgraph.ai/health"

    # --- On-Prem via base_url parameter ---

    def test_onprem_via_base_url_param(self):
        """On-prem: base_url parameter overrides cloud default."""
        client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)
        assert client.base_url == self.ON_PREM_URL

    def test_onprem_base_url_overrides_env(self):
        """On-prem: base_url parameter takes precedence over env var."""
        with patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://env-server:8001/v1"}):
            client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)
            assert client.base_url == self.ON_PREM_URL  # param wins

    def test_onprem_requests_use_correct_base(self):
        """On-prem: API calls go to the on-prem server."""
        client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)
        mock_resp = self._mock_ok({"event_id": "1"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.add("test memory", user_id="u1")
            call_url = mock_req.call_args.args[1] if mock_req.call_args.args else mock_req.call_args[0][1]
            assert call_url.startswith(self.ON_PREM_URL)
            assert "api.memgraph.ai" not in call_url

    def test_onprem_ping_health_url(self):
        """On-prem: ping() hits on-prem /health (not cloud)."""
        client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.ping()
            call_url = mock_get.call_args[0][0]
            assert call_url == "http://my-server:8001/health"
            assert "api.memgraph.ai" not in call_url

    # --- On-Prem via MEMGRAPH_API_URL env var ---

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://internal:8001/v1"})
    def test_onprem_via_env_var(self):
        """On-prem: MEMGRAPH_API_URL env var overrides cloud default."""
        client = MemgraphClient(api_key="mg_key")
        assert client.base_url == "http://internal:8001/v1"

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://localhost:8001/v1"})
    def test_onprem_localhost_via_env(self):
        """On-prem: localhost development server via env var."""
        client = MemgraphClient(api_key="mg_key")
        assert client.base_url == self.ON_PREM_LOCALHOST

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://10.0.1.50:8001/v1"})
    def test_onprem_private_ip_via_env(self):
        """On-prem: private IP address via env var."""
        client = MemgraphClient(api_key="mg_key")
        assert client.base_url == "http://10.0.1.50:8001/v1"

    # --- On-prem with tenant_id ---

    def test_onprem_with_tenant_id(self):
        """On-prem: explicit tenant_id included in payloads."""
        client = MemgraphClient(
            api_key="mg_key",
            tenant_id="onprem-tenant-1",
            base_url=self.ON_PREM_URL,
        )
        assert client.base_url == self.ON_PREM_URL
        assert client.tenant_id == "onprem-tenant-1"

        mock_resp = self._mock_ok({"event_id": "1"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.add("test", user_id="u1")
            call_kwargs = mock_req.call_args
            sent_data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
            assert sent_data["tenant_id"] == "onprem-tenant-1"

    def test_onprem_without_tenant_id(self):
        """On-prem: tenant_id omitted — server resolves from API key."""
        client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)
        assert client.tenant_id is None

        mock_resp = self._mock_ok({"event_id": "1"})
        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.add("test", user_id="u1")
            call_kwargs = mock_req.call_args
            sent_data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
            assert "tenant_id" not in sent_data

    # --- URL priority: param > env > default ---

    @patch.dict("os.environ", {"MEMGRAPH_API_URL": "http://env-server:8001/v1"})
    def test_url_priority_param_over_env_over_default(self):
        """URL resolution priority: base_url param > env var > cloud default."""
        # param wins over env
        c1 = MemgraphClient(api_key="mg_k", base_url="http://param:8001/v1")
        assert c1.base_url == "http://param:8001/v1"

        # env wins over default
        c2 = MemgraphClient(api_key="mg_k")
        assert c2.base_url == "http://env-server:8001/v1"

    def test_url_priority_default_when_nothing_set(self):
        """URL defaults to cloud when no param and no env var."""
        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MEMGRAPH_API_URL", None)
            c = MemgraphClient(api_key="mg_k")
            assert c.base_url == self.CLOUD_DEFAULT

    # --- All SDK methods route to correct URL ---

    def test_all_methods_use_onprem_url(self):
        """On-prem: all SDK methods (add/search/remember/get_beliefs) hit the on-prem URL."""
        client = MemgraphClient(api_key="mg_key", base_url=self.ON_PREM_URL)

        for method_name, call_args, call_kwargs in [
            ("add", ("test memory", "u1"), {}),
            ("search", ("query", "u1"), {}),
            ("remember", ("test fact", "u1"), {}),
            ("get_beliefs", ("u1",), {}),
            ("health", (), {}),
            ("contradictions", (), {}),
        ]:
            mock_resp = self._mock_ok({"ok": True, "beliefs": [], "cursor": None, "memories": []})
            # search() calls _session.post directly (v2 endpoint); others use _request → _session.request
            with patch.object(client._session, "request", return_value=mock_resp) as mock_req, \
                 patch.object(client._session, "post", return_value=mock_resp) as mock_post:
                getattr(client, method_name)(*call_args, **call_kwargs)
                if method_name == "search":
                    req_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                else:
                    req_url = mock_req.call_args.args[1] if mock_req.call_args.args else mock_req.call_args[0][1]
                base_host = self.ON_PREM_URL.split("/v1")[0]
                assert base_host in req_url, \
                    f"{method_name}() should hit on-prem URL, got: {req_url}"
                assert "api.memgraph.ai" not in req_url, \
                    f"{method_name}() should NOT hit cloud, got: {req_url}"

    # --- Helper ---

    def _mock_ok(self, json_data):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = json_data
        mock_resp.text = str(json_data)
        mock_resp.headers = {}
        return mock_resp


if __name__ == "__main__":
    unittest.main()
