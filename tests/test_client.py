"""Unit tests for MemgraphClient."""
import unittest
from unittest.mock import MagicMock, patch

from memgraph_sdk import MemgraphClient, __version__
from memgraph_sdk.exceptions import (
    MemgraphAuthError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __version__ == "0.3.0"


class TestClientInit(unittest.TestCase):
    def test_defaults(self):
        client = MemgraphClient(api_key="test_key", tenant_id="test_tenant")
        assert client.api_key == "test_key"
        assert client.tenant_id == "test_tenant"
        assert client.base_url == "http://localhost:8001/v1"
        assert client.timeout == 30.0
        assert client.max_retries == 3

    def test_custom_url(self):
        client = MemgraphClient(api_key="k", tenant_id="t", base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_custom_retries(self):
        client = MemgraphClient(api_key="k", tenant_id="t", max_retries=5, timeout=10.0)
        assert client.max_retries == 5
        assert client.timeout == 10.0

    def test_headers(self):
        client = MemgraphClient(api_key="my_key", tenant_id="t")
        assert client._session.headers.get("X-API-KEY") == "my_key"


def _make_mock_response(json_data, status_code=200):
    """Create a mock response object."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.ok = status_code < 400
    resp.status_code = status_code
    resp.text = str(json_data)
    resp.headers = {}
    return resp


class TestClientMethods(unittest.TestCase):
    def setUp(self):
        self.client = MemgraphClient(api_key="k", tenant_id="t", max_retries=1)

    @patch.object(MemgraphClient, "_request")
    def test_add(self, mock_request):
        mock_request.return_value = _make_mock_response({"event_id": "123"})
        result = self.client.add("test memory", user_id="u1")
        assert result == {"event_id": "123"}
        mock_request.assert_called_once()

    @patch.object(MemgraphClient, "_request")
    def test_search(self, mock_request):
        mock_request.return_value = _make_mock_response({"memories": []})
        result = self.client.search("query", user_id="u1")
        assert result == {"memories": []}
        mock_request.assert_called_once()

    @patch.object(MemgraphClient, "_request")
    def test_remember(self, mock_request):
        mock_request.return_value = _make_mock_response({"id": "belief-1"})
        result = self.client.remember("User likes dark mode", user_id="u1", category="preference")
        assert result == {"id": "belief-1"}
        mock_request.assert_called_once()

    @patch.object(MemgraphClient, "_request")
    def test_get_beliefs(self, mock_request):
        mock_request.return_value = _make_mock_response({"items": [], "next_cursor": None})
        result = self.client.get_beliefs(user_id="u1", limit=20)
        assert result == {"items": [], "next_cursor": None}

    @patch.object(MemgraphClient, "_request")
    def test_health(self, mock_request):
        mock_request.return_value = _make_mock_response({"status": "healthy"})
        result = self.client.health()
        assert result == {"status": "healthy"}

    @patch.object(MemgraphClient, "_request")
    def test_contradictions(self, mock_request):
        mock_request.return_value = _make_mock_response({"contradictions": []})
        result = self.client.contradictions(user_id="u1")
        assert result == {"contradictions": []}

    @patch.object(MemgraphClient, "_request")
    def test_mcis(self, mock_request):
        mock_request.return_value = _make_mock_response({"score": 85})
        result = self.client.mcis(save=True)
        assert result == {"score": 85}

    @patch.object(MemgraphClient, "_request")
    def test_evaluate(self, mock_request):
        mock_request.return_value = _make_mock_response({"results": []})
        result = self.client.evaluate(query="test", user_id="u1")
        assert result == {"results": []}

    @patch.object(MemgraphClient, "_request")
    def test_benchmark(self, mock_request):
        mock_request.return_value = _make_mock_response({"passed": True})
        result = self.client.benchmark(scenario="preference_recall")
        assert result == {"passed": True}


class TestClientErrorHandling(unittest.TestCase):
    def setUp(self):
        self.client = MemgraphClient(api_key="k", tenant_id="t", max_retries=1)

    def test_auth_error(self):
        resp = _make_mock_response({"detail": "Invalid key"}, status_code=401)
        with self.assertRaises(MemgraphAuthError):
            self.client._raise_for_status(resp)

    def test_validation_error(self):
        resp = _make_mock_response({"detail": "Bad request"}, status_code=422)
        with self.assertRaises(MemgraphValidationError):
            self.client._raise_for_status(resp)

    def test_rate_limit_error(self):
        resp = _make_mock_response({"detail": "Rate limited"}, status_code=429)
        resp.headers = {"Retry-After": "30"}
        with self.assertRaises(MemgraphRateLimitError) as ctx:
            self.client._raise_for_status(resp)
        assert ctx.exception.retry_after == 30

    def test_ok_response_no_error(self):
        resp = _make_mock_response({"ok": True}, status_code=200)
        # Should not raise
        self.client._raise_for_status(resp)


class TestAsyncClientImport(unittest.TestCase):
    def test_import_does_not_fail(self):
        # AsyncMemgraphClient may be None if httpx not installed
        # but the import itself should not fail
        from memgraph_sdk import AsyncMemgraphClient  # noqa: F401


if __name__ == "__main__":
    unittest.main()
