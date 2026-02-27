"""Unit tests for MemgraphClient."""
import unittest
from unittest.mock import patch, MagicMock

from memgraph_sdk import MemgraphClient, __version__


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestClientInit(unittest.TestCase):
    def test_defaults(self):
        client = MemgraphClient(api_key="test_key", tenant_id="test_tenant")
        assert client.api_key == "test_key"
        assert client.tenant_id == "test_tenant"
        assert client.base_url == "http://localhost:8001/v1"

    def test_custom_url(self):
        client = MemgraphClient(api_key="k", tenant_id="t", base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_headers(self):
        client = MemgraphClient(api_key="my_key", tenant_id="t")
        assert client.headers == {"X-API-KEY": "my_key"}


class TestClientMethods(unittest.TestCase):
    @patch("memgraph_sdk.client.requests.post")
    def test_add(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"event_id": "123"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MemgraphClient(api_key="k", tenant_id="t")
        result = client.add("test memory", user_id="u1")

        assert result == {"event_id": "123"}
        mock_post.assert_called_once()

    @patch("memgraph_sdk.client.requests.post")
    def test_search(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"memories": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MemgraphClient(api_key="k", tenant_id="t")
        result = client.search("query", user_id="u1")

        assert result == {"memories": []}
        mock_post.assert_called_once()

    @patch("memgraph_sdk.client.requests.get")
    def test_health(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = MemgraphClient(api_key="k", tenant_id="t")
        result = client.health()

        assert result == {"status": "healthy"}

    @patch("memgraph_sdk.client.requests.get")
    def test_contradictions(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"contradictions": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = MemgraphClient(api_key="k", tenant_id="t")
        result = client.contradictions(user_id="u1")

        assert result == {"contradictions": []}


class TestAsyncClientImport(unittest.TestCase):
    def test_import_does_not_fail(self):
        from memgraph_sdk import AsyncMemgraphClient
        # AsyncMemgraphClient may be None if httpx not installed
        # but the import itself should not fail


if __name__ == "__main__":
    unittest.main()
