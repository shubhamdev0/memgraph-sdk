"""Tests for the MCP server module."""
import asyncio
import os
import sys
import unittest
from unittest.mock import patch

try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _run(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


@unittest.skipUnless(HAS_MCP, "mcp library not installed (pip install 'memgraph-sdk[mcp]')")
class TestMCPServerModule(unittest.TestCase):
    """Test MCP server can be imported and configured correctly."""

    def test_mcp_module_exists(self):
        """The mcp module file exists in the SDK package."""
        from pathlib import Path
        mcp_path = Path(__file__).parent.parent / "memgraph_sdk" / "mcp.py"
        assert mcp_path.exists(), f"MCP server not found at {mcp_path}"

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_test_key_123"})
    def test_mcp_requires_api_key(self):
        """MCP server should have expected attributes when API key is set."""
        # Force reimport with env var set
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]

        try:
            import memgraph_sdk.mcp as mcp_module
            assert hasattr(mcp_module, "app")
            assert hasattr(mcp_module, "main")
            assert hasattr(mcp_module, "run")
        except SystemExit:
            self.fail("MCP module exited even with MEMGRAPH_API_KEY set")
        finally:
            # Clean up to avoid polluting other tests
            if "memgraph_sdk.mcp" in sys.modules:
                del sys.modules["memgraph_sdk.mcp"]

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_test_key_123"})
    def test_mcp_client_uses_sdk(self):
        """MCP server should use MemgraphClient from the SDK."""
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]

        try:
            import memgraph_sdk.mcp as mcp_module
            from memgraph_sdk import MemgraphClient
            # memgraph is lazy-initialized — test _get_client() instead
            client = mcp_module._get_client()
            assert isinstance(client, MemgraphClient)
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            if "memgraph_sdk.mcp" in sys.modules:
                del sys.modules["memgraph_sdk.mcp"]

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_test_key_123"})
    def test_mcp_default_user_id(self):
        """Default agent user ID should be 'ai_agent'."""
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]

        try:
            import memgraph_sdk.mcp as mcp_module
            assert mcp_module.AGENT_USER_ID == os.getenv("MEMGRAPH_AGENT_USER_ID", "ai_agent")
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            if "memgraph_sdk.mcp" in sys.modules:
                del sys.modules["memgraph_sdk.mcp"]

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_test_key_123"})
    def test_mcp_tools_defined(self):
        """MCP server should define 3 tools: remember, search, profile."""
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]

        try:
            import memgraph_sdk.mcp as mcp_module
            tool_names = [t.name for t in mcp_module.TOOLS]
            assert "memgraph_remember" in tool_names
            assert "memgraph_search" in tool_names
            assert "memgraph_profile" in tool_names
            assert len(tool_names) == 3
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            if "memgraph_sdk.mcp" in sys.modules:
                del sys.modules["memgraph_sdk.mcp"]


@unittest.skipUnless(HAS_MCP, "mcp library not installed (pip install 'memgraph-sdk[mcp]')")
class TestMCPToolHandlers(unittest.TestCase):
    """Test MCP tool handler implementations."""

    @classmethod
    def setUpClass(cls):
        """Import MCP module once with API key set."""
        os.environ["MEMGRAPH_API_KEY"] = "mg_test_key_123"
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]
        try:
            import memgraph_sdk.mcp
            cls.mcp_module = memgraph_sdk.mcp
        except SystemExit:
            cls.mcp_module = None

    @classmethod
    def tearDownClass(cls):
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]
        os.environ.pop("MEMGRAPH_API_KEY", None)

    def setUp(self):
        if self.mcp_module is None:
            self.skipTest("MCP module failed to import (missing mcp library?)")

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_remember_calls_sdk(self, mock_client):
        """handle_remember should call client.remember()."""
        mock_client.return_value.remember.return_value = {"id": "belief-123"}
        result = _run(self.mcp_module.handle_remember("test memory", category="general"))

        assert result["success"] is True
        assert result["belief_id"] == "belief-123"
        mock_client.return_value.remember.assert_called_once()

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_search_calls_sdk(self, mock_client):
        """handle_search should call client.search()."""
        mock_client.return_value.search.return_value = {
            "results": [
                {"type": "belief", "content": "uses PostgreSQL", "score": 0.95, "metadata": {}}
            ],
            "total": 1,
        }
        result = _run(self.mcp_module.handle_search("database choice"))

        assert result["success"] is True
        assert result["results_count"] == 1
        assert result["results"][0]["content"] == "uses PostgreSQL"
        mock_client.return_value.search.assert_called_once()

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_search_handles_error(self, mock_client):
        """handle_search should handle errors gracefully."""
        from memgraph_sdk.exceptions import MemgraphConnectionError

        mock_client.return_value.search.side_effect = MemgraphConnectionError("Connection refused")
        result = _run(self.mcp_module.handle_search("test query"))

        assert result["success"] is False
        assert "Connection refused" in result["error"]

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_remember_handles_error(self, mock_client):
        """handle_remember should handle errors gracefully."""
        mock_client.return_value.remember.side_effect = Exception("Server error")
        result = _run(self.mcp_module.handle_remember("test"))

        assert result["success"] is False
        assert "Server error" in result["error"]

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_profile_calls_get_beliefs(self, mock_client):
        """handle_profile should call client.get_beliefs()."""
        mock_client.return_value.get_beliefs.return_value = {
            "items": [
                {"key": "preference_dark_mode", "value": "prefers dark mode", "confidence": 0.9, "belief_type": "belief"},
                {"key": "tech_stack", "value": "uses PostgreSQL", "confidence": 0.95, "belief_type": "fact"},
            ]
        }
        result = _run(self.mcp_module.handle_profile())

        assert result["success"] is True
        assert len(result["profile"]["preferences"]) == 1  # "prefer" in key
        assert len(result["profile"]["facts"]) == 1  # belief_type == "fact"
        mock_client.return_value.get_beliefs.assert_called_once()

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_search_limits_results(self, mock_client):
        """Search results should be limited to 5."""
        # search() enforces limit internally, so mock returns limited set
        mock_client.return_value.search.return_value = {
            "results": [{"type": "belief", "content": f"item {i}", "score": 0.5, "metadata": {}} for i in range(5)],
            "total": 5,
        }
        result = _run(self.mcp_module.handle_search("test", limit=5))

        assert result["results_count"] == 5
        mock_client.return_value.search.assert_called_once_with(query="test", user_id=unittest.mock.ANY, limit=5)

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_profile_handles_error(self, mock_client):
        """handle_profile should handle errors gracefully."""
        mock_client.return_value.get_beliefs.side_effect = Exception("DB error")
        result = _run(self.mcp_module.handle_profile())

        assert result["success"] is False
        assert "DB error" in result["error"]

    @patch("memgraph_sdk.mcp._get_client")
    def test_handle_remember_truncates_message(self, mock_client):
        """handle_remember should truncate long text in success message."""
        long_text = "x" * 200
        mock_client.return_value.remember.return_value = {"id": "belief-456"}
        result = _run(self.mcp_module.handle_remember(long_text))

        assert result["success"] is True
        assert len(result["message"]) < 200  # truncated


# ======================================================================
# Cloud vs On-Prem URL Resolution Tests — MCP Server
# ======================================================================


@unittest.skipUnless(HAS_MCP, "mcp library not installed (pip install 'memgraph-sdk[mcp]')")
class TestMCPCloudVsOnPremURL(unittest.TestCase):
    """Verify MCP server URL resolution works for both cloud and on-prem."""

    def _reimport_mcp(self):
        """Force reimport of MCP module to pick up new env vars."""
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]
        import memgraph_sdk.mcp
        return memgraph_sdk.mcp

    def _cleanup_mcp(self):
        if "memgraph_sdk.mcp" in sys.modules:
            del sys.modules["memgraph_sdk.mcp"]

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_cloud_key"}, clear=False)
    def test_mcp_cloud_default_url(self):
        """MCP server without MEMGRAPH_API_URL should use cloud default."""
        os.environ.pop("MEMGRAPH_API_URL", None)
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().base_url == "https://api.memgraph.ai/v1"
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {
        "MEMGRAPH_API_KEY": "mg_onprem_key",
        "MEMGRAPH_API_URL": "http://my-server:8001/v1",
    })
    def test_mcp_onprem_url_from_env(self):
        """MCP server with MEMGRAPH_API_URL should use on-prem URL."""
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().base_url == "http://my-server:8001/v1"
            assert "api.memgraph.ai" not in mcp._get_client().base_url
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {
        "MEMGRAPH_API_KEY": "mg_local_key",
        "MEMGRAPH_API_URL": "http://localhost:8001/v1",
    })
    def test_mcp_localhost_url(self):
        """MCP server with localhost URL for local development."""
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().base_url == "http://localhost:8001/v1"
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {
        "MEMGRAPH_API_KEY": "mg_key",
        "MEMGRAPH_TENANT_ID": "explicit-tenant",
    })
    def test_mcp_with_explicit_tenant_id(self):
        """MCP server with MEMGRAPH_TENANT_ID should pass it to client."""
        os.environ.pop("MEMGRAPH_API_URL", None)
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().tenant_id == "explicit-tenant"
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_key"}, clear=False)
    def test_mcp_without_tenant_id(self):
        """MCP server without MEMGRAPH_TENANT_ID should have None tenant_id."""
        os.environ.pop("MEMGRAPH_TENANT_ID", None)
        os.environ.pop("MEMGRAPH_API_URL", None)
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().tenant_id is None
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {
        "MEMGRAPH_API_KEY": "mg_key",
        "MEMGRAPH_API_URL": "http://10.0.1.100:8001/v1",
        "MEMGRAPH_TENANT_ID": "private-tenant",
    })
    def test_mcp_private_network_full_config(self):
        """MCP server with full on-prem config (private IP + tenant + key)."""
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().base_url == "http://10.0.1.100:8001/v1"
            assert mcp._get_client().tenant_id == "private-tenant"
            assert mcp._get_client().api_key == "mg_key"
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()

    @patch.dict(os.environ, {"MEMGRAPH_API_KEY": "mg_key"}, clear=False)
    def test_mcp_api_key_in_client_headers(self):
        """MCP client should include X-API-KEY header for both cloud and on-prem."""
        os.environ.pop("MEMGRAPH_API_URL", None)
        os.environ.pop("MEMGRAPH_TENANT_ID", None)
        try:
            mcp = self._reimport_mcp()
            assert mcp._get_client().headers == {"X-API-KEY": "mg_key"}
        except SystemExit:
            self.fail("MCP module exited unexpectedly")
        finally:
            self._cleanup_mcp()


if __name__ == "__main__":
    unittest.main()
