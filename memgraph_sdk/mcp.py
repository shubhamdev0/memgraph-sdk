#!/usr/bin/env python3
"""
Memgraph MCP Server - Cognitive Memory Integration for AI agents.

Exposes Memgraph memory tools via MCP for Claude Desktop, Cursor, VS Code, etc.

v2 UPGRADE: "Always-On" cognitive layer with:
- memgraph_think: Process FULL conversation context, auto-recall + auto-learn in one call
- memgraph_recall: Mandatory auto-recall before every response (enhanced search)
- memgraph_remember: Store a memory (unchanged)
- memgraph_profile: Get user profile (unchanged)

The key change: Tool descriptions now MANDATE memory usage, not suggest it.

Usage:
    python -m memgraph_sdk.mcp

Environment variables:
    MEMGRAPH_API_URL    - Backend URL (default: https://api.memgraph.ai/v1)
    MEMGRAPH_API_KEY    - API key (required, format: mg_...)
    MEMGRAPH_TENANT_ID  - Tenant ID (optional, resolved from API key if not set)
    MEMGRAPH_AGENT_USER_ID - Default user ID for memories (default: ai_agent)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

# Configure logging to stderr (MCP uses stdout for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("memgraph-mcp")

# MCP library import
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, TextContent, Tool
except ImportError:
    print(
        "Error: MCP library not installed. Install with: pip install 'memgraph-sdk[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

# Memgraph SDK import (must be after MCP import check above)
from memgraph_sdk import MemgraphClient  # noqa: E402

# --- Configuration (lazy initialization) ---
API_KEY = os.getenv("MEMGRAPH_API_KEY")
TENANT_ID = os.getenv("MEMGRAPH_TENANT_ID")  # optional
API_URL = os.getenv("MEMGRAPH_API_URL")  # optional, SDK defaults to cloud
AGENT_USER_ID = os.getenv("MEMGRAPH_AGENT_USER_ID", "ai_agent")

# Lazy client initialization — only created when first used.
# Avoids sys.exit(1) at import time which breaks testing and IDE import scanning.
memgraph: Optional[MemgraphClient] = None


def _get_client() -> MemgraphClient:
    """Get or create the Memgraph client. Raises RuntimeError if no API key."""
    global memgraph
    if memgraph is not None:
        return memgraph

    if not API_KEY:
        raise RuntimeError(
            "MEMGRAPH_API_KEY is required. Set it as an environment variable. "
            "Get one at https://memgraph.ai"
        )

    kwargs: Dict[str, Any] = {"api_key": API_KEY}
    if TENANT_ID:
        kwargs["tenant_id"] = TENANT_ID
    if API_URL:
        kwargs["base_url"] = API_URL

    memgraph = MemgraphClient(**kwargs)
    logger.info("Memgraph MCP client initialized: %s", _get_client().base_url)
    return memgraph

# Create MCP server
app = Server("memgraph-memory")


# ============================================================================
# Tool definitions — v2 with MANDATORY memory behavior
# ============================================================================

TOOLS = [
    Tool(
        name="memgraph_remember",
        description=(
            "\n\nStore a memory in Memgraph for persistent recall across sessions.\n\n"
            "Use this to remember:\n"
            "- User preferences and coding style\n"
            "- Project decisions and rationale\n"
            "- Architecture choices and patterns\n"
            "- Bug fixes and solutions found\n"
            "- Important context about the codebase\n"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The memory to store (be specific and descriptive)",
                },
                "category": {
                    "type": "string",
                    "description": "One of: decision, architecture, bug_fix, preference, general",
                    "enum": ["decision", "architecture", "bug_fix", "preference", "general"],
                    "default": "general",
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="memgraph_search",
        description=(
            "\n\nSearch memories using semantic similarity.\n\n"
            "Use this to recall:\n"
            "- Past decisions and why they were made\n"
            "- User preferences and coding patterns\n"
            "- Previous bug fixes and solutions\n"
            "- Architecture and design choices\n"
            "- Any previously stored context\n"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (natural language)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memgraph_profile",
        description=(
            "\n\nGet the user's memory profile - beliefs, preferences, and facts that Memgraph\n"
            "has consolidated from past interactions.\n"
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ============================================================================
# Tool handlers
# ============================================================================


async def handle_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search memories with semantic similarity and result limiting."""
    try:
        search_result = _get_client().search(query=query, user_id=AGENT_USER_ID, limit=limit)
        results = search_result.get("results", [])
        return {"success": True, "query": query, "results_count": len(results), "results": results}
    except Exception as e:
        logger.error("Error searching memories: %s", e)
        return {"success": False, "error": str(e), "results": []}


async def handle_remember(text: str, category: str = "general") -> Dict[str, Any]:
    """Store a memory."""
    try:
        result = _get_client().remember(text=text, user_id=AGENT_USER_ID, category=category)
        return {"success": True, "message": f"Remembered: {text[:80]}...", "belief_id": result.get("id")}
    except Exception as e:
        logger.error("Error storing memory: %s", e)
        return {"success": False, "error": str(e)}


async def handle_forget(belief_id: str = None, domain: str = None, soft: bool = False) -> Dict[str, Any]:
    """Delete memories — specific belief or all for user."""
    try:
        if belief_id:
            result = _get_client().forget(belief_id=belief_id)
            return {"success": True, "message": f"Deleted belief {belief_id}", "result": result}
        else:
            result = _get_client().forget_all(user_id=AGENT_USER_ID, domain=domain, soft=soft)
            return {"success": True, "message": "Bulk deleted memories", "result": result}
    except Exception as e:
        logger.error("Error deleting memory: %s", e)
        return {"success": False, "error": str(e)}


async def handle_recall(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search memories (alias for handle_search)."""
    return await handle_search(query=query, limit=limit)


async def handle_profile() -> Dict[str, Any]:
    """Get user profile."""
    try:
        beliefs_data = _get_client().get_beliefs(user_id=AGENT_USER_ID, limit=20)
        items = beliefs_data.get("items", [])
        if not isinstance(items, list):
            items = []
        profile: Dict[str, list] = {"beliefs": [], "preferences": [], "facts": []}
        for belief in items:
            entry = {
                "key": belief.get("key", ""),
                "value": belief.get("value", ""),
                "confidence": belief.get("confidence", 1.0),
            }
            key = belief.get("key", "").lower()
            if "prefer" in key or "like" in key:
                profile["preferences"].append(entry)
            elif belief.get("belief_type") == "fact":
                profile["facts"].append(entry)
            else:
                profile["beliefs"].append(entry)
        return {"success": True, "profile": profile}
    except Exception as e:
        logger.error("Error getting profile: %s", e)
        return {"success": False, "error": str(e)}


async def handle_think(messages: List[Dict[str, str]], current_query: str = None) -> Dict[str, Any]:
    """
    Process full conversation through cognitive engine.
    Does BOTH recall (for current topic) AND learning (from exchange) in one call.
    """
    try:
        # Determine the query for recall
        if not current_query:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    current_query = msg.get("content", "")
                    break
            if not current_query and messages:
                current_query = messages[-1].get("content", "")

        # RECALL: Get relevant memories for current topic
        recall_result = {"results": []}
        if current_query:
            try:
                search_result = _get_client().search(query=current_query, user_id=AGENT_USER_ID, limit=8)
                recall_result["results"] = search_result.get("results", [])
            except Exception as e:
                logger.warning("Recall step failed in think: %s", e)

        # LEARN: Trigger background learning from the conversation
        learning_triggered = False
        # Only learn from completed exchanges (need at least user + assistant)
        has_user = any(m.get("role") == "user" for m in messages)
        has_assistant = any(m.get("role") == "assistant" for m in messages)

        if has_user and has_assistant:
            try:
                _get_client().sidecar_post_flight(
                    messages=messages, user_id=AGENT_USER_ID, agent_id="mcp_think",
                )
                learning_triggered = True
            except Exception as e:
                logger.warning("Learning step failed in think: %s", e)

        return {
            "success": True,
            "recall": {
                "query": current_query,
                "memories_found": len(recall_result["results"]),
                "results": recall_result["results"],
            },
            "learning": {
                "triggered": learning_triggered,
                "messages_processed": len(messages),
            },
        }
    except Exception as e:
        logger.error("Error in think: %s", e)
        return {"success": False, "error": str(e)}


# ============================================================================
# MCP Protocol handlers
# ============================================================================


@app.list_tools()
async def list_tools() -> List[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
    if name == "memgraph_remember":
        result = await handle_remember(
            text=arguments["text"],
            category=arguments.get("category", "general"),
        )
    elif name == "memgraph_search":
        result = await handle_search(
            query=arguments["query"],
            limit=arguments.get("limit", 5),
        )
    elif name == "memgraph_recall":
        # Backward compatibility alias for memgraph_search
        result = await handle_search(
            query=arguments["query"],
            limit=arguments.get("limit", 5),
        )
    elif name == "memgraph_forget":
        result = await handle_forget(
            belief_id=arguments.get("belief_id"),
            domain=arguments.get("domain"),
            soft=arguments.get("soft", False),
        )
    elif name == "memgraph_profile":
        result = await handle_profile()
    elif name == "memgraph_think":
        result = await handle_think(
            messages=arguments["messages"],
            current_query=arguments.get("current_query"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


@app.list_resources()
async def list_resources() -> List[Resource]:
    return [
        Resource(
            uri="memgraph://project/status",
            name="Memgraph Status",
            description="Current memory server status and configuration",
            mimeType="text/plain",
        ),
        Resource(
            uri="memgraph://memory/recent",
            name="Recent Memories",
            description="Recent memories and beliefs from past interactions",
            mimeType="text/plain",
        ),
        Resource(
            uri="memgraph://memory/profile",
            name="User Memory Profile",
            description="Consolidated user profile from all stored memories",
            mimeType="text/plain",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    uri_str = str(uri)
    if uri_str == "memgraph://project/status":
        try:
            health = _get_client().ping()
            return (
                f"# Memgraph Status\n\n"
                f"**Server**: {_get_client().base_url}\n"
                f"**Status**: {health.get('status', 'unknown')}\n"
                f"**Version**: Cognitive Sidecar v2\n"
            )
        except Exception as e:
            return f"# Memgraph Status\n\n**Error**: {e}"

    elif uri_str == "memgraph://memory/recent":
        try:
            beliefs = _get_client().get_beliefs(user_id=AGENT_USER_ID, limit=15)
            items = beliefs.get("items", [])
            if not isinstance(items, list):
                items = []
            output = "# Recent Memories\n\n"
            for b in items:
                btype = b.get("belief_type", "belief")
                conf = b.get("confidence", 0)
                output += f"- [{btype}] **{b.get('key', '?')}**: {b.get('value', '')[:120]} (confidence: {conf})\n"
            if not items:
                output += "_No memories yet. Memories are automatically created as you work._\n"
            return output
        except Exception as e:
            return f"# Recent Memories\n\n**Error**: {e}"

    elif uri_str == "memgraph://memory/profile":
        try:
            result = await handle_profile()
            if not result.get("success"):
                return f"# Memory Profile\n\n**Error**: {result.get('error', 'Unknown error')}"

            profile = result.get("profile", {})
            output = "# Memory Profile\n\n"

            facts = profile.get("facts", [])
            if facts:
                output += "## Facts\n"
                for f in facts:
                    output += f"- **{f['key']}**: {f['value']}\n"
                output += "\n"

            prefs = profile.get("preferences", [])
            if prefs:
                output += "## Preferences\n"
                for p in prefs:
                    output += f"- **{p['key']}**: {p['value']}\n"
                output += "\n"

            beliefs = profile.get("beliefs", [])
            if beliefs:
                output += "## Beliefs\n"
                for b in beliefs:
                    output += f"- **{b['key']}**: {b['value']} (confidence: {b.get('confidence', 0)})\n"
                output += "\n"

            if not facts and not prefs and not beliefs:
                output += "_No profile data yet. Memories build up automatically over time._\n"

            return output
        except Exception as e:
            return f"# Memory Profile\n\n**Error**: {e}"

    else:
        return f"Unknown resource: {uri_str}"


# ============================================================================
# Main
# ============================================================================


async def main():
    """Start the MCP server on stdio."""
    logger.info("Starting Memgraph MCP Server (v2 Cognitive)...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def run():
    """Entry point for `python -m memgraph_sdk.mcp`."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
