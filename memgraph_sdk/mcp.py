#!/usr/bin/env python3
"""
Memgraph MCP Server - Model Context Protocol integration for AI agents.

Exposes Memgraph memory tools via MCP for Claude Desktop, Cursor, VS Code, etc.

Usage:
    python -m memgraph_sdk.mcp

Environment variables:
    MEMGRAPH_API_URL    - Backend URL (default: https://api.memgraph.ai/v1)
    MEMGRAPH_API_KEY    - API key (required, format: mg_...)
    MEMGRAPH_TENANT_ID  - Tenant ID (optional, resolved from API key if not set)
    MEMGRAPH_AGENT_USER_ID - Default user ID for memories (default: ai_agent)
"""

import asyncio
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
    from mcp.types import Tool, TextContent, Resource
except ImportError:
    print(
        "Error: MCP library not installed. Install with: pip install 'memgraph-sdk[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

# Memgraph SDK import
from memgraph_sdk import MemgraphClient
from memgraph_sdk.exceptions import MemgraphConnectionError

# --- Configuration ---
API_KEY = os.getenv("MEMGRAPH_API_KEY")
TENANT_ID = os.getenv("MEMGRAPH_TENANT_ID")  # optional
API_URL = os.getenv("MEMGRAPH_API_URL")  # optional, SDK defaults to cloud
AGENT_USER_ID = os.getenv("MEMGRAPH_AGENT_USER_ID", "ai_agent")

if not API_KEY:
    logger.error("MEMGRAPH_API_KEY is required. Get one at https://memgraph.ai")
    sys.exit(1)

# Initialize client — tenant_id is optional (resolved server-side from API key)
_client_kwargs: Dict[str, Any] = {"api_key": API_KEY}
if TENANT_ID:
    _client_kwargs["tenant_id"] = TENANT_ID
if API_URL:
    _client_kwargs["base_url"] = API_URL

memgraph = MemgraphClient(**_client_kwargs)
logger.info("Memgraph MCP Server initialized: %s", memgraph.base_url)

# Create MCP server
app = Server("memgraph-memory")


# ============================================================================
# Tool definitions
# ============================================================================

TOOLS = [
    Tool(
        name="memgraph_remember",
        description=(
            "Store a memory in Memgraph. Use this to remember: user preferences, "
            "project decisions, architecture choices, bug fixes, and important context."
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
                    "description": "Category: decision, architecture, bug_fix, preference, general",
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
            "Search memories using semantic similarity. Use this to recall: "
            "past decisions, user preferences, previous bug fixes, architecture choices."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (natural language)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memgraph_profile",
        description=(
            "Get the user's memory profile - beliefs, preferences, and facts "
            "that Memgraph has consolidated from past interactions."
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


async def handle_remember(text: str, category: str = "general") -> Dict[str, Any]:
    """Store a memory."""
    try:
        result = memgraph.remember(text=text, user_id=AGENT_USER_ID, category=category)
        return {"success": True, "message": f"Remembered: {text[:80]}...", "belief_id": result.get("id")}
    except Exception as e:
        logger.error("Error storing memory: %s", e)
        return {"success": False, "error": str(e)}


async def handle_search(query: str) -> Dict[str, Any]:
    """Search memories."""
    try:
        context = memgraph.search(query=query, user_id=AGENT_USER_ID)
        retrieved_items = context.get("retrieved_items", [])
        results = []
        for item in retrieved_items[:5]:
            if not isinstance(item, dict):
                continue
            results.append({
                "type": item.get("type", "memory"),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            })
        return {"success": True, "query": query, "results_count": len(results), "results": results}
    except Exception as e:
        logger.error("Error searching memories: %s", e)
        return {"success": False, "error": str(e), "results": []}


async def handle_profile() -> Dict[str, Any]:
    """Get user profile."""
    try:
        beliefs_data = memgraph.get_beliefs(user_id=AGENT_USER_ID, limit=20)
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


# ============================================================================
# MCP Protocol handlers
# ============================================================================

import json


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
        result = await handle_search(query=arguments["query"])
    elif name == "memgraph_profile":
        result = await handle_profile()
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


@app.list_resources()
async def list_resources() -> List[Resource]:
    return [
        Resource(
            uri="memgraph://project/status",
            name="Memgraph Status",
            description="Current memory server status",
            mimeType="text/plain",
        ),
        Resource(
            uri="memgraph://memory/recent",
            name="Recent Memories",
            description="Recent memories from the current session",
            mimeType="text/plain",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    uri_str = str(uri)
    if uri_str == "memgraph://project/status":
        try:
            health = memgraph.ping()
            return f"# Memgraph Status\n\n**Server**: {memgraph.base_url}\n**Status**: {health.get('status', 'unknown')}\n"
        except Exception as e:
            return f"# Memgraph Status\n\n**Error**: {e}"
    elif uri_str == "memgraph://memory/recent":
        try:
            beliefs = memgraph.get_beliefs(user_id=AGENT_USER_ID, limit=10)
            items = beliefs.get("items", [])
            if not isinstance(items, list):
                items = []
            output = "# Recent Memories\n\n"
            for b in items:
                output += f"- **{b.get('key', '?')}**: {b.get('value', '')[:100]}\n"
            if not items:
                output += "_No memories yet._\n"
            return output
        except Exception as e:
            return f"# Recent Memories\n\n**Error**: {e}"
    else:
        return f"Unknown resource: {uri_str}"


# ============================================================================
# Main
# ============================================================================


async def main():
    """Start the MCP server on stdio."""
    logger.info("Starting Memgraph MCP Server...")
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
