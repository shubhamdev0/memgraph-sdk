#!/usr/bin/env python3
"""
Memgraph OS MCP Server
Model Context Protocol server for Claude Desktop, Cursor, and other MCP clients.

This server exposes Memgraph's memory capabilities through the MCP protocol,
enabling IDEs and AI assistants to remember context across sessions.

Features:
- Full CRUD operations (create, read, update, delete)
- Semantic search across all memories
- User profile and belief access
- Episode and event management

Setup:
1. Install dependencies: pip install mcp anthropic-mcp requests
2. Set environment variables:
   export MEMGRAPH_API_URL=http://localhost:8001/v1
   export MEMGRAPH_TENANT_ID=your-tenant-id
   export MEMGRAPH_API_KEY=vel_your_key

3. Add to Claude Desktop config (~/.config/claude/config.json):
   {
     "mcpServers": {
       "memgraph": {
         "command": "python3",
         "args": ["/path/to/mcp_server.py"],
         "env": {
           "MEMGRAPH_API_URL": "http://localhost:8001/v1",
           "MEMGRAPH_TENANT_ID": "your-tenant-id",
           "MEMGRAPH_API_KEY": "vel_your_key"
         }
       }
     }
   }

4. Restart Claude Desktop
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP Protocol imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import Tool
except ImportError:
    print("Error: MCP library not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Memgraph SDK import
try:
    from memgraph_sdk import MemgraphClient
except ImportError:
    print("Error: Memgraph SDK not found. Install with: pip install memgraph-sdk", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("memgraph-mcp")

# Initialize Memgraph client from environment
API_URL = os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
TENANT_ID = os.getenv("MEMGRAPH_TENANT_ID")
API_KEY = os.getenv("MEMGRAPH_API_KEY")

if not TENANT_ID or not API_KEY:
    logger.error("Missing required environment variables: MEMGRAPH_TENANT_ID and MEMGRAPH_API_KEY")
    sys.exit(1)

memgraph = MemgraphClient(
    api_key=API_KEY,
    tenant_id=TENANT_ID,
    base_url=API_URL
)

logger.info(f"Memgraph MCP Server initialized: {API_URL}")

# Create MCP server
app = Server("memgraph-memory")


# ============================================================================
# MCP Tools - Available functions that can be called by the AI
# ============================================================================

@app.tool()
async def memgraph_remember(text: str, user_id: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Store a memory in Memgraph OS.

    Use this to remember:
    - User preferences and settings
    - Project decisions and rationale
    - Important conversations and outcomes
    - Code patterns and architectural choices
    - Bug fixes and solutions

    Args:
        text: The memory text to store (e.g., "User prefers Python for backend")
        user_id: Identifier for the user/project (e.g., "john_doe" or "project_alpha")
        category: Optional category (e.g., "preference", "decision", "bug_fix")

    Returns:
        Confirmation with event ID

    Example:
        memgraph_remember(
            text="Team decided to use PostgreSQL as primary database",
            user_id="project_alpha",
            category="architecture"
        )
    """
    try:
        metadata = {"category": category} if category else {}
        result = memgraph.add(text=text, user_id=user_id, metadata=metadata)

        return {
            "success": True,
            "message": "Memory stored successfully",
            "event_id": result.get("event_id"),
            "text": text,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.tool()
async def memgraph_search(query: str, user_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search for relevant memories using semantic similarity.

    Use this to recall:
    - Past decisions and their context
    - User preferences and settings
    - Similar problems and solutions
    - Project history and conversations

    Args:
        query: Search query (e.g., "What database are we using?")
        user_id: User/project identifier
        limit: Maximum number of results (default: 5)

    Returns:
        Relevant beliefs, episodes, and events with context

    Example:
        memgraph_search(
            query="authentication implementation decisions",
            user_id="project_alpha",
            limit=3
        )
    """
    try:
        context = memgraph.search(query=query, user_id=user_id)

        # Extract and format results
        beliefs = context.get("beliefs", [])[:limit]
        memories = context.get("memories", [])[:limit]

        results = []

        # Add beliefs (long-term facts)
        for belief in beliefs:
            if isinstance(belief, dict):
                results.append({
                    "type": "belief",
                    "content": belief.get("value", belief.get("key", str(belief))),
                    "confidence": belief.get("confidence_score", 1.0),
                    "source": "long-term memory"
                })

        # Add memories (episodes)
        for memory in memories:
            if isinstance(memory, dict):
                text = memory.get("text", memory.get("content", ""))
                if isinstance(text, dict):
                    text = text.get("text", str(text))
                results.append({
                    "type": "episode",
                    "content": text,
                    "source": "episodic memory"
                })

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@app.tool()
async def memgraph_list_memories(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    List recent memories for a user.

    Args:
        user_id: User/project identifier
        limit: Maximum number of memories to return (default: 10)

    Returns:
        List of recent memories with metadata

    Example:
        memgraph_list_memories(user_id="project_alpha", limit=5)
    """
    try:
        import requests

        # Use raw API call since SDK might not have list endpoint
        headers = {"X-API-KEY": API_KEY}
        params = {"user_id": user_id, "limit": limit}

        response = requests.get(
            f"{API_URL}/events",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        events = response.json()

        memories = []
        for event in events[:limit]:
            content = event.get("content_payload", {})
            if isinstance(content, dict):
                text = content.get("text", str(content))
            else:
                text = str(content)

            memories.append({
                "id": event.get("id"),
                "text": text,
                "timestamp": event.get("timestamp"),
                "type": event.get("event_type")
            })

        return {
            "success": True,
            "user_id": user_id,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "memories": []
        }


@app.tool()
async def memgraph_get_profile(user_id: str) -> Dict[str, Any]:
    """
    Get user profile with their beliefs and preferences.

    Args:
        user_id: User/project identifier

    Returns:
        User profile with beliefs, preferences, and stats

    Example:
        memgraph_get_profile(user_id="john_doe")
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}

        # Get beliefs for this user
        response = requests.get(
            f"{API_URL}/beliefs",
            headers=headers,
            params={"user_id": user_id}
        )
        response.raise_for_status()
        beliefs = response.json()

        profile = {
            "user_id": user_id,
            "beliefs": [],
            "preferences": [],
            "facts": []
        }

        # Categorize beliefs
        for belief in beliefs[:20]:  # Limit to top 20
            belief_data = {
                "key": belief.get("key", ""),
                "value": belief.get("value", ""),
                "confidence": belief.get("confidence_score", 1.0),
                "last_updated": belief.get("updated_at")
            }

            # Categorize by key pattern
            key = belief.get("key", "").lower()
            if "prefer" in key or "like" in key or "favorite" in key:
                profile["preferences"].append(belief_data)
            elif "use" in key or "is" in key or "has" in key:
                profile["facts"].append(belief_data)
            else:
                profile["beliefs"].append(belief_data)

        return {
            "success": True,
            "profile": profile
        }
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.tool()
async def memgraph_get_episodes(user_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Get recent episodes (conversation sessions) for a user.

    Args:
        user_id: User/project identifier
        limit: Maximum number of episodes (default: 5)

    Returns:
        Recent episodes with summaries

    Example:
        memgraph_get_episodes(user_id="project_alpha", limit=3)
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}
        params = {"user_id": user_id, "limit": limit}

        response = requests.get(
            f"{API_URL}/episodes",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        episodes_data = response.json()

        episodes = []
        for ep in episodes_data[:limit]:
            episodes.append({
                "id": ep.get("id"),
                "summary": ep.get("summary", "No summary yet"),
                "event_count": ep.get("event_count", 0),
                "start_time": ep.get("start_time"),
                "end_time": ep.get("end_time"),
                "status": ep.get("consolidation_status", "pending")
            })

        return {
            "success": True,
            "user_id": user_id,
            "count": len(episodes),
            "episodes": episodes
        }
    except Exception as e:
        logger.error(f"Error getting episodes: {e}")
        return {
            "success": False,
            "error": str(e),
            "episodes": []
        }


@app.tool()
async def memgraph_delete_memory(event_id: str) -> Dict[str, Any]:
    """
    Delete a specific memory by event ID.

    Args:
        event_id: The ID of the event to delete

    Returns:
        Confirmation of deletion

    Example:
        memgraph_delete_memory(event_id="123e4567-e89b-12d3-a456-426614174000")
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}

        response = requests.delete(
            f"{API_URL}/events/{event_id}",
            headers=headers
        )
        response.raise_for_status()

        return {
            "success": True,
            "message": f"Memory {event_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.tool()
async def memgraph_update_belief(belief_id: str, value: str, confidence: Optional[float] = None) -> Dict[str, Any]:
    """
    Update an existing belief with new value or confidence.

    Args:
        belief_id: The ID of the belief to update
        value: New value for the belief
        confidence: Optional confidence score (0.0 to 1.0)

    Returns:
        Confirmation of update

    Example:
        memgraph_update_belief(
            belief_id="123e4567-e89b-12d3-a456-426614174000",
            value="User prefers TypeScript over JavaScript",
            confidence=0.95
        )
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

        payload = {"value": value}
        if confidence is not None:
            payload["confidence_score"] = confidence

        response = requests.patch(
            f"{API_URL}/beliefs/{belief_id}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        return {
            "success": True,
            "message": f"Belief {belief_id} updated successfully",
            "value": value
        }
    except Exception as e:
        logger.error(f"Error updating belief: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# MCP Resources - URIs that can be read by the AI
# ============================================================================

@app.resource("memgraph://project/summary")
async def get_project_summary() -> str:
    """
    Get a summary of the entire project's memory.
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}

        # Get stats
        response = requests.get(f"{API_URL}/stats", headers=headers)
        stats = response.json() if response.ok else {}

        return f"""# Memgraph Project Summary

**Total Events**: {stats.get('total_events', 0)}
**Total Episodes**: {stats.get('total_episodes', 0)}
**Total Beliefs**: {stats.get('total_beliefs', 0)}
**Tenant**: {TENANT_ID}

This project is using Memgraph OS for persistent memory across AI sessions.
"""
    except Exception as e:
        return f"Error getting project summary: {e}"


@app.resource("memgraph://project/status")
async def get_project_status() -> str:
    """
    Get current status of cognitive dreaming and memory consolidation.
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}

        # Get dreaming health
        response = requests.get(f"{API_URL}/dreaming/health", headers=headers)
        health = response.json() if response.ok else {}

        return f"""# Cognitive Dreaming Status

**Status**: {health.get('status', 'unknown')}
**Worker Running**: {health.get('worker_running', False)}
**Last Cycle**: {health.get('last_cycle', 'never')}
**Pending Episodes**: {health.get('pending_episodes', 0)}

The cognitive dreaming worker automatically consolidates episodes into beliefs.
"""
    except Exception as e:
        return f"Error getting status: {e}"


@app.resource("memgraph://memory/recent")
async def get_recent_memories() -> str:
    """
    Get recent memories across all users.
    """
    try:
        import requests

        headers = {"X-API-KEY": API_KEY}

        response = requests.get(
            f"{API_URL}/events",
            headers=headers,
            params={"limit": 10}
        )
        events = response.json() if response.ok else []

        output = "# Recent Memories\n\n"
        for event in events:
            content = event.get("content_payload", {})
            text = content.get("text", str(content)) if isinstance(content, dict) else str(content)
            timestamp = event.get("timestamp", "unknown")
            output += f"- **{timestamp}**: {text[:100]}...\n"

        return output
    except Exception as e:
        return f"Error getting recent memories: {e}"


# ============================================================================
# Main - Start the MCP server
# ============================================================================

async def main():
    """Start the MCP server"""
    logger.info("Starting Memgraph MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
