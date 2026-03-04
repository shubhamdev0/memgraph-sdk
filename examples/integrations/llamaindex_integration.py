"""
Memgraph + LlamaIndex Integration

Provides a custom retriever and tool for LlamaIndex query engines and agents.

Usage:
    pip install memgraph-sdk llama-index

Example:
    from llamaindex_integration import MemgraphRetriever, MemgraphToolSpec

    retriever = MemgraphRetriever(api_key="mg_...", tenant_id="...", user_id="u1")
    nodes = retriever.retrieve("What auth method do we use?")
"""

import os
from typing import Any, Dict, List, Optional

from memgraph_sdk import MemgraphClient

try:
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.tools import FunctionTool, ToolMetadata
except ImportError:
    raise ImportError(
        "llama-index is required. Install with: pip install llama-index-core"
    )


class MemgraphRetriever(BaseRetriever):
    """LlamaIndex retriever backed by Memgraph's context/search API.

    Converts Memgraph search results into LlamaIndex nodes for use
    in query engines and chat engines.
    """

    def __init__(
        self,
        api_key: str = None,
        tenant_id: str = None,
        user_id: str = "llamaindex_user",
        base_url: str = None,
        top_k: int = 10,
    ):
        super().__init__()
        self._client = MemgraphClient(
            api_key=api_key or os.getenv("MEMGRAPH_API_KEY", ""),
            tenant_id=tenant_id or os.getenv("MEMGRAPH_TENANT_ID", ""),
            base_url=base_url,
        )
        self._user_id = user_id
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from Memgraph."""
        result = self._client.search(
            query=query_bundle.query_str,
            user_id=self._user_id,
        )
        nodes = []
        for item in result.get("retrieved_items", [])[:self._top_k]:
            text = item.get("content", item.get("text", str(item)))
            score = item.get("score", item.get("relevance_score", 0.5))
            metadata = {
                "type": item.get("type", "memory"),
                "source": "memgraph",
            }
            if item.get("key"):
                metadata["key"] = item["key"]
            if item.get("domain"):
                metadata["domain"] = item["domain"]

            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes


class MemgraphToolSpec:
    """LlamaIndex tool spec providing search and remember tools.

    Use with LlamaIndex agents for persistent memory.
    """

    def __init__(
        self,
        api_key: str = None,
        tenant_id: str = None,
        user_id: str = "llamaindex_agent",
        base_url: str = None,
    ):
        self._client = MemgraphClient(
            api_key=api_key or os.getenv("MEMGRAPH_API_KEY", ""),
            tenant_id=tenant_id or os.getenv("MEMGRAPH_TENANT_ID", ""),
            base_url=base_url,
        )
        self._user_id = user_id

    def search_memory(self, query: str) -> str:
        """Search Memgraph for relevant memories and past context."""
        result = self._client.search(query=query, user_id=self._user_id)
        items = result.get("retrieved_items", [])
        if not items:
            return "No relevant memories found."
        return "\n".join(
            f"- {it.get('content', it.get('text', str(it)))}" for it in items[:10]
        )

    def remember(self, text: str, category: str = "general") -> str:
        """Store a new memory in Memgraph for future recall."""
        self._client.remember(text, user_id=self._user_id, category=category)
        return f"Stored: {text[:100]}"

    def to_tool_list(self) -> List[FunctionTool]:
        """Convert to LlamaIndex FunctionTool list."""
        return [
            FunctionTool.from_defaults(
                fn=self.search_memory,
                name="memgraph_search",
                description="Search persistent memory for relevant context, past decisions, and knowledge.",
            ),
            FunctionTool.from_defaults(
                fn=self.remember,
                name="memgraph_remember",
                description="Store a new fact or decision in persistent memory for future recall.",
            ),
        ]


# --- Example Usage ---

if __name__ == "__main__":
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.query_engine import RetrieverQueryEngine

    # 1. As a retriever in a query engine
    retriever = MemgraphRetriever(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="dev_user",
    )

    query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
    response = query_engine.query("What authentication method are we using?")
    print("Query Engine Response:", response)

    # 2. As tools in an agent
    tool_spec = MemgraphToolSpec(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
    )
    tools = tool_spec.to_tool_list()
    print(f"Created {len(tools)} tools: {[t.metadata.name for t in tools]}")
