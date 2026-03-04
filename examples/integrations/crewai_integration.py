"""
Memgraph + CrewAI Integration

Gives CrewAI agents persistent memory across tasks, crews, and sessions.

Usage:
    pip install memgraph-sdk crewai crewai-tools

Example:
    from crewai import Agent, Task, Crew
    from crewai_integration import MemgraphSearchTool, MemgraphRememberTool

    search = MemgraphSearchTool(api_key="mg_...", tenant_id="...")
    remember = MemgraphRememberTool(api_key="mg_...", tenant_id="...")

    agent = Agent(
        role="Research Analyst",
        goal="Provide insights based on accumulated knowledge",
        tools=[search, remember],
    )
"""

import os
from typing import Optional, Type

from pydantic import BaseModel, Field

try:
    from crewai_tools import BaseTool
except ImportError:
    raise ImportError("crewai-tools is required. Install with: pip install crewai-tools")

from memgraph_sdk import MemgraphClient

# --- Tool Input Schemas ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant memories")


class RememberInput(BaseModel):
    text: str = Field(description="The text/fact to remember")
    category: str = Field(
        default="general",
        description="Category: decision, architecture, bug_fix, preference, or general",
    )


# --- CrewAI Tools ---

class MemgraphSearchTool(BaseTool):
    name: str = "memgraph_search"
    description: str = (
        "Search Memgraph for relevant memories, past decisions, context, and knowledge. "
        "Use this before answering questions about past work, architecture, or decisions."
    )
    args_schema: Type[BaseModel] = SearchInput

    api_key: str = ""
    tenant_id: str = ""
    user_id: str = "crewai_agent"
    base_url: Optional[str] = None

    def __init__(self, api_key: str = None, tenant_id: str = None, **kwargs):
        super().__init__(
            api_key=api_key or os.getenv("MEMGRAPH_API_KEY", ""),
            tenant_id=tenant_id or os.getenv("MEMGRAPH_TENANT_ID", ""),
            **kwargs,
        )

    def _run(self, query: str) -> str:
        client = MemgraphClient(
            api_key=self.api_key,
            tenant_id=self.tenant_id,
            base_url=self.base_url,
        )
        result = client.search(query, user_id=self.user_id)
        items = result.get("retrieved_items", [])
        if not items:
            return "No relevant memories found."
        return "\n".join(
            f"- [{it.get('type', 'memory')}] {it.get('content', it.get('text', str(it)))}"
            for it in items[:10]
        )


class MemgraphRememberTool(BaseTool):
    name: str = "memgraph_remember"
    description: str = (
        "Store a new memory/fact in Memgraph for future recall. "
        "Use after completing tasks, making decisions, or learning something new."
    )
    args_schema: Type[BaseModel] = RememberInput

    api_key: str = ""
    tenant_id: str = ""
    user_id: str = "crewai_agent"
    base_url: Optional[str] = None

    def __init__(self, api_key: str = None, tenant_id: str = None, **kwargs):
        super().__init__(
            api_key=api_key or os.getenv("MEMGRAPH_API_KEY", ""),
            tenant_id=tenant_id or os.getenv("MEMGRAPH_TENANT_ID", ""),
            **kwargs,
        )

    def _run(self, text: str, category: str = "general") -> str:
        client = MemgraphClient(
            api_key=self.api_key,
            tenant_id=self.tenant_id,
            base_url=self.base_url,
        )
        client.remember(text, user_id=self.user_id, category=category)
        return f"Remembered: {text[:100]}"


# --- Example Usage ---

if __name__ == "__main__":
    from crewai import Agent, Crew, Process, Task

    # Create tools
    search = MemgraphSearchTool(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
    )
    remember = MemgraphRememberTool(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
    )

    # Create agent with memory tools
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Provide insights using accumulated knowledge and learn from new findings",
        backstory="You have access to a persistent memory system. Always search for relevant "
                  "context before answering, and remember important findings for future use.",
        tools=[search, remember],
        verbose=True,
    )

    # Create task
    task = Task(
        description="Research the current authentication approach and suggest improvements. "
                    "First, search for any past decisions about auth. Then provide recommendations.",
        expected_output="A report with current auth approach and improvement suggestions.",
        agent=researcher,
    )

    # Run crew
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    print(result)
