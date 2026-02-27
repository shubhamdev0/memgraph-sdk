# Memgraph SDK

The official Python SDK for **[Memgraph](https://memgraph.ai)** -- the memory graph for AI agents.

[![PyPI version](https://badge.fury.io/py/memgraph-sdk.svg)](https://pypi.org/project/memgraph-sdk/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install memgraph-sdk
```

For async support:

```bash
pip install "memgraph-sdk[async]"
```

## Quick Start

```python
from memgraph_sdk import MemgraphClient

client = MemgraphClient(
    api_key="mg_your_api_key",
    tenant_id="your-tenant-id",
)

# Store a memory
client.add(
    text="User prefers dark mode and uses PyTorch.",
    user_id="user_123",
)

# Search for relevant context
result = client.search(
    query="What does this user prefer?",
    user_id="user_123",
)
print(result)
```

## Async Client

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(
    api_key="mg_your_api_key",
    tenant_id="your-tenant-id",
) as client:
    await client.add("User prefers dark mode", user_id="user_123")
    result = await client.search("user preferences", user_id="user_123")
```

## Memory Intelligence API

```python
# Memory health metrics
health = client.health(user_id="user_123")

# Detect contradictions
contradictions = client.contradictions(user_id="user_123")

# Evaluate retrieval quality
evaluation = client.evaluate(query="test query", user_id="user_123")

# Cognitive Integrity Score (0-100)
score = client.mcis(user_id="user_123")

# Run benchmarks
result = client.benchmark(scenario="contradiction_storm")
scenarios = client.benchmark_scenarios()
```

## CLI

```bash
# Initialize Memgraph in your project
memgraph init

# Store a memory
memgraph remember "We decided to use PostgreSQL" -c decision

# Search memories
memgraph recall "database choice"

# Check connection
memgraph status
```

## Configuration

The client reads the API URL from the `MEMGRAPH_API_URL` environment variable, defaulting to `http://localhost:8001/v1`. You can also pass it explicitly:

```python
client = MemgraphClient(
    api_key="mg_your_key",
    tenant_id="your-tenant-id",
    base_url="https://api.memgraph.ai/v1",
)
```

## Examples

See the [examples/](examples/) directory for complete integration examples:

- [Quick Start](examples/quick_start.py) -- Basic add and search
- [Agent Integration](examples/agent_integration.py) -- OpenAI-powered agent with memory
- [MCP Server](examples/integrations/mcp_server.py) -- Model Context Protocol server for Claude/Cursor
- [LangChain](examples/integrations/langchain_integration.py) -- LangChain integration
- [OpenAI](examples/integrations/openai_integration.py) -- OpenAI integration
- [Benchmarks](examples/integrations/benchmark.py) -- Performance benchmarking

## License

MIT License. See [LICENSE](LICENSE) for details.
