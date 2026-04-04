<p align="center">
  <img src="assets/logo.png" alt="Memgraph AI" width="400">
</p>

<h3 align="center">Memory that helps AI agents learn from their mistakes</h3>

<p align="center">
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/v/memgraph-sdk?color=%2334D058&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/dm/memgraph-sdk" alt="Downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="https://memgraph.ai">Website</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/tree/main/examples">Examples</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/issues">Issues</a>
</p>

---

**Memgraph AI** is a persistent memory layer for AI agents. Not a vector store with a wrapper — a cognitive engine that extracts beliefs from conversations, tracks how they evolve, and lets agents learn from their mistakes.

**What makes it different:**
- **Beliefs, not blobs** — Stores structured facts with confidence scores and types (fact / belief / tenet)
- **Learns from corrections** — "Actually, I moved to Manchester" supersedes "Lives in London"
- **Decision traces** — Track *why* your agent made each decision and what beliefs informed it
- **Background consolidation** — Cognitive Dreaming merges, deduplicates, and resolves contradictions while your agents sleep
- **Multi-tenant** — Each customer gets isolated memory with their own API key

## Installation

```bash
pip install memgraph-sdk
```

## Quick Start

```python
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key="mg_your_api_key")

# Store memories (immediately searchable)
mg.remember("Customer is on the Pro plan, $49/month", user_id="alice")
mg.remember("Customer is allergic to peanuts", user_id="alice")

# Search memories
result = mg.search("What plan is Alice on?", user_id="alice")
print(result["results"][0]["content"])
# → "Customer is on the Pro plan, $49/month" (score: 0.76)

# Get all beliefs for a user
beliefs = mg.get_beliefs(user_id="alice")
```

3 lines to set up. Store and search immediately. `tenant_id` is resolved automatically from your API key.

## How It Works

```
Conversations → Events → Episodes → Beliefs → Decisions
                             ↑                      ↓
                     Cognitive Dreaming      Outcome Feedback
                     (consolidation)         (learning loop)
```

| Layer | What it stores | Retention |
|---|---|---|
| **Events** | Raw messages, timestamps | Permanent |
| **Episodes** | Grouped conversations with summaries | Permanent |
| **Beliefs** | Extracted facts with confidence + type | Active (superseded archived) |
| **Decisions** | What the agent decided + why + outcome | Permanent audit trail |

## Core Methods

```python
# Store (immediately searchable)
mg.remember("User prefers dark mode", user_id="alice", category="preference")

# Store (async, goes through extraction pipeline)
mg.add("Full conversation text here", user_id="alice")

# Search (returns scored results)
result = mg.search("UI preferences", user_id="alice")
# → {"results": [{"content": "...", "score": 0.76, "metadata": {...}}]}

# Get all beliefs
beliefs = mg.get_beliefs(user_id="alice", limit=50)

# Health check
status = mg.health()
```

## Async Client

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(api_key="mg_your_api_key") as mg:
    await mg.add("User prefers dark mode", user_id="alice")
    result = await mg.search("preferences", user_id="alice")
```

## MCP Server (Claude / Cursor)

Give your AI IDE persistent memory with one command:

```bash
memgraph setup --key mg_your_api_key
```

Auto-detects Cursor, Claude Desktop, VS Code. Or configure manually:

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "python3",
      "args": ["-m", "memgraph_sdk.mcp"],
      "env": { "MEMGRAPH_API_KEY": "mg_your_api_key" }
    }
  }
}
```

## CLI

```bash
memgraph setup --key mg_your_api_key    # Set up MCP for your IDE
memgraph remember "We chose PostgreSQL"  # Store a memory
memgraph recall "database choice"        # Search memories
memgraph status                          # Check connection
```

## Self-Hosted

```python
# Point to your own server
mg = MemgraphClient(
    api_key="mg_your_key",
    base_url="http://your-server:8001/v1",
)
```

Or via environment:

```bash
export MEMGRAPH_API_URL=http://your-server:8001/v1
export MEMGRAPH_API_KEY=mg_your_key
```

## Integrations

Works with any AI framework:

- **OpenAI** — Function calling + Assistants API
- **LangChain** — Memory, Retriever, Toolkit
- **CrewAI** — Search and Remember tools
- **LlamaIndex** — Retriever and ToolSpec

See [examples/](examples/) for runnable integration code.

## Examples

**Tested on production (api.memgraph.ai):**

| Example | Description | Status |
|---|---|---|
| [quick_start.py](examples/quick_start.py) | Store, search, update — 2 minutes | ✅ Tested |
| [agent.py](examples/agent.py) | Interactive chat agent with memory | ✅ Tested |
| [sdk_demo.py](examples/sdk_demo.py) | Core SDK in 30 lines | ✅ Tested |
| [OpenAI](examples/integrations/openai_integration.py) | Function calling + memory | ✅ Tested |

**Integration examples (require additional deps):**

| Example | Description | Requires |
|---|---|---|
| [LangChain](examples/integrations/langchain_integration.py) | Memory, Retriever, Toolkit | `langchain` |
| [CrewAI](examples/integrations/crewai_integration.py) | Search and Remember tools | `crewai` |
| [LlamaIndex](examples/integrations/llamaindex_integration.py) | Retriever and ToolSpec | `llama-index` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

Report vulnerabilities to security@memgraph.ai. See [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
