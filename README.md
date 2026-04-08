<p align="center">
  <img src="https://raw.githubusercontent.com/shubhamdev0/memgraph-sdk/main/assets/logo.png" alt="Memgraph AI" width="400">
</p>

<h3 align="center">Memory that helps AI agents learn from their mistakes</h3>

<p align="center">
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/v/memgraph-sdk?color=%2334D058&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/dm/memgraph-sdk" alt="Downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
  <a href="https://github.com/shubhamdev0/memgraph-sdk/actions"><img src="https://github.com/shubhamdev0/memgraph-sdk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://memgraph.ai">Website</a> ·
  <a href="https://memgraph.ai/docs">Docs</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/issues">Issues</a>
</p>

---

Not a vector store with a wrapper. A three-layer cognitive engine that distills raw events into episodes, crystallizes them into beliefs, and tracks how those beliefs evolve — with background consolidation that improves memory while your agents sleep.

## Table of Contents

- [Installation](#installation)
- [Quick Start (30 seconds)](#quick-start-30-seconds)
- [Authentication](#authentication)
- [Core Methods](#core-methods)
- [Decisions & Reasoning Traces](#decisions--reasoning-traces)
- [Entities & Knowledge Graph](#entities--knowledge-graph)
- [Cognitive Sidecar (Always-On Memory)](#cognitive-sidecar-always-on-memory)
- [Memory Intelligence](#memory-intelligence)
- [Error Handling](#error-handling)
- [Async Client](#async-client)
- [MCP Server (Claude / Cursor)](#mcp-server-claude--cursor)
- [CLI](#cli)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Integrations](#integrations)

## Installation

```bash
pip install memgraph-sdk
```

With optional extras:

```bash
pip install "memgraph-sdk[async]"   # Async client (httpx)
pip install "memgraph-sdk[mcp]"     # MCP server for Claude/Cursor
pip install "memgraph-sdk[all]"     # Everything
```

**Upgrading:** Use `pip install --upgrade memgraph-sdk` (not `--force-reinstall`, which can cause dependency conflicts with other packages like CrewAI).

**Works with:** CrewAI, LangChain, OpenAI SDK, LlamaIndex — tested in shared environments.

## Quick Start (30 seconds)

**Step 1: Get your API key** — sign up at [memgraph.ai](https://memgraph.ai), or via CLI:

```bash
# Cloud (recommended)
pip install memgraph-sdk
memgraph setup --key mg_your_api_key

# Self-hosted
docker compose up -d              # Start PostgreSQL + Memgraph
memgraph setup --key mg_your_key  # Point to your server
```

> Your API key starts with `mg_`. Find it in Settings > API Keys after signing up.

**Step 2: Use it:**

```python
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key="mg_your_api_key")

# Store a memory (immediately searchable)
mg.remember("Customer prefers dark mode and uses PyTorch", user_id="alice")

# Search memories (returns scored results)
result = mg.search("What does Alice prefer?", user_id="alice")
print(result["results"][0]["content"])
# → "Customer prefers dark mode and uses PyTorch" (score: 0.78)

# Get all beliefs for a user
beliefs = mg.get_beliefs(user_id="alice")
```

Three lines to set up. The `tenant_id` is resolved automatically from your API key.

## Authentication

```bash
export MEMGRAPH_API_KEY=mg_your_api_key
```

```python
import os
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key=os.environ["MEMGRAPH_API_KEY"])
```

Get your API key at [memgraph.ai](https://memgraph.ai) — sign up and it's on the Settings > API Keys page.

## Core Methods

### `remember()` — Immediate storage

Creates a belief directly with a vector embedding. **Immediately searchable.**

```python
mg.remember(
    "User prefers dark mode",
    user_id="alice",
    category="preference",   # "general", "decision", "architecture", "bug_fix", "preference"
    domain="general",         # optional domain tag
    confidence=0.90,          # 0.0 - 1.0 (default: 0.90)
)
```

### `add()` — Async extraction pipeline

Sends text through the background extraction pipeline (entity extraction, belief crystallization, episode grouping). **May take 5-10 seconds before results are searchable.**

```python
mg.add("Full conversation text here", user_id="alice")
```

> Use `remember()` when you need immediate searchability.
> Use `add()` when you want the full extraction pipeline (entities, episodes, beliefs from raw text).

### `search()` — Semantic memory retrieval

Returns scored results with semantic similarity, recency, confidence, frequency, and keyword signals.

```python
result = mg.search("UI preferences", user_id="alice")
# Returns:
# {
#   "results": [
#     {"content": "User prefers dark mode", "score": 0.76, "metadata": {"key": "...", "domain": "general"}},
#   ],
#   "total": 1
# }
```

Optional parameters: `agent_id` (scope to a specific agent), `limit` (default 10).

### `get_beliefs()` — List all beliefs

```python
beliefs = mg.get_beliefs(user_id="alice", limit=50)
```

### `forget()` / `forget_all()`

```python
mg.forget(belief_id="uuid-of-the-belief")       # Delete one belief by ID
mg.forget_all(user_id="alice")                    # Delete all beliefs for a user
mg.forget_all(user_id="alice", domain="work")     # Delete beliefs in a domain
```

### `belief_history()` / `belief_timeline()`

```python
# How a specific belief changed over time
history = mg.belief_history(user_id="alice", key="preference_dark_mode_abc123")

# Timeline of all belief changes
timeline = mg.belief_timeline(user_id="alice", domain="work")
```

### Context manager

```python
with MemgraphClient(api_key="mg_your_key") as mg:
    mg.remember("User likes Python", user_id="alice")
    # Session closed automatically
```

## Decisions & Reasoning Traces

Record, inspect, and debug AI agent decisions. This is Memgraph's unique feature — no other memory system tracks *why* your agent made a decision.

### Record a decision

```python
decision = mg.record_decision(
    goal="Choose database for analytics service",
    reasoning_steps=[
        {"step": 1, "description": "Evaluated PostgreSQL vs MongoDB vs ClickHouse"},
        {"step": 2, "description": "Ran cost analysis — $50/mo vs $200/mo vs $150/mo"},
        {"step": 3, "description": "Checked team expertise — strong PostgreSQL skills"},
    ],
    tools_used=[
        {"tool_name": "benchmark_runner", "tool_input": "pg vs mongo", "tool_output": "pg wins"},
        {"tool_name": "cost_calculator", "tool_input": "3 options", "tool_output": "$50/mo"},
    ],
    beliefs_used=["PostgreSQL is our production DB", "Team has PostgreSQL expertise"],
    confidence=0.92,
    outcome="SUCCESS",           # SUCCESS, FAILURE, PARTIAL, UNKNOWN, REVERTED
    outcome_assessment="PostgreSQL selected, 3x faster than MongoDB for our workload",
    agent_id="my-agent",
    user_id="alice",
)
print(decision["id"])  # UUID of the decision
```

**Field reference for `reasoning_steps`:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step` | int | yes | Step number |
| `description` | str | yes | What was done |
| `tool` | str | no | Tool used in this step |
| `input` | any | no | Input to the tool |
| `output` | any | no | Output from the tool |
| `confidence` | float | no | Step-level confidence |

**Field reference for `tools_used`:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_name` | str | yes | Name of the tool |
| `tool_input` | any | no | What was passed to the tool |
| `tool_output` | any | no | What the tool returned |

### Inspect & explain decisions

```python
# Get decision by ID
d = mg.get_decision(decision["id"])

# Get full explanation (reasoning + beliefs + context snapshot)
explanation = mg.explain_decision(decision["id"])

# List all decisions (with optional filters)
all_decisions = mg.list_decisions(agent_id="my-agent", outcome="FAILURE", limit=20)
```

## Entities & Knowledge Graph

Build a knowledge graph of people, organizations, products, and concepts.

```python
# Create entities
person = mg.create_entity(
    name="John Smith",
    entity_type="person",
    properties={"role": "tech lead", "preference": "TypeScript"},
)

org = mg.create_entity(
    name="Acme Corp",
    entity_type="organization",
    properties={"industry": "technology"},
)

# Create a relationship
mg.create_relationship(
    source_entity_id=person["id"],
    target_entity_id=org["id"],
    relation_type="works_at",
    confidence=0.95,
    valid_from="2025-01-01",       # optional temporal bounds
)

# Search entities
results = mg.search_entities("tech lead")

# Traverse the graph
graph = mg.traverse_graph(entity_ids=[person["id"]], max_depth=2)

# List & manage
entities = mg.list_entities()
relationships = mg.list_relationships(entity_id=person["id"])
mg.delete_entity(person["id"])
```

## Cognitive Sidecar (Always-On Memory)

Drop-in middleware that auto-recalls before every LLM call and auto-learns after.

```python
# Pre-flight: recall relevant memories before sending to LLM
context = mg.sidecar_pre_flight(
    message="What database should I use?",
    user_id="alice",
    token_budget=4000,            # max tokens for injected context
)
# → Returns memory context to inject as system message

# Post-flight: extract learnable signals from the conversation
mg.sidecar_post_flight(
    messages=[
        {"role": "user", "content": "What database should I use?"},
        {"role": "assistant", "content": "PostgreSQL with pgvector."},
    ],
    user_id="alice",
)
# → Learning happens in background, returns immediately

# Process: combined pre-flight + post-flight in one call (recommended)
result = mg.sidecar_process(
    messages=[
        {"role": "user", "content": "What database should I use?"},
        {"role": "assistant", "content": "PostgreSQL with pgvector."},
    ],
    user_id="alice",
)
```

## Memory Intelligence

```python
# Health check
mg.ping()

# Memory health stats (belief count, episode count, etc.)
mg.health()

# MCIS — Memgraph Cognitive Integrity Score (0-100)
score = mg.mcis()
# → {"mcis": 79.3, "grade": "B", "sub_scores": {"accuracy": 100, ...}}

# MCIS history over time
mg.mcis_history()

# Contradiction detection
mg.contradictions()

# Evaluate retrieval quality for a query
mg.evaluate("What is our database?", user_id="alice")

# Run a benchmark scenario
mg.benchmark("contradiction_detection")
```

## Error Handling

```python
from memgraph_sdk.exceptions import (
    MemgraphAuthError,         # 401/403 — bad API key
    MemgraphConnectionError,   # Network error / timeout
    MemgraphRateLimitError,    # 429 — e.retry_after has wait time
    MemgraphValidationError,   # 422 — bad request parameters
    MemgraphAPIError,          # 5xx — server error (auto-retried)
)

try:
    result = mg.search("query", user_id="alice")
except MemgraphRateLimitError as e:
    print(f"Rate limited. Retry in {e.retry_after}s")
except MemgraphAuthError:
    print("Check your MEMGRAPH_API_KEY")
```

The SDK automatically retries transient errors (500, 502, 503, 504) with exponential backoff.

## Async Client

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(api_key="mg_your_api_key") as mg:
    await mg.remember("User prefers dark mode", user_id="alice")
    result = await mg.search("preferences", user_id="alice")
```

Requires: `pip install "memgraph-sdk[async]"`

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

## Configuration

### Cloud (default)

```python
mg = MemgraphClient(api_key="mg_your_key")
# Connects to https://api.memgraph.ai/v1
```

### Self-hosted

```python
mg = MemgraphClient(
    api_key="mg_your_key",
    base_url="http://your-server:8001/v1",
)
```

### Environment variables

```bash
export MEMGRAPH_API_KEY=mg_your_key
export MEMGRAPH_API_URL=http://your-server:8001/v1  # optional
```

**URL resolution priority:**
1. `base_url` parameter (highest)
2. `MEMGRAPH_API_URL` environment variable
3. `https://api.memgraph.ai/v1` (default)

### Using a `.env` file

Create a `.env` file (add to `.gitignore`!):

```bash
# .env
MEMGRAPH_API_KEY=mg_your_key
MEMGRAPH_API_URL=https://api.memgraph.ai/v1  # or your self-hosted URL
```

Load it in your app:

```python
from dotenv import load_dotenv
load_dotenv()

mg = MemgraphClient(api_key=os.environ["MEMGRAPH_API_KEY"])
```

### Rate limits

| Tier | Requests/min | Beliefs | Entities |
|------|-------------|---------|----------|
| Free | 120 | 1,000 | 100 |
| Pro | 600 | 50,000 | 5,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

The SDK auto-retries on 429 with exponential backoff. Catch `MemgraphRateLimitError` for custom handling.

### Input validation

The SDK validates inputs before sending requests:

- **API key** must start with `mg_` — raises `MemgraphValidationError` if not
- **user_id** must be a non-empty string — raises `MemgraphValidationError` if empty
- **ping()** validates both connectivity AND API key authenticity

## How It Works

```
Raw Input → Events → Episodes → Beliefs → Decisions
              │          │          │          │
          (short-term) (grouped)  (long-term) (traced)
                                     │
                              Cognitive Dreaming
                         (consolidation while idle)
```

- **Events** — Raw, immutable records with vector embeddings
- **Episodes** — Auto-grouped sequences with LLM summaries
- **Beliefs** — Extracted facts, preferences, decisions with confidence scores and types (fact / belief / tenet)
- **Decisions** — Full reasoning traces: goal → steps → tools → beliefs → outcome
- **Cognitive Dreaming** — Background worker that consolidates, deduplicates, and resolves contradictions

## Integrations

Works with any AI framework:

| Framework | Integration | Docs |
|---|---|---|
| **OpenAI Agents SDK** | `MemgraphAgentHooks`, `MemgraphRunHooks` | [Docs](https://memgraph.ai/docs/integrations) |
| **LangChain / LangGraph** | Memory + Retriever | [Docs](https://memgraph.ai/docs/integrations) |
| **CrewAI** | Search + Remember tools | [Docs](https://memgraph.ai/docs/integrations) |
| **Claude Code (MCP)** | `memgraph setup` | [Docs](https://memgraph.ai/docs/mcp) |
| **Cursor / VS Code** | MCP auto-config | [Docs](https://memgraph.ai/docs/mcp) |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Report vulnerabilities to **security@memgraph.ai**. See [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
