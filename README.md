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
- [Learn From Mistakes](#learn-from-mistakes)
- [Entities & Knowledge Graph](#entities--knowledge-graph)
- [Cognitive Sidecar (Always-On Memory)](#cognitive-sidecar-always-on-memory)
- [Memory Intelligence](#memory-intelligence)
- [Deleting User Data](#deleting-user-data)
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
pip install memgraph-sdk
export MEMGRAPH_API_KEY=mg_your_api_key
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

Sends raw text through the extraction pipeline, which pulls out facts and preferences as beliefs in the background. **Results are searchable after ~5-10 seconds.**

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
#     {"id": "6f1c…", "content": "User prefers dark mode", "score": 0.76,
#      "metadata": {"key": "...", "value": "User prefers dark mode", "domain": "general"}},
#   ],
#   "total": 1
# }
```

Optional parameters: `agent_id` (scope to a specific agent), `limit` (default 10).

Remembering an updated fact replaces the old one: after `remember("I moved to Pune")`, a search for where Alice lives returns Pune, not the old city. The old value stays in `belief_history()` and only appears in results marked `[PREVIOUSLY]`.

If the server is unreachable or returns an error, `search()` raises (see [Error Handling](#error-handling)) — an outage never looks like "no memories".

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

Record, inspect, and debug AI agent decisions: Memgraph AI keeps *why* your agent did something, not just what it remembered.

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
    beliefs_used=[r["id"] for r in mg.search("our database", user_id="alice")["results"]],
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

# Delete a decision
mg.delete_decision(decision["id"])
```

`beliefs_used` takes belief **IDs** (the `id` of `search()` results). When the decision has an outcome, Memgraph AI adjusts those beliefs' confidence: down on `FAILURE`, slightly up on `SUCCESS`.

## Learn From Mistakes

Record what your agent did and how it turned out. The next time a similar task comes in, the past attempt — and why it failed — is part of the context.

```python
# 1. The agent acts; you learn later that it went wrong
d = mg.record_decision(goal="Refund order #4411 bought 45 days ago", user_id="alice")
mg.record_outcome(d["id"], "FAILURE", feedback="Refunds are only allowed within 30 days")

# 2. A similar request arrives
ctx = mg.sidecar_pre_flight("Please refund my order #5520 from 50 days ago", user_id="alice")
print(ctx["memory_context"])
# LESSONS FROM PAST ATTEMPTS AT SIMILAR TASKS:
#   - "Refund order #4411 bought 45 days ago" → FAILED: Refunds are only allowed within 30 days
#   Do not repeat an approach that FAILED; adjust it using the reason given.

# Same data from the v2 API:
mg.get_context_graph("refund order bought 50 days ago", user_id="alice")["decisions"]
```

Past decisions are matched by how similar their goal is to the current message, and only the user's own decisions (plus agent-level ones recorded without a `user_id`) are shown.

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
context["memory_context"]   # text block to add as a system message
context["system_messages"]  # or: a ready-made messages list
context["past_decisions"]   # similar earlier attempts and their outcomes

# Post-flight: extract learnable signals from the conversation
mg.sidecar_post_flight(
    messages=[
        {"role": "user", "content": "What database should I use?"},
        {"role": "assistant", "content": "PostgreSQL with pgvector."},
    ],
    user_id="alice",
)
# → {"status": "queued", ...} — returns immediately, learns in the background.
#   Pass wait=True to learn synchronously and get the belief counts back.

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
mg.benchmark("contradiction_storm")
# Scenarios: contradiction_storm, tenet_violation, retrieval_accuracy,
#            locomo, deep_memory_retrieval  (mg.benchmark_scenarios() lists them)
```

## Deleting User Data

When one of your users asks to be forgotten:

```python
mg.forget_all(user_id="alice")                       # every belief for the user
for d in mg.list_decisions(user_id="alice", limit=200):
    mg.delete_decision(d["id"])                       # their decision records
```

`forget_all(user_id, soft=True)` deactivates instead of deleting, if you need an audit trail.

## Error Handling

```python
from memgraph_sdk.exceptions import (
    MemgraphAuthError,         # 401/403 — bad API key
    MemgraphConnectionError,   # Network error / timeout
    MemgraphRateLimitError,    # 429 — e.retry_after has wait time
    MemgraphValidationError,   # 4xx — bad request (404, 409, 422 …)
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

Available: `add`, `remember`, `search`, `get_beliefs`, `forget`, `forget_all`, `sidecar_pre_flight`, `sidecar_post_flight`, `get_context_graph`, `record_decision`, `record_outcome`, `delete_decision`, plus the memory-intelligence calls (`health`, `contradictions`, `evaluate`, `mcis`, `benchmark`).

Requires: `pip install "memgraph-sdk[async]"`

## MCP Server (Claude / Cursor)

Give your AI IDE persistent memory with one command:

```bash
pip install "memgraph-sdk[mcp]"
memgraph setup --key mg_your_api_key
```

Auto-detects Cursor, Claude Desktop, VS Code. Or configure manually:

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "python3",
      "args": ["-m", "memgraph_sdk.mcp"],
      "env": {
        "MEMGRAPH_API_KEY": "mg_your_api_key",
        "MEMGRAPH_AGENT_USER_ID": "your-name"
      }
    }
  }
}
```

Tools: `memgraph_search`, `memgraph_remember`, `memgraph_forget` (delete a wrong memory by the `id` from a search result), `memgraph_think` (recall + learn from the conversation in one call) and `memgraph_profile`.

`MEMGRAPH_AGENT_USER_ID` decides whose memory the IDE reads and writes (default `ai_agent`). Everyone on your team who uses the same API key with the default shares one memory — give each person their own value.

## CLI

```bash
memgraph setup --key mg_your_api_key    # Set up MCP for your IDE
memgraph remember "We chose PostgreSQL"  # Store a memory
memgraph recall "database choice"        # Search memories
memgraph status                          # Check connection
```

The CLI reads `MEMGRAPH_API_KEY` / `MEMGRAPH_API_URL` from the environment, so `export MEMGRAPH_API_KEY=mg_...` is enough — no setup file needed. Environment variables override `.memgraph.env`.

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

Each API key can make **120 requests per minute**. Need more? Email hello@memgraph.ai.

Over the limit the API returns `429` with a `Retry-After` header; the SDK waits and retries automatically. Catch `MemgraphRateLimitError` for custom handling.

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
