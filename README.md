<p align="center">
  <img src="https://raw.githubusercontent.com/shubhamdev0/memgraph-sdk/main/assets/logo.png" alt="Memgraph AI" width="360">
</p>

<p align="center"><strong>Memory for AI agents that learns from outcomes.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/v/memgraph-sdk?color=34D058&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/pyversions/memgraph-sdk" alt="Python versions"></a>
  <a href="https://github.com/shubhamdev0/memgraph-sdk/actions/workflows/ci.yml"><img src="https://github.com/shubhamdev0/memgraph-sdk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shubhamdev0/memgraph-sdk/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://memgraph.ai/docs">Documentation</a> ·
  <a href="https://memgraph.ai/docs/quickstart">Quickstart</a> ·
  <a href="https://api.memgraph.ai/docs">API reference</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/blob/main/CHANGELOG.md">Changelog</a>
</p>

---

The Memgraph AI Python SDK gives your agents long-term memory: store what users tell them, retrieve it by meaning, keep it current as facts change, and record decisions so an agent doesn't repeat a mistake that already failed.

## Installation

```bash
pip install memgraph-sdk
```

Optional extras:

```bash
pip install "memgraph-sdk[async]"   # AsyncMemgraphClient (httpx)
pip install "memgraph-sdk[mcp]"     # MCP server for Claude, Cursor, VS Code (Python 3.10+)
pip install "memgraph-sdk[all]"
```

Requires Python 3.8+.

## Usage

```python
import os
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key=os.environ["MEMGRAPH_API_KEY"])

mg.remember("Prefers dark mode and uses PyTorch", user_id="alice")

results = mg.search("What does Alice prefer?", user_id="alice")
print(results["results"][0]["content"])
# Prefers dark mode and uses PyTorch
```

Get an API key (it starts with `mg_`) from **Settings → API Keys** at [memgraph.ai](https://memgraph.ai). Keep it in the `MEMGRAPH_API_KEY` environment variable rather than in source code.

## Contents

- [Memories](#memories)
- [Learning from outcomes](#learning-from-outcomes)
- [Always-on memory (sidecar)](#always-on-memory-sidecar)
- [Knowledge graph](#knowledge-graph)
- [Deleting user data](#deleting-user-data)
- [Async usage](#async-usage)
- [Errors, retries and rate limits](#errors-retries-and-rate-limits)
- [Configuration](#configuration)
- [MCP server](#mcp-server)
- [CLI](#cli)
- [Framework integrations](#framework-integrations)

## Memories

### Store

| Method | When to use |
|---|---|
| `remember(text, user_id)` | Store a fact you already know. Searchable immediately. |
| `add(text, user_id)` | Send raw conversation text; Memgraph AI extracts the facts in the background (~5–10 s). |

```python
mg.remember(
    "Moved to Pune in September",
    user_id="alice",
    category="general",   # general | preference | decision | architecture | bug_fix
    confidence=0.9,
)

mg.add("User: I just started at Razorpay as a backend engineer.", user_id="alice")
```

When a new memory updates an existing one, the old one is archived instead of competing with it: after "Moved to Pune", a search for where Alice lives returns Pune. The previous value stays available through `belief_history()` and only appears in results marked `[PREVIOUSLY]`. Storing the same fact twice keeps one copy.

### Search

```python
results = mg.search("Where does Alice live?", user_id="alice", limit=5)
```

```python
{
    "results": [
        {
            "id": "6f1c2b7e-…",
            "content": "Moved to Pune in September",
            "score": 0.81,
            "type": "belief",
            "metadata": {"key": "…", "value": "Moved to Pune in September", "domain": "general"},
        }
    ],
    "total": 1,
}
```

Results are ranked by semantic similarity, recency, confidence, how often a memory is used, and keyword overlap. Pass `agent_id` to scope retrieval to one agent.

### Inspect

```python
mg.get_beliefs(user_id="alice", limit=50)                # current memories
mg.belief_history(user_id="alice", key="<metadata.key>") # versions of one memory
mg.belief_timeline(user_id="alice", domain="work")       # chronological changes
```

## Learning from outcomes

Record what your agent decided and how it turned out. When a similar task comes in later, the earlier attempt, and why it failed, is part of the context.

```python
# The agent acted, and you later learn it went wrong
decision = mg.record_decision(
    goal="Refund order #4411 bought 45 days ago",
    user_id="alice",
    beliefs_used=[r["id"] for r in mg.search("refund policy", user_id="alice")["results"]],
)
mg.record_outcome(decision["id"], "FAILURE", feedback="Refunds are only allowed within 30 days")

# A similar request arrives
ctx = mg.sidecar_pre_flight("Please refund my order #5520 from 50 days ago", user_id="alice")
print(ctx["memory_context"])
```

```text
LESSONS FROM PAST ATTEMPTS AT SIMILAR TASKS:
  - "Refund order #4411 bought 45 days ago" → FAILED: Refunds are only allowed within 30 days
  Do not repeat an approach that FAILED; adjust it using the reason given.
```

- **Outcomes adjust confidence.** `FAILURE` lowers the confidence of the memories listed in `beliefs_used` (pass memory IDs); `SUCCESS` raises it slightly.
- **Past decisions are matched by goal similarity** and scoped to the same user, plus agent-level decisions recorded without a `user_id`.
- `get_context_graph(query, user_id=…)["decisions"]` returns the same matches.

<details>
<summary>Full <code>record_decision</code> reference</summary>

```python
decision = mg.record_decision(
    goal="Choose a database for the analytics service",
    reasoning_steps=[
        {"step": 1, "description": "Compared PostgreSQL, MongoDB and ClickHouse"},
        {"step": 2, "description": "Cost analysis: $50 vs $200 vs $150 per month"},
    ],
    tools_used=[{"tool_name": "cost_calculator", "tool_input": "3 options", "tool_output": "$50/mo"}],
    beliefs_used=["<memory id>", "<memory id>"],
    confidence=0.92,
    outcome="SUCCESS",            # SUCCESS | FAILURE | PARTIAL | UNKNOWN | REVERTED
    outcome_assessment="3x faster than MongoDB for our workload",
    agent_id="planner",
    user_id="alice",
)

mg.get_decision(decision["id"])
mg.explain_decision(decision["id"])          # reasoning, memories used, context snapshot
mg.list_decisions(agent_id="planner", outcome="FAILURE", limit=20)
mg.delete_decision(decision["id"])
```

| `reasoning_steps` field | Type | Required |
|---|---|---|
| `step` | int | yes |
| `description` | str | yes |
| `tool`, `input`, `output` | any | no |
| `confidence` | float | no |

</details>

## Always-on memory (sidecar)

Call `sidecar_pre_flight` before each LLM call and `sidecar_post_flight` after it. Memory is then present on every turn without the model having to decide to look it up.

```python
ctx = mg.sidecar_pre_flight(message=user_message, user_id="alice", token_budget=4000)

messages = ctx["system_messages"] + [{"role": "user", "content": user_message}]
reply = llm(messages)

mg.sidecar_post_flight(
    messages=[{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}],
    user_id="alice",
)  # returns immediately; learning runs in the background (wait=True to block)
```

`sidecar_pre_flight` returns:

| Field | Description |
|---|---|
| `memory_context` | Text block to add as a system message |
| `system_messages` | The same context as a ready-made messages list |
| `past_decisions` | Similar earlier attempts and their outcomes |
| `profile` | Consolidated facts and preferences for the user |

`sidecar_process(messages, user_id)` does both steps in one call.

## Knowledge graph

```python
person = mg.create_entity(entity_type="person", name="John Smith", properties={"role": "tech lead"})
org = mg.create_entity(entity_type="organization", name="Acme Corp")
mg.create_relationship(person["id"], org["id"], relation_type="works_at", valid_from="2025-01-01")

mg.search_entities("tech lead")            # matches names, aliases and properties
mg.traverse_graph(entity_ids=[person["id"]], max_depth=2)
mg.delete_entity(person["id"])             # also removes its relationships
```

## Deleting user data

When a user asks to be forgotten:

```python
mg.forget_all(user_id="alice")                    # all memories
for d in mg.list_decisions(user_id="alice", limit=200):
    mg.delete_decision(d["id"])                   # decision records
```

Use `mg.forget(memory_id)` to delete a single memory, `forget_all(user_id, domain="work")` to limit the scope, or `soft=True` to deactivate instead of delete.

## Async usage

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(api_key=os.environ["MEMGRAPH_API_KEY"]) as mg:
    await mg.remember("Prefers dark mode", user_id="alice")
    results = await mg.search("preferences", user_id="alice")
```

The async client supports `add`, `remember`, `search`, `get_beliefs`, `forget`, `forget_all`, `sidecar_pre_flight`, `sidecar_post_flight`, `get_context_graph`, `record_decision`, `record_outcome`, `delete_decision` and the memory-health methods. Install it with `pip install "memgraph-sdk[async]"`.

## Errors, retries and rate limits

All exceptions inherit from `memgraph_sdk.exceptions.MemgraphError`:

| Exception | Raised on |
|---|---|
| `MemgraphAuthError` | 401 / 403: missing or invalid API key |
| `MemgraphValidationError` | other 4xx, e.g. 404 not found, 409 conflict, 422 invalid parameters |
| `MemgraphRateLimitError` | 429; `e.retry_after` is the wait in seconds |
| `MemgraphAPIError` | 5xx server error |
| `MemgraphConnectionError` | network failure or timeout |

```python
from memgraph_sdk.exceptions import MemgraphAPIError, MemgraphRateLimitError

try:
    mg.search("query", user_id="alice")
except MemgraphRateLimitError as e:
    print(f"Retry in {e.retry_after}s")
except MemgraphAPIError as e:
    print(e.status_code, e)
```

- **Retries.** Connection errors, timeouts, 429 and 500/502/503/504 responses are retried with backoff, up to `max_retries` attempts (default 3).
- **Timeouts.** 30 seconds per request by default; set `timeout=` on the client.
- **Rate limits.** 120 requests per minute per API key. Contact hello@memgraph.ai for higher limits.
- **No silent failures.** `search()` raises on errors, so an outage never looks like "no memories".

The client is safe to share across threads.

## Configuration

```python
MemgraphClient(
    api_key="mg_…",
    base_url="https://api.memgraph.ai/v1",  # or your own deployment
    timeout=30.0,
    max_retries=3,
)
```

| Environment variable | Purpose |
|---|---|
| `MEMGRAPH_API_KEY` | API key for the CLI and MCP server. Pass it to the client explicitly. |
| `MEMGRAPH_API_URL` | Base URL when `base_url` is not given. Default: `https://api.memgraph.ai/v1` |

The SDK validates input before sending a request: API keys must start with `mg_` and `user_id` must be a non-empty string (otherwise `MemgraphValidationError`). `mg.ping()` checks connectivity and that the key is valid.

## MCP server

Give Claude Desktop, Cursor or VS Code persistent memory:

```bash
pip install "memgraph-sdk[mcp]"
memgraph setup --key mg_your_api_key   # detects your IDE and writes its MCP config
```

Or configure it manually:

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

| Tool | What it does |
|---|---|
| `memgraph_search` | Semantic search over stored memories |
| `memgraph_remember` | Store a memory |
| `memgraph_forget` | Delete a memory by the `id` from a search result |
| `memgraph_think` | Recall for the current topic and learn from the conversation |
| `memgraph_profile` | Consolidated profile |

`MEMGRAPH_AGENT_USER_ID` sets whose memory the IDE reads and writes (default `ai_agent`). Teammates who share an API key should each set their own value.

## CLI

```bash
memgraph setup --key mg_your_api_key
memgraph remember "We chose PostgreSQL for analytics"
memgraph recall "database choice"
memgraph status
```

The CLI reads `MEMGRAPH_API_KEY` and `MEMGRAPH_API_URL` from the environment; they override the values `memgraph setup` saves in `.memgraph.env`.

## Framework integrations

| Framework | Integration |
|---|---|
| OpenAI Agents SDK | `from memgraph_sdk.openai_agents import MemgraphAgentHooks, MemgraphRunHooks` |
| LangChain / LangGraph | Memory and retriever adapters ([examples](https://github.com/shubhamdev0/memgraph-sdk/tree/main/examples/integrations)) |
| CrewAI | Search and remember tools ([examples](https://github.com/shubhamdev0/memgraph-sdk/tree/main/examples/integrations)) |
| LlamaIndex | Memory adapter ([examples](https://github.com/shubhamdev0/memgraph-sdk/tree/main/examples/integrations)) |
| Claude Code, Cursor, VS Code | [MCP server](#mcp-server) |

Guides: [memgraph.ai/docs/integrations](https://memgraph.ai/docs/integrations).

## Versioning

This package follows [semantic versioning](https://semver.org/); changes are listed in the [changelog](https://github.com/shubhamdev0/memgraph-sdk/blob/main/CHANGELOG.md). Upgrade with `pip install --upgrade memgraph-sdk`. Avoid `--force-reinstall`, which can break other packages in the same environment.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/shubhamdev0/memgraph-sdk/blob/main/CONTRIBUTING.md).

## Security

Report vulnerabilities privately to **security@memgraph.ai**, not in a public issue. See [SECURITY.md](https://github.com/shubhamdev0/memgraph-sdk/blob/main/SECURITY.md).

## License

[MIT](https://github.com/shubhamdev0/memgraph-sdk/blob/main/LICENSE)
