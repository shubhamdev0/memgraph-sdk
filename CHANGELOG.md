# Changelog

All notable changes to the Memgraph SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.8.1] - 2026-04-08

### Added
- `--version` / `-V` flag on CLI (`memgraph --version`)
- API key format validation — must start with `mg_`, raises `MemgraphValidationError` on init
- `user_id` validation — rejects empty strings in `add()`, `remember()`, `search()`
- `ping()` now validates API key authenticity (calls `/auth/whoami`, not just `/health`)
- Welcome banner in `memgraph setup` with next-step instructions
- CLI `recall` shows visual score bars and truncates long content
- CLI `remember` shows category in confirmation (`Remembered (architecture): ...`)
- CLI `setup` validates API key format before connecting
- Signup flow documented in README with step-by-step instructions
- `.env` file setup guide in README
- Rate limit table in README
- Input validation section in README

### Fixed
- `ping()` with a fake API key no longer silently succeeds — raises `MemgraphAuthError`
- All 4 example files rewritten with correct SDK v0.8 API calls
- CLI help epilog now includes "Get started" instructions
- `memgraph setup` success message shows Python quick-start snippet

## [0.8.0] - 2026-04-08

### Added
- **Decisions & Reasoning Traces**: `record_decision()`, `get_decision()`, `explain_decision()`, `list_decisions()` — full decision debugging with reasoning steps, tools used, and outcome tracking
- **Entities & Knowledge Graph**: `create_entity()`, `get_entity()`, `list_entities()`, `search_entities()`, `delete_entity()`
- **Relationships**: `create_relationship()`, `list_relationships()`
- **Graph Traversal**: `traverse_graph()` with depth control and temporal filtering
- **Context Graph**: `get_context_graph()` — 6-stage retrieval pipeline (v2 API)
- **Contradictions**: `list_contradictions()`, `resolve_contradiction()`
- **Analytics**: `analytics()` endpoint for decision observability
- **Cognitive Sidecar**: `sidecar_pre_flight()`, `sidecar_post_flight()`, `sidecar_process()` — always-on memory middleware
- Comprehensive README with decision recording field reference tables
- `forget()` and `forget_all()` methods for belief deletion

### Fixed
- CLI `recall` command now falls back to v1/context when v2/context is unavailable
- CLI `recall` shows proper error messages instead of generic "Internal Server Error"
- `record_decision()` docstring now documents exact field names for `reasoning_steps` and `tools_used`
- `add()` docstring clarifies async extraction timing vs `remember()` for immediate searchability

## [0.7.2] - 2026-04-04

### Added
- OpenAI Agents SDK integration: `MemgraphAgentHooks`, `MemgraphRunHooks`, `create_memgraph_agent()`
- `memgraph_instructions()` — dynamic system prompt injector
- `memgraph_tools()` — memory tool set for agents
- `DecisionCapture` class for structured decision recording

### Fixed
- `search()` now uses v2/context endpoint for structured JSON responses
- CLI `setup` command auto-detects Claude Code and writes MCP config
- All 83 tests passing

## [0.7.0] - 2026-04-03

### Added
- `MemgraphMemory` middleware for agent framework integration
- `CognitiveSidecar` middleware for always-on memory
- MCP server rewrite with 3 canonical tools
- Async client parity with sync client

## [0.6.0] - 2026-03-30

### Added
- gpt-5.4 model support across extraction and reasoning pipelines
- Web search integration via Tavily for grounding memories against live data
- Smart memory-first search: checks memory before triggering web search, reducing latency
- Conflict detection improvements: better handling of contradictory beliefs across sessions
- Belief stability scoring: tracks how frequently a belief is reinforced vs. contradicted
- Playground upgrades: interactive belief inspection, confidence history charts, search debugger
- `handle_search` function in MCP server (explicit, testable — was previously only an alias)
- MCP TOOLS list simplified to 3 canonical tools: `memgraph_remember`, `memgraph_search`, `memgraph_profile`

### Fixed
- MCP server `memgraph_search` tool now routes to dedicated `handle_search` handler
- `memgraph_recall` retained as backward-compatibility alias in `call_tool` dispatcher

## [0.2.0] - 2026-02-27

### Added
- `remember()` method on both `MemgraphClient` and `AsyncMemgraphClient` for immediate belief storage with vector embedding
- Memory Intelligence API on `AsyncMemgraphClient`: `health()`, `contradictions()`, `evaluate()`, `mcis()`, `mcis_history()`
- Benchmark API on `AsyncMemgraphClient`: `benchmark()`, `benchmark_scenarios()`
- `log_event()` method on `AsyncMemgraphClient` for raw event logging
- `get_beliefs()` method on `AsyncMemgraphClient`

### Fixed
- Memories stored via `remember()` are now immediately searchable (bypasses event pipeline)

## [0.1.0] - 2026-02-27

### Added
- Initial public release
- `MemgraphClient` for synchronous API calls
- `AsyncMemgraphClient` for async/await usage (requires `httpx`)
- Memory operations: `add()`, `search()`
- Memory Intelligence API: `health()`, `contradictions()`, `evaluate()`, `mcis()`, `mcis_history()`
- Benchmarking: `benchmark()`, `benchmark_scenarios()`
- CLI tool: `memgraph init`, `memgraph remember`, `memgraph recall`, `memgraph status`
