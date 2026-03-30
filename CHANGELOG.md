# Changelog

All notable changes to the Memgraph SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
