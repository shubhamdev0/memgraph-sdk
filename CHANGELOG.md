# Changelog

All notable changes to the Memgraph SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.1] - 2026-02-27

### Fixed
- Fix import sorting and lint issues for CI compliance
- Clean up unused imports and trailing whitespace

## [0.1.0] - 2026-02-27

### Added
- Initial public release
- `MemgraphClient` for synchronous API calls
- `AsyncMemgraphClient` for async/await usage (requires `httpx`)
- Memory operations: `add()`, `search()`
- Memory Intelligence API: `health()`, `contradictions()`, `evaluate()`, `mcis()`, `mcis_history()`
- Benchmarking: `benchmark()`, `benchmark_scenarios()`
- CLI tool: `memgraph init`, `memgraph remember`, `memgraph recall`, `memgraph status`
