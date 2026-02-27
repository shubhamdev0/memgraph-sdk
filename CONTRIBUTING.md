# Contributing to Memgraph SDK

Thank you for your interest in contributing to the Memgraph SDK!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shubhamdev0/memgraph-sdk.git
   cd memgraph-sdk
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

```bash
pytest
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check memgraph_sdk/
ruff format memgraph_sdk/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting
5. Commit your changes with a descriptive message
6. Push to your fork and submit a pull request

## Reporting Issues

Please use [GitHub Issues](https://github.com/shubhamdev0/memgraph-sdk/issues) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
