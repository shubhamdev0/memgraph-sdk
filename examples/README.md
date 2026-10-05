# Memgraph AI Examples

## Setup

```bash
pip install memgraph-sdk
export MEMGRAPH_API_KEY=mg_your_key_here   # Get from memgraph.ai
```

## Examples

| Example | Description | Extra deps |
|---|---|---|
| [quick_start.py](quick_start.py) | Store, search, update memories | None |
| [sdk_demo.py](sdk_demo.py) | Core SDK operations in 30 lines | None |
| [agent.py](agent.py) | Interactive chat agent with memory | `openai` |
| [agent_integration.py](agent_integration.py) | Full agent with Memgraph sidecar | `openai` |

## Integration Examples

| Example | Description | Extra deps |
|---|---|---|
| [openai_integration.py](integrations/openai_integration.py) | OpenAI function calling + memory | `openai` |
| [langchain_integration.py](integrations/langchain_integration.py) | LangChain memory adapter | `langchain` |
| [crewai_integration.py](integrations/crewai_integration.py) | CrewAI shared agent memory | `crewai` |
| [llamaindex_integration.py](integrations/llamaindex_integration.py) | LlamaIndex memory integration | `llama-index` |
| MCP server for Claude/Cursor | Built in: `pip install "memgraph-sdk[mcp]"` then `memgraph setup --key mg_...` (or `python -m memgraph_sdk.mcp`) | `mcp` |

## Quick Test

```bash
# Store and search in 4 lines:
python -c "
from memgraph_sdk import MemgraphClient
mg = MemgraphClient(api_key='$MEMGRAPH_API_KEY')
mg.remember('I love pizza', user_id='test')
import time; time.sleep(2)
print(mg.search('food preferences', user_id='test'))
"
```
