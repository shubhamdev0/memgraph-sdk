"""
Memgraph AI — SDK Demo

Shows the 3 core SDK operations: remember, search, get_beliefs.

Setup:
    pip install memgraph-sdk
    export MEMGRAPH_API_KEY=mg_your_key_here

Run:
    python examples/sdk_demo.py
"""

import os
from memgraph_sdk import MemgraphClient


def main():
    api_key = os.getenv("MEMGRAPH_API_KEY")
    if not api_key:
        print("Set MEMGRAPH_API_KEY first: export MEMGRAPH_API_KEY=mg_your_key")
        return

    client = MemgraphClient(api_key=api_key)

    # Store
    client.remember("User prefers dark mode", user_id="demo", category="preference")
    client.remember("User works at Acme Corp", user_id="demo", category="work")
    print("✅ 2 memories stored")

    # Search
    import time; time.sleep(2)
    result = client.search("UI preferences", user_id="demo")
    for r in result.get("results", []):
        print(f"🔍 [{r['score']:.2f}] {r['content']}")

    # List all
    beliefs = client.get_beliefs(user_id="demo")
    items = beliefs if isinstance(beliefs, list) else beliefs.get("items", [])
    print(f"📋 {len(items)} total beliefs stored")


if __name__ == "__main__":
    main()
