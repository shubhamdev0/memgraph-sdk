"""
Memgraph AI — Quick Start

Store memories, search them, update them. Takes 2 minutes.

Setup:
    pip install memgraph-sdk
    export MEMGRAPH_API_KEY=mg_your_key_here    # Get from memgraph.ai

Run:
    python examples/quick_start.py
"""

import os
import time
from memgraph_sdk import MemgraphClient


def main():
    print("🚀 Memgraph AI — Quick Start")
    print("=" * 50)

    # Step 1: Connect
    api_key = os.getenv("MEMGRAPH_API_KEY")
    if not api_key:
        print("\n❌ Set MEMGRAPH_API_KEY first:")
        print("   export MEMGRAPH_API_KEY=mg_your_key_here")
        print("   Get one at: https://memgraph.ai")
        return

    client = MemgraphClient(api_key=api_key)
    print(f"\n✅ Connected to {client.base_url}")

    user_id = f"quickstart_{int(time.time())}"

    # Step 2: Store memories (immediately searchable)
    print("\n📝 Storing memories...")
    memories = [
        ("User prefers Python over JavaScript for backend", "preference"),
        ("Project uses PostgreSQL as primary database", "architecture"),
        ("Fixed auth token expiration by increasing TTL to 1 week", "bug_fix"),
        ("Team decided to deploy on AWS ECS with Docker", "decision"),
        ("User is allergic to peanuts", "general"),
    ]

    for text, category in memories:
        client.remember(text=text, user_id=user_id, category=category)
        print(f"   ✅ {text[:50]}...")

    # Small delay for embedding to complete
    time.sleep(2)

    # Step 3: Search memories
    print("\n🔍 Searching memories...")
    queries = [
        "What programming language does the user prefer?",
        "How did we fix the authentication issue?",
        "What database are we using?",
        "Any health concerns?",
    ]

    for query in queries:
        result = client.search(query=query, user_id=user_id)
        hits = result.get("results", [])
        print(f"\n   Q: {query}")
        if hits:
            top = hits[0]
            print(f"   → [{top['score']:.2f}] {top['content'][:60]}")
        else:
            print("   → No results (try waiting a moment and rerun)")

    # Step 4: Get all beliefs
    print("\n📋 All stored beliefs:")
    beliefs = client.get_beliefs(user_id=user_id, limit=10)
    items = beliefs if isinstance(beliefs, list) else beliefs.get("items", [])
    for b in items:
        print(f"   • {b['key']}: {b['value'][:50]}")

    # Step 5: Update a memory
    print("\n🔄 Updating: PostgreSQL → MySQL")
    client.remember(
        text="Project migrated from PostgreSQL to MySQL",
        user_id=user_id,
        category="architecture",
    )
    time.sleep(1)

    result = client.search("What database are we using?", user_id=user_id)
    if result.get("results"):
        print(f"   → Latest: {result['results'][0]['content'][:60]}")

    print("\n" + "=" * 50)
    print("✅ Quick start complete!")
    print(f"\n📊 View your data: https://memgraph.ai")
    print(f"📖 Full docs: https://memgraph.ai/docs")
    print(f"💬 Questions? https://github.com/shubhamdev0/memgraph-sdk/issues")


if __name__ == "__main__":
    main()
