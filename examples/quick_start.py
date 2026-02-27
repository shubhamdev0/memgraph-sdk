"""
Memgraph OS Quick Start Example

This script demonstrates the complete integration flow:
1. Initialize the client
2. Store some memories
3. Search for relevant context
4. View what was learned

Run this after:
1. Starting the Memgraph OS server
2. Creating an account in the dashboard
3. Getting your API key from Settings

Usage:
    export MEMGRAPH_API_KEY=vel_your_key_here
    export MEMGRAPH_TENANT_ID=your-tenant-uuid
    python examples/quick_start.py
"""

import os
import time
from memgraph_sdk import MemgraphClient


def main():
    """Run the quick start demo"""

    STUDIO_URL = os.getenv("MEMGRAPH_STUDIO_URL", "http://localhost:3000")

    print("🚀 Memgraph OS Quick Start Demo")
    print("=" * 50)

    # Step 1: Initialize Client
    print("\n1️⃣ Initializing Memgraph Client...")

    api_key = os.getenv("MEMGRAPH_API_KEY")
    tenant_id = os.getenv("MEMGRAPH_TENANT_ID")

    if not api_key or not tenant_id:
        print("❌ Error: Missing credentials!")
        print("\nPlease set environment variables:")
        print("  export MEMGRAPH_API_KEY=vel_your_key")
        print("  export MEMGRAPH_TENANT_ID=your-tenant-id")
        print(f"\nGet these from: {STUDIO_URL}/studio/settings")
        return

    try:
        client = MemgraphClient(
            api_key=api_key,
            tenant_id=tenant_id,
            base_url=os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        )
        print("✅ Client initialized successfully!")
        print(f"   Connected to: {client.base_url}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Step 2: Store Some Memories
    print("\n2️⃣ Storing sample memories...")

    user_id = "demo_user_" + str(int(time.time()))
    memories = [
        {
            "text": "User mentioned they prefer Python over JavaScript for backend development",
            "metadata": {"category": "preference", "topic": "programming"}
        },
        {
            "text": "Project decision: We will use PostgreSQL as the primary database",
            "metadata": {"category": "architecture", "topic": "database"}
        },
        {
            "text": "Bug fix: Resolved authentication token expiration issue by increasing TTL to 1 week",
            "metadata": {"category": "bug_fix", "topic": "authentication"}
        },
        {
            "text": "User asked about the weather in San Francisco. Provided current weather forecast.",
            "metadata": {"category": "general", "topic": "weather"}
        },
        {
            "text": "Discussion: Team decided to deploy using Docker containers on AWS ECS",
            "metadata": {"category": "decision", "topic": "deployment"}
        }
    ]

    for i, memory in enumerate(memories, 1):
        try:
            result = client.add(
                text=memory["text"],
                user_id=user_id,
                metadata=memory["metadata"]
            )
            print(f"   ✓ Memory {i}/5 stored")
        except Exception as e:
            print(f"   ✗ Memory {i}/5 failed: {e}")

    print(f"✅ {len(memories)} memories stored for user: {user_id}")

    # Step 3: Search for Relevant Context
    print("\n3️⃣ Searching for relevant context...")

    queries = [
        "What programming language should I use?",
        "How did we fix the authentication issue?",
        "What database are we using?"
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        try:
            context = client.search(
                query=query,
                user_id=user_id
            )

            # Display results
            memories_found = context.get('memories', [])
            if memories_found:
                print(f"   📝 Found {len(memories_found)} relevant memories:")
                for mem in memories_found[:2]:  # Show top 2
                    text = mem.get('text', mem.get('content', 'N/A'))
                    if isinstance(text, dict):
                        text = text.get('text', str(text))
                    print(f"      - {text[:80]}...")
            else:
                print("   📝 No memories found (might need time to process)")

        except Exception as e:
            print(f"   ✗ Search failed: {e}")

    # Step 4: What You Can Do Next
    print("\n" + "=" * 50)
    print("🎉 Quick Start Complete!")
    print("\n📊 Next Steps:")
    print("\n1. View your data in the dashboard:")
    print(f"   {STUDIO_URL}/dashboard")
    print("\n2. Check the episode that was created:")
    print(f"   {STUDIO_URL}/studio/episodes")
    print("\n3. Wait for Cognitive Dreaming to extract beliefs:")
    print(f"   {STUDIO_URL}/studio/beliefs")
    print("\n4. Integrate into your agent:")
    print("   See examples/agent_integration.py")
    print("\n5. Read the full guide:")
    print("   TEAM_ONBOARDING.md")

    print("\n" + "=" * 50)
    print("🔑 Your Test User ID:", user_id)
    print("💡 Use this ID to filter data in the dashboard")


if __name__ == "__main__":
    main()
