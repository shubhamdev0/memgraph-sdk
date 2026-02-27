import os
import time

from memgraph_sdk import MemgraphClient

API_KEY = os.getenv("MEMGRAPH_API_KEY", "vel_test_sdk")
TENANT_ID = os.getenv("MEMGRAPH_TENANT_ID", "demo-tenant")

def main():
    print("🚀 Initializing Memgraph Client...")
    client = MemgraphClient(api_key=API_KEY, tenant_id=TENANT_ID)

    user_id = "sdk_user_01"

    # 1. Add a Memory
    print("\n📝 Adding Memory...")
    try:
        res = client.add(
            text="Hello from Python SDK! Feeling excited about Memgraph.",
            user_id=user_id,
            metadata={"mood": "Excited"}
        )
        print(f"✅ Memory Added: {res}")
    except Exception as e:
        print(f"❌ Add Failed: {e}")

    # 2. Search Memory
    print("\n🧠 Searching Memory...")
    try:
        time.sleep(1)  # Wait for ingestion
        results = client.search(query="Hello SDK", user_id=user_id)
        memories = results.get('memories', [])
        print(f"✅ Memories Found: {len(memories)}")
        for m in memories[:2]:
            text = m.get('text', m.get('content', str(m)))
            if isinstance(text, dict):
                text = text.get('text', str(text))
            print(f"   - {text[:80]}...")
    except Exception as e:
        print(f"❌ Search Failed: {e}")

if __name__ == "__main__":
    main()
