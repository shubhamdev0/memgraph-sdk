"""
Memgraph AI — Interactive Agent with Memory

Chat with an AI agent that remembers everything you tell it.
Uses OpenAI for responses and Memgraph for persistent memory.

Setup:
    pip install memgraph-sdk openai
    export MEMGRAPH_API_KEY=mg_your_key_here
    export OPENAI_API_KEY=sk-your_key_here

Run:
    python examples/agent.py
"""

import os
from memgraph_sdk import MemgraphClient

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    exit(1)


def main():
    api_key = os.getenv("MEMGRAPH_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Set MEMGRAPH_API_KEY: export MEMGRAPH_API_KEY=mg_your_key")
        return
    if not openai_key:
        print("Set OPENAI_API_KEY: export OPENAI_API_KEY=sk-your_key")
        return

    mg = MemgraphClient(api_key=api_key)
    oai = OpenAI(api_key=openai_key)
    user_id = "interactive_user"

    print("🤖 Memgraph Agent (type 'quit' to exit)")
    print("   I remember everything you tell me.\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # 1. Recall memories
        memories = mg.get_beliefs(user_id=user_id, limit=20)
        items = memories if isinstance(memories, list) else memories.get("items", [])

        mem_text = ""
        if items:
            mem_text = "Known about this user:\n"
            for b in items[:10]:
                mem_text += f"  - {b['key']}: {b['value']}\n"

        # 2. Generate response
        messages = [
            {"role": "system", "content": f"You are a helpful assistant with memory.\n{mem_text}"},
        ]
        messages.extend(history[-6:])  # Last 3 turns
        messages.append({"role": "user", "content": user_input})

        response = oai.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=200,
        )
        reply = response.choices[0].message.content
        print(f"Agent: {reply}\n")

        # 3. Store this interaction
        mg.remember(f"User said: {user_input}", user_id=user_id, category="conversation")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
