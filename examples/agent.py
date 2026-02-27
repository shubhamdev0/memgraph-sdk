import os

from memgraph_sdk import MemgraphClient

# Config (Usually from env)
API_KEY = "demo-key"
TENANT_ID = "41d60ab2-2d26-4dcb-a0cc-d67e32215c71" # Nebula Corp from db seed

def simulated_agent_loop():
    client = MemgraphClient(api_key=API_KEY, tenant_id=TENANT_ID)
    
    user_id = "user_example_1"
    print(f"--- Memgraph Memory Agent (User: {user_id}) ---")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # 1. Get Context (RAG + Beliefs)
        print("... Recalling memories ...")
        context = client.get_context(query=user_input, user_id=user_id)
        
        # In a real app, you'd pass 'context' to LLM system prompt
        beliefs = context.get('beliefs', [])
        history = context.get('history', [])
        
        print("\n[🧠 Memory Context]")
        if beliefs:
            print("  Beliefs:")
            for b in beliefs:
                print(f"  - {b['key']}: {b['value']}")
        else:
            print("  - No relevant beliefs found.")
            
        print("\n[🤖 Agent Thought]")
        print(f"I should use this context to answer: '{user_input}'")
        
        # 2. Log Interaction (to learn for next time)
        client.log_event(
            event_type="user_message",
            content={"text": user_input},
            metadata={"user_id": user_id}
        )
        print("(Event logged for background learning)\n")

if __name__ == "__main__":
    simulated_agent_loop()
