"""
Complete Agent Integration Example

This example shows how to integrate Memgraph OS into a production AI agent.
It demonstrates:
- Context retrieval before LLM calls
- Memory storage after interactions
- Error handling
- Multi-turn conversations
- Belief-aware responses

Usage:
    export MEMGRAPH_API_KEY=vel_your_key
    export MEMGRAPH_TENANT_ID=your-tenant-id
    export OPENAI_API_KEY=sk-your-openai-key

    python examples/agent_integration.py
"""

import os
from typing import List, Dict
from memgraph_sdk import MemgraphClient
from openai import OpenAI


class MemoryEnabledAgent:
    """An AI agent with persistent memory via Memgraph OS"""

    def __init__(self, memgraph_api_key: str, tenant_id: str, openai_api_key: str):
        """Initialize the agent with Memgraph and OpenAI clients"""

        self.memgraph = MemgraphClient(
            api_key=memgraph_api_key,
            tenant_id=tenant_id
        )

        self.openai = OpenAI(api_key=openai_api_key)

        self.conversation_history: List[Dict] = []

    def chat(self, user_id: str, message: str) -> str:
        """
        Handle a user message with memory integration

        Args:
            user_id: Unique identifier for the user
            message: The user's message

        Returns:
            The agent's response
        """

        print(f"\n💬 User: {message}")

        # Step 1: Retrieve relevant context from memory
        print("🧠 Retrieving context from memory...")
        context = self._get_context(user_id, message)

        # Step 2: Build prompt with context
        system_prompt = self._build_system_prompt(context)

        # Step 3: Generate response using LLM
        print("🤖 Generating response...")
        response = self._generate_response(system_prompt, message)

        # Step 4: Store the interaction for future context
        print("💾 Storing interaction in memory...")
        self._store_interaction(user_id, message, response)

        print(f"✅ Agent: {response}")
        return response

    def _get_context(self, user_id: str, query: str) -> Dict:
        """Retrieve relevant context from Memgraph"""

        try:
            context = self.memgraph.search(
                query=query,
                user_id=user_id
            )
            return context
        except Exception as e:
            print(f"⚠️ Context retrieval failed: {e}")
            # Fallback to empty context
            return {"memories": [], "beliefs": [], "history": []}

    def _build_system_prompt(self, context: Dict) -> str:
        """Build a system prompt incorporating memory context"""

        beliefs = context.get('beliefs', [])
        memories = context.get('memories', [])
        history = context.get('history', [])

        prompt = """You are a helpful AI assistant with memory.

You remember past conversations and user preferences.
"""

        # Add user profile/beliefs
        if beliefs:
            prompt += "\n## User Profile (What you know about this user):\n"
            for belief in beliefs[:5]:  # Top 5 most relevant
                if isinstance(belief, dict):
                    key = belief.get('key', '')
                    value = belief.get('value', '')
                    prompt += f"- {key}: {value}\n"
                else:
                    prompt += f"- {belief}\n"

        # Add relevant past context
        if memories:
            prompt += "\n## Relevant Past Context:\n"
            for memory in memories[:3]:  # Top 3 most relevant
                if isinstance(memory, dict):
                    text = memory.get('text', memory.get('content', ''))
                    if isinstance(text, dict):
                        text = text.get('text', str(text))
                    prompt += f"- {text}\n"
                else:
                    prompt += f"- {memory}\n"

        # Add recent conversation history
        if history:
            prompt += "\n## Recent Conversation:\n"
            for item in history[-3:]:  # Last 3 exchanges
                if isinstance(item, dict):
                    event_type = item.get('event_type', '')
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        text = content.get('text', str(content))
                    else:
                        text = str(content)

                    if 'user' in event_type.lower():
                        prompt += f"User: {text}\n"
                    elif 'agent' in event_type.lower():
                        prompt += f"You: {text}\n"

        prompt += "\n## Instructions:\n"
        prompt += "- Use the context above to provide personalized responses\n"
        prompt += "- Reference past conversations when relevant\n"
        prompt += "- Respect user preferences from their profile\n"
        prompt += "- Be helpful, accurate, and concise\n"

        return prompt

    def _generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a response using OpenAI"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return "I'm having trouble generating a response right now. Please try again."

    def _store_interaction(self, user_id: str, message: str, response: str):
        """Store the interaction in Memgraph for future context"""

        try:
            # Store the complete interaction
            self.memgraph.add(
                text=f"User: {message}\nAgent: {response}",
                user_id=user_id,
                metadata={
                    "type": "conversation",
                    "user_message": message,
                    "agent_response": response,
                    "timestamp": "auto"
                }
            )
        except Exception as e:
            print(f"⚠️ Memory storage failed: {e}")
            # Continue even if storage fails


def main():
    """Run the agent integration demo"""

    STUDIO_URL = os.getenv("MEMGRAPH_STUDIO_URL", "http://localhost:3000")

    print("🤖 Memory-Enabled Agent Demo")
    print("=" * 60)

    # Check environment variables
    memgraph_key = os.getenv("MEMGRAPH_API_KEY")
    tenant_id = os.getenv("MEMGRAPH_TENANT_ID")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not all([memgraph_key, tenant_id, openai_key]):
        print("\n❌ Missing required environment variables!")
        print("\nPlease set:")
        print("  export MEMGRAPH_API_KEY=vel_your_key")
        print("  export MEMGRAPH_TENANT_ID=your-tenant-id")
        print("  export OPENAI_API_KEY=sk-your-openai-key")
        return

    # Initialize agent
    agent = MemoryEnabledAgent(
        memgraph_api_key=memgraph_key,
        tenant_id=tenant_id,
        openai_api_key=openai_key
    )

    user_id = "demo_user_integration"

    # Simulate a multi-turn conversation
    conversation = [
        "Hi! My name is Alex and I'm a software engineer.",
        "I prefer Python for backend development.",
        "Can you help me with database design?",
        "What programming language did I say I prefer?",  # Tests memory
        "Thanks! By the way, what's my name?"  # Tests memory
    ]

    print("\n📝 Starting conversation...")
    print("=" * 60)

    for message in conversation:
        response = agent.chat(user_id, message)
        print("-" * 60)

    print("\n" + "=" * 60)
    print("✅ Conversation Complete!")
    print("\n📊 What happened:")
    print("1. Agent retrieved context before each response")
    print("2. Agent used beliefs and history to personalize answers")
    print("3. All interactions were stored in memory")
    print("\n🔍 View the results:")
    print(f"   Dashboard: {STUDIO_URL}/dashboard")
    print(f"   Episodes: {STUDIO_URL}/studio/episodes")
    print(f"   User ID: {user_id}")


if __name__ == "__main__":
    main()
