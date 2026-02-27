"""
Memgraph OS + OpenAI SDK Integration

This example demonstrates how to integrate Memgraph OS with the OpenAI SDK
to create AI agents with persistent memory.

Key Features:
- Automatic memory storage for all conversations
- Context retrieval before LLM calls
- Belief-aware responses
- Multi-turn conversation support

Installation:
    pip install openai memgraph-sdk

Usage:
    export MEMGRAPH_API_KEY=vel_your_key
    export MEMGRAPH_TENANT_ID=your-tenant-id
    export OPENAI_API_KEY=sk-your-openai-key

    python openai_integration.py
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI SDK not installed. Install with: pip install openai")
    sys.exit(1)

# Memgraph SDK
try:
    from memgraph_sdk import MemgraphClient
except ImportError:
    print("Memgraph SDK not found. Install with: pip install memgraph-sdk")
    sys.exit(1)


# ============================================================================
# MemgraphOpenAIAgent - OpenAI Agent with Persistent Memory
# ============================================================================

class MemgraphOpenAIAgent:
    """
    OpenAI agent with persistent memory via Memgraph OS.

    This agent automatically:
    1. Retrieves relevant context before generating responses
    2. Stores all conversations in Memgraph
    3. Uses beliefs to personalize responses
    4. Maintains conversation history across sessions

    Example:
        agent = MemgraphOpenAIAgent(
            openai_api_key="sk-...",
            memgraph_api_key="vel_...",
            memgraph_tenant_id="tenant-id",
            user_id="user123"
        )

        response = agent.chat("What programming language should I use?")
        print(response)
    """

    def __init__(
        self,
        openai_api_key: str,
        memgraph_api_key: str,
        memgraph_tenant_id: str,
        user_id: str,
        model: str = "gpt-4",
        memgraph_base_url: str = None
    ):
        """
        Initialize the agent.

        Args:
            openai_api_key: OpenAI API key
            memgraph_api_key: Memgraph API key
            memgraph_tenant_id: Memgraph tenant ID
            user_id: User identifier
            model: OpenAI model to use (default: gpt-4)
            memgraph_base_url: Memgraph API URL
        """
        memgraph_base_url = memgraph_base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.openai = OpenAI(api_key=openai_api_key)
        self.memgraph = MemgraphClient(
            api_key=memgraph_api_key,
            tenant_id=memgraph_tenant_id,
            base_url=memgraph_base_url
        )
        self.user_id = user_id
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Send a message and get a response with memory.

        This method:
        1. Retrieves relevant context from Memgraph
        2. Builds a prompt with beliefs and history
        3. Generates a response using OpenAI
        4. Stores the interaction in Memgraph

        Args:
            message: User message
            system_prompt: Optional custom system prompt
            temperature: OpenAI temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            Agent response string
        """
        print(f"\n💬 User: {message}")

        # Step 1: Retrieve context from memory
        print("🧠 Retrieving context from memory...")
        context = self._get_context(message)

        # Step 2: Build prompt with context
        system_prompt = system_prompt or self._build_system_prompt(context)

        # Step 3: Generate response
        print("🤖 Generating response...")
        response = self._generate_response(system_prompt, message, temperature, max_tokens)

        # Step 4: Store interaction
        print("💾 Storing interaction in memory...")
        self._store_interaction(message, response)

        print(f"✅ Agent: {response}")
        return response

    def _get_context(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant context from Memgraph."""
        try:
            context = self.memgraph.search(
                query=query,
                user_id=self.user_id
            )
            return context
        except Exception as e:
            print(f"⚠️  Context retrieval failed: {e}")
            return {"memories": [], "beliefs": [], "history": []}

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with memory context."""
        beliefs = context.get("beliefs", [])
        memories = context.get("memories", [])
        history = context.get("history", [])

        prompt = """You are a helpful AI assistant with persistent memory.

You remember past conversations, user preferences, and project details.
Use the context below to provide personalized, context-aware responses.
"""

        # Add beliefs (long-term facts)
        if beliefs:
            prompt += "\n## What You Know About This User:\n"
            for belief in beliefs[:5]:
                if isinstance(belief, dict):
                    key = belief.get("key", "")
                    value = belief.get("value", "")
                    confidence = belief.get("confidence_score", 1.0)
                    prompt += f"- {key}: {value} (confidence: {confidence:.0%})\n"

        # Add relevant memories (episodic context)
        if memories:
            prompt += "\n## Relevant Past Context:\n"
            for memory in memories[:3]:
                if isinstance(memory, dict):
                    text = memory.get("text", memory.get("content", ""))
                    if isinstance(text, dict):
                        text = text.get("text", str(text))
                    prompt += f"- {text}\n"

        # Add recent conversation history
        if history:
            prompt += "\n## Recent Conversation:\n"
            for item in history[-3:]:
                if isinstance(item, dict):
                    event_type = item.get("event_type", "")
                    content = item.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", str(content))
                    else:
                        text = str(content)

                    if "user" in event_type.lower():
                        prompt += f"User: {text}\n"
                    elif "agent" in event_type.lower():
                        prompt += f"You: {text}\n"

        prompt += "\n## Instructions:\n"
        prompt += "- Use the context above to provide personalized responses\n"
        prompt += "- Reference past conversations when relevant\n"
        prompt += "- Respect user preferences and beliefs\n"
        prompt += "- Be helpful, accurate, and concise\n"

        return prompt

    def _generate_response(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using OpenAI."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Add conversation history
            for msg in self.conversation_history[-5:]:  # Last 5 exchanges
                messages.insert(-1, msg)

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return "I'm having trouble generating a response. Please try again."

    def _store_interaction(self, message: str, response: str):
        """Store interaction in Memgraph."""
        try:
            # Store user message
            self.memgraph.add(
                text=f"User: {message}\nAgent: {response}",
                user_id=self.user_id,
                metadata={
                    "type": "conversation",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            print(f"⚠️  Memory storage failed: {e}")

    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile from Memgraph.

        Returns:
            Dictionary with beliefs, preferences, and stats
        """
        try:
            context = self.memgraph.search(
                query=f"user profile for {self.user_id}",
                user_id=self.user_id
            )

            profile = {
                "user_id": self.user_id,
                "beliefs": context.get("beliefs", []),
                "recent_memories": context.get("memories", [])[:5],
                "conversation_turns": len(self.conversation_history) // 2
            }

            return profile

        except Exception as e:
            print(f"Error getting profile: {e}")
            return {}


# ============================================================================
# OpenAI Assistants API Integration
# ============================================================================

class MemgraphOpenAIAssistant:
    """
    OpenAI Assistants API integration with Memgraph memory.

    This class extends OpenAI Assistants with persistent memory across sessions.

    Example:
        assistant = MemgraphOpenAIAssistant(
            openai_api_key="sk-...",
            memgraph_api_key="vel_...",
            memgraph_tenant_id="tenant-id",
            assistant_id="asst_...",
            user_id="user123"
        )

        response = assistant.chat("What did we discuss last time?")
    """

    def __init__(
        self,
        openai_api_key: str,
        memgraph_api_key: str,
        memgraph_tenant_id: str,
        assistant_id: str,
        user_id: str,
        memgraph_base_url: str = None
    ):
        """
        Initialize assistant with memory.

        Args:
            openai_api_key: OpenAI API key
            memgraph_api_key: Memgraph API key
            memgraph_tenant_id: Memgraph tenant ID
            assistant_id: OpenAI Assistant ID
            user_id: User identifier
            memgraph_base_url: Memgraph API URL
        """
        memgraph_base_url = memgraph_base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.openai = OpenAI(api_key=openai_api_key)
        self.memgraph = MemgraphClient(
            api_key=memgraph_api_key,
            tenant_id=memgraph_tenant_id,
            base_url=memgraph_base_url
        )
        self.assistant_id = assistant_id
        self.user_id = user_id
        self.thread_id = None

    def chat(self, message: str) -> str:
        """
        Send a message to the assistant with memory context.

        Args:
            message: User message

        Returns:
            Assistant response
        """
        # Create thread if needed
        if not self.thread_id:
            thread = self.openai.beta.threads.create()
            self.thread_id = thread.id

        # Retrieve context from Memgraph
        context = self.memgraph.search(query=message, user_id=self.user_id)

        # Add context as additional instructions
        context_text = self._format_context(context)

        # Send message
        self.openai.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=f"{context_text}\n\nUser message: {message}"
        )

        # Run assistant
        run = self.openai.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id
        )

        # Wait for completion
        while run.status in ["queued", "in_progress"]:
            run = self.openai.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )

        # Get response
        messages = self.openai.beta.threads.messages.list(
            thread_id=self.thread_id
        )
        response = messages.data[0].content[0].text.value

        # Store in Memgraph
        self.memgraph.add(
            text=f"User: {message}\nAssistant: {response}",
            user_id=self.user_id,
            metadata={"type": "assistant_conversation"}
        )

        return response

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for assistant."""
        text = "Context from memory:\n"

        for belief in context.get("beliefs", [])[:3]:
            if isinstance(belief, dict):
                text += f"- {belief.get('key')}: {belief.get('value')}\n"

        return text


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_chat():
    """Example: Basic chat with memory"""
    agent = MemgraphOpenAIAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        memgraph_api_key=os.getenv("MEMGRAPH_API_KEY"),
        memgraph_tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user_openai"
    )

    # Multi-turn conversation
    agent.chat("Hi, my name is Sarah and I'm learning Python")
    agent.chat("I'm building a web application with Flask")
    agent.chat("What's my name and what framework am I using?")

    # Get profile
    profile = agent.get_profile()
    print("\n📊 User Profile:")
    print(f"Beliefs: {len(profile.get('beliefs', []))}")
    print(f"Memories: {len(profile.get('recent_memories', []))}")


def example_function_calling():
    """Example: OpenAI function calling with memory"""
    agent = MemgraphOpenAIAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        memgraph_api_key=os.getenv("MEMGRAPH_API_KEY"),
        memgraph_tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user_functions"
    )

    # Define functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_preference",
                "description": "Save a user preference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preference_type": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["preference_type", "value"]
                }
            }
        }
    ]

    # Chat with function support
    response = agent.openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Remember that I prefer dark mode"}
        ],
        tools=tools
    )

    print(response.choices[0].message)


def example_streaming_response():
    """Example: Streaming responses with memory"""
    agent = MemgraphOpenAIAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        memgraph_api_key=os.getenv("MEMGRAPH_API_KEY"),
        memgraph_tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user_streaming"
    )

    # Get context
    message = "Tell me about my preferences"
    context = agent._get_context(message)
    system_prompt = agent._build_system_prompt(context)

    # Stream response
    print("\n🤖 Streaming response:")
    stream = agent.openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    # Store after streaming
    agent._store_interaction(message, full_response)


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run the OpenAI integration demo"""
    print("🤖 Memgraph OS + OpenAI SDK Integration Demo")
    print("=" * 60)

    # Check environment variables
    if not all([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("MEMGRAPH_API_KEY"),
        os.getenv("MEMGRAPH_TENANT_ID")
    ]):
        print("\n❌ Missing required environment variables!")
        print("\nPlease set:")
        print("  export OPENAI_API_KEY=sk-your-key")
        print("  export MEMGRAPH_API_KEY=vel_your_key")
        print("  export MEMGRAPH_TENANT_ID=your-tenant-id")
        return

    # Run basic example
    example_basic_chat()

    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("\n💡 Other examples available:")
    print("  - example_function_calling() - Function calling with memory")
    print("  - example_streaming_response() - Streaming with memory")
    STUDIO_URL = os.getenv("MEMGRAPH_STUDIO_URL", "http://localhost:3000")
    print("\n🔍 View results in Studio:")
    print(f"   {STUDIO_URL}/dashboard")


if __name__ == "__main__":
    main()
