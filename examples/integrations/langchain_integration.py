"""
Memgraph OS LangChain Integration

This module provides LangChain-compatible classes for integrating Memgraph OS
with LangChain agents and chains.

Includes:
- MemgraphMemory: Chat message history storage
- MemgraphRetriever: Semantic search retriever
- MemgraphVectorStore: Vector store interface

Installation:
    pip install langchain langchain-core memgraph-sdk

Usage Example:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from memgraph_langchain import MemgraphMemory, MemgraphRetriever

    # Initialize Memgraph memory
    memory = MemgraphMemory(
        api_key="vel_your_key",
        tenant_id="your-tenant-id",
        user_id="user123"
    )

    # Use with LangChain agent
    llm = ChatOpenAI(model="gpt-4")
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

    # Run agent - memory is automatically stored
    result = executor.invoke({"input": "What did we discuss about databases?"})
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

# LangChain imports
try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.memory import BaseMemory
except ImportError:
    raise ImportError(
        "LangChain not installed. Install with: pip install langchain langchain-core"
    )

# Memgraph SDK import
try:
    from memgraph_sdk import MemgraphClient
except ImportError:
    raise ImportError(
        "Memgraph SDK not found. Install with: pip install memgraph-sdk"
    )


# ============================================================================
# MemgraphMemory - Chat Message History Storage
# ============================================================================

class MemgraphMemory(BaseChatMessageHistory):
    """
    LangChain-compatible chat message history stored in Memgraph OS.

    This class stores all conversation messages as events in Memgraph,
    and retrieves relevant historical context for each interaction.

    Example:
        memory = MemgraphMemory(
            api_key="vel_your_key",
            tenant_id="tenant_id",
            user_id="user123",
            session_id="session_abc"
        )

        # Add messages
        memory.add_user_message("What databases do you support?")
        memory.add_ai_message("We support PostgreSQL, MySQL, and MongoDB")

        # Get messages
        messages = memory.messages  # Returns all messages for this session
    """

    def __init__(
        self,
        api_key: str,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        base_url: str = None
    ):
        """
        Initialize Memgraph memory.

        Args:
            api_key: Memgraph API key (starts with 'vel_')
            tenant_id: Tenant identifier
            user_id: User identifier
            session_id: Optional session identifier (defaults to user_id)
            base_url: Memgraph API URL
        """
        base_url = base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.client = MemgraphClient(
            api_key=api_key,
            tenant_id=tenant_id,
            base_url=base_url
        )
        self.user_id = user_id
        self.session_id = session_id or user_id
        self._message_cache: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the history.

        Args:
            message: LangChain message (HumanMessage, AIMessage, etc.)
        """
        # Determine event type based on message type
        if isinstance(message, HumanMessage):
            event_type = "user_message"
        elif isinstance(message, AIMessage):
            event_type = "agent_message"
        elif isinstance(message, SystemMessage):
            event_type = "system_message"
        else:
            event_type = "message"

        # Store in Memgraph
        self.client.add(
            text=message.content,
            user_id=self.user_id,
            metadata={
                "event_type": event_type,
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": message.type
            }
        )

        # Update cache
        self._message_cache.append(message)

    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Clear message cache (does not delete from Memgraph)."""
        self._message_cache = []

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Retrieve all messages for this session.

        Returns relevant historical context from Memgraph.
        """
        if self._message_cache:
            return self._message_cache

        # Retrieve from Memgraph
        try:
            context = self.client.search(
                query=f"session:{self.session_id} conversation history",
                user_id=self.user_id
            )

            messages = []
            for event in context.get("history", []):
                if isinstance(event, dict):
                    content = event.get("content", {})
                    text = content.get("text", "") if isinstance(content, dict) else str(content)
                    event_type = event.get("event_type", "")

                    if "user" in event_type.lower():
                        messages.append(HumanMessage(content=text))
                    elif "agent" in event_type.lower() or "ai" in event_type.lower():
                        messages.append(AIMessage(content=text))

            self._message_cache = messages
            return messages
        except Exception as e:
            print(f"Error retrieving messages: {e}")
            return []


# ============================================================================
# MemgraphRetriever - Semantic Search
# ============================================================================

class MemgraphRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for semantic search in Memgraph OS.

    This retriever searches across all Memgraph memory layers (events, episodes, beliefs)
    and returns relevant documents.

    Example:
        retriever = MemgraphRetriever(
            api_key="vel_your_key",
            tenant_id="tenant_id",
            user_id="user123",
            k=5
        )

        # Use in a chain
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever
        )

        result = qa_chain.invoke("What databases are we using?")
    """

    def __init__(
        self,
        api_key: str,
        tenant_id: str,
        user_id: str,
        k: int = 5,
        base_url: str = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Memgraph retriever.

        Args:
            api_key: Memgraph API key
            tenant_id: Tenant identifier
            user_id: User identifier
            k: Number of documents to retrieve (default: 5)
            base_url: Memgraph API URL
            search_kwargs: Additional search parameters
        """
        super().__init__()
        base_url = base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.client = MemgraphClient(
            api_key=api_key,
            tenant_id=tenant_id,
            base_url=base_url
        )
        self.user_id = user_id
        self.k = k
        self.search_kwargs = search_kwargs or {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents from Memgraph.

        Args:
            query: Search query
            run_manager: Callback manager (unused)

        Returns:
            List of LangChain Documents
        """
        try:
            # Search Memgraph
            context = self.client.search(
                query=query,
                user_id=self.user_id
            )

            documents = []

            # Add beliefs (long-term facts)
            for belief in context.get("beliefs", [])[:self.k]:
                if isinstance(belief, dict):
                    content = f"{belief.get('key', '')}: {belief.get('value', '')}"
                    metadata = {
                        "source": "belief",
                        "confidence": belief.get("confidence_score", 1.0),
                        "user_id": self.user_id
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            # Add memories (episodes)
            for memory in context.get("memories", [])[:self.k]:
                if isinstance(memory, dict):
                    text = memory.get("text", memory.get("content", ""))
                    if isinstance(text, dict):
                        text = text.get("text", str(text))

                    metadata = {
                        "source": "episode",
                        "episode_id": memory.get("id"),
                        "user_id": self.user_id
                    }
                    documents.append(Document(page_content=text, metadata=metadata))

            # Add recent events if not enough results
            if len(documents) < self.k:
                for event in context.get("history", [])[:self.k - len(documents)]:
                    if isinstance(event, dict):
                        content = event.get("content", {})
                        text = content.get("text", "") if isinstance(content, dict) else str(content)

                        metadata = {
                            "source": "event",
                            "event_type": event.get("event_type"),
                            "user_id": self.user_id
                        }
                        documents.append(Document(page_content=text, metadata=metadata))

            return documents[:self.k]

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


# ============================================================================
# MemgraphConversationMemory - For Chains and Agents
# ============================================================================

class MemgraphConversationMemory(BaseMemory):
    """
    LangChain conversation memory backed by Memgraph OS.

    This memory class stores and retrieves conversation context for chains and agents.

    Example:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        memory = MemgraphConversationMemory(
            api_key="vel_your_key",
            tenant_id="tenant_id",
            user_id="user123"
        )

        conversation = ConversationChain(
            llm=ChatOpenAI(),
            memory=memory,
            verbose=True
        )

        conversation.predict(input="Hi, I'm building a web app")
        conversation.predict(input="What did I just say I'm building?")
    """

    def __init__(
        self,
        api_key: str,
        tenant_id: str,
        user_id: str,
        memory_key: str = "history",
        base_url: str = None
    ):
        """
        Initialize conversation memory.

        Args:
            api_key: Memgraph API key
            tenant_id: Tenant identifier
            user_id: User identifier
            memory_key: Key to use for storing memory in chain context
            base_url: Memgraph API URL
        """
        super().__init__()
        base_url = base_url or os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
        self.chat_memory = MemgraphMemory(
            api_key=api_key,
            tenant_id=tenant_id,
            user_id=user_id,
            base_url=base_url
        )
        self.memory_key = memory_key
        self.user_id = user_id

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables.

        Args:
            inputs: Input variables

        Returns:
            Dictionary with memory key and chat history
        """
        messages = self.chat_memory.messages
        return {self.memory_key: messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context to memory.

        Args:
            inputs: Input dictionary with user message
            outputs: Output dictionary with AI response
        """
        # Save user message
        if "input" in inputs:
            self.chat_memory.add_user_message(inputs["input"])

        # Save AI response
        if "output" in outputs:
            self.chat_memory.add_ai_message(outputs["output"])

    def clear(self) -> None:
        """Clear memory cache."""
        self.chat_memory.clear()


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_usage():
    """Example: Basic memory storage and retrieval"""
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain

    # Initialize memory
    memory = MemgraphConversationMemory(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user"
    )

    # Create conversation chain
    llm = ChatOpenAI(model="gpt-4")
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Have conversation
    print(conversation.predict(input="My name is Alice and I'm a Python developer"))
    print(conversation.predict(input="What's my name and what do I do?"))


def example_retrieval_qa():
    """Example: Retrieval-augmented QA"""
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    # Initialize retriever
    retriever = MemgraphRetriever(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user",
        k=5
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        return_source_documents=True
    )

    # Ask questions
    result = qa_chain.invoke("What technologies are we using?")
    print("Answer:", result["result"])
    print("Sources:", result["source_documents"])


def example_agent_with_memory():
    """Example: Agent with persistent memory"""
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Initialize memory
    memory = MemgraphMemory(
        api_key=os.getenv("MEMGRAPH_API_KEY"),
        tenant_id=os.getenv("MEMGRAPH_TENANT_ID"),
        user_id="demo_user"
    )

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    llm = ChatOpenAI(model="gpt-4")
    agent = create_openai_functions_agent(llm, [], prompt)
    executor = AgentExecutor(agent=agent, tools=[], memory=memory)

    # Run agent
    result = executor.invoke({"input": "Remember: I prefer PostgreSQL"})
    print(result)


if __name__ == "__main__":
    print("Memgraph LangChain Integration Examples")
    print("=" * 60)
    print("\nRun these examples:")
    print("1. example_basic_usage() - Basic conversation with memory")
    print("2. example_retrieval_qa() - QA with semantic search")
    print("3. example_agent_with_memory() - Agent with persistent memory")
    print("\nMake sure to set environment variables:")
    print("  export MEMGRAPH_API_KEY=vel_your_key")
    print("  export MEMGRAPH_TENANT_ID=your-tenant-id")
