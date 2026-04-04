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
        api_key="mg_your_key",
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
            api_key="mg_your_key",
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
        user_id: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        base_url: str = None
    ):
        """
        Initialize Memgraph memory.

        Args:
            api_key: Memgraph API key (starts with 'mg_')
            user_id: User identifier
            tenant_id: Tenant identifier (optional — resolved from API key if omitted)
            session_id: Optional session identifier (defaults to user_id)
            base_url: Memgraph API URL (defaults to MEMGRAPH_API_URL env or cloud)
        """
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
    LangChain-compatible retriever for semantic search in Memgraph AI.

    Example:
        retriever = MemgraphRetriever(
            api_key="mg_your_key",
            user_id="user123",
        )
        docs = retriever.invoke("What databases are we using?")
    """

    # Pydantic v2 fields (required by LangChain BaseRetriever)
    client: Any = None
    user_id: str = ""
    k: int = 5
    search_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        api_key: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        k: int = 5,
        base_url: str = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            client=MemgraphClient(api_key=api_key, tenant_id=tenant_id, base_url=base_url),
            user_id=user_id,
            k=k,
            search_kwargs=search_kwargs or {},
            **kwargs,
        )

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
            # Search Memgraph — returns {results: [{content, score, metadata}]}
            result = self.client.search(
                query=query,
                user_id=self.user_id,
                limit=self.k,
            )

            documents = []

            for item in result.get("results", []):
                content = item.get("content", "")
                if content:
                    metadata = {
                        "source": "memgraph",
                        "score": item.get("score", 0),
                        "user_id": self.user_id,
                    }
                    metadata.update(item.get("metadata", {}))
                    documents.append(Document(page_content=content, metadata=metadata))

            # Legacy: also check beliefs/memories format for backward compat
            for belief in result.get("beliefs", [])[:self.k]:
                if isinstance(belief, dict):
                    content = f"{belief.get('key', '')}: {belief.get('value', '')}"
                    metadata = {
                        "source": "belief",
                        "confidence": belief.get("confidence_score", 1.0),
                        "user_id": self.user_id
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            for memory in result.get("memories", [])[:self.k]:
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

            return documents[:self.k]

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


# ============================================================================
# NOTE: MemgraphConversationMemory removed — BaseMemory was removed from
# LangChain in v1.x. Use MemgraphMemory (BaseChatMessageHistory) instead,
# which works with RunnableWithMessageHistory.
# ============================================================================

class _MemgraphConversationMemoryLegacy:
    """
    LangChain conversation memory backed by Memgraph OS.

    This memory class stores and retrieves conversation context for chains and agents.

    Example:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        memory = MemgraphConversationMemory(
            api_key="mg_your_key",
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
        user_id: str,
        tenant_id: Optional[str] = None,
        memory_key: str = "history",
        base_url: str = None
    ):
        """
        Initialize conversation memory.

        Args:
            api_key: Memgraph API key
            user_id: User identifier
            tenant_id: Tenant identifier (optional — resolved from API key if omitted)
            memory_key: Key to use for storing memory in chain context
            base_url: Memgraph API URL (defaults to MEMGRAPH_API_URL env or cloud)
        """
        super().__init__()
        self.chat_memory = MemgraphMemory(
            api_key=api_key,
            user_id=user_id,
            tenant_id=tenant_id,
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

class MemgraphToolkit:
    """LangChain toolkit providing search and remember as function-calling tools.

    Compatible with create_openai_functions_agent and similar agents that use
    structured tool definitions.

    Example::

        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_functions_agent

        toolkit = MemgraphToolkit(api_key="mg_...", tenant_id="...", user_id="u1")
        tools = toolkit.get_tools()
        agent = create_openai_functions_agent(ChatOpenAI(model="gpt-4o"), tools, prompt)
    """

    def __init__(self, api_key: str, user_id: str, tenant_id: str = None, base_url: str = None):
        self.client = MemgraphClient(api_key=api_key, tenant_id=tenant_id, base_url=base_url)
        self.user_id = user_id

    def _search(self, query: str) -> str:
        """Search Memgraph for relevant memories and past context."""
        result = self.client.search(query=query, user_id=self.user_id)
        items = result.get("retrieved_items", result.get("beliefs", []))
        if not items:
            return "No relevant memories found."
        lines = []
        for it in items[:10]:
            if isinstance(it, dict):
                lines.append(f"- {it.get('content', it.get('value', it.get('text', str(it))))}")
            else:
                lines.append(f"- {it}")
        return "\n".join(lines)

    def _remember(self, text: str, category: str = "general") -> str:
        """Store a fact or decision in Memgraph for future recall."""
        self.client.remember(text, user_id=self.user_id, category=category)
        return f"Remembered: {text[:100]}"

    def get_tools(self):
        """Return LangChain StructuredTool instances."""
        from langchain.tools import StructuredTool
        from pydantic import BaseModel as LCBaseModel, Field as LCField

        class SearchInput(LCBaseModel):
            query: str = LCField(description="The search query to find relevant memories")

        class RememberInput(LCBaseModel):
            text: str = LCField(description="The text/fact to remember")
            category: str = LCField(default="general", description="Category: decision, architecture, bug_fix, preference, general")

        return [
            StructuredTool.from_function(
                func=self._search,
                name="memgraph_search",
                description="Search persistent memory for relevant context, past decisions, and knowledge.",
                args_schema=SearchInput,
            ),
            StructuredTool.from_function(
                func=self._remember,
                name="memgraph_remember",
                description="Store a new fact or decision in persistent memory for future recall.",
                args_schema=RememberInput,
            ),
        ]


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
    print("  export MEMGRAPH_API_KEY=mg_your_key")
