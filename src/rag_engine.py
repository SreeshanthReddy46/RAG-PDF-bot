"""Conversational retrieval augmented generation chain."""

from __future__ import annotations

from threading import Lock
from typing import Any

from src.config import settings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory
    try:
        # LangChain v1 moved the legacy chain helpers into langchain-classic.
        from langchain_classic.chains import create_history_aware_retriever
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        # Keep compatibility with LangChain 0.2/0.3 installations.
        from langchain.chains import create_history_aware_retriever
        from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as exc:
    # Preserve the import failure so application entry points can import without
    # the optional RAG dependencies installed, while reporting it on use.
    _DEPENDENCY_ERROR: ImportError | None = exc
else:
    _DEPENDENCY_ERROR = None

session_histories: dict[str, Any] = {}
_history_lock = Lock()
_chain: Any | None = None
_chain_lock = Lock()


def get_session_history(session_id: str) -> Any:
    """Return stable, thread-safe in-memory chat history for a session."""
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError("Install the project's LangChain dependencies to use chat.") from _DEPENDENCY_ERROR
    normalized_id = str(session_id).strip()
    if not normalized_id:
        normalized_id = "default"
    with _history_lock:
        history = session_histories.get(normalized_id)
        if history is None:
            history = InMemoryChatMessageHistory()
            session_histories[normalized_id] = history
        return history


def _build_chain() -> Any:
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError("Install the project's LangChain dependencies to use chat.") from _DEPENDENCY_ERROR
    if not settings.GOOGLE_API_KEY.strip():
        raise RuntimeError("GOOGLE_API_KEY is not configured. Set it in the environment or .env file.")
    if settings.TOP_K_RETRIEVAL < 1:
        raise RuntimeError("TOP_K_RETRIEVAL must be a positive integer.")

    try:
        from src.ingestion import get_vector_store
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are incomplete. Install the packages in requirements.txt."
        ) from exc

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RETRIEVAL},
    )
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.2,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question which might reference context "
                "in the chat history, formulate a standalone question which can be understood "
                "without the chat history. Do not answer the question; reformulate it only if needed.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document assistant. Answer using only the retrieved context. "
                "If the context does not contain the answer, say that the documents do not "
                "provide this information.\n\nContext:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Keep the history and original input alongside the retrieved documents.
    # create_retrieval_chain only forwards the retriever's context and input,
    # while this prompt also needs chat_history.
    retrieval_chain = (
        RunnablePassthrough.assign(context=history_aware_retriever)
        | answer_chain
    )
    return RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def build_conversational_rag_chain() -> Any:
    """Build (once) and return the conversational RAG chain.

    Expensive setup is deferred until explicitly requested, allowing API and UI
    modules to import even when credentials or optional dependencies are absent.
    """
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                _chain = _build_chain()
    return _chain


def reset_conversational_rag_chain() -> None:
    """Clear the cached chain; useful after configuration changes and in tests."""
    global _chain
    with _chain_lock:
        _chain = None




