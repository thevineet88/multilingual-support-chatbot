import json
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Resolve paths relative to this file so it works regardless of cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level cache so the index is built once per process
_vector_store = None
_llm = None


def _load_knowledge_base(path: str = None) -> list[Document]:
    if path is None:
        path = os.path.join(_BASE_DIR, "knowledge_base.json")
    """Read the FAQ JSON and convert each entry into a LangChain Document."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for entry in data:
        content = f"Q: {entry['question']}\nA: {entry['answer']}"
        documents.append(Document(page_content=content))
    return documents


def _get_vector_store() -> FAISS:
    """Build (or return cached) FAISS index from the knowledge base."""
    global _vector_store
    if _vector_store is None:
        docs = _load_knowledge_base()
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        _vector_store = FAISS.from_documents(docs, embeddings)
    return _vector_store


def _get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    return _llm


SYSTEM_PROMPT = """You are a helpful, polite multilingual customer support assistant for an e-commerce store.

Rules:
1. Answer ONLY based on the provided context below. Do not use outside knowledge.
2. If the answer is not found in the context, respond with: "I'm sorry, I don't have information about that. Let me connect you to a human agent for further assistance."
3. Respond in the SAME language the customer wrote in. If they wrote in Hindi (transliterated), reply in Hindi (transliterated). If English, reply in English.
4. Be concise, friendly, and professional.

Context:
{context}
"""


def get_rag_response(user_query: str) -> str:
    """Retrieve relevant FAQ entries and generate an LLM answer.
    Returns the assistant's response string.
    """
    try:
        store = _get_vector_store()
        # Retrieve top 3 most relevant documents
        docs = store.similarity_search(user_query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=user_query),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"I'm experiencing a technical issue right now. Please try again shortly. (Error: {e})"
