"""
query.py
--------
Query pipeline: loads the pre-existing ChromaDB vector store from disk,
embeds the user's question with the same embedding model used during
ingestion, retrieves the top-k most relevant chunks, builds a strictly
grounded prompt, and calls Gemini to generate a cited answer.

This module makes no fresh embedding calls for the source documents —
those were already computed once by ingest.py.
"""

import google.generativeai as genai
import chromadb

from src import config
from src.embeddings import GeminiEmbeddingFunction

# Configure the Gemini client once at import time.
genai.configure(api_key=config.GEMINI_API_KEY)

# System prompt enforcing strict grounding to prevent hallucinations.
SYSTEM_PROMPT = (
    "You are a professional, accurate document Q&A assistant. "
    "Answer the user's question using ONLY the provided document context below. "
    "Cite the sources (filenames and pages) inline next to the facts you state, "
    "in the form (filename, Page X). "
    "If the answer cannot be found in the context, clearly state: "
    "'I am sorry, but the provided documents do not contain the answer to your question.' "
    "Do not make up facts or use any external knowledge."
)


def get_collection(db_path: str = config.DB_DIR):
    """Loads the persisted ChromaDB collection from disk."""
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = GeminiEmbeddingFunction(
        api_key=config.GEMINI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )
    return client.get_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def retrieve_context(user_query: str, k: int = config.TOP_K, db_path: str = config.DB_DIR) -> dict:
    """Embeds the query and retrieves the top-k closest chunks from the vector store."""
    collection = get_collection(db_path)
    results = collection.query(
        query_texts=[user_query],
        n_results=k,
    )
    return results


def build_grounded_prompt(user_query: str, results: dict) -> tuple[str, list[str]]:
    """Formats retrieved chunks into a citation-labeled context block and full prompt."""
    context_blocks = []
    citations = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        source_name = meta.get("source", "unknown")
        page_num = meta.get("page", "?")
        citation_str = f"Source: {source_name}, Page: {page_num}"

        context_blocks.append(f"[{citation_str}]\nContext: {doc}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    return prompt, citations


def query_rag_pipeline(user_query: str, db_path: str = config.DB_DIR, k: int = config.TOP_K) -> dict:
    """
    Full RAG query flow: retrieve relevant chunks, build a grounded prompt,
    and call Gemini to produce a cited answer.
    """
    results = retrieve_context(user_query, k=k, db_path=db_path)
    prompt, citations = build_grounded_prompt(user_query, results)

    model = genai.GenerativeModel(config.GENERATION_MODEL)
    response = model.generate_content(prompt)

    documents = results.get("documents", [[]])[0]

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": documents,
    }