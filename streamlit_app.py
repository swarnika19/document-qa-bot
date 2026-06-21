"""
streamlit_app.py
-----------------
Optional bonus web UI for the Document Q&A Bot, built with Streamlit.
Loads the existing vector database from disk (built by ingest.py) and
provides a simple chat-style interface for asking questions.

Run with:

    streamlit run streamlit_app.py
"""

import streamlit as st
from src import config
from src.query import query_rag_pipeline, get_collection

st.set_page_config(page_title="Document Q&A Bot", page_icon="📚", layout="centered")

st.title("📚 Document Q&A Bot")
st.caption("RAG-powered Q&A over your own documents — Gemini + ChromaDB")

# --- Startup checks -----------------------------------------------------
try:
    config.validate_config()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

try:
    collection = get_collection()
    doc_count = collection.count()
    if doc_count == 0:
        st.warning("Your vector database is empty. Run `python -m src.ingest` first, then refresh this page.")
        st.stop()
    st.sidebar.success(f"Loaded {doc_count} indexed chunk(s) from the vector database.")
except Exception:
    st.error("No vector database found. Run `python -m src.ingest` first to index your documents.")
    st.stop()

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=config.TOP_K)

# --- Chat history --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- {c}")

# --- Chat input -----------------------------------------------------------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = query_rag_pipeline(user_query, k=top_k)
                answer = result["answer"]
                citations = sorted(set(result["citations"]))
            except Exception as e:
                answer = f"Something went wrong: {e}"
                citations = []

        st.markdown(answer)
        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- {c}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
