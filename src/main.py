"""
main.py
-------
Interactive command-line interface for the Document Q&A Bot.
Loads the existing vector database from disk (built by ingest.py) and lets
the user ask questions in a loop, printing grounded answers with citations.

Run with:

    python -m src.main
"""

import sys
from src import config
from src.query import query_rag_pipeline


def print_banner():
    print("=" * 60)
    print("  DOCUMENT Q&A BOT (RAG-powered, Gemini + ChromaDB)")
    print("=" * 60)
    print("Ask a question about your documents below.")
    print("Type 'exit' or 'quit' to leave.\n")


def main():
    config.validate_config()

    print_banner()

    # Sanity check: make sure the DB has actually been built.
    try:
        from src.query import get_collection
        collection = get_collection()
        count = collection.count()
        if count == 0:
            print("Warning: the vector database is empty. Run 'python -m src.ingest' first.\n")
    except Exception:
        print("No vector database found. Run 'python -m src.ingest' first to index your documents.\n")
        sys.exit(1)

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = query_rag_pipeline(user_query)
        except Exception as e:
            print(f"\n[Error] Something went wrong while answering: {e}\n")
            continue

        print(f"\nBot: {result['answer']}\n")

        if result["citations"]:
            print("Sources used:")
            for c in sorted(set(result["citations"])):
                print(f"  - {c}")
        print()


if __name__ == "__main__":
    main()
