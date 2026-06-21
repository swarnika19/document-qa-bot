"""
ingest.py
---------
Pipeline script: scans the data/ folder, extracts text from PDFs, DOCX,
and TXT files, splits it into overlapping chunks, embeds those chunks with
Gemini's text-embedding-004 model, and persists everything into a local
ChromaDB vector store on disk.

Run this once whenever you add/change documents in data/:

    python -m src.ingest
"""

import os
import sys
import chromadb
from tqdm import tqdm
from pypdf import PdfReader
from docx import Document as DocxDocument

from src import config
from src.embeddings import GeminiEmbeddingFunction


# ---------------------------------------------------------------------------
# Step 2: Document Ingestion & Text Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extracts text page-by-page from a PDF, tracking page numbers and source file."""
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1,
                    }
                })
    except Exception as e:
        print(f"  ! Error reading PDF {file_name}: {e}")

    return extracted_data


def extract_docx_pages(file_path: str) -> list[dict]:
    """
    Extracts text from a Word document. DOCX has no native "page" concept,
    so we treat the whole document as a single logical unit (page 1) and
    join all non-empty paragraphs together.
    """
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        doc = DocxDocument(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = " ".join(paragraphs)
        if full_text:
            extracted_data.append({
                "text": full_text,
                "metadata": {
                    "source": file_name,
                    "page": 1,
                }
            })
    except Exception as e:
        print(f"  ! Error reading DOCX {file_name}: {e}")

    return extracted_data


def extract_txt_pages(file_path: str) -> list[dict]:
    """Extracts text from a plain .txt file."""
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        clean_text = " ".join(text.split())
        if clean_text:
            extracted_data.append({
                "text": clean_text,
                "metadata": {
                    "source": file_name,
                    "page": 1,
                }
            })
    except Exception as e:
        print(f"  ! Error reading TXT {file_name}: {e}")

    return extracted_data


def load_all_documents(data_dir: str) -> list[dict]:
    """Scans data_dir for supported files and extracts page-level text from each."""
    all_pages = []

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return all_pages

    file_names = sorted(os.listdir(data_dir))
    supported_files = [f for f in file_names if f.lower().endswith((".pdf", ".docx", ".txt"))]

    if not supported_files:
        print(f"No supported files (.pdf, .docx, .txt) found in {data_dir}")
        return all_pages

    print(f"Found {len(supported_files)} document(s) to ingest:")
    for file_name in supported_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"  - {file_name}")

        if file_name.lower().endswith(".pdf"):
            pages = extract_pdf_pages(file_path)
        elif file_name.lower().endswith(".docx"):
            pages = extract_docx_pages(file_path)
        else:
            pages = extract_txt_pages(file_path)

        all_pages.extend(pages)

    return all_pages


# ---------------------------------------------------------------------------
# Step 3: Text Chunking Strategy (Recursive-style character splitting)
# ---------------------------------------------------------------------------

def chunk_extracted_pages(
    pages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Splits page-level documents into smaller, overlapping chunks.
    Source metadata (file name, page number) is carried over to every chunk
    so answers can always be traced back to where they came from.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        start = 0
        text_length = len(text)

        if text_length == 0:
            continue

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}",
                }
            })

            # Slide window forward; guard against chunk_overlap >= chunk_size
            step = max(chunk_size - chunk_overlap, 1)
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Step 4: Persisting the Vector Database
# ---------------------------------------------------------------------------

def save_to_vector_db(chunks: list[dict], db_path: str = config.DB_DIR):
    """Embeds text chunks (via Gemini) and saves them into a persistent ChromaDB."""
    if not chunks:
        print("No chunks to index. Aborting.")
        return

    client = chromadb.PersistentClient(path=db_path)

    embedding_fn = GeminiEmbeddingFunction(
        api_key=config.GEMINI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )

    # Start fresh each time ingest.py runs, so stale chunks never linger.
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Batch in groups to keep individual API calls reasonably sized.
    batch_size = 50
    print(f"\nEmbedding and indexing {len(chunks)} chunks...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"\nSuccessfully indexed {len(chunks)} chunks into '{config.COLLECTION_NAME}' at {db_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config.validate_config()

    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    pages = load_all_documents(config.DATA_DIR)
    if not pages:
        print("\nNo text extracted. Add .pdf, .docx, or .txt files to the data/ folder and re-run.")
        sys.exit(1)

    print(f"\nExtracted {len(pages)} page-level text block(s).")

    chunks = chunk_extracted_pages(pages)
    print(f"Generated {len(chunks)} chunk(s) (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    save_to_vector_db(chunks)

    print("\nDone! You can now run: python -m src.main")


if __name__ == "__main__":
    main()