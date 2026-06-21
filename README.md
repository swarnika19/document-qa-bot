# Document Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about
your own documents (PDF, DOCX, TXT) using Google Gemini for generation and
embeddings, with ChromaDB as a local, persistent vector store.

The bot only answers from the content of your documents — if the answer
isn't in there, it says so instead of guessing, and every fact in its
answer is tagged with the source filename and page number it came from.

## How it works

```
Documents (PDF/DOCX/TXT)
        |
        v
  Text Extraction  ->  Chunking  ->  Embeddings  ->  ChromaDB (on disk)
                                                          |
User question  ->  Embed query  ->  Similarity search  --+
                                          |
                                          v
                            Top-k relevant chunks + citations
                                          |
                                          v
                        Grounded prompt  ->  Gemini  ->  Cited answer
```

## Project Structure

```
document-qa-bot/
├── .env.example          # Copy to .env and add your Gemini API key
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py       # Bonus web UI
├── data/                   # Put your PDF / DOCX / TXT files here
│   └── sample_factsheet.txt   # A sample doc so you can test immediately
├── db/                     # Persistent ChromaDB storage (auto-generated)
└── src/
    ├── __init__.py
    ├── config.py            # Constants & configuration
    ├── ingest.py             # Extracts, chunks, embeds, and stores docs
    ├── query.py               # Retrieval + grounded answer generation
    └── main.py                 # Interactive CLI chat loop
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Get a free key at https://aistudio.google.com/app/apikey, then:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
GEMINI_API_KEY=your_actual_key_here
```

### 4. Add your documents

Drop your `.pdf`, `.docx`, or `.txt` files into the `data/` folder. A sample
file (`sample_factsheet.txt`) is already included so you can test the
pipeline immediately without adding your own files first.

### 5. Run ingestion (builds the vector database)

```bash
python -m src.ingest
```

This reads every file in `data/`, splits the text into overlapping chunks,
embeds them with Gemini's `text-embedding-004` model, and saves everything
to the local `db/` folder. You only need to re-run this when you add or
change documents.

### 6. Ask questions

**Command line:**

```bash
python -m src.main
```

**Or the bonus Streamlit web UI:**

```bash
streamlit run streamlit_app.py
```

Try asking (against the included sample document):
- "What was Aether Robotics' net revenue in fiscal year 2025?"
- "Who is the CEO of the company?"
- "What are the company's expansion plans?"

## Configuration

Tunable values live in `src/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K` | 4 | Number of chunks retrieved per question |
| `GENERATION_MODEL` | `gemini-2.5-flash-preview-09-2025` | LLM used for answers |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |

## Design Notes

- **Why chunk with overlap?** If a key fact sits right at a chunk boundary,
  overlap ensures it still appears whole in at least one chunk, so it isn't
  lost during retrieval.
- **Why a separate ingest.py and query.py?** Embedding costs API calls and
  time. Splitting ingestion from querying means the vector database is built
  once and reused instantly on every later question.
- **Why strict grounding in the prompt?** The system prompt explicitly
  forbids the model from using outside knowledge, which is the main defense
  against hallucinated answers in a RAG pipeline.

## Troubleshooting

- **"GEMINI_API_KEY is not set"** — make sure you copied `.env.example` to
  `.env` and filled in a real key.
- **"No vector database found"** — run `python -m src.ingest` before
  `python -m src.main` or the Streamlit app.
- **Empty/garbled PDF text** — some PDFs are scanned images rather than
  real text; `pypdf` can only extract text that is actually embedded in the
  file, not text inside a scanned image.
