# DocuChat — AI Chatbot for Your Documents (Free Stack)

> **RAG-powered document assistant — FastAPI · LangChain · Groq · FAISS · sentence-transformers**
>
> Upload PDFs or text files. Ask questions. Get answers grounded strictly in your documents.
> **$0/month to run.** No OpenAI billing. No credit card required.

---

## Free Stack at a Glance

| Component | Free Option | Notes |
|-----------|------------|-------|
| LLM | Groq — `llama-3.3-70b-versatile` | Free tier, no credit card |
| Embeddings | `all-MiniLM-L6-v2` (local) | Runs on CPU, ~90 MB, downloaded once |
| Vector DB | FAISS (local) | Persisted to disk |
| Framework | FastAPI + LangChain | Open source |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Get Your Free Groq API Key](#get-your-free-groq-api-key)
5. [Configuration](#configuration)
6. [Running the App](#running-the-app)
7. [API Reference + curl Examples](#api-reference--curl-examples)
8. [How It Works](#how-it-works)
9. [Best Document Types for Demos](#best-document-types-for-demos)
10. [Fiverr Demo Video Script](#fiverr-demo-video-script)
11. [Project Structure](#project-structure)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
User Browser
    │
    ▼
FastAPI (main.py)
    │
    ├── POST /upload ──► RAGEngine.add_documents()
    │                       ├── PyMuPDF (PDF parsing)
    │                       ├── RecursiveCharacterTextSplitter (chunking)
    │                       ├── all-MiniLM-L6-v2 via sentence-transformers (FREE, local)
    │                       └── FAISS (vector store, persisted to disk)
    │
    ├── POST /chat ───► RAGEngine.query()
    │                       ├── FAISS similarity_search (top-4 chunks, local)
    │                       ├── SessionManager (conversation history)
    │                       └── Groq llama-3.3-70b-versatile (FREE API)
    │
    ├── DELETE /reset ► Wipes FAISS index + all sessions
    └── GET /health  ► Liveness check with chunk count
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | 23+ |
| Groq account | Free — no credit card |
| RAM | 2 GB+ (embedding model loads into memory) |

---

## Installation

```bash
# 1. Clone or download the project
git clone https://github.com/yourname/docuchat.git
cd docuchat

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first `pip install` pulls `sentence-transformers` (~500 MB with PyTorch).
> The actual embedding model (~90 MB) downloads on first server startup.

---

## Get Your Free Groq API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up — no credit card required
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

Groq's free tier gives you generous rate limits on Llama 3.3 70B — more than enough for demos and portfolio use.

---

## Configuration

```bash
cp .env.example .env
# Open .env and paste your Groq API key
```

Your `.env` should look like:

```
GROQ_API_KEY=gsk_...your-real-key-here...
```

Everything else has sensible defaults. See `.env.example` for all options.

---

## Running the App

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**First startup** will print something like:
```
Loading local embedding model 'all-MiniLM-L6-v2' (downloads once, then cached)…
Downloading: 100%|████████████| 90.9M/90.9M
Embedding model ready.
```

Then open **http://localhost:8000** in your browser. Swagger docs at **/docs**.

Subsequent startups load the model from cache in ~2 seconds.

---

## API Reference + curl Examples

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "indexed_chunks": 0, "sessions_active": 0 }
```

---

### `POST /upload`

```bash
# Single file
curl -X POST http://localhost:8000/upload \
  -F "files=@/path/to/document.pdf"

# Multiple files
curl -X POST http://localhost:8000/upload \
  -F "files=@report.pdf" \
  -F "files=@notes.txt"
```

```json
{
  "status": "success",
  "chunks_indexed": 147,
  "files_processed": ["report.pdf", "notes.txt"],
  "errors": []
}
```

---

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "session_id": "demo-1"}'
```

```json
{
  "answer": "According to the document, the key findings include...",
  "sources": ["[report.pdf] The key findings show a 23% increase in..."]
}
```

**Follow-up (same session_id keeps conversation context):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you expand on the second point?", "session_id": "demo-1"}'
```

---

### `DELETE /reset`

```bash
curl -X DELETE http://localhost:8000/reset
```

```json
{ "status": "cleared" }
```

---

## How It Works

### 1. Document Ingestion
- **PDF**: PyMuPDF extracts text page-by-page, handling multi-column layouts and tables.
- **TXT**: UTF-8 / latin-1 decoded automatically.
- **Chunking**: `RecursiveCharacterTextSplitter` splits into 500-character chunks with 50-character overlap. Overlap prevents answers that span a chunk boundary from being missed.

### 2. Embedding (Free, Local)
- Each chunk is embedded with `all-MiniLM-L6-v2` via `sentence-transformers`.
- Produces 384-dimensional vectors. Runs on CPU — no GPU needed.
- Zero API calls, zero cost, works offline after first download.
- Vectors are stored in FAISS and saved to `faiss_index/` on disk.

### 3. Retrieval
- At query time the question is embedded with the same local model.
- FAISS returns the 4 most similar chunks by cosine similarity.
- Chunks are formatted into a numbered context block with source filenames.

### 4. Generation (Free Groq API)
- System prompt enforces document-only answers.
- Prompt: `[system] → [conversation history] → [context + question]`.
- Groq runs `llama-3.3-70b-versatile` at extremely low latency (~1-2s responses).
- Temperature 0 = deterministic, auditable answers.

### 5. Session History
- Each `session_id` keeps a rolling window of the last 6 turns.
- History is injected before the current question so follow-ups work naturally.

---

## Best Document Types for Demos

### 1. Company Annual Report (PDF)
Dense financial data, multiple sections — great for showing precision.

**Example questions:**
- "What was the net revenue in 2023?"
- "How many employees does the company have?"
- "What are the three biggest risk factors mentioned?"

**Source:** Any public company's investor relations page (Apple, Tesla, NVIDIA).

---

### 2. Research Paper (PDF)
Specific claims, methodology, citations — demonstrates accuracy on technical content.

**Example questions:**
- "What methodology did the authors use?"
- "What were the limitations of the study?"
- "Summarize the key findings in plain English."

**Source:** [arXiv.org](https://arxiv.org) — free, thousands of papers.

---

### 3. Product Manual / Technical Documentation
Immediately shows the commercial use case to potential clients.

**Example questions:**
- "How do I reset the device to factory settings?"
- "What accessories are compatible with this product?"
- "What safety warnings are mentioned?"

**Source:** Any manufacturer's website.

---

## Fiverr Demo Video Script

**Duration:** 2–3 minutes. Record screen with voiceover.

---

**[0:00 – 0:15] Hook**
> "What if you could chat with any PDF — a 200-page report, a legal contract, a technical manual — and get instant, accurate answers? That's what I built. This is DocuChat, and it runs completely free."

**[0:15 – 0:45] Show the upload**
> "I'll upload an annual report PDF. Watch how fast it indexes."
> [Drag PDF onto the drop zone. Show the progress pill: '147 chunks indexed'.]
> "The document is parsed, split into chunks, and turned into vectors using a local AI model — all on my own machine, no paid API."

**[0:45 – 1:30] Ask questions**
> [Type: "What was the total revenue in 2023?"]
> [Read the answer, point to the source chips.]
> "Notice the sources — it shows you exactly which part of the document it used."
>
> [Type a question the doc doesn't cover.]
> "If the answer isn't in the document, it says so. No hallucinations."

**[1:30 – 1:50] Follow-up questions**
> [Type: "Tell me more about that."]
> "It remembers the conversation — follow-up questions work naturally."

**[1:50 – 2:20] Multi-document**
> [Upload a second PDF.]
> "You can upload multiple documents and query across all of them at once."

**[2:20 – 2:50] Tech credibility**
> "Under the hood: FastAPI backend, LangChain orchestration, Groq's free Llama 3.3 API for answers, local sentence-transformers for embeddings, FAISS for vector search. The index saves to disk — documents survive server restarts."

**[2:50 – 3:00] Close**
> "I can build this for you — customised with your branding, your document types, your workflow. Check the packages below."

---

## Project Structure

```
DocuChat/
├── main.py              # FastAPI app + all 5 endpoints
├── rag_engine.py        # PDF parsing, chunking, FAISS, Groq call
├── session_manager.py   # Conversation history per session_id
├── config.py            # Settings from .env (Groq key, model names, etc.)
├── requirements.txt     # Pinned dependencies
├── .env.example         # Env var template
├── faiss_index/         # FAISS persistence (auto-created on first upload)
│   ├── index.faiss
│   └── index.pkl
├── frontend/
│   └── index.html       # Complete chat UI (served at GET /)
└── README.md
```

---

## Troubleshooting

**`GROQ_API_KEY is not set`**
→ Copy `.env.example` to `.env` and paste your key from [console.groq.com](https://console.groq.com).

**Slow first startup**
→ The embedding model downloads once (~90 MB). After that, startup takes ~2 seconds.

**`No extractable text found in 'file.pdf'`**
→ The PDF is scanned (image-only). Use a PDF with a real text layer.

**`No documents have been uploaded yet` on /chat**
→ POST to `/upload` first. The chat endpoint requires an indexed document.

**Groq `429 Rate Limited`**
→ Groq's free tier is generous but has per-minute limits. Wait a few seconds and retry.

**Port 8000 in use**
```bash
uvicorn main:app --reload --port 8001
# Update API_BASE in frontend/index.html to 'http://localhost:8001'
```

**Out of memory on embedding**
→ `all-MiniLM-L6-v2` needs ~500 MB RAM. Close other apps or upgrade to a machine with more RAM.
