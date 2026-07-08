"""
main.py — FastAPI application entry point for DocuChat.

Endpoints:
    POST   /upload   — Ingest one or more PDF/.txt files into the vector store.
    POST   /chat     — Ask a question; get a grounded answer + source previews.
    DELETE /reset    — Wipe the vector store and all session histories.
    GET    /health   — Liveness + readiness check.
    GET    /         — Serve the frontend index.html.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import settings
from rag_engine import rag_engine
from session_manager import session_manager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocuChat",
    description="RAG chatbot — ask questions grounded in your uploaded documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the frontend (served on the same origin in production, but any origin
# during local dev) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the frontend HTML file
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question in natural language.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Opaque session identifier for conversation continuity.",
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str = Field(..., description="LLM-generated answer grounded in the documents.")
    sources: list[str] = Field(
        default_factory=list,
        description="Short previews of the retrieved document chunks used to generate the answer.",
    )


class UploadResponse(BaseModel):
    """Response body for POST /upload."""

    status: str
    chunks_indexed: int
    files_processed: list[str]
    errors: list[dict] = Field(default_factory=list)


class ResetResponse(BaseModel):
    """Response body for DELETE /reset."""

    status: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    indexed_chunks: int
    sessions_active: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_documents_loaded() -> None:
    """
    Raise HTTP 400 if no documents have been indexed yet.

    This prevents the chat endpoint from calling the LLM when there is
    nothing in the vector store to ground the answer.
    """
    if not rag_engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No documents have been uploaded yet. "
                "Please POST one or more files to /upload first."
            ),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """
    Serve the single-page frontend from frontend/index.html.

    This lets you access the full UI by opening http://localhost:8000/ in a
    browser without running a separate web server.
    """
    if not FRONTEND_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frontend/index.html not found. Make sure the frontend/ folder exists.",
        )
    return FileResponse(str(FRONTEND_PATH), media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Liveness and readiness check.

    Returns the number of indexed chunks so monitoring tools can verify
    that documents were successfully ingested.
    """
    return HealthResponse(
        status="ok",
        indexed_chunks=rag_engine.chunk_count,
        sessions_active=session_manager.session_count,
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents"],
    summary="Upload and index PDF or .txt files",
)
async def upload_files(
    files: list[UploadFile] = File(..., description="One or more PDF or .txt files"),
) -> UploadResponse:
    """
    Parse, chunk, embed, and index one or more uploaded files.

    - Accepts PDF and .txt files (other types are rejected per-file).
    - Chunks are sized by token count (chunk_size / chunk_overlap from config).
    - Embeddings are generated with OpenAI text-embedding-3-small.
    - The FAISS index is saved to disk after each successful upload so it
      survives server restarts.

    Returns the number of new chunks added and the list of processed filenames.
    If some files fail (e.g. image-only PDFs) they are reported in `errors`
    but do not prevent other files from being processed.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files received. Attach at least one PDF or .txt file.",
        )

    # Validate extensions before reading bytes
    allowed_extensions = {".pdf", ".txt"}
    invalid = [
        f.filename for f in files
        if Path(f.filename or "").suffix.lower() not in allowed_extensions
    ]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type(s): {invalid}. Only PDF and .txt are accepted.",
        )

    t0 = time.perf_counter()

    # Read all file bytes (done async in the request handler)
    file_bytes_list: list[bytes] = []
    filenames: list[str] = []
    for upload in files:
        content = await upload.read()
        file_bytes_list.append(content)
        filenames.append(upload.filename or "unknown")
        logger.info("Received file '%s' (%d bytes)", upload.filename, len(content))

    # Heavy lifting (CPU/IO bound) — runs synchronously; consider BackgroundTasks
    # for very large files in a production setting.
    try:
        result = rag_engine.add_documents(file_bytes_list, filenames)
    except RuntimeError as exc:
        logger.exception("add_documents raised RuntimeError")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "Upload complete in %.2fs — %d chunks indexed from %s",
        elapsed,
        result["chunks_indexed"],
        result["files_processed"],
    )

    return UploadResponse(
        status="success",
        chunks_indexed=result["chunks_indexed"],
        files_processed=result["files_processed"],
        errors=result.get("errors", []),
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question grounded in uploaded documents",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Retrieve the most relevant document chunks and generate a grounded answer.

    The answer is produced exclusively from the retrieved context. If the
    relevant information is absent from the indexed documents, the LLM returns
    a standard "I couldn't find..." message rather than hallucinating.

    Conversation history (last N turns) is injected into the prompt to support
    follow-up questions within the same session.
    """
    _assert_documents_loaded()

    # Fetch prior conversation history for this session
    history = session_manager.get_history(request.session_id)

    try:
        answer, sources = rag_engine.query(
            question=request.question,
            conversation_history=history,
        )
    except RuntimeError as exc:
        logger.exception("RAG query failed for session '%s'", request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Persist this exchange to the session
    session_manager.add_turn(
        session_id=request.session_id,
        user_message=request.question,
        assistant_message=answer,
    )

    return ChatResponse(answer=answer, sources=sources)


@app.delete(
    "/reset",
    response_model=ResetResponse,
    tags=["System"],
    summary="Clear the vector store and all session histories",
)
async def reset() -> ResetResponse:
    """
    Wipe the FAISS index (in-memory + on-disk) and all session histories.

    Use this to start fresh with a new set of documents. After calling this
    endpoint, /upload must be called again before /chat will work.
    """
    rag_engine.reset()
    session_manager.clear_all()
    logger.info("Full reset performed — index and sessions cleared.")
    return ResetResponse(status="cleared")


# ---------------------------------------------------------------------------
# Global exception handler — ensures all unhandled errors return JSON
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler that converts unexpected exceptions into a structured
    JSON error response instead of an HTML traceback.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Check server logs for details."},
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
