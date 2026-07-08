"""
rag_engine.py — Core RAG engine for DocuChat (free stack).

Responsibilities:
1. Parse uploaded PDF and .txt files into raw text (via PyMuPDF).
2. Split text into overlapping chunks (via LangChain RecursiveCharacterTextSplitter).
3. Embed chunks locally using sentence-transformers all-MiniLM-L6-v2 — no API key needed.
4. Store and retrieve vectors with FAISS (persisted to disk).
5. Build a grounded prompt (system + history + retrieved context + question).
6. Call Groq's free API (llama-3.3-70b-versatile) and return the answer + source previews.

Free stack changes vs the OpenAI version:
- OpenAIEmbeddings   → HuggingFaceEmbeddings (local sentence-transformers, zero cost)
- openai.OpenAI      → groq.Groq             (free tier, same chat completions API shape)
- tiktoken splitter  → character-based splitter (no tiktoken dependency needed)
- chunk_size unit    → characters (not tokens); 500 chars ≈ 100-120 tokens for English prose

Design notes:
- The HuggingFace model is downloaded once (~90 MB) to ~/.cache/huggingface on first run.
- Groq's API is OpenAI-compatible so the completion call looks almost identical.
- FAISS and all persistence logic are unchanged.
- Temperature stays 0 for deterministic, factual answers.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from groq import Groq
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Iterates every page and concatenates text blocks. PyMuPDF preserves
    reading order better than most alternatives for multi-column layouts.

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename: Original filename, used only for log messages.

    Returns:
        A single string containing all extracted text, pages separated by
        double newlines.

    Raises:
        ValueError: If PyMuPDF cannot open the bytes as a valid PDF.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot parse '{filename}' as a PDF: {exc}") from exc

    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
        else:
            logger.debug(
                "Page %d of '%s' yielded no text (possibly image-only).",
                page_num, filename,
            )

    doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The PDF may be image-only (scanned). OCR is not supported."
        )

    return "\n\n".join(pages)


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    """
    Decode a plain-text file to a string.

    Tries UTF-8 first (the common case), then falls back to latin-1 which
    can decode any byte sequence without errors.

    Args:
        file_bytes: Raw bytes of the text file.
        filename: Original filename, used only for error messages.

    Returns:
        Decoded string content.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode '{filename}' as text.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route a file to the correct text extractor based on its extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename including extension.

    Returns:
        Extracted text string.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if suffix == ".txt":
        return extract_text_from_txt(file_bytes, filename)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Upload PDF or .txt files only."
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, source: str) -> list[Document]:
    """
    Split a long text string into overlapping chunks suitable for embedding.

    Uses character-based RecursiveCharacterTextSplitter (no tiktoken required).
    500 characters ≈ 100-120 tokens for typical English prose, which sits well
    within all-MiniLM-L6-v2's 256-token context window.

    Args:
        text: The full extracted text from one document.
        source: The original filename, stored in each chunk's metadata.

    Returns:
        A list of LangChain Document objects, each with:
            - page_content: The chunk text.
            - metadata["source"]: The original filename.
            - metadata["chunk_index"]: Zero-based position within this file.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]

    logger.info(
        "'%s' → %d chunks (size=%d chars, overlap=%d chars)",
        source, len(documents), settings.chunk_size, settings.chunk_overlap,
    )
    return documents


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """
    Stateful RAG engine that owns the FAISS vector store and LLM client.

    Lifecycle:
        engine = RAGEngine()               # downloads embedding model on first run,
                                           # loads persisted FAISS index if present
        engine.add_documents(...)          # upload → parse → chunk → embed → store
        answer, sources = engine.query(q, history)   # retrieval + Groq generation
        engine.reset()                     # wipe everything

    Attributes:
        _embeddings:    HuggingFaceEmbeddings — local all-MiniLM-L6-v2 model.
        _groq_client:   Groq client for chat completions (free API).
        _vector_store:  LangChain FAISS wrapper, or None before first upload.
        _chunk_count:   Running total of indexed chunks across all uploads.
    """

    def __init__(self) -> None:
        """
        Initialise the engine.

        Downloads the sentence-transformers model on first run (~90 MB, cached
        to ~/.cache/huggingface/). Subsequent startups load from cache instantly.
        Attempts to reload a persisted FAISS index from disk.
        """
        logger.info(
            "Loading local embedding model '%s' (downloads once, then cached)…",
            settings.embedding_model,
        )
        self._embeddings = HuggingFaceInferenceAPIEmbeddings(
		    api_key=settings.hf_token,
    		    model_name=settings.embedding_model,
        )
        logger.info("Embedding model ready.")

        self._groq_client = Groq(api_key=settings.groq_api_key)

        self._vector_store: Optional[FAISS] = None
        self._chunk_count: int = 0

        self._try_load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _try_load_index(self) -> None:
        """
        Attempt to deserialise a previously persisted FAISS index from disk.

        Silently skips if no saved index exists. Logs a warning if the
        directory is present but loading fails (corrupt / incompatible index).
        """
        index_dir = settings.faiss_index_dir
        if not index_dir.exists():
            logger.info("No persisted FAISS index at '%s'. Starting fresh.", index_dir)
            return

        try:
            self._vector_store = FAISS.load_local(
                str(index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._chunk_count = self._vector_store.index.ntotal
            logger.info(
                "Loaded persisted FAISS index from '%s' (%d vectors).",
                index_dir, self._chunk_count,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load FAISS index from '%s': %s. Starting fresh.",
                index_dir, exc,
            )
            self._vector_store = None
            self._chunk_count = 0

    def _save_index(self) -> None:
        """
        Persist the current FAISS index to disk.

        Creates the target directory if needed. Called after every successful
        add_documents() call so the index survives server restarts.
        """
        if self._vector_store is None:
            return
        index_dir = settings.faiss_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(index_dir))
        logger.info(
            "FAISS index saved to '%s' (%d vectors).",
            index_dir, self._chunk_count,
        )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        file_bytes_list: list[bytes],
        filenames: list[str],
    ) -> dict:
        """
        Parse, chunk, embed, and index a batch of uploaded files.

        Embedding runs locally via sentence-transformers — no API call, no cost.

        Args:
            file_bytes_list: Raw bytes of each uploaded file, same order as filenames.
            filenames: Original filenames with extensions.

        Returns:
            A dict with:
                - "chunks_indexed":   Total new chunks added this call.
                - "files_processed":  List of successfully processed filenames.
                - "errors":           List of {"file": ..., "error": ...} dicts for
                                      files that failed (one bad file won't abort all).

        Raises:
            RuntimeError: If FAISS indexing fails unexpectedly.
        """
        all_documents: list[Document] = []
        processed_files: list[str] = []
        errors: list[dict] = []

        for file_bytes, filename in zip(file_bytes_list, filenames):
            try:
                text = extract_text(file_bytes, filename)
                docs = chunk_text(text, source=filename)
                all_documents.extend(docs)
                processed_files.append(filename)
            except ValueError as exc:
                logger.error("Failed to process '%s': %s", filename, exc)
                errors.append({"file": filename, "error": str(exc)})

        if not all_documents:
            return {
                "chunks_indexed": 0,
                "files_processed": processed_files,
                "errors": errors,
            }

        try:
            if self._vector_store is None:
                self._vector_store = FAISS.from_documents(all_documents, self._embeddings)
            else:
                self._vector_store.add_documents(all_documents)
        except Exception as exc:
            logger.exception("FAISS embedding/indexing failed.")
            raise RuntimeError(f"Failed to index documents: {exc}") from exc

        new_chunk_count = len(all_documents)
        self._chunk_count += new_chunk_count
        self._save_index()

        return {
            "chunks_indexed": new_chunk_count,
            "files_processed": processed_files,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Retrieval + Generation
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> tuple[str, list[str]]:
        """
        Run a full RAG cycle: retrieve relevant chunks, build a grounded prompt,
        call Groq, and return the answer with source previews.

        Args:
            question: The user's natural language question.
            conversation_history: List of {"role": ..., "content": ...} dicts
                                  from the session manager (oldest first).

        Returns:
            A 2-tuple:
                - answer (str): The LLM's grounded response.
                - sources (list[str]): 200-char previews of each retrieved chunk.

        Note:
            Returns the no-context reply immediately (no Groq call) when the
            vector store is empty.
        """
        if self._vector_store is None or self._chunk_count == 0:
            return settings.no_context_reply, []

        # --- 1. Retrieve top-K relevant chunks (local FAISS, free) ---
        try:
            retrieved_docs = self._vector_store.similarity_search(
                question,
                k=settings.top_k_results,
            )
        except Exception as exc:
            logger.exception("FAISS similarity_search failed.")
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        if not retrieved_docs:
            return settings.no_context_reply, []

        # --- 2. Build context block ---
        context_parts: list[str] = []
        source_previews: list[str] = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source_name = doc.metadata.get("source", "unknown")
            chunk_content = doc.page_content.strip()
            context_parts.append(f"[Source {i} — {source_name}]\n{chunk_content}")
            preview = chunk_content[:200].replace("\n", " ")
            source_previews.append(f"[{source_name}] {preview}...")

        context_block = "\n\n---\n\n".join(context_parts)

        # --- 3. Assemble messages: system → history → current question+context ---
        messages: list[dict] = [
            {"role": "system", "content": settings.system_prompt},
        ]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": (
                f"Context from uploaded documents:\n\n"
                f"{context_block}\n\n"
                f"---\n\n"
                f"Question: {question}"
            ),
        })

        # --- 4. Call Groq (free API, OpenAI-compatible shape) ---
        try:
            response = self._groq_client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as exc:
            logger.exception("Groq chat completion failed.")
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        answer = response.choices[0].message.content or settings.no_context_reply
        return answer.strip(), source_previews

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Destroy the in-memory FAISS index and delete persisted files from disk.

        The embedding model itself stays loaded in memory (no need to re-download).
        After this call, the engine is ready for a fresh set of documents.
        """
        self._vector_store = None
        self._chunk_count = 0

        index_dir = settings.faiss_index_dir
        if index_dir.exists():
            shutil.rmtree(str(index_dir))
            logger.info("Deleted persisted FAISS index at '%s'.", index_dir)

        logger.info("RAG engine reset complete.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Total number of document chunks currently indexed."""
        return self._chunk_count

    @property
    def is_ready(self) -> bool:
        """True if at least one document has been indexed."""
        return self._vector_store is not None and self._chunk_count > 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
rag_engine = RAGEngine()
