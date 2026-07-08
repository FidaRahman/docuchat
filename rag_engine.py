"""
rag_engine.py - Core RAG engine for DocuChat.
Uses HuggingFace Inference API for embeddings (no local model, fits in 512MB RAM).
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import fitz
from groq import Groq
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot parse '{filename}' as a PDF: {exc}") from exc

    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
    doc.close()

    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'.")

    return "\n\n".join(pages)


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    """Decode a plain text file."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode '{filename}' as text.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route file to correct extractor based on extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if suffix == ".txt":
        return extract_text_from_txt(file_bytes, filename)
    raise ValueError(f"Unsupported file type '{suffix}'.")


def chunk_text(text: str, source: str) -> list[Document]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]


class RAGEngine:
    """RAG engine using HuggingFace Inference API embeddings and Groq LLM."""

    def __init__(self) -> None:
        logger.info("Initialising embeddings via HuggingFace Inference API...")
        self._embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.hf_token,
            model_name=settings.embedding_model,
        )
        self._groq_client = Groq(api_key=settings.groq_api_key)
        self._vector_store: Optional[FAISS] = None
        self._chunk_count: int = 0
        self._try_load_index()

    def _try_load_index(self) -> None:
        """Load persisted FAISS index from disk if it exists."""
        index_dir = settings.faiss_index_dir
        if not index_dir.exists():
            return
        try:
            self._vector_store = FAISS.load_local(
                str(index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._chunk_count = self._vector_store.index.ntotal
            logger.info("Loaded FAISS index with %d vectors.", self._chunk_count)
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            self._vector_store = None
            self._chunk_count = 0

    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        if self._vector_store is None:
            return
        index_dir = settings.faiss_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(index_dir))

    def add_documents(self, file_bytes_list: list[bytes], filenames: list[str]) -> dict:
        """Parse, chunk, embed and index uploaded files."""
        all_documents = []
        processed_files = []
        errors = []

        for file_bytes, filename in zip(file_bytes_list, filenames):
            try:
                text = extract_text(file_bytes, filename)
                docs = chunk_text(text, source=filename)
                all_documents.extend(docs)
                processed_files.append(filename)
            except ValueError as exc:
                errors.append({"file": filename, "error": str(exc)})

        if not all_documents:
            return {"chunks_indexed": 0, "files_processed": processed_files, "errors": errors}

        try:
            if self._vector_store is None:
                self._vector_store = FAISS.from_documents(all_documents, self._embeddings)
            else:
                self._vector_store.add_documents(all_documents)
        except Exception as exc:
            raise RuntimeError(f"Failed to index documents: {exc}") from exc

        self._chunk_count += len(all_documents)
        self._save_index()

        return {
            "chunks_indexed": len(all_documents),
            "files_processed": processed_files,
            "errors": errors,
        }

    def query(self, question: str, conversation_history: list[dict]) -> tuple[str, list[str]]:
        """Retrieve relevant chunks and generate a grounded answer."""
        if self._vector_store is None or self._chunk_count == 0:
            return settings.no_context_reply, []

        try:
            retrieved_docs = self._vector_store.similarity_search(question, k=settings.top_k_results)
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        if not retrieved_docs:
            return settings.no_context_reply, []

        context_parts = []
        source_previews = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source_name = doc.metadata.get("source", "unknown")
            chunk_content = doc.page_content.strip()
            context_parts.append(f"[Source {i} - {source_name}]\n{chunk_content}")
            preview = chunk_content[:200].replace("\n", " ")
            source_previews.append(f"[{source_name}] {preview}...")

        context_block = "\n\n---\n\n".join(context_parts)

        messages = [{"role": "system", "content": settings.system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": f"Context:\n\n{context_block}\n\n---\n\nQuestion: {question}"
        })

        try:
            response = self._groq_client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        answer = response.choices[0].message.content or settings.no_context_reply
        return answer.strip(), source_previews

    def reset(self) -> None:
        """Wipe the FAISS index from memory and disk."""
        self._vector_store = None
        self._chunk_count = 0
        index_dir = settings.faiss_index_dir
        if index_dir.exists():
            shutil.rmtree(str(index_dir))

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None and self._chunk_count > 0


rag_engine = RAGEngine()
