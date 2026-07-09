"""
rag_engine.py - Core RAG engine with page number tracking.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import fitz
from groq import Groq
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from PDF with page number tracking.
    Returns list of {text, page} dicts instead of one big string.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot parse '{filename}' as a PDF: {exc}") from exc

    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text.strip(), "page": page_num})

    doc.close()

    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'.")

    return pages


def extract_text_from_txt(file_bytes: bytes, filename: str) -> list[dict]:
    """Decode plain text file — no page numbers, use page 1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return [{"text": file_bytes.decode(encoding), "page": 1}]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode '{filename}' as text.")


def extract_pages(file_bytes: bytes, filename: str) -> list[dict]:
    """Route file to correct extractor."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if suffix == ".txt":
        return extract_text_from_txt(file_bytes, filename)
    raise ValueError(f"Unsupported file type '{suffix}'.")


def chunk_pages(pages: list[dict], source: str) -> list[Document]:
    """
    Split pages into chunks, preserving page number in metadata.
    Each chunk knows which page it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents = []
    chunk_index = 0

    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "page": page_data["page"],
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

    return documents


class RAGEngine:

    def __init__(self) -> None:
        logger.info("Loading local sentence-transformers model...")
        from langchain_huggingface import HuggingFaceEmbeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._groq_client = Groq(api_key=settings.groq_api_key)
        self._vector_store: Optional[FAISS] = None
        self._chunk_count: int = 0
        self._try_load_index()
        logger.info("RAG engine ready.")

    def _try_load_index(self) -> None:
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
        if self._vector_store is None:
            return
        index_dir = settings.faiss_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(index_dir))

    def add_documents(self, file_bytes_list: list[bytes], filenames: list[str]) -> dict:
        all_documents = []
        processed_files = []
        errors = []

        for file_bytes, filename in zip(file_bytes_list, filenames):
            try:
                pages = extract_pages(file_bytes, filename)
                docs = chunk_pages(pages, source=filename)
                all_documents.extend(docs)
                processed_files.append(filename)
                logger.info("'%s' → %d chunks across %d pages", filename, len(docs), len(pages))
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

    def query(self, question: str, conversation_history: list[dict]) -> tuple[str, list[dict]]:
        """
        Returns answer and list of source dicts with keys:
        - text: chunk preview
        - source: filename
        - page: page number
        """
        if self._vector_store is None or self._chunk_count == 0:
            return settings.no_context_reply, []

        try:
            retrieved_docs = self._vector_store.similarity_search(
                question, k=settings.top_k_results
            )
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        if not retrieved_docs:
            return settings.no_context_reply, []

        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source_name = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", "?")
            chunk_content = doc.page_content.strip()

            context_parts.append(
                f"[Source {i} — {source_name}, Page {page_num}]\n{chunk_content}"
            )
            sources.append({
                "text": chunk_content,
                "source": source_name,
                "page": page_num,
            })

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
        return answer.strip(), sources

    def reset(self) -> None:
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
