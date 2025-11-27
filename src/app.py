# src/app.py
import os
import re
import logging
from pathlib import Path
from typing import List

import pdfplumber
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from src.vectordb import VectorDB
from src.metrics import timed

load_dotenv()

# Logging setup
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

# Auto-discover ALL PDFs in data/ folder
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATHS = sorted(DATA_DIR.glob("*.pdf"))

if not PDF_PATHS:
    logger.error("No PDF files found in data/ folder!")
    exit(1)

logger.info(f"Found {len(PDF_PATHS)} PDF(s) in data/ folder:")
for p in PDF_PATHS:
    logger.info(f"  → {p.name}")

# Optional: small KB hints (still works, but less needed now)
KB_HINTS = {
    "it act": ["section 66", "section 66a", "section 66f", "section 67", "section 43"],
    "environment": ["section 15", "section 7", "section 8"],
    "consumer": ["section 2(9)", "section 94", "e-commerce"]
}

LEGAL_SAFE_STOPWORDS = {"please", "can you", "could you", "tell me", "explain", "briefly", "summary"}


def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    for sw in LEGAL_SAFE_STOPWORDS:
        q = q.replace(sw, " ")
    return re.sub(r"\s+", " ", q).strip()


def detect_domain(q: str) -> str | None:
    ql = q.lower()
    if any(x in ql for x in ["it act", "information technology", "cyber", "66f", "43a"]):
        return "it act"
    if any(x in ql for x in ["environment", "pollution", "epa", "env"]):
        return "environment"
    if "consumer" in ql or "online" in ql or "e-commerce" in ql:
        return "consumer"
    return None


class RAGAssistant:
    def __init__(self):
        self.llm = self._initialize_llm()
        if not self.llm:
            raise RuntimeError("No LLM available. Set API key for OpenAI/Groq/Gemini.")

        embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        rerank_model = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

        self.vector_db = VectorDB(
            embedding_model=embed_model,
            rerank_model=rerank_model,
            persist_path=persist_path
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are an expert Indian legal research assistant.

Instructions:
- Answer using ONLY the provided statutory context
- Cite exact section numbers
- Be precise, direct, and authoritative
- Never say "not found" — the context IS the law

Context:
{context}

Question: {question}

Answer:
"""
        )

        logger.info(f"[READY] RAG Assistant ({type(self.llm).__name__})")

    def _initialize_llm(self):
        if key := os.getenv("OPENAI_API_KEY"):
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"[LLM] Using OpenAI: {model}")
            return ChatOpenAI(api_key=key, model=model, temperature=0.0)

        if key := os.getenv("GROQ_API_KEY"):
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info(f"[LLM] Using Groq: {model}")
            return ChatGroq(model=model, groq_api_key=key, temperature=0.0)

        if key := os.getenv("GOOGLE_API_KEY"):
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
            logger.info(f"[LLM] Using Google Gemini: {model}")
            return ChatGoogleGenerativeAI(google_api_key=key, model=model, temperature=0.0)

        return None

    @timed
    def invoke(self, raw_query: str, n_results: int = 10) -> str:
        if not raw_query or not raw_query.strip():
            return "Please provide a question."

        q = normalize_query(raw_query)
        logger.info(f"[QUERY] preprocessed: {q}")

        # SEARCH — now returns FLAT lists
        res = self.vector_db.search(q, n_results=n_results * 3)

        # ←←← THIS IS THE FINAL CORRECT WAY (NO [0] ANYWHERE!)
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        distances = res.get("distances", [])

        if not docs:
            logger.warning("No documents returned from search")
            return "No relevant legal provisions found."

        sources = set()
        context_parts = []

        for doc, meta in zip(docs, metas):
            text = doc.strip() if isinstance(doc, str) else ""
            if len(text) < 40:  # lowered for short acts
                continue

            context_parts.append(text)

            if isinstance(meta, dict):
                src = meta.get("source_doc", "Unknown Act")
                sources.add(src.replace(".pdf", "").replace("_", " ").title())

        if not context_parts:
            logger.warning("All retrieved chunks were too short or empty")
            return "No sufficiently relevant sections found."

        source_hint = f"**Source: {', '.join(sorted(sources))}**\n\n" if sources else ""
        context = source_hint + "\n\n---\n\n".join(context_parts[:n_results * 2])

        if len(context) > 12_000:
            context = context[:12_000] + "\n\n... (truncated)"

        logger.info(f"[CONTEXT] using {len(context_parts)} chunks, {len(context):,} chars")

        prompt = self.prompt_template.format(context=context, question=raw_query)

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            answer = getattr(response, "content", str(response)).strip()

            # Deduplicate lines
            seen = set()
            lines = []
            for line in answer.split("\n"):
                if line.strip() and line not in seen:
                    seen.add(line)
                    lines.append(line)
                elif not line.strip():
                    lines.append("")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[LLM ERROR] {e}")
            return "Error generating answer."

def main():
    logger.info("Starting RAG Legal Assistant")

    assistant = RAGAssistant()

    logger.info("\nLoading and indexing PDFs from data/ folder...")
    documents = []
    source_names = []

    for pdf_path in PDF_PATHS:
        filename = pdf_path.name
        logger.info(f"Loading {filename}...")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(re.sub(r"\s+", " ", p).strip() for p in pages if p.strip())
                documents.append(text)
                source_names.append(filename)
                logger.info(f"[LOAD] {filename} → {len(text):,} chars")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    if documents:
        assistant.vector_db.add_documents(documents, source_names=source_names)
        logger.info("[INGEST] Completed successfully.")
    else:
        logger.error("No documents loaded!")

    logger.info("Ready. Ask questions (type 'quit' to exit).")

    while True:
        try:
            q = input("\nQuestion > ").strip()
            if q.lower() in {"quit", "exit", "bye", ""}:
                break
            answer = assistant.invoke(q)
            print("\n----- ANSWER -----\n")
            print(answer)
            print("\n" + "-" * 50)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()