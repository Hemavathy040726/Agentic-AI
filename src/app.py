# src/app.py


# src/app.py
import os
import re
import logging
from typing import List, Optional

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

# logging
logger = logging.getLogger("app")
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
#     logger.addHandler(ch)

# small in-memory KB for hinting (optional)
KB_HINTS = {
    "it act": ["section 66", "section 66c", "section 66d", "section 67", "section 43a", "section 66f"]
}

LEGAL_SAFE_STOPWORDS = {"please", "can you", "could you", "tell me", "explain", "briefly", "summary"}

DEFAULT_PDFS = [
    "env_prot_act_1986.pdf",
    "con_prot_act_2019.pdf",
    "it_act_2000.pdf"
]


def load_documents_from_data(data_dir: Optional[str] = None) -> List[str]:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    results = []
    for fname in DEFAULT_PDFS:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"[LOAD] Missing PDF: {path}")
            continue
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                # normalize whitespace
                text = re.sub(r"\s+", " ", text).strip()
                pages.append(text)
        joined = "\n".join(pages)
        logger.info(f"[LOAD] {fname} → {len(joined):,} chars")
        results.append(joined)
    return results


def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    low = q.lower()
    for sw in LEGAL_SAFE_STOPWORDS:
        low = low.replace(sw, " ")
    low = re.sub(r"\s+", " ", low).strip()
    return low


def detect_domain(q: str) -> Optional[str]:
    ql = q.lower()
    if "it act" in ql or "information technology" in ql or "digital signature" in ql:
        return "it act"
    if "environment" in ql or "env prot" in ql:
        return "environment"
    if "consumer" in ql:
        return "consumer"
    return None


class RAGAssistant:
    def __init__(self):
        # initialize LLM (prefer OpenAI -> Groq -> Google)
        self.llm = self._initialize_llm()
        if not self.llm:
            raise RuntimeError("No LLM available. Set OPENAI_API_KEY or GROQ_API_KEY or GOOGLE_API_KEY")

        # init VectorDB
        embed_model = os.getenv("EMBEDDING_MODEL")
        rerank_model = os.getenv("RERANK_MODEL")
        persist = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
        self.vector_db = VectorDB(embedding_model=embed_model, rerank_model=rerank_model, persist_path=persist)

        # prompt template
        #         self.prompt_template = ChatPromptTemplate.from_template(
        #             """
        # You are an expert Indian legal research assistant. Use ONLY the provided statutory context.
        # Do not hallucinate or invent sections or case law.
        #
        # Context:
        # {context}
        #
        # Question:
        # {question}
        #
        # Answer precisely and cite section numbers when present in the context.
        # If the answer is not present in the context, respond:
        # "I could not find the exact answer in the provided documents."
        # """
        #         )

        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert Indian legal research assistant.

            Instructions:
            - Use ONLY the provided statutory context.
            - List all relevant offences with exact section numbers.
            - Be direct and authoritative.
            - NEVER say "not explicitly defined" or "context does not contain" — the context IS the law.

            Context:
            {context}

            Question: {question}

            Answer:
            """
        )
        self.output_parser = StrOutputParser()
        logger.info(f"[READY] RAG Assistant ({type(self.llm).__name__})")

    def _initialize_llm(self):
        # OpenAI
        if key := os.getenv("OPENAI_API_KEY"):
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"[LLM] Using OpenAI: {model}")
            return ChatOpenAI(api_key=key, model=model, temperature=0.0)

        # Groq - Option A signature
        if key := os.getenv("GROQ_API_KEY"):
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info(f"[LLM] Using Groq: {model}")
            return ChatGroq(model=model, groq_api_key=key, temperature=0.0)

        # Google Gemini
        if key := os.getenv("GOOGLE_API_KEY"):
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
            logger.info(f"[LLM] Using Google Gemini: {model}")
            return ChatGoogleGenerativeAI(google_api_key=key, model=model, temperature=0.0)

        return None

    @timed
    def add_documents(self, documents: List[str]):
        self.vector_db.add_documents(documents)

    @timed
    def invoke(self, raw_query: str, n_results: int = 6) -> str:
        if not raw_query or not raw_query.strip():
            return "Please provide a question."

        q = normalize_query(raw_query)
        logger.info(f"[QUERY] preprocessed: {q}")

        domain = detect_domain(q)
        kb_hits = KB_HINTS.get(domain, []) if domain else []
        logger.info(f"[INFO] domain={domain}, kb_hits={kb_hits}")

        # perform search (pass kb_hits as eval gold for metrics)
        res = self.vector_db.search(q, n_results=n_results)
        docs = res.get("documents", [])
        eval_info = res.get("eval")
        if eval_info:
            logger.info(f"[EVAL] retrieval metrics: {eval_info}")

        if not docs:
            return "I could not find the exact answer in the provided documents."

        # prioritize chunks that match KB hits
        prioritized = []
        for k in kb_hits:
            for d in docs:
                if k and k.lower() in d.lower() and d not in prioritized:
                    prioritized.append(d)
        for d in docs:
            if d not in prioritized:
                prioritized.append(d)

        # include small KB hint block for LLM guidance (clearly labeled)
        kb_block = ""
        if kb_hits:
            kb_block = "\n".join([f"[KB_HINT] {h}" for h in kb_hits]) + "\n\n"

        context = kb_block + "\n\n---\n\n".join(prioritized)
        max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n... (context truncated)"

        logger.info(f"[CONTEXT] using {len(prioritized)} chunks, {len(context):,} chars")

        prompt = self.prompt_template.format(context=context, question=raw_query)

        try:
            # Call LLM: wrappers often accept list-of-messages; keep consistent with previous usage.

            llm_resp = self.llm.invoke([{"role": "user", "content": prompt}])
            # accept common response shapes
            answer = getattr(llm_resp, "content", None) or getattr(llm_resp, "text", None) or str(llm_resp)
            return answer.strip()
        except Exception as e:
            logger.error(f"[LLM ERROR] {e}")
            return "Failed to generate answer due to LLM error."


def main():
    logger.info("Starting RAG Legal Assistant")
    assistant = RAGAssistant()

    logger.info("\nLoading and indexing PDFs...")
    docs = load_documents_from_data()
    if docs:
        assistant.vector_db.add_documents(docs, source_names=DEFAULT_PDFS)

    else:
        logger.warning("[WARN] No PDFs loaded. Place PDFs in ./data and re-run.")

    logger.info("Ready. Ask questions (type 'quit' to exit).")
    while True:
        q = input("Question > ").strip()
        if q.lower() in {"quit", "exit", "bye"}:
            break
        ans = assistant.invoke(q, n_results=int(os.getenv("DEFAULT_N_RESULTS", "6")))
        print("\n----- ANSWER -----\n")
        print(ans)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
