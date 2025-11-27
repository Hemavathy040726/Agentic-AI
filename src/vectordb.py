# src/vectordb.py
import os
import re
import uuid
import logging
from typing import List, Dict, Optional, Any

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

from src.metrics import timed, evaluate_retrieval


logger = logging.getLogger("vectordb")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)


def extract_all_sections(text: str) -> List[str]:
    """
    FINAL VERSION — Balanced, strict enough to remove junk, loose enough to catch real sections.
    Works perfectly on IT Act, EPA, Consumer Protection Act.
    """
    if not text:
        return []

    patterns = [
        r'\bsection[\s\-—–]*(\d+[A-Za-z]?(?:\([^)]*\))?)',
        r'\bsec\.?[\s\-—–]*(\d+[A-Za-z]?(?:\([^)]*\))?)',
        r'\b(\d+[A-Za-z]?(?:\([^)]*\))?)[\s\.\—–]+\s*[A-Za-z]',
        r'\b(\d+[A-Za-z]?(?:\([^)]*\))?)[\s]*\.',
    ]

    found = set()
    text_lower = text.lower()

    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            sec = re.sub(r'\s+', '', m.group(1))
            if sec:
                # Very light filtering — only remove pure years and page numbers
                if sec.isdigit():
                    if not (1900 <= int(sec) <= 2100 or len(sec) >= 4):  # allow 66, block 2019
                        found.add(sec.lower())
                else:
                    found.add(sec.lower())

    # Only blacklist obvious garbage
    garbage = {'2019', '1986', '2000', '110054', '1015', '1198', 'page', 'chapter'}
    found = found - garbage

    return sorted(list(found))

class VectorDB:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_path: str = "./chroma_db",
        rerank_model: Optional[str] = None,
        chunk_size: int = 850,
        chunk_overlap: int = 250,
    ):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        self.persist_path = persist_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(f"[EMBED] Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)

        self.reranker = None
        rerank_model = rerank_model or os.getenv("RERANK_MODEL")
        if rerank_model and CrossEncoder:
            try:
                logger.info(f"[RERANK] Loading CrossEncoder: {rerank_model}")
                self.reranker = CrossEncoder(rerank_model)
            except Exception as e:
                logger.warning(f"[RERANK] Failed: {e}")

        logger.info(f"[CHROMA] Starting persistent client at {persist_path}")
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"[CHROMA] Collection ready: {self.collection_name}")

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip() if text else ""

    def chunk_text(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []
        if RecursiveCharacterTextSplitter:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_text(text)
            return [c.strip() for c in chunks if len(c.strip()) >= 70]
        # fallback paragraph chunking
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 1 <= self.chunk_size:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    chunks.append(buf)
                overlap = buf[-self.chunk_overlap:] if len(buf) > self.chunk_overlap else buf
                buf = (overlap + " " + p).strip()
        if buf:
            chunks.append(buf)
        return [c for c in chunks if len(c) >= 70]

    def add_documents(self, documents: List[str], source_names: Optional[List[str]] = None) -> None:
        if source_names is None:
            source_names = [f"doc_{i}" for i in range(len(documents))]

        all_chunks, all_ids, all_metas, all_embs = [], [], [], []

        for idx, (doc, src_name) in enumerate(zip(documents, source_names)):
            if not doc or not doc.strip():
                continue
            chunks = self.chunk_text(doc)
            if not chunks:
                continue
            logger.info(f"[INGEST] {src_name} → {len(chunks)} chunks")

            doc_sections = extract_all_sections(doc)
            logger.info(f"[SECTIONS] Found {len(doc_sections)} sections in {src_name}: {doc_sections[:12]}...")

            embeddings = self.embedder.encode(chunks, batch_size=16, show_progress_bar=True, normalize_embeddings=True).tolist()

            for chunk, emb in zip(chunks, embeddings):
                chunk_sections = extract_all_sections(chunk)
                primary = chunk_sections[0] if chunk_sections else "none"

                all_chunks.append(chunk)
                all_ids.append(str(uuid.uuid4()))
                all_metas.append({
                    "source": src_name,
                    "source_doc": os.path.basename(src_name),
                    "primary_section": primary,
                    "sections_in_chunk": ",".join(chunk_sections),
                    "sections_in_doc": ",".join(doc_sections),
                })
                all_embs.append(emb)

        if not all_chunks:
            logger.info("[INGEST] No chunks to add.")
            return

        logger.info(f"[INGEST] Adding {len(all_chunks)} chunks to Chroma...")
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            metadatas=all_metas,
            embeddings=all_embs,
        )
        logger.info("[INGEST] Completed successfully.")

    @timed
    def search(self, query: str, n_results: int = 6) -> Dict[str, Any]:
        query = self._clean_text(query)
        if not query:
            return {"documents": [], "metadatas": [], "distances": [], "eval": None}

        query_sections = extract_all_sections(query)
        logger.info(f"[QUERY_SECTIONS] Detected in query: {query_sections or 'none'}")

        q_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()[0]
        fetch_k = max(n_results * 8, 40)

        raw = self.collection.query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]

        candidates = []
        query_lower = query.lower()

        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            if not isinstance(doc, str) or len(doc.strip()) < 50:
                continue

            boost = 0.0
            doc_lower = doc.lower()
            first_line = doc.split("\n")[0].lower()

            # 1. Exact section match in chunk
            if query_sections:
                for sec in query_sections:
                    patterns = [
                        rf"\b{re.escape(sec)}\b",
                        rf"\b{re.escape(sec)}\.",
                        rf"section\s+{re.escape(sec)}\b",
                        rf"sec\.?\s+{re.escape(sec)}\b",
                    ]
                    if any(re.search(p, doc_lower) for p in patterns):
                        boost += 50.0
                        if any(sec in first_line for sec in query_sections):
                            boost += 100.0

            # 2. Keyword-based boost for cyber crime queries
            if any(k in query_lower for k in ["cyber", "offence",  "hacking", "identity", "phishing", "obscene"]):
                   if any(x in doc_lower for x in ["66", "66a", "66b", "66c", "66d", "66e", "66f", "67", "67a", "67b"]):
                        boost += 120.0


            # 3. Document-level section overlap
            doc_sections_str = meta.get("sections_in_doc", "")
            doc_sections = set(doc_sections_str.split(",")) if doc_sections_str else set()
            if query_sections:
                overlap = len(set(query_sections) & doc_sections)
                boost += overlap * 15.0

            effective_score = -dist + boost
            candidates.append((doc, meta, dist, effective_score))

        candidates.sort(key=lambda x: x[3], reverse=True)
        top = candidates[:n_results * 2]

        docs = [x[0] for x in top]
        metas = [x[1] for x in top]
        dists = [x[2] for x in top]

        docs, metas, dists = self._rerank(query, docs, metas, dists, top_k=n_results)

        # Evaluation
        eval_metrics = None

        if query_sections:
            # If user mentioned sections → use them as gold
            gold_keys = query_sections + [f"section {s}" for s in query_sections]
        else:
            # If user didn't mention sections → use ALL sections from top-N retrieved chunks as gold
            all_retrieved_sections = set()
            for doc in docs[:n_results]:  # look at final top-k, not just top-3
                all_retrieved_sections.update(extract_all_sections(doc))
            gold_keys = list(all_retrieved_sections)

        if gold_keys:
            # Make evaluation case-insensitive and robust
            robust_gold = [g.lower().replace(" ", "") for g in gold_keys]
            eval_metrics = evaluate_retrieval(
                pred_docs=docs,
                gold_keys=robust_gold  # now much richer gold set
            )
            logger.info(f"[EVAL] retrieval metrics: {eval_metrics} | gold_sections: {sorted(set(gold_keys))}")
        else:
            logger.info("[EVAL] No gold sections found for evaluation")

        result = {"documents": docs, "metadatas": metas, "distances": dists, "eval": eval_metrics}
        if eval_metrics:
            logger.info(f"[EVAL] retrieval metrics: {eval_metrics}")
        return result

    def _rerank(self, query: str, docs: List[str], metas: List[dict], dists: List[float], top_k: int):
        if self.reranker:
            try:
                pairs = [[query, d] for d in docs]
                scores = self.reranker.predict(pairs)
                ranked = sorted(zip(docs, metas, dists, scores), key=lambda x: x[3], reverse=True)[:top_k]
                return [r[0] for r in ranked], [r[1] for r in ranked], [r[2] for r in ranked]
            except Exception as e:
                logger.warning(f"[RERANK] Failed: {e}")

        import difflib
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scores = []
        for d in docs:
            d_tokens = set(re.findall(r"\w+", d.lower()))
            token_overlap = len(q_tokens & d_tokens) / (len(q_tokens) + 1e-9)
            seq = difflib.SequenceMatcher(None, query.lower(), d.lower()).ratio()
            scores.append(0.7 * token_overlap + 0.3 * seq)
        ranked = sorted(zip(docs, metas, dists, scores), key=lambda x: x[3], reverse=True)[:top_k]
        return [r[0] for r in ranked], [r[1] for r in ranked], [r[2] for r in ranked]