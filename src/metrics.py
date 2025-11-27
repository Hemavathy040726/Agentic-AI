# ----------------------------
# Simple timing decorator for metrics
# ----------------------------
# # src/metrics.py
import time
from typing import List, Dict


def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.time() - start
            name = getattr(fn, "__name__", "fn")
            print(f"[PERF] {name} took {elapsed:.3f}s")
    return wrapper


def evaluate_retrieval(pred_docs: List[str], gold_keys: List[str]) -> Dict:
    if not gold_keys or not pred_docs:
        return {"hit@1": 0, "hit@3": 0, "hit@5": 0, "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0}

    gold = {g.lower().replace(" ", "") for g in gold_keys if g}

    def contains_gold(text: str) -> bool:
        t = text.lower().replace(" ", "")
        return any(g in t for g in gold)

    pred_lower = [d.lower().replace(" ", "") for d in pred_docs]

    ks = [1, 3, 5]
    results = {}
    first_hit = None

    found_sections = set()
    for i, doc in enumerate(pred_lower):
        if contains_gold(doc):
            if first_hit is None:
                first_hit = i + 1
            # collect which gold sections were found
            for g in gold:
                if g in doc:
                    found_sections.add(g)

        for k in ks:
            if i < k:
                results[f"hit@{k}"] = 1
                results[f"recall@{k}"] = len(found_sections) / len(gold)

    for k in ks:
        results.setdefault(f"hit@{k}", 0)
        results.setdefault(f"recall@{k}", len(found_sections) / len(gold) if gold else 0.0)

    results["mrr"] = 1.0 / first_hit if first_hit else 0.0

    return results