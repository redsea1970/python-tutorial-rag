import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX = Path("data/faiss.index")
META = Path("data/meta.json")
CHUNKS = Path("data/chunks.jsonl")

THRESHOLD = 0.35
CANDIDATES = 40
RRF_K = 60
TITLE_WEIGHT = 2.0
EXACT_TITLE_WEIGHT = 0.01

_model = None
_index = None
_meta = None
_chunks = None
_bm25 = None
_title_bm25 = None


def _normalize_token(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]

    if word.endswith("ing") and len(word) > 5:
        return word[:-3]

    if word.endswith("ed") and len(word) > 4:
        return word[:-2]

    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]

    return word


def _tokenize(text):
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [_normalize_token(word) for word in words]


STOPWORDS = {
    "what", "is", "a", "an", "the", "how", "do", "does",
    "i", "to", "with", "and", "of", "in", "for", "from",
    "on", "have", "has", "are", "can", "my", "your",
}


def _exact_title_score(query, title):
    def prepare(text):
        words = _tokenize(text)
        words = [
            word[:-1] if word.endswith("e") and len(word) > 4 else word
            for word in words
        ]
        return {word for word in words if word not in STOPWORDS}

    query_words = prepare(query)
    title_words = prepare(title)

    if not query_words or not title_words:
        return 0.0

    overlap = len(query_words & title_words)
    precision = overlap / len(title_words)
    recall = overlap / len(query_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _load():
    global _model, _index, _meta, _chunks, _bm25, _title_bm25

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX))
        _meta = json.loads(META.read_text(encoding="utf-8"))

        with CHUNKS.open(encoding="utf-8") as f:
            _chunks = [json.loads(line) for line in f if line.strip()]

        documents = [
            _tokenize(c["page_title"] + " " + c["section_title"] + " " + c["text"])
            for c in _chunks
        ]

        titles = [
            _tokenize(c["section_title"] + " " + c["page_title"])
            for c in _chunks
        ]

        _bm25 = BM25Okapi(documents)
        _title_bm25 = BM25Okapi(titles)


def _rank_map(scores, limit):
    ids = np.argsort(scores)[::-1][:limit]
    return {
        int(i): float(scores[i])
        for i in ids
        if scores[i] > 0
    }


def _rank_positions(scores):
    ordered = sorted(scores, key=scores.get, reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(ordered)}


def retrieve(query, k=5):
    _load()

    tokens = _tokenize(query)

    bm25_scores = _bm25.get_scores(tokens)
    title_scores = _title_bm25.get_scores(tokens)

    bm25 = _rank_map(bm25_scores, CANDIDATES)
    title = _rank_map(title_scores, CANDIDATES)

    q = _model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q)

    search_k = min(CANDIDATES, _index.ntotal)
    semantic_scores, semantic_ids = _index.search(q, search_k)

    semantic = {
        int(i): float(score)
        for score, i in zip(semantic_scores[0], semantic_ids[0])
        if i >= 0
    }

    semantic_ranks = _rank_positions(semantic)
    bm25_ranks = _rank_positions(bm25)
    title_ranks = _rank_positions(title)

    candidates = set(semantic) | set(bm25) | set(title)

    ranked = []

    for i in candidates:
        rrf_score = 0.0

        if i in semantic_ranks:
            rrf_score += 1.0 / (RRF_K + semantic_ranks[i])

        if i in bm25_ranks:
            rrf_score += 1.0 / (RRF_K + bm25_ranks[i])

        if i in title_ranks:
            rrf_score += TITLE_WEIGHT / (RRF_K + title_ranks[i])

        exact_title_score = _exact_title_score(
            query,
            _chunks[i]["section_title"],
        )
        rrf_score += EXACT_TITLE_WEIGHT * exact_title_score

        ranked.append({
            "rrf_score": rrf_score,
            "semantic_score": semantic.get(i, 0.0),
            "bm25_score": bm25.get(i, 0.0),
            "title_score": title.get(i, 0.0),
            "exact_title_score": exact_title_score,
            "id": i,
        })

    ranked.sort(
        key=lambda x: (
            x["rrf_score"],
            x["semantic_score"],
            x["id"],
        ),
        reverse=True,
    )

    out = []

    for item in ranked[:k]:
        i = item["id"]
        m = _meta[i]
        c = _chunks[i]

        out.append({
            "score": float(item["rrf_score"]),
            "rrf_score": float(item["rrf_score"]),
            "semantic_score": float(item["semantic_score"]),
            "bm25_score": float(item["bm25_score"]),
            "title_score": float(item["title_score"]),
            "exact_title_score": float(item["exact_title_score"]),
            "id": m["id"],
            "url": m["url"],
            "section_title": m["section_title"],
            "text": c["text"],
        })

    return out


def answer(query, k=5):
    hits = retrieve(query, k=k)

    if not hits:
        return {
            "text": "\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445.",
            "source": None,
            "score": 0.0,
        }

    best = hits[0]

    if best["semantic_score"] < THRESHOLD:
        titles = ", ".join(h["section_title"] for h in hits[:3])
        return {
            "text": (
                "\u041d\u0435 \u043d\u0430\u0448\u0451\u043b "
                "\u0442\u043e\u0447\u043d\u043e\u0433\u043e "
                "\u043e\u0442\u0432\u0435\u0442\u0430 "
                "\u0432 \u0442\u0443\u0442\u043e\u0440\u0438\u0430\u043b\u0435. "
                "\u0411\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0435 "
                "\u0442\u0435\u043c\u044b: "
                + titles
            ),
            "source": best["url"],
            "score": best["semantic_score"],
        }

    return {
        "text": best["text"][:1200],
        "source": best["url"],
        "score": best["semantic_score"],
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are f-strings?"
    hits = retrieve(q)

    print("Q:", q)

    for h in hits:
        print(
            "\n",
            h["section_title"],
            "| RRF:", round(h["rrf_score"], 5),
            "| Semantic:", round(h["semantic_score"], 3),
            "| BM25:", round(h["bm25_score"], 3),
            "| Title:", round(h["title_score"], 3),
            "| URL:", h["url"],
        )
