import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS = Path("data/chunks.jsonl")
INDEX_OUT = Path("data/faiss.index")
META_OUT = Path("data/meta.json")

def load_chunks():
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    chunks = load_chunks()
    print("chunks:", len(chunks))
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_OUT))
    meta = [{"id": c["id"], "url": c["url"], "anchor": c["anchor"], "section_title": c["section_title"], "page_title": c["page_title"]} for c in chunks]
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dim:", emb.shape[1], "saved:", INDEX_OUT, META_OUT)

if __name__ == "__main__":
    main()
