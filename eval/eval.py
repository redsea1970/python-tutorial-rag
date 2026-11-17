import json
from pathlib import Path
from src.rag import retrieve

QS = Path("eval/questions.jsonl")
OUT = Path("eval/results.json")


def main():
    qs = [json.loads(l) for l in QS.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(qs)
    hit1 = hit3 = 0
    mrr = 0.0
    rows = []
    for q in qs:
        hits = retrieve(q["q"], k=5)
        got = [(h.get("url") or "").split("#")[-1] for h in hits]
        gold = {a.split("#")[-1] for a in q["anchors"]}
        r1 = bool(got) and got[0] in gold
        r3 = any(a in gold for a in got[:3])
        rank = next((i + 1 for i, a in enumerate(got) if a in gold), 0)
        if r1:
            hit1 += 1
        if r3:
            hit3 += 1
        if rank:
            mrr += 1.0 / rank
        rows.append({"id": q["id"], "q": q["q"], "gold": sorted(gold), "top": got, "rank": rank})
    res = {
        "n": n,
        "hit@1": round(hit1 / n, 4),
        "hit@3": round(hit3 / n, 4),
        "mrr": round(mrr / n, 4),
        "rows": rows,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n", "hit@1", "hit@3", "mrr")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
