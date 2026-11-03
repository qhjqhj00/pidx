"""Page search: rank pages for one or more questions by embeddings alone (no LLM call).

Each page has a bge-m3 vector of "file path · page label / title path / page text" (indexer.embed_pages).
A page's score blends its own similarity with that of the tree node that best covers it, whose vector
carries the node summary:  score = 0.7 * page + 0.3 * node  (cosine, 0..1 in practice).

This is the fast first stage only. The tree search in search.py is still what reads and reasons.
"""

import time

import numpy as np

from . import llm, store
from .indexer import page_nodes
from .search import _unit, node_scores

PAGE_WEIGHT = 0.7
_cache: dict[str, tuple] = {}


def _page_vectors(doc):
    key = doc.get("updated_at")
    hit = _cache.get(doc["id"])
    if hit and hit[0] == key:
        return hit[1], hit[2], hit[3]
    pages, m = store.load_page_vectors(doc["id"])
    m = _unit(m) if m is not None else None
    deepest = page_nodes(doc.get("tree") or [])
    _cache[doc["id"]] = (key, pages, m, deepest)
    return pages, m, deepest


def _match(meta: dict, want: dict) -> bool:
    return all(meta.get(k) == v for k, v in want.items())


async def page_search(queries: list[str], doc_ids=None, collection_id=None, top_k=10, metadata=None) -> dict:
    t0 = time.time()
    if doc_ids:
        docs = [store.get_document(i, full=True) for i in doc_ids]
    else:
        docs, _ = store.list_documents(limit=100000, full=True, collection_id=collection_id)
    docs = [d for d in docs if d and d.get("tree") and (not metadata or _match(d.get("metadata") or {}, metadata))]
    if not docs:
        return {"error": "no completed documents to search", "results": []}

    t1 = time.time()
    vecs = await llm.embed(queries)
    if vecs is None:
        return {"error": "embedding service unavailable", "results": []}
    qvs = [_unit(v) for v in vecs]
    t2 = time.time()

    per_query = [[] for _ in queries]       # (score, page_sim, node_sim, doc, page, path)
    searched, missing = 0, []
    for d in docs:
        pages, m, deepest = _page_vectors(d)
        if m is None:
            missing.append(d["id"])
            continue
        searched += len(pages)
        sims = m @ np.stack(qvs).T                       # pages x queries
        for qi, qv in enumerate(qvs):
            ns = node_scores(d, [qv])
            col = sims[:, qi]
            for idx in np.argsort(-col)[: top_k * 2]:
                pg = pages[idx]
                nid, path = deepest.get(pg, (None, []))
                node_sim = ns.get(nid, float(col[idx]))
                score = PAGE_WEIGHT * float(col[idx]) + (1 - PAGE_WEIGHT) * node_sim
                per_query[qi].append((score, float(col[idx]), node_sim, d, pg, path))
    t3 = time.time()

    results = []
    for q, hits in zip(queries, per_query):
        hits.sort(key=lambda h: -h[0])
        hits = hits[:top_k]
        texts = {}
        for d_id in {h[3]["id"] for h in hits}:
            want = [h[4] for h in hits if h[3]["id"] == d_id]
            texts.update({(d_id, p["page"]): p for p in store.get_pages(d_id, want)})
        out = []
        for score, ps, nsim, d, pg, path in hits:
            p = texts.get((d["id"], pg), {})
            out.append({"doc_id": d["id"], "doc_name": d["name"], "rel_path": d.get("rel_path"),
                        "format": d.get("format"), "page": pg, "label": p.get("label"),
                        "score": round(score, 4), "page_score": round(ps, 4), "node_score": round(nsim, 4),
                        "title_path": path, "snippet": " ".join((p.get("text") or "").split())[:240],
                        "image": f"v1/documents/{d['id']}/pages/{pg}/image" if d.get("pdf_path") else None})
        results.append({"query": q, "pages": out})
    t4 = time.time()
    ms = lambda a, b: round((b - a) * 1000)
    return {"results": results, "retrieval": "embedding", "documents": len(docs), "pages_searched": searched,
            "documents_without_page_vectors": missing,
            "timing_ms": {"load": ms(t0, t1), "embed": ms(t1, t2), "score": ms(t2, t3), "fetch": ms(t3, t4),
                          "total": ms(t0, t4)}}
