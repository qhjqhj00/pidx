"""Reasoning-based retrieval over document trees, with node embeddings for recall.

1. Pick documents: rank by embedding similarity (doc description and best-matching node), then
   the LLM reads a shortlist (descriptions, and the folder tree for documents from a zip collection).
2. Tree search, two modes:
   - hybrid (default): the tree is cut into anchor subtrees that each fit in one small prompt; node
     vectors recall the best anchors, and the LLM reads those subtrees (titles, pages, summaries) in
     parallel and names the nodes to read.
   - tree: the LLM reads the whole skeleton at once; trees too big for one prompt are explored
     level by level. Also the fallback when there are no vectors.
3. Read: node text is assembled from layout blocks, each tagged with its block id.
4. Optional answer with block-level citations; if the LLM says the evidence is
   insufficient, it describes what is missing and another tree-search round runs.
"""

import asyncio
import json
import re
import time

import numpy as np

from . import config, images as imagemod, llm, store, tree
from .llm import approx_tokens, chat_json

FULL_TREE_BUDGET = 40000      # tokens of skeleton we are happy to send in one prompt
NODE_TEXT_BUDGET = 12000      # per node, when reading
CONTEXT_BUDGET = 60000        # all nodes, when answering


# --------------------------------------------------------------------------- tree rendering

def _node_line(n, depth, collapsed=False):
    s = n.get("summary") or n.get("prefix_summary") or ""
    pages = f"p.{n['start_index']}" + (f"-{n['end_index']}" if n["end_index"] != n["start_index"] else "")
    more = f" [+{sum(1 for _ in tree.walk(n['nodes']))} subsections, collapsed]" if collapsed and n["nodes"] else ""
    return "  " * depth + f"[{n['node_id']}] {n['title']} ({pages}){more}" + (f" — {s}" if s else "")


def render(nodes, max_depth=99, expanded=frozenset(), depth=0):
    out = []
    for n in nodes:
        collapsed = depth + 1 >= max_depth and n["node_id"] not in expanded
        out.append(_node_line(n, depth, collapsed))
        if n["nodes"] and not collapsed:
            out.extend(render(n["nodes"], max_depth, expanded, depth + 1))
    return out


def index_nodes(nodes):
    return {n["node_id"]: n for n in tree.walk(nodes)}


# --------------------------------------------------------------------------- doc selection

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


_vec_cache: dict[str, tuple] = {}


def vectors_current(doc) -> bool:
    """Vectors made by another embedding model live in another space (and size): ignore them until
    the background backfill has recomputed them."""
    return (doc.get("embed_model") or store.LEGACY_EMBED_MODEL) == llm.embed_model()


def node_vectors(doc) -> tuple[list[str], np.ndarray | None]:
    """Unit-normalised node vectors of a document, cached until the document changes."""
    if not vectors_current(doc):
        return [], None
    key = doc.get("updated_at")
    hit = _vec_cache.get(doc["id"])
    if hit and hit[0] == key:
        return hit[1], hit[2]
    ids, m = store.load_node_vectors(doc["id"])
    m = _unit(m) if m is not None else None
    _vec_cache[doc["id"]] = (key, ids, m)
    return ids, m


def node_scores(doc, qvs) -> dict[str, float]:
    """node_id -> best cosine similarity to any of the query vectors."""
    ids, m = node_vectors(doc)
    if m is None or not qvs:
        return {}
    sims = (m @ np.stack(qvs).T).max(axis=1)
    return dict(zip(ids, sims.tolist()))


def doc_score(doc, qv) -> float:
    """Best of: document description, best node, and (Gemma path) best image when it clears
    IMAGE_MIN_SCORE, so a photo with no text to describe it can still reach the shortlist."""
    s = [max(node_scores(doc, [qv]).values(), default=0.0)]
    if doc.get("embedding") and vectors_current(doc) and len(doc["embedding"]) == len(qv):
        s.append(float(_unit(doc["embedding"]) @ qv))
    im = doc_image_score(doc, qv)
    if im >= config.IMAGE_MIN_SCORE:
        s.append(im)
    return max(s)


_img_cache: dict[str, tuple] = {}


def _image_matrix(doc):
    """(meta, unit-normalised image vectors) of a document, cached until it changes; (.., None) if none."""
    if not config.EMBED_MULTIMODAL or not vectors_current(doc):
        return [], None
    hit = _img_cache.get(doc["id"])
    if not hit or hit[0] != doc.get("updated_at"):
        meta, m = store.load_image_vectors(doc["id"])
        hit = (doc.get("updated_at"), meta, _unit(m) if m is not None else None)
        _img_cache[doc["id"]] = hit
    return hit[1], hit[2]


def doc_image_score(doc, qv) -> float:
    _, m = _image_matrix(doc)
    if m is None or qv is None or m.shape[1] != len(qv):
        return 0.0
    return float((m @ qv).max())


def image_hits(docs, qvs, k=None) -> list[dict]:
    """EmbeddingGemma path: the images (cropped figures, charts, photos) closest to the query across
    the given documents, best first, above IMAGE_MIN_SCORE."""
    if not config.EMBED_MULTIMODAL or not qvs:
        return []
    q = np.stack(qvs)
    out = []
    for d in docs:
        meta, m = _image_matrix(d)
        if m is None or m.shape[1] != q.shape[1]:
            continue
        sims = (m @ q.T).max(axis=1)
        by_id = index_nodes(d["tree"])
        for im, sc in zip(meta, sims.tolist()):
            if sc < config.IMAGE_MIN_SCORE:
                continue
            n = by_id.get(im["node_id"])
            out.append({"doc_id": d["id"], "doc_name": d["name"], "rel_path": d.get("rel_path"),
                        "image_id": im["image_id"], "page": im["page"], "score": round(sc, 4),
                        "bbox_pt": im["bbox"], "node_id": im["node_id"], "node_title": n["title"] if n else None,
                        "caption": im.get("caption") or "",
                        "url": f"/v1/documents/{d['id']}/images/{im['image_id']}"})
    out.sort(key=lambda x: -x["score"])
    if out:
        out = [x for x in out if x["score"] >= out[0]["score"] - config.IMAGE_REL_MARGIN]
    return out[:k or config.IMAGE_TOP_K]


def _image_note(d, img_scores) -> str:
    sc = (img_scores or {}).get(d["id"], 0.0)
    return f" [contains an image that visually matches the question, similarity {sc:.2f}]" \
        if sc >= config.IMAGE_MIN_SCORE else ""


def _doc_listing(docs, img_scores=None) -> str:
    """Shortlisted documents, standalone ones as a list and zip-collection ones under their folders.
    img_scores (Gemma path): documents whose images match the question get a note, because their
    text description may say nothing about what the picture shows."""
    lines, colls = [], {}
    for i, d in enumerate(docs):
        if d.get("collection_id"):
            colls.setdefault(d["collection_id"], []).append(i)
        else:
            lines.append(f"[{i}] {d['name']} ({d['page_count']} pages): {d.get('description') or ''}"
                         + _image_note(d, img_scores))
    for cid, idx in colls.items():
        c = store.get_collection(cid)
        if not c:
            for i in idx:
                lines.append(f"[{i}] {docs[i]['name']}: {docs[i].get('description') or ''}"
                             + _image_note(docs[i], img_scores))
            continue
        want = {docs[i]["id"]: i for i in idx}
        lines.append(f"Collection \"{c['name']}\" (folder tree; only matching files shown):")

        def show(node, depth):
            out = []
            for ch in node["nodes"]:
                if ch["type"] == "file":
                    if ch["doc_id"] in want:
                        d = docs[want[ch["doc_id"]]]
                        out.append("  " * depth + f"[{want[ch['doc_id']]}] {ch['title']} ({d['page_count']} pages): "
                                   f"{d.get('description') or ''}" + _image_note(d, img_scores))
                else:
                    sub = show(ch, depth + 1)
                    if sub:
                        out.append("  " * depth + f"{ch['title']}/ — {ch.get('summary') or ''}")
                        out.extend(sub)
            return out

        lines.extend(show(c["tree"], 1))
    return "\n".join(lines)


async def select_docs(query, docs, qv=None, max_docs=5):
    """Returns (picked docs, thinking, shortlist [(doc, score)])."""
    ranked = [(d, None) for d in docs]
    if qv is not None:
        ranked = sorted(((d, doc_score(d, qv)) for d in docs), key=lambda x: -x[1])
    if len(docs) <= 1:
        return docs, "", ranked
    short = ranked[:config.DOC_SHORTLIST]
    cand = [d for d, _ in short]
    img_scores = {d["id"]: doc_image_score(d, qv) for d in cand} if config.EMBED_MULTIMODAL and qv is not None else {}
    js = await chat_json(f"""Question: {query}

Documents in the library:
{_doc_listing(cand, img_scores)}

Which documents should be searched to answer the question? Pick all that are needed (e.g. several for comparisons), at most {max_docs}, most relevant first.
Return JSON {{"thinking": "...", "docs": [<index>, ...]}}""", max_tokens=1000, label="选文档")
    picked = [cand[i] for i in js.get("docs", []) if isinstance(i, int) and 0 <= i < len(cand)]
    picked = list({d["id"]: d for d in picked}.values()) or cand[:max_docs]
    # the documents holding the images that search would return are searched too, even when the LLM
    # passed them over (a photo's text description can be empty or wrong); at most 2 added
    if img_scores:
        best = max(img_scores.values())
        have = {d["id"] for d in picked}
        extra = [d for d in cand if d["id"] not in have and img_scores[d["id"]] >= config.IMAGE_MIN_SCORE
                 and img_scores[d["id"]] >= best - config.IMAGE_REL_MARGIN]
        picked += sorted(extra, key=lambda d: -img_scores[d["id"]])[:2]
    return picked, js.get("thinking", ""), short


# --------------------------------------------------------------------------- tree search

SEARCH_PROMPT = """You are searching the tree index of the document "{name}" to answer a question.
Document: {desc}
Each line is a node: [node_id] title (pages) — summary. Indentation shows nesting.
{collapsed_note}
Question: {query}
{extra}
Tree:
{tree}

Find the nodes whose text most likely contains the information needed. Prefer the most specific nodes that cover it;
include several if the answer may be spread out (comparisons, totals, definitions, footnotes, tables in another section).
Return JSON {{"thinking": "<where the answer should be and why>", "node_list": ["<node_id>", ...]{expand_field}}}
with at most {k} node ids in node_list, most relevant first."""


async def tree_search(query, doc, k=5, exclude=(), extra=""):
    """Pure reasoning over the whole skeleton (layered expansion for big trees)."""
    nodes = doc["tree"]
    by_id = index_nodes(nodes)
    full = "\n".join(render(nodes))
    if approx_tokens(full) <= FULL_TREE_BUDGET:
        js = await chat_json(SEARCH_PROMPT.format(name=doc["name"], desc=doc.get("description") or "", query=query,
                                                  extra=extra, tree=full, k=k, collapsed_note="", expand_field=""),
                             max_tokens=2000, label=f"树检索 · {doc['name']}")
        thinking, picked = js.get("thinking", ""), js.get("node_list", [])
        how = {"mode": "full", "tree_tokens": approx_tokens(full)}
    else:
        # big tree: start shallow, let the model expand collapsed nodes it cares about
        expanded, thinking, picked = set(), "", []
        for _ in range(4):
            view = "\n".join(render(nodes, max_depth=2, expanded=frozenset(expanded)))
            js = await chat_json(SEARCH_PROMPT.format(
                name=doc["name"], desc=doc.get("description") or "", query=query, extra=extra, tree=view, k=k,
                collapsed_note="Nodes marked 'collapsed' have hidden subsections; ask to expand them if needed.",
                expand_field=', "expand": ["<node_id of a collapsed node to open>", ...]'), max_tokens=2000,
                label=f"树检索 · {doc['name']} · 展开 {len(expanded)}")
            thinking, picked = js.get("thinking", ""), js.get("node_list", [])
            new = [i for i in js.get("expand", []) if i in by_id and i not in expanded and by_id[i]["nodes"]]
            if not new:
                break
            for i in new:   # open the node and its path
                expanded.add(i)
        how = {"mode": "layered", "expanded": sorted(expanded)}
    picked = [i for i in dict.fromkeys(map(str, picked)) if i in by_id and i not in exclude][:k]
    return thinking, drop_overlaps([by_id[i] for i in picked], doc), how


# --------------------------------------------------------------------------- hybrid: anchors + vectors

_anchor_cache: dict[str, tuple] = {}


def anchors(doc) -> list[dict]:
    """Cut the tree into anchor subtrees whose skeleton fits ANCHOR_TOKENS.

    Consecutive small siblings are packed together; a node too big to fit becomes context (its
    ancestors' lines are shown above every anchor) and its children are packed instead.
    Anchor: {"path": [ancestor nodes], "roots": [sibling nodes], "ids": set of node ids inside}."""
    key = doc.get("updated_at")
    hit = _anchor_cache.get(doc["id"])
    if hit and hit[0] == key:
        return hit[1]
    size = {}

    def measure(n):
        size[n["node_id"]] = approx_tokens(_node_line(n, 0)) + sum(measure(c) for c in n["nodes"])
        return size[n["node_id"]]

    for n in doc["tree"]:
        measure(n)
    budget = config.ANCHOR_TOKENS
    out = []

    def pack(siblings, path):
        cur, tok = [], 0
        for n in siblings:
            t = size[n["node_id"]]
            if t <= budget or not n["nodes"]:
                if cur and tok + t > budget:
                    out.append((path, cur))
                    cur, tok = [], 0
                cur.append(n)
                tok += t
            else:
                if cur:
                    out.append((path, cur))
                    cur, tok = [], 0
                pack(n["nodes"], path + [n])
        if cur:
            out.append((path, cur))

    pack(doc["tree"], [])
    res = [{"path": p, "roots": r, "ids": {m["node_id"] for x in r for m in tree.walk([x])}} for p, r in out]
    _anchor_cache[doc["id"]] = (key, res)
    return res


def _anchor_label(a) -> str:
    t = " > ".join(n["title"] for n in a["path"])
    first, last = a["roots"][0]["title"], a["roots"][-1]["title"]
    span = first if len(a["roots"]) == 1 else f"{first} … {last}"
    return f"{t} > {span}" if t else span


ANCHOR_PROMPT = """You are searching the tree index of the document "{name}" to answer a question.
Document: {desc}
Below is ONE PART of the document's tree (other parts are not shown). Each line is a node:
[node_id] title (pages) — summary. Indentation shows nesting; the first lines are the ancestors of this part.

Question: {query}
{extra}
Tree part:
{tree}

Find the nodes in this part whose text most likely contains the information needed. Prefer the most specific nodes.
Rate how relevant this part is: 0 = nothing useful, 1 = maybe, 2 = likely, 3 = certainly contains the answer.
Return JSON {{"thinking": "<short reason>", "relevance": <0-3>, "node_list": ["<node_id>", ...]}}
with at most {k} node ids, most relevant first; empty node_list if relevance is 0."""


async def hybrid_search(query, doc, qvs, k=5, exclude=(), extra="", skip_anchors=(), image_nodes=()):
    """Recall anchor subtrees by node vectors, read them in parallel. Returns
    (thinking, nodes, how, examined anchor indexes); nodes=None means no vectors for this doc.
    image_nodes: nodes holding images that matched the query; the subtree with the best of them is
    read too, even when its text scores low (a chart can answer what the prose does not say)."""
    scores = node_scores(doc, qvs)
    if not scores:
        return None
    by_id = index_nodes(doc["tree"])
    al = anchors(doc)
    anchor_score = [max((scores.get(n, -1.0) for n in a["ids"]), default=-1.0) for a in al]
    fresh = [i for i in range(len(al)) if i not in skip_anchors] or list(range(len(al)))   # all read: re-read
    ranked = sorted(((i, anchor_score[i]) for i in fresh), key=lambda x: -x[1])
    chosen = ranked[:config.ANCHORS_PER_DOC]
    taken = {i for i, _ in chosen}
    for nid in image_nodes:
        i = next((i for i in fresh if nid in al[i]["ids"]), None)
        if i is not None and i not in taken:
            chosen.append((i, anchor_score[i]))
            taken.add(i)
            break

    async def read(i, score):
        a = al[i]
        lines = [_node_line(n, d) for d, n in enumerate(a["path"])]
        for r in a["roots"]:
            lines.extend(render([r], depth=len(a["path"])))
        js = await chat_json(ANCHOR_PROMPT.format(name=doc["name"], desc=doc.get("description") or "", query=query,
                                                  extra=extra, tree="\n".join(lines), k=k),
                             max_tokens=800, label=f"树检索 · {doc['name']} · 子树 {_anchor_label(a)[:30]}")
        allowed = a["ids"] | {n["node_id"] for n in a["path"]}
        picked = [x for x in dict.fromkeys(map(str, js.get("node_list") or [])) if x in allowed and x not in exclude]
        try:
            rel = int(js.get("relevance", 1))
        except (TypeError, ValueError):
            rel = 1
        return {"anchor": i, "label": _anchor_label(a), "score": round(score, 3), "relevance": rel,
                "nodes": picked, "thinking": js.get("thinking", "")}

    results = await asyncio.gather(*(read(i, sc) for i, sc in chosen))
    flat = []
    for rank, r in enumerate(results):
        if r["relevance"] <= 0:
            continue
        for pos, nid in enumerate(r["nodes"]):
            flat.append(((-r["relevance"], pos, rank), nid))
    picked = [nid for _, nid in sorted(flat)]
    picked = list(dict.fromkeys(picked))[:k]
    thinking = "\n".join(f"〔{r['label']}｜相关度 {r['relevance']}〕{r['thinking']}" for r in results)
    how = {"mode": "hybrid", "anchors_total": len(al),
           "anchors": [{k2: r[k2] for k2 in ("label", "score", "relevance", "nodes")} for r in results]}
    return thinking, drop_overlaps([by_id[i] for i in picked], doc), how, {i for i, _ in chosen}


def _covers(a, b):
    """Does node a's range contain node b's?"""
    a_end = tuple(a["end"]) if a["end"] else (10 ** 9, 0)
    b_end = tuple(b["end"]) if b["end"] else (10 ** 9, 0)
    return a is not b and tuple(a["start"]) <= tuple(b["start"]) and b_end <= a_end


def drop_overlaps(nodes, doc):
    """A node and its ancestor would be read twice: keep the ancestor if it fits the
    reading budget, otherwise keep the more specific descendants."""
    pages = None
    keep = list(nodes)
    for a in nodes:
        inner = [b for b in nodes if _covers(a, b)]
        if not inner or a not in keep:
            continue
        if pages is None:
            pages = store.get_pages(doc["id"])
        size = approx_tokens(tree.text_between(pages, tuple(a["start"]), tuple(a["end"]) if a["end"] else None))
        if size <= NODE_TEXT_BUDGET:
            keep = [n for n in keep if n not in inner]
        else:
            keep = [n for n in keep if n is not a]
    return keep


# --------------------------------------------------------------------------- reading

def node_blocks(doc_id, node, pages_cache):
    """Blocks of a node's full range (start .. end), as (block, page, text)."""
    sp, so = node["start"]
    ep, eo = node["end"] if node["end"] else (node["end_index"], 10 ** 9)
    out = []
    for p in range(sp, ep + 1):
        page = pages_cache[p]
        for b in page["blocks"]:
            if b["end"] <= b["start"]:
                continue
            if p == sp and b["end"] <= so:
                continue
            if p == ep and b["start"] >= eo:
                continue
            out.append((b, page, page["text"][b["start"]:b["end"]]))
    return out


def node_images(doc_id, node, pages_cache):
    """Image / chart blocks inside a node's range that have a crop on disk, as (block, page)."""
    sp, so = node["start"]
    ep, eo = node["end"] if node["end"] else (node["end_index"], 10 ** 9)
    out = []
    for p in range(sp, ep + 1):
        page = pages_cache.get(p)
        if not page:
            continue
        for b in page["blocks"]:
            if b.get("type") not in imagemod.IMAGE_TYPES:
                continue
            if (p == sp and b["start"] < so) or (p == ep and b["start"] >= eo):
                continue
            if imagemod.crop_path(doc_id, b["id"]).exists():
                out.append((b, page))
    return out


def bbox_k(b, page):
    """bbox in thousandths of page width/height, origin top-left (PageIndex convention)."""
    if not b.get("bbox") or not page.get("width"):
        return None
    x0, y0, x1, y1 = b["bbox"]
    w, h = page["width"], page["height"]
    return [round(x0 / w * 1000), round(y0 / h * 1000), round(x1 / w * 1000), round(y1 / h * 1000)]


def _node_brief(d, n):
    return {"doc_id": d["id"], "doc_name": d["name"], "node_id": n["node_id"], "title": n["title"],
            "start_index": n["start_index"], "end_index": n["end_index"], "rel_path": d.get("rel_path")}


async def search(query: str, doc_ids=None, top_k=5, answer=False, max_rounds=2, include_text=True, emit=None,
                 llm_backend=None, collection_id=None, retrieval=None):
    """See module docstring. `emit` (async callable) receives progress events for streaming UIs."""
    t0 = time.time()
    llm.backend.set(llm_backend or config.SEARCH_BACKEND)
    used = llm.new_scope()
    tr = llm.Tracer(emit)
    llm.tracer.set(tr)
    phases = []          # (name, start, end) wall-clock spans
    retrieval = retrieval or config.RETRIEVAL

    # any doc with a tree is searchable, including one being re-indexed (old tree until the new one lands)
    if doc_ids:
        docs = [store.get_document(i, full=True) for i in doc_ids]
    else:
        docs, _ = store.list_documents(limit=100000, full=True, collection_id=collection_id)
    docs = [d for d in docs if d and d.get("tree")]
    if not docs:
        return {"query": query, "error": "no completed documents to search", "nodes": []}
    await tr.event({"type": "start", "query": query, "llm": config.LLM_BACKENDS[llm.backend.get()]["label"],
                    "retrieval": retrieval,
                    "library": [{"doc_id": d["id"], "name": d["name"]} for d in docs]})

    qv = None
    if retrieval == "hybrid" or len(docs) > 1 or config.EMBED_MULTIMODAL:   # Gemma: images need the query vector
        t = time.time()
        v = await llm.embed([query], kind="query")
        qv = _unit(v[0]) if v else None
        phases.append(("向量召回", t - t0, time.time() - t0))
        await tr.event({"type": "embed", "ok": qv is not None, "ms": round((time.time() - t) * 1000)})

    t = time.time()
    n_lib = len(docs)
    docs, why, short = await select_docs(query, docs, qv)
    phases.append(("选文档", t - t0, time.time() - t0))
    await tr.event({"type": "docs", "skipped": n_lib <= 1, "thinking": why,
                    "shortlist": [{"name": d.get("rel_path") or d["name"], "score": round(sc, 3) if sc is not None else None}
                                  for d, sc in short[:10]] if n_lib > 1 else [],
                    "selected": [{"doc_id": d["id"], "name": d["name"], "rel_path": d.get("rel_path"),
                                  "description": d.get("description")} for d in docs]})

    pages = {d["id"]: {p["page"]: p for p in store.get_pages(d["id"])} for d in docs}
    hits = []          # (doc, node, thinking)
    reasoning = {}
    extra = ""
    qvs = [qv] if qv is not None else []
    examined = {}      # doc_id -> anchor indexes already read (hybrid)
    images = image_hits(docs, qvs)
    if config.EMBED_MULTIMODAL:
        await tr.event({"type": "images", "images": images})
    result = {}
    for rnd in range(max_rounds if answer else 1):
        seen = {(d["id"], n["node_id"]) for d, n, _ in hits}

        async def one(d):
            excl = {nid for did, nid in seen if did == d["id"]}
            r = None
            if retrieval == "hybrid" and qvs:
                r = await hybrid_search(query, d, qvs, top_k, exclude=excl, extra=extra,
                                        skip_anchors=examined.get(d["id"], set()),
                                        image_nodes=[im["node_id"] for im in images
                                                     if im["doc_id"] == d["id"] and im["node_id"]] if rnd == 0 else ())
            if r is not None:
                th, nodes, how, ex = r
                examined.setdefault(d["id"], set()).update(ex)
            else:
                th, nodes, how = await tree_search(query, d, top_k, exclude=excl, extra=extra)
            await tr.event({"type": "tree_search", "round": rnd + 1, "doc_id": d["id"], "doc_name": d["name"],
                            "thinking": th, "how": how, "nodes": [_node_brief(d, n) for n in nodes]})
            return th, nodes

        t = time.time()
        await tr.event({"type": "tree_search_start", "round": rnd + 1, "docs": [d["name"] for d in docs],
                        "mode": retrieval if qvs else "tree"})
        found = await asyncio.gather(*(one(d) for d in docs))
        phases.append((f"树检索 · 第 {rnd + 1} 轮", t - t0, time.time() - t0))
        new = 0
        for d, (thinking, nodes) in zip(docs, found):
            reasoning.setdefault(d["id"], []).append(thinking)
            for n in nodes:
                if (d["id"], n["node_id"]) not in seen:
                    hits.append((d, n, thinking))
                    new += 1
        if not answer or new == 0 and rnd > 0:
            break
        t = time.time()
        result = await _answer(query, docs, hits, pages, tr, images)
        phases.append((f"作答 · 第 {rnd + 1} 轮", t - t0, time.time() - t0))
        if result.get("sufficient", True) or not result.get("missing"):
            break
        await tr.event({"type": "insufficient", "round": rnd + 1, "missing": result["missing"]})
        extra = ("Already read (not sufficient): " + "; ".join(n["title"] for _, n, _ in hits) +
                 f"\nStill missing: {result['missing']}\nPoint to OTHER nodes that may contain the missing information.")
        if qvs:   # recall again around what is missing
            v = await llm.embed([f"{query}\n{result['missing']}"], kind="query")
            if v:
                qvs = [_unit(v[0])]

    out_nodes = []
    for d, n, _ in hits:
        item = {**_node_brief(d, n), "summary": n.get("summary") or n.get("prefix_summary") or ""}
        if include_text:
            item["text"] = tree.text_between(list(pages[d["id"]][p] for p in sorted(pages[d["id"]])),
                                             tuple(n["start"]), tuple(n["end"]) if n["end"] else None)
        out_nodes.append(item)
    res = {"query": query, "documents": [{"doc_id": d["id"], "name": d["name"]} for d in docs],
           "reasoning": reasoning, "nodes": out_nodes}
    if config.EMBED_MULTIMODAL:
        for im in images:
            pg = pages.get(im["doc_id"], {}).get(im["page"])
            im["bbox"] = bbox_k({"bbox": im["bbox_pt"]}, pg) if pg else None
        res["images"] = images
    if answer:
        res.update(answer=result.get("answer", ""), citations=result.get("citations", []),
                   sufficient=result.get("sufficient", True))
    res["llm"] = llm.backend.get()
    res["retrieval"] = retrieval if qv is not None else "tree"
    res["usage"] = used
    res["seconds"] = round(time.time() - t0, 1)
    res["timing"] = timing_summary(tr.calls, phases, res["seconds"])
    return res


def timing_summary(calls, phases, total):
    """Where the time went: per phase wall-clock, and per LLM call queue / prefill / decode."""
    ttft = sum(c["ttft_ms"] for c in calls) / 1000
    decode = sum(c["decode_ms"] for c in calls) / 1000
    wait = sum(c["wait_ms"] for c in calls) / 1000
    return {"total_s": total,
            "phases": [{"name": n, "start": round(a, 2), "end": round(b, 2), "seconds": round(b - a, 2)} for n, a, b in phases],
            "calls": calls,
            "llm_sum": {"wait_s": round(wait, 2), "ttft_s": round(ttft, 2), "decode_s": round(decode, 2),
                        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                        "completion_tokens": sum(c["completion_tokens"] for c in calls)}}


STATUS_MARK = "@@STATUS@@"
ANSWER_PROMPT = """Answer the question using only the document excerpts below.
Each excerpt block starts with an id in square brackets like [1:p12_b3] (document 1, page 12, block 3).
Cite the blocks that support each statement by putting their ids in square brackets right after it, e.g. "... was $3.4 billion [1:p12_b3]."
{images_note}Be precise with numbers, units and periods. Answer in the language of the question. If the excerpts do not contain enough information, answer what you can and say what is missing.

Question: {query}

{context}

Write the answer in markdown first. Then, on the very last line, write {mark} followed by JSON:
{mark} {{"sufficient": <true|false>, "missing": "<what information is still missing, empty if sufficient>"}}"""


def _answer_images(docs, hits, pages, matched) -> list[tuple]:
    """Images for a vision answer: (doc, block, page, why), matched-by-vector first, then the
    figures inside the sections being read; capped at ANSWER_IMAGES."""
    if not config.ANSWER_IMAGES or not config.LLM_BACKENDS[llm.backend.get()].get("vision"):
        return []
    by_id = {d["id"]: d for d in docs}
    out, seen = [], set()

    def add(d, b, page, why):
        if (d["id"], b["id"]) not in seen and len(out) < config.ANSWER_IMAGES:
            seen.add((d["id"], b["id"]))
            out.append((d, b, page, why))

    for im in matched or []:
        d = by_id.get(im["doc_id"])
        page = pages.get(im["doc_id"], {}).get(im["page"])
        if d and page:
            b = next((x for x in page["blocks"] if x["id"] == im["image_id"]), None) or \
                {"id": im["image_id"], "type": "image", "bbox": im["bbox_pt"]}
            add(d, b, page, f"matched the question (score {im['score']})")
    for d, n, _ in hits:
        for b, page in node_images(d["id"], n, pages[d["id"]]):
            add(d, b, page, f"figure in section \"{n['title']}\"")
    return out


def _image_part(doc_id, image_id) -> dict:
    import base64
    data = imagemod.crop_path(doc_id, image_id).read_bytes()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


async def _answer(query, docs, hits, pages, tr, matched_images=None):
    didx = {d["id"]: i + 1 for i, d in enumerate(docs)}
    parts, used, block_map, n_blocks = [], 0, {}, 0
    for d, n, _ in hits:
        blocks = node_blocks(d["id"], n, pages[d["id"]])
        body, tok = [], 0
        for b, page, text in blocks:
            t = approx_tokens(text)
            if tok + t > NODE_TEXT_BUDGET or used + tok + t > CONTEXT_BUDGET:
                body.append("[... truncated]")
                break
            tag = f"{didx[d['id']]}:{b['id']}"
            block_map[tag] = (d, b, page, text)
            body.append(f"[{tag}] {text}")
            tok += t
            n_blocks += 1
        used += tok
        parts.append(f"=== Document {didx[d['id']]}: {d.get('rel_path') or d['name']} — section \"{n['title']}\" "
                     f"(pages {n['start_index']}-{n['end_index']})\n" + "\n".join(body))
    pics = _answer_images(docs, hits, pages, matched_images)
    image_parts = []
    for d, b, page, why in pics:
        tag = f"{didx[d['id']]}:{b['id']}"
        caption = page["text"][b.get("start", 0):b.get("end", 0)].strip()
        block_map[tag] = (d, b, page, "[image] " + caption)
        image_parts.append({"type": "text", "text": f"[{tag}] image on page {page['page']} of document "
                                                    f"{didx[d['id']]} ({d.get('rel_path') or d['name']}), {why}"
                                                    + (f"; caption: {caption[:200]}" if caption else "")})
        image_parts.append(_image_part(d["id"], b["id"]))
    await tr.event({"type": "read", "nodes": [_node_brief(d, n) for d, n, _ in hits], "blocks": n_blocks,
                    "context_tokens": used, "images": [f"{didx[d['id']]}:{b['id']}" for d, b, _, _ in pics],
                    "sample": [f"[{k}] {v[3][:90]}" for k, v in list(block_map.items())[:3]]})

    buf, sent = "", 0

    async def on_delta(piece):   # forward answer text, hold back the trailing status line
        nonlocal buf, sent
        buf += piece
        cut = buf.find(STATUS_MARK)
        limit = cut if cut >= 0 else max(sent, len(buf) - len(STATUS_MARK))
        if limit > sent:
            await tr.event({"type": "answer_delta", "text": buf[sent:limit]})
            sent = limit

    await tr.event({"type": "answer_start"})
    context = "\n\n".join(parts) or ("(No section of the searched documents looked relevant to the question. "
                                       "Say that the documents do not contain this information.)")
    note = ("Images from the documents follow the text, each preceded by its id; read them like the text "
            "(values in charts, labels, photos) and cite them the same way.\n") if image_parts else ""
    prompt = ANSWER_PROMPT.format(query=query, context=context, mark=STATUS_MARK, images_note=note)
    content = [{"type": "text", "text": prompt}, *image_parts] if image_parts else prompt
    out = await llm.chat([{"role": "user", "content": content}],
                         max_tokens=4000, label="作答" + (f" · {len(pics)} 张图" if pics else ""), on_delta=on_delta)
    ans, _, tail = out.partition(STATUS_MARK)
    ans = ans.strip()
    try:
        status = llm.parse_json(tail) if tail.strip() else {}
    except (ValueError, json.JSONDecodeError):
        status = {}
    cites = []
    for tag in dict.fromkeys(re.findall(r"\[?(\d+:p\d+_(?:b\d+|full))\]?", ans)):
        if tag in block_map:
            d, b, page, text = block_map[tag]
            cite = {"tag": tag, "doc_id": d["id"], "doc_name": d["name"], "block_id": b["id"],
                    "page": page["page"], "page_label": page.get("label"), "bbox": bbox_k(b, page),
                    "text": text[:500]}
            if text.startswith("[image]"):
                cite["image_url"] = f"/v1/documents/{d['id']}/images/{b['id']}"
            cites.append(cite)
    res = {"answer": ans, "citations": cites, "sufficient": bool(status.get("sufficient", True)),
           "missing": status.get("missing", "")}
    await tr.event({"type": "answer_done", **res})
    return res
