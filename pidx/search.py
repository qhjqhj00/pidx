"""Reasoning-based retrieval over document trees (no chunking, no vector store).

1. Pick documents: embedding prefilter (bge-m3, only for big libraries) + LLM over doc descriptions.
2. Tree search: the LLM reads each tree's skeleton (titles, pages, summaries) and names the
   nodes to read; trees too big for one prompt are explored level by level.
3. Read: node text is assembled from layout blocks, each tagged with its block id.
4. Optional answer with block-level citations; if the LLM says the evidence is
   insufficient, it describes what is missing and another tree-search round runs.
"""

import asyncio
import json
import math
import re
import time

from . import config, llm, store, tree
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

def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    return num / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


async def select_docs(query, docs, max_docs=5):
    if len(docs) <= 1:
        return docs, ""
    if len(docs) > 20:
        qv = await llm.embed([query])
        if qv:
            docs = sorted(docs, key=lambda d: -_cos(qv[0], d["embedding"]) if d.get("embedding") else 0)[:20]
        else:
            docs = docs[:20]
    listing = "\n".join(f"[{i}] {d['name']} ({d['page_count']} pages): {d.get('description') or ''}"
                        for i, d in enumerate(docs))
    js = await chat_json(f"""Question: {query}

Documents in the library:
{listing}

Which documents should be searched to answer the question? Pick all that are needed (e.g. several for comparisons), at most {max_docs}, most relevant first.
Return JSON {{"thinking": "...", "docs": [<index>, ...]}}""", max_tokens=1000, label="选文档")
    picked = [docs[i] for i in js.get("docs", []) if isinstance(i, int) and 0 <= i < len(docs)]
    return picked or docs[:max_docs], js.get("thinking", "")


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


def bbox_k(b, page):
    """bbox in thousandths of page width/height, origin top-left (PageIndex convention)."""
    x0, y0, x1, y1 = b["bbox"]
    w, h = page["width"], page["height"]
    return [round(x0 / w * 1000), round(y0 / h * 1000), round(x1 / w * 1000), round(y1 / h * 1000)]


def _node_brief(d, n):
    return {"doc_id": d["id"], "doc_name": d["name"], "node_id": n["node_id"], "title": n["title"],
            "start_index": n["start_index"], "end_index": n["end_index"]}


async def search(query: str, doc_ids=None, top_k=5, answer=False, max_rounds=2, include_text=True, emit=None,
                 llm_backend=None):
    """See module docstring. `emit` (async callable) receives progress events for streaming UIs."""
    t0 = time.time()
    llm.backend.set(llm_backend or config.SEARCH_BACKEND)
    used = llm.new_scope()
    tr = llm.Tracer(emit)
    llm.tracer.set(tr)
    phases = []          # (name, start, end) wall-clock spans

    # any doc with a tree is searchable, including one being re-indexed (old tree until the new one lands)
    if doc_ids:
        docs = [store.get_document(i, full=True) for i in doc_ids]
    else:
        docs, _ = store.list_documents(limit=10000, full=True)
    docs = [d for d in docs if d and d.get("tree")]
    if not docs:
        return {"query": query, "error": "no completed documents to search", "nodes": []}
    await tr.event({"type": "start", "query": query, "llm": config.LLM_BACKENDS[llm.backend.get()]["label"], "library": [{"doc_id": d["id"], "name": d["name"]} for d in docs]})

    t = time.time()
    n_lib = len(docs)
    docs, why = await select_docs(query, docs)
    phases.append(("选文档", t - t0, time.time() - t0))
    await tr.event({"type": "docs", "skipped": n_lib <= 1, "thinking": why,
                    "selected": [{"doc_id": d["id"], "name": d["name"], "description": d.get("description")} for d in docs]})

    pages = {d["id"]: {p["page"]: p for p in store.get_pages(d["id"])} for d in docs}
    hits = []          # (doc, node, thinking)
    reasoning = {}
    extra = ""
    result = {}
    for rnd in range(max_rounds if answer else 1):
        seen = {(d["id"], n["node_id"]) for d, n, _ in hits}

        async def one(d):
            th, nodes, how = await tree_search(query, d, top_k, exclude={nid for did, nid in seen if did == d["id"]},
                                               extra=extra)
            await tr.event({"type": "tree_search", "round": rnd + 1, "doc_id": d["id"], "doc_name": d["name"],
                            "thinking": th, "how": how, "nodes": [_node_brief(d, n) for n in nodes]})
            return th, nodes

        t = time.time()
        await tr.event({"type": "tree_search_start", "round": rnd + 1, "docs": [d["name"] for d in docs]})
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
        result = await _answer(query, docs, hits, pages, tr)
        phases.append((f"作答 · 第 {rnd + 1} 轮", t - t0, time.time() - t0))
        if result.get("sufficient", True) or not result.get("missing"):
            break
        await tr.event({"type": "insufficient", "round": rnd + 1, "missing": result["missing"]})
        extra = ("Already read (not sufficient): " + "; ".join(n["title"] for _, n, _ in hits) +
                 f"\nStill missing: {result['missing']}\nPoint to OTHER nodes that may contain the missing information.")

    out_nodes = []
    for d, n, _ in hits:
        item = {**_node_brief(d, n), "summary": n.get("summary") or n.get("prefix_summary") or ""}
        if include_text:
            item["text"] = tree.text_between(list(pages[d["id"]][p] for p in sorted(pages[d["id"]])),
                                             tuple(n["start"]), tuple(n["end"]) if n["end"] else None)
        out_nodes.append(item)
    res = {"query": query, "documents": [{"doc_id": d["id"], "name": d["name"]} for d in docs],
           "reasoning": reasoning, "nodes": out_nodes}
    if answer:
        res.update(answer=result.get("answer", ""), citations=result.get("citations", []),
                   sufficient=result.get("sufficient", True))
    res["llm"] = llm.backend.get()
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
Be precise with numbers, units and periods. Answer in the language of the question. If the excerpts do not contain enough information, answer what you can and say what is missing.

Question: {query}

{context}

Write the answer in markdown first. Then, on the very last line, write {mark} followed by JSON:
{mark} {{"sufficient": <true|false>, "missing": "<what information is still missing, empty if sufficient>"}}"""


async def _answer(query, docs, hits, pages, tr):
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
        parts.append(f"=== Document {didx[d['id']]}: {d['name']} — section \"{n['title']}\" "
                     f"(pages {n['start_index']}-{n['end_index']})\n" + "\n".join(body))
    await tr.event({"type": "read", "nodes": [_node_brief(d, n) for d, n, _ in hits], "blocks": n_blocks,
                    "context_tokens": used, "sample": [f"[{k}] {v[3][:90]}" for k, v in list(block_map.items())[:3]]})

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
    out = await llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(query=query, context="\n\n".join(parts),
                                                                          mark=STATUS_MARK)}],
                         max_tokens=4000, label="作答", on_delta=on_delta)
    ans, _, tail = out.partition(STATUS_MARK)
    ans = ans.strip()
    try:
        status = llm.parse_json(tail) if tail.strip() else {}
    except (ValueError, json.JSONDecodeError):
        status = {}
    cites = []
    for tag in dict.fromkeys(re.findall(r"\[?(\d+:p\d+_b\d+)\]?", ans)):
        if tag in block_map:
            d, b, page, text = block_map[tag]
            cites.append({"tag": tag, "doc_id": d["id"], "doc_name": d["name"], "block_id": b["id"],
                          "page": page["page"], "bbox": bbox_k(b, page), "text": text[:500]})
    res = {"answer": ans, "citations": cites, "sufficient": bool(status.get("sufficient", True)),
           "missing": status.get("missing", "")}
    await tr.event({"type": "answer_done", **res})
    return res
