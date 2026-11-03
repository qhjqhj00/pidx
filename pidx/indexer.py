"""Indexing pipeline and the background job queue.

Queue items are document ids ("pi-…") or collection ids ("pc-…"); a collection job writes the
directory summaries once all of its files are indexed.
"""

import asyncio
import logging
import time
import traceback

from . import collection, config, llm, parse, progress, store, tree

log = logging.getLogger("pidx.indexer")

_queue: asyncio.Queue | None = None
_coll_pending: set[str] = set()


def node_embed_text(doc: dict, path: list[str], n: dict, pages=None) -> str:
    """What a node vector represents: where it sits (doc, folder, title path) and what it says."""
    where = doc["name"] if not doc.get("rel_path") else doc["rel_path"]
    s = n.get("summary") or n.get("prefix_summary") or ""
    if not s and pages is not None:
        s = tree.text_between(pages, tuple(n["start"]), tuple(n["text_end"]) if n.get("text_end") else None)[:600]
    return f"{where}\n{' > '.join(path + [n['title']])}\n{s}".strip()


async def embed_nodes(doc: dict, nodes: list, pages=None) -> int:
    """One bge-m3 vector per tree node. Returns the count (0 if the embedding service is off)."""
    items = []

    def visit(ns, path):
        for n in ns:
            items.append((n["node_id"], node_embed_text(doc, path, n, pages)))
            visit(n["nodes"], path + [n["title"]])

    visit(nodes, [])
    if not items:
        return 0
    vecs = []
    for i in range(0, len(items), 32):
        v = await llm.embed([t for _, t in items[i:i + 32]])
        if v is None:
            return 0
        vecs.extend(v)
    store.save_node_vectors(doc["id"], [(nid, v) for (nid, _), v in zip(items, vecs)])
    return len(vecs)


def page_nodes(nodes) -> dict[int, tuple[str, list[str]]]:
    """page -> (node_id, title path) of the node that best describes it: the narrowest page range
    covering it, then the deepest, then the first. (A one-page Markdown file has many nodes on page 1;
    its root heading, not its last section, describes the page.)"""
    best = {}

    def visit(ns, path):
        for n in ns:
            p = path + [n["title"]]
            key = (n["end_index"] - n["start_index"], -len(p))
            for pg in range(n["start_index"], n["end_index"] + 1):
                if pg not in best or key < best[pg][0]:
                    best[pg] = (key, n["node_id"], p)
            visit(n["nodes"], p)

    visit(nodes, [])
    return {pg: (nid, p) for pg, (_, nid, p) in best.items()}


def page_embed_text(doc: dict, page: dict, path: list[str]) -> str:
    where = doc.get("rel_path") or doc["name"]
    head = page.get("label") or f"p.{page['page']}"
    return f"{where} · {head}\n{' > '.join(path)}\n{(page.get('text') or '')[:3000]}".strip()


async def embed_pages(doc: dict, nodes: list, pages: list) -> int:
    """One bge-m3 vector per page, for page search. Returns the count (0 if the service is off)."""
    paths = page_nodes(nodes)
    items = [(p["page"], page_embed_text(doc, p, paths.get(p["page"], (None, []))[1])) for p in pages]
    vecs = []
    for i in range(0, len(items), 32):
        v = await llm.embed([t for _, t in items[i:i + 32]])
        if v is None:
            return 0
        vecs.extend(v)
    store.save_page_vectors(doc["id"], [(pg, v) for (pg, _), v in zip(items, vecs)])
    return len(vecs)


async def index_document(doc_id: str) -> None:
    doc = store.get_document(doc_id)
    opts = doc["options"] or {}
    store.update_document(doc_id, status="processing", error=None)
    t0 = time.time()
    used = llm.new_scope()
    progress.begin(doc_id, used)
    progress.stage("parse")
    parsed = await asyncio.to_thread(parse.parse_file, doc["path"], doc["format"], opts.get("mode", "auto"),
                                     doc.get("pdf_path"))
    pages = parsed["pages"]
    store.save_pages(doc_id, pages)
    store.update_document(doc_id, page_count=len(pages))
    t1 = time.time()

    progress.stage("tree", {"pages": len(pages)})
    nodes, info = await tree.build_tree(parsed)
    t2 = time.time()
    progress.stage("summary", {"nodes": info["nodes"]})
    if opts.get("summary", True):
        await tree.summarize(nodes, pages, doc["name"])
    description = await tree.describe(nodes, pages, doc["name"])
    t3 = time.time()

    progress.stage("embed", {"nodes": info["nodes"]})
    where = doc.get("rel_path") or doc["name"]
    emb = await llm.embed([f"{where}\n{description}"])
    n_vec = await embed_nodes(doc, nodes, pages)
    n_pvec = await embed_pages(doc, nodes, pages)
    t4 = time.time()
    info.update(parser=parsed["parser"], format=doc["format"], node_vectors=n_vec, page_vectors=n_pvec,
                seconds={"parse": round(t1 - t0, 1), "tree": round(t2 - t1, 1), "summary": round(t3 - t2, 1),
                         "embed": round(t4 - t3, 1)},
                llm=used, stages=progress.end(doc_id))
    store.update_document(doc_id, status="completed", tree=nodes, info=info, description=description,
                          embedding=emb[0] if emb else None)
    log.info("indexed %s (%s, %d pages, %d nodes) in %.0fs", doc_id, doc["name"], len(pages), info["nodes"], t4 - t0)


def _after_document(doc_id: str):
    """Queue the collection's directory summaries once its last file is done."""
    d = store.get_document(doc_id)
    cid = d and d.get("collection_id")
    if not cid or cid in _coll_pending:
        return
    counts = store.collection_status_counts(cid)
    if not counts.get("queued") and not counts.get("processing"):
        _coll_pending.add(cid)
        _queue.put_nowait(cid)


async def _worker(n: int):
    while True:
        job = await _queue.get()
        try:
            if job.startswith("pc-"):
                _coll_pending.discard(job)
                await collection.summarize(job)
            else:
                await index_document(job)
        except Exception as e:
            log.error("job %s failed: %s", job, traceback.format_exc())
            if job.startswith("pc-"):
                store.update_collection(job, status="failed", error=repr(e)[:2000])
            else:
                stages = progress.end(job)
                d = store.get_document(job) or {}
                info = {**(d.get("info") or {}), "stages": stages, "failed_stage": stages[-1][0] if stages else None}
                store.update_document(job, status="failed", error=repr(e)[:2000], info=info)
        finally:
            if not job.startswith("pc-"):
                try:
                    _after_document(job)
                except Exception:
                    log.exception("collection bookkeeping for %s", job)
            _queue.task_done()


async def _backfill_vectors():
    """Documents indexed before node vectors existed get them now (cheap: embeddings only)."""
    for doc_id in store.docs_without_vectors():
        d = store.get_document(doc_id, full=True)
        try:
            n = await embed_nodes(d, d["tree"], store.get_pages(doc_id))
        except Exception:
            log.exception("vector backfill %s", doc_id)
            continue
        if n == 0:
            return          # embedding service unavailable; try again on next start
        log.info("backfilled %d node vectors for %s", n, d["name"])
    for doc_id in store.docs_without_page_vectors():
        d = store.get_document(doc_id, full=True)
        try:
            n = await embed_pages(d, d["tree"], store.get_pages(doc_id))
        except Exception:
            log.exception("page vector backfill %s", doc_id)
            continue
        if n == 0:
            return
        log.info("backfilled %d page vectors for %s", n, d["name"])


def start_workers():
    global _queue
    _queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    for n in range(config.INDEX_WORKERS):
        loop.create_task(_worker(n))
    # resume jobs interrupted by a restart
    for status in ("queued", "processing"):
        docs, _ = store.list_documents(limit=100000, status=status)
        for d in reversed(docs):
            _queue.put_nowait(d["id"])
    for c in store.list_collections():
        if c["status"] == "processing":
            counts = store.collection_status_counts(c["id"])
            if not counts.get("queued") and not counts.get("processing"):
                _coll_pending.add(c["id"])
                _queue.put_nowait(c["id"])
    loop.create_task(_backfill_vectors())


def enqueue(job_id: str):
    _queue.put_nowait(job_id)
