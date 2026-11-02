"""Indexing pipeline and the background job queue.

Queue items are document ids ("pi-…") or collection ids ("pc-…"); a collection job writes the
directory summaries once all of its files are indexed.
"""

import asyncio
import logging
import time
import traceback

from . import collection, config, images, llm, parse, store, tree

log = logging.getLogger("pidx.indexer")

_queue: asyncio.Queue | None = None
_coll_pending: set[str] = set()


def node_embed_text(doc: dict, path: list[str], n: dict, pages=None) -> tuple[str, str]:
    """What a node vector represents, as (title, text): where it sits (doc, folder, title path) and
    what it says. llm.embed joins them for bge-m3 exactly as before, or prefixes them for Gemma."""
    where = doc["name"] if not doc.get("rel_path") else doc["rel_path"]
    s = n.get("summary") or n.get("prefix_summary") or ""
    if not s and pages is not None:
        s = tree.text_between(pages, tuple(n["start"]), tuple(n["text_end"]) if n.get("text_end") else None)[:600]
    return f"{where}\n{' > '.join(path + [n['title']])}", s.strip()


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


async def embed_images(doc: dict, nodes: list, pages: list) -> int:
    """EmbeddingGemma path: crop image blocks and embed the crops. Returns the count (0 when off)."""
    if not config.EMBED_MULTIMODAL:
        return 0
    found = await asyncio.to_thread(images.extract, doc, pages, nodes)
    vecs = []
    for i in range(0, len(found), 8):
        v = await llm.embed_images([f["png"] for f in found[i:i + 8]])
        if v is None:
            return 0
        vecs.extend(v)
    store.save_image_vectors(doc["id"], [{**f, "vec": v} for f, v in zip(found, vecs)])
    return len(vecs)


async def embed_document(doc: dict, nodes: list, pages: list, description: str) -> dict:
    """All vectors of a document under the current embedding model: description, nodes, images."""
    where = doc.get("rel_path") or doc["name"]
    emb = await llm.embed([(where, description or "")])
    n_vec = await embed_nodes(doc, nodes, pages)
    n_img = await embed_images(doc, nodes, pages) if n_vec else 0
    return {"embedding": emb[0] if emb else None, "node_vectors": n_vec, "image_vectors": n_img,
            "embed_model": llm.embed_model() if n_vec else None}


async def index_document(doc_id: str) -> None:
    doc = store.get_document(doc_id)
    opts = doc["options"] or {}
    store.update_document(doc_id, status="processing", error=None)
    t0 = time.time()
    used = llm.new_scope()
    parsed = await asyncio.to_thread(parse.parse_file, doc["path"], doc["format"], opts.get("mode", "auto"),
                                     doc.get("pdf_path"))
    pages = parsed["pages"]
    store.save_pages(doc_id, pages)
    store.update_document(doc_id, page_count=len(pages))
    t1 = time.time()

    nodes, info = await tree.build_tree(parsed)
    t2 = time.time()
    if opts.get("summary", True):
        await tree.summarize(nodes, pages, doc["name"])
    description = await tree.describe(nodes, pages, doc["name"])
    t3 = time.time()

    store.clear_vectors(doc_id)
    ev = await embed_document(doc, nodes, pages, description)
    t4 = time.time()
    info.update(parser=parsed["parser"], format=doc["format"], node_vectors=ev["node_vectors"],
                image_vectors=ev["image_vectors"], embed_model=ev["embed_model"],
                seconds={"parse": round(t1 - t0, 1), "tree": round(t2 - t1, 1), "summary": round(t3 - t2, 1),
                         "embed": round(t4 - t3, 1)},
                llm=used)
    store.update_document(doc_id, status="completed", tree=nodes, info=info, description=description,
                          embedding=ev["embedding"], embed_model=ev["embed_model"])
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
                store.update_document(job, status="failed", error=repr(e)[:2000])
        finally:
            if not job.startswith("pc-"):
                try:
                    _after_document(job)
                except Exception:
                    log.exception("collection bookkeeping for %s", job)
            _queue.task_done()


async def _backfill_vectors():
    """Documents without vectors, or with vectors from another embedding model (bge-m3 <-> Gemma),
    get them recomputed now. Cheap: embeddings only, the trees and summaries are kept. Until then
    search treats such a document as having no vectors (pure tree reading)."""
    model = llm.embed_model()
    for doc_id in store.docs_needing_vectors(model):
        d = store.get_document(doc_id, full=True)
        try:
            pages = store.get_pages(doc_id)
            store.clear_vectors(doc_id)
            ev = await embed_document(d, d["tree"], pages, d.get("description") or "")
        except Exception:
            log.exception("vector backfill %s", doc_id)
            continue
        if not ev["node_vectors"]:
            return          # embedding service unavailable; try again on next start
        store.update_document(doc_id, embedding=ev["embedding"], embed_model=ev["embed_model"])
        log.info("backfilled %s: %d node + %d image vectors (%s)", d["name"], ev["node_vectors"],
                 ev["image_vectors"], model)


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
