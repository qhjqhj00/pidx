"""Indexing pipeline and the background job queue."""

import asyncio
import logging
import time
import traceback

from . import config, llm, parse, store, tree

log = logging.getLogger("pidx.indexer")

_queue: asyncio.Queue | None = None


async def index_document(doc_id: str) -> None:
    doc = store.get_document(doc_id)
    opts = doc["options"] or {}
    store.update_document(doc_id, status="processing", error=None)
    t0 = time.time()
    used = llm.new_scope()
    parsed = await asyncio.to_thread(parse.parse_pdf, doc["path"], opts.get("mode", "auto"))
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

    emb = await llm.embed([f"{doc['name']}\n{description}"])
    info.update(parser=parsed["parser"], seconds={"parse": round(t1 - t0, 1), "tree": round(t2 - t1, 1),
                                                  "summary": round(t3 - t2, 1)},
                llm=used)
    store.update_document(doc_id, status="completed", tree=nodes, info=info, description=description,
                          embedding=emb[0] if emb else None)
    log.info("indexed %s (%s, %d pages, %d nodes) in %.0fs", doc_id, doc["name"], len(pages), info["nodes"], t3 - t0)


async def _worker(n: int):
    while True:
        doc_id = await _queue.get()
        try:
            await index_document(doc_id)
        except Exception as e:
            log.error("index %s failed: %s", doc_id, traceback.format_exc())
            store.update_document(doc_id, status="failed", error=repr(e)[:2000])
        finally:
            _queue.task_done()


def start_workers():
    global _queue
    _queue = asyncio.Queue()
    for n in range(config.INDEX_WORKERS):
        asyncio.get_running_loop().create_task(_worker(n))
    # resume jobs interrupted by a restart
    for status in ("queued", "processing"):
        docs, _ = store.list_documents(limit=10000, status=status)
        for d in reversed(docs):
            _queue.put_nowait(d["id"])


def enqueue(doc_id: str):
    _queue.put_nowait(doc_id)
