"""HTTP API.

  POST   /v1/documents                     upload a file (PDF / image / Office) -> {doc_id}; indexing runs in the background
  GET    /v1/documents                     list
  GET    /v1/documents/{id}                info + status
  DELETE /v1/documents/{id}
  POST   /v1/documents/{id}/reindex        ?mode=&summary=
  GET    /v1/documents/{id}/tree           ?node_summary=&include_text=
  GET    /v1/documents/{id}/pages          ?pages=5-7,9
  GET    /v1/documents/{id}/blocks/{bid}   one layout block with bbox
  GET    /v1/documents/{id}/file           the original file
  POST   /v1/collections                   upload a zip -> {collection_id, doc_ids}; one document per supported file
  GET    /v1/collections[/{id}]            list / folder tree with per-file status
  DELETE /v1/collections/{id}              the collection and all its documents
  GET    /v1/documents/{id}/pages/{n}/image?highlight=p3_b1,p3_b4
  POST   /v1/search                        {query, doc_ids?, top_k?, answer?, max_rounds?, include_text?}
  POST   /v1/search/stream                 same body, Server-Sent Events (progress + answer deltas + result)
"""

import asyncio
import hmac
import json
import logging
import mimetypes
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import collection, config, indexer, pagesearch, parse, progress, search as searchmod, store, tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app):
    config.FILES_DIR.mkdir(parents=True, exist_ok=True)
    indexer.start_workers()
    yield


app = FastAPI(title="pidx — tree-index RAG", lifespan=lifespan)
WEB = Path(__file__).parent / "web"
NO_CACHE = {"Cache-Control": "no-cache"}   # pages change with releases; always revalidate (ETag)

DENIED = """<!doctype html><meta charset="utf-8"><title>pidx</title>
<body style="font:15px system-ui;padding:48px;color:#444">需要访问 token：请使用带 <code>?token=…</code> 的完整链接打开。</body>"""


def _token_ok(given: str | None) -> bool:
    return bool(given) and hmac.compare_digest(given, config.PUBLIC_TOKEN)


@app.middleware("http")
async def token_gate(request: Request, call_next):
    """Gate traffic that comes in through the public proxy; tailnet clients go straight through."""
    if not config.PUBLIC_TOKEN or "x-forwarded-for" not in request.headers:
        return await call_next(request)
    q = request.query_params.get("token")
    if _token_ok(q):
        resp = await call_next(request)
        resp.set_cookie("pidx_token", q, max_age=30 * 86400, httponly=True, secure=True, samesite="lax")
        return resp
    bearer = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if _token_ok(request.cookies.get("pidx_token")) or _token_ok(bearer):
        return await call_next(request)
    if request.url.path.startswith("/v1/"):
        return Response('{"detail":"token required"}', status_code=401, media_type="application/json")
    return HTMLResponse(DENIED, status_code=401)


def _doc_or_404(doc_id, full=False):
    d = store.get_document(doc_id, full=full)
    if not d:
        raise HTTPException(404, f"document {doc_id} not found")
    return d


def _public(d):
    return {"id": d["id"], "name": d["name"], "status": d["status"], "error": d.get("error"),
            "pageNum": d.get("page_count"), "description": d.get("description"), "metadata": d.get("metadata"),
            "createdAt": d.get("created_at"), "info": d.get("info"), "format": d.get("format"),
            "pageImages": bool(d.get("pdf_path")), "collectionId": d.get("collection_id"), "relPath": d.get("rel_path"),
            "progress": progress.get(d["id"]) if d["status"] == "processing" else None}


def parse_pages(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


@app.post("/v1/documents")
async def submit_document(file: UploadFile = File(...), mode: str = Form("auto"), summary: bool = Form(True),
                          metadata: str = Form("{}")):
    fname = Path(file.filename or "").name
    ext = Path(fname).suffix.lower()
    if ext == ".zip":
        raise HTTPException(400, "zip files go to POST /v1/collections")
    fmt = parse.file_format(fname)
    if not fmt:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'}; supported: "
                                 + " ".join(sorted(config.SUPPORTED_EXTS)) + " (and .zip via /v1/collections)")
    name, stem, n = fname, fname[:-len(ext)], 1
    while store.find_by_name(name):
        name = f"{stem}_{n}{ext}"
        n += 1
    doc_id = store.new_doc_id()
    path = config.FILES_DIR / f"{doc_id}{ext}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    pdf_path = None
    try:
        parse.check_file(str(path), fmt)
        if fmt == "image":
            pdf_path = str(config.FILES_DIR / f"{doc_id}.pdf")
            parse.image_to_pdf(str(path), pdf_path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    store.create_document(doc_id, name, str(path), json.loads(metadata or "{}"), {"mode": mode, "summary": summary},
                          format=fmt, pdf_path=pdf_path or (str(path) if fmt == "pdf" else None))
    indexer.enqueue(doc_id)
    return {"doc_id": doc_id, "name": name, "format": fmt, "status": "queued"}


@app.post("/v1/collections")
async def submit_collection(file: UploadFile = File(...), summary: bool = Form(True)):
    fname = Path(file.filename or "upload.zip").name
    if not fname.lower().endswith(".zip"):
        raise HTTPException(400, "upload a .zip file")
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=config.DATA_DIR, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        cid, doc_ids, info = await asyncio.to_thread(collection.ingest_zip, tmp.name, fname[:-4],
                                                     {"mode": "auto", "summary": summary})
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    for d in doc_ids:
        indexer.enqueue(d)
    return {"collection_id": cid, "doc_ids": doc_ids, "files": info["files"], "skipped": info["skipped"],
            "status": "processing"}


def _coll_public(c, with_tree=False):
    counts = store.collection_status_counts(c["id"])
    o = {"id": c["id"], "name": c["name"], "status": c["status"], "error": c.get("error"), "info": c.get("info"),
         "counts": counts, "createdAt": c.get("created_at")}
    if with_tree:
        docs = {d["id"]: d for d in store.list_documents(limit=100000, collection_id=c["id"])[0]}

        def conv(n):
            if n["type"] == "file":
                d = docs.get(n["doc_id"]) or {}
                return {**n, "status": d.get("status"), "pageNum": d.get("page_count"), "format": d.get("format"),
                        "description": d.get("description")}
            return {**n, "nodes": [conv(x) for x in n["nodes"]]}

        o["tree"] = conv(c["tree"])
    return o


@app.get("/v1/collections")
def list_collections():
    return {"collections": [_coll_public(c) for c in store.list_collections()]}


@app.get("/v1/collections/{cid}")
def get_collection(cid: str):
    c = store.get_collection(cid)
    if not c:
        raise HTTPException(404, f"collection {cid} not found")
    return _coll_public(c, with_tree=True)


@app.get("/v1/collections/{cid}/progress")
def collection_progress(cid: str):
    """Everything the processing timeline needs, per file: stage timestamps, live stage + detail, outcome."""
    c = store.get_collection(cid)
    if not c:
        raise HTTPException(404, f"collection {cid} not found")
    docs = {d["id"]: d for d in store.list_documents(limit=100000, collection_id=cid)[0]}
    files = []
    for n in collection.walk(c["tree"]):
        if n["type"] != "file" or n["doc_id"] not in docs:
            continue
        d = docs[n["doc_id"]]
        i = d.get("info") or {}
        live = progress.get(d["id"]) if d["status"] == "processing" else None
        files.append({"doc_id": d["id"], "path": n["path"], "name": d["name"], "format": d["format"],
                      "status": d["status"], "error": d.get("error"), "queued_at": d.get("created_at"),
                      "pages": d.get("page_count"), "nodes": i.get("nodes"), "vectors": i.get("node_vectors"),
                      "parser": i.get("parser"), "method": i.get("method"), "seconds": i.get("seconds"),
                      "vision": i.get("vision"),
                      "llm": (live or {}).get("llm") or i.get("llm"),
                      "stages": (live or {}).get("stages") or i.get("stages") or [],
                      "stage": (live or {}).get("stage"), "detail": (live or {}).get("detail")})
    info = c.get("info") or {}
    live = progress.get(cid)
    return {"id": cid, "name": c["name"], "status": c["status"], "error": c.get("error"),
            "created_at": c.get("created_at"), "now": time.time(), "workers": config.INDEX_WORKERS,
            "ingest": info.get("ingest"), "zip_bytes": info.get("zip_bytes"), "dirs": info.get("dirs"),
            "skipped": info.get("skipped") or [], "files": files,
            "dirs_stage": (live or {}).get("stages") or info.get("stages") or [],
            "dirs_detail": (live or {}).get("detail")}


@app.delete("/v1/collections/{cid}")
def delete_collection(cid: str):
    if not store.get_collection(cid):
        raise HTTPException(404, f"collection {cid} not found")
    for d in store.list_documents(limit=100000, collection_id=cid)[0]:
        _delete_files(d)
        store.delete_document(d["id"])
    store.delete_collection(cid)
    return {"message": "Collection deleted successfully."}


def _delete_files(d):
    for p in {d["path"], d.get("pdf_path")}:
        if p:
            Path(p).unlink(missing_ok=True)


@app.get("/v1/documents")
def list_documents(limit: int = Query(50, ge=1, le=10000), offset: int = 0, collection_id: str | None = None):
    docs, total = store.list_documents(limit, offset, collection_id=collection_id)
    return {"documents": [_public(d) for d in docs], "total": total, "limit": limit, "offset": offset}


@app.get("/v1/documents/{doc_id}")
def get_document(doc_id: str):
    return _public(_doc_or_404(doc_id))


@app.delete("/v1/documents/{doc_id}")
def delete_document(doc_id: str):
    d = _doc_or_404(doc_id)
    store.delete_document(doc_id)
    _delete_files(d)
    return {"message": "Document deleted successfully."}


@app.post("/v1/documents/{doc_id}/reindex")
def reindex_document(doc_id: str, mode: str | None = None, summary: bool | None = None):
    d = _doc_or_404(doc_id)
    opts = d["options"] or {}
    if mode is not None:
        opts["mode"] = mode
    if summary is not None:
        opts["summary"] = summary
    store.update_document(doc_id, status="queued", error=None, options=opts)
    indexer.enqueue(doc_id)
    return {"doc_id": doc_id, "status": "queued"}


@app.get("/v1/documents/{doc_id}/tree")
def get_tree(doc_id: str, node_summary: bool = False, include_text: bool = False):
    d = _doc_or_404(doc_id, full=True)
    if not d["tree"]:   # a re-indexing doc keeps serving its previous tree
        return {"doc_id": doc_id, "status": d["status"]}
    pages = store.get_pages(doc_id) if include_text else None

    def conv(n):
        o = {"title": n["title"], "node_id": n["node_id"], "page_index": n["start_index"],
             "start_index": n["start_index"], "end_index": n["end_index"]}
        if node_summary:
            o["summary" if not n["nodes"] else "prefix_summary"] = n.get("summary") or n.get("prefix_summary") or ""
        if include_text:
            o["text"] = tree.text_between(pages, tuple(n["start"]), tuple(n["text_end"]) if n["text_end"] else None)
        if n["nodes"]:
            o["nodes"] = [conv(c) for c in n["nodes"]]
        return o

    return {"doc_id": doc_id, "status": d["status"], "result": [conv(n) for n in d["tree"]]}


@app.get("/v1/documents/{doc_id}/pages")
def get_page_content(doc_id: str, pages: str = Query(..., description='e.g. "5-7" or "3,8"')):
    _doc_or_404(doc_id)
    return {"doc_id": doc_id, "pages": [{"page_index": p["page"], "label": p.get("label"), "markdown": p["text"]}
                                        for p in store.get_pages(doc_id, parse_pages(pages))]}


@app.get("/v1/documents/{doc_id}/blocks/{block_id}")
def get_block(doc_id: str, block_id: str):
    _doc_or_404(doc_id)
    try:
        pno = int(block_id.split("_")[0][1:])
    except ValueError:
        raise HTTPException(400, "block id looks like p3_b5")
    for page in store.get_pages(doc_id, [pno]):
        for b in page["blocks"]:
            if b["id"] == block_id:
                return {"doc_id": doc_id, "page": pno, "block_id": block_id, "bbox": searchmod.bbox_k(b, page),
                        "block_type": b["type"], "text": page["text"][b["start"]:b["end"]]}
    raise HTTPException(404, "block not found")


@app.get("/v1/documents/{doc_id}/file")
def get_file(doc_id: str):
    d = _doc_or_404(doc_id)
    return FileResponse(d["path"], media_type=mimetypes.guess_type(d["name"])[0] or "application/octet-stream",
                        filename=d["name"], content_disposition_type="inline")


@app.get("/v1/documents/{doc_id}/pages/{page}/image")
def page_image(doc_id: str, page: int, highlight: str = "", zoom: float = 1.5):
    d = _doc_or_404(doc_id)
    if not d.get("pdf_path"):
        raise HTTPException(404, f"no page images for {d['format']} documents")
    pdf = pymupdf.open(d["pdf_path"])
    if not 1 <= page <= pdf.page_count:
        raise HTTPException(404, "page out of range")
    pg = pdf[page - 1]
    ids = {h for h in highlight.split(",") if h}
    if ids:
        for p in store.get_pages(doc_id, [page]):
            for b in p["blocks"]:
                if b["id"] in ids:
                    r = pymupdf.Rect(b["bbox"]) + (-2, -2, 2, 2)
                    pg.draw_rect(r, color=(0.95, 0.55, 0), fill=(1, 0.85, 0.2), fill_opacity=0.25, width=1.5)
    png = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=300"})


class PageSearchReq(BaseModel):
    query: str | list[str]                 # one question, or up to 10
    doc_ids: list[str] | None = None
    collection_id: str | None = None
    top_k: int = 10
    metadata: dict | None = None           # exact-match filter on document metadata


@app.post("/v1/pages/search")
async def page_search(req: PageSearchReq):
    """Rank pages by embeddings only (no LLM): ~0.3 s, returns scored pages to read next."""
    qs = [req.query] if isinstance(req.query, str) else req.query
    qs = [q.strip() for q in qs if q and q.strip()]
    if not qs or len(qs) > 10:
        raise HTTPException(400, "give 1 to 10 non-empty questions")
    if not 1 <= req.top_k <= 50:
        raise HTTPException(400, "top_k must be 1..50")
    r = await pagesearch.page_search(qs, req.doc_ids, req.collection_id, req.top_k, req.metadata)
    if r.get("error"):
        raise HTTPException(503 if "unavailable" in r["error"] else 404, r["error"])
    return r


class SearchReq(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    top_k: int = 5
    answer: bool = False
    max_rounds: int = 2
    include_text: bool = True
    llm: str | None = None      # backend name, see GET /v1/llm; default PIDX_SEARCH_BACKEND
    collection_id: str | None = None   # search only the files of one zip collection
    retrieval: str | None = None       # "hybrid" (vectors recall subtrees) | "tree" (pure LLM); default PIDX_RETRIEVAL


def _check_llm(name, retrieval=None):
    if retrieval and retrieval not in ("hybrid", "tree"):
        raise HTTPException(400, "retrieval must be 'hybrid' or 'tree'")
    if name and name not in config.LLM_BACKENDS:
        raise HTTPException(400, f"unknown llm backend {name!r}; available: {list(config.LLM_BACKENDS)}")


@app.get("/v1/llm")
def llm_backends():
    return {"default": config.SEARCH_BACKEND,
            "backends": [{"name": k, "label": v["label"], "model": v["model"]} for k, v in config.LLM_BACKENDS.items()]}


@app.post("/v1/search")
async def search(req: SearchReq):
    _check_llm(req.llm, req.retrieval)
    return await searchmod.search(req.query, req.doc_ids, req.top_k, req.answer, req.max_rounds, req.include_text,
                                  llm_backend=req.llm, collection_id=req.collection_id, retrieval=req.retrieval)


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def static_no_cache(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.post("/v1/search/stream")
async def search_stream(req: SearchReq):
    """Same as /v1/search, as Server-Sent Events: progress events, answer deltas, then {"type": "done", "result"}."""
    _check_llm(req.llm, req.retrieval)
    q: asyncio.Queue = asyncio.Queue()

    async def emit(ev):
        await q.put(ev)

    async def run():
        try:
            res = await searchmod.search(req.query, req.doc_ids, req.top_k, req.answer, req.max_rounds,
                                         req.include_text, emit=emit, llm_backend=req.llm,
                                         collection_id=req.collection_id, retrieval=req.retrieval)
            await q.put({"type": "done", "result": res})
        except Exception as e:
            logging.exception("search failed")
            await q.put({"type": "error", "message": repr(e)})
        finally:
            await q.put(None)

    task = asyncio.create_task(run())

    async def events():
        try:
            while (ev := await q.get()) is not None:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():   # client went away
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index_page():
    return FileResponse(WEB / "index.html", headers=NO_CACHE)


@app.get("/pipeline")
def pipeline_page():
    return FileResponse(WEB / "pipeline.html", headers=NO_CACHE)


@app.get("/api-docs")
def api_docs_page():
    return FileResponse(WEB / "api.html", headers=NO_CACHE)
