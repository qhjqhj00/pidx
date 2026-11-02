"""Zip uploads -> collections.

A collection is a directory tree whose file leaves are ordinary documents (each indexed on its own,
with its own pages, tree and citations). Directory nodes get an LLM summary built bottom-up from
their children once every file in the collection is indexed.

Collection tree node:  {"type": "dir", "title", "path", "summary", "nodes": [...]}
                       {"type": "file", "title", "path", "doc_id"}
"""

import asyncio
import logging
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from . import config, parse, store
from .llm import chat

log = logging.getLogger("pidx.collection")

JUNK = {".ds_store", "thumbs.db", "desktop.ini"}


def _decode_name(info: zipfile.ZipInfo) -> str:
    """Zips made on Chinese Windows store GBK names without the UTF-8 flag; Python reads them as cp437."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def _skip_reason(parts: tuple[str, ...]) -> str | None:
    if any(p == "__MACOSX" for p in parts):
        return "macos metadata"
    if any(p.startswith(".") for p in parts) or parts[-1].lower() in JUNK or parts[-1].startswith("~$"):
        return "hidden / system file"
    return None


def ingest_zip(zip_path: str, name: str, options: dict) -> tuple[str, list[str], dict]:
    """Unpack a zip into documents + a collection tree. Returns (collection_id, doc_ids, info).

    Files are written to FILES_DIR under generated names, so paths inside the zip never touch the
    filesystem (no zip-slip). Limits: file count, per-file and total uncompressed size."""
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise ValueError("not a valid zip file")
    entries, skipped = [], []
    total = 0
    for info in z.infolist():
        if info.is_dir():
            continue
        path = _decode_name(info).replace("\\", "/")
        parts = tuple(p for p in PurePosixPath(path).parts if p not in ("", ".", "/"))
        if not parts or ".." in parts:
            skipped.append({"path": path, "reason": "bad path"})
            continue
        why = _skip_reason(parts)
        if why:
            continue                       # silently: nobody wants to see __MACOSX in the report
        fmt = parse.file_format(parts[-1])
        if not fmt:
            skipped.append({"path": "/".join(parts), "reason": "unsupported format"})
            continue
        if info.file_size > config.ZIP_MAX_FILE_MB * 2 ** 20:
            skipped.append({"path": "/".join(parts), "reason": f"larger than {config.ZIP_MAX_FILE_MB} MB"})
            continue
        total += info.file_size
        if total > config.ZIP_MAX_TOTAL_MB * 2 ** 20:
            raise ValueError(f"zip expands to more than {config.ZIP_MAX_TOTAL_MB} MB")
        entries.append((info, parts, fmt))
    if len(entries) > config.ZIP_MAX_FILES:
        raise ValueError(f"zip has {len(entries)} supported files, the limit is {config.ZIP_MAX_FILES}")
    if not entries:
        raise ValueError("zip contains no supported files")

    # a single top-level folder is the collection root, not an extra level
    tops = {parts[0] for _, parts, _ in entries}
    if len(tops) == 1 and all(len(parts) > 1 for _, parts, _ in entries):
        root_name = next(iter(tops))
        entries = [(i, parts[1:], f) for i, parts, f in entries]
        for s in skipped:
            s["path"] = s["path"].removeprefix(root_name + "/")
    else:
        root_name = name

    cid = store.new_collection_id()
    root = {"type": "dir", "title": root_name, "path": "", "summary": "", "nodes": []}
    dirs = {(): root}
    doc_ids = []
    for info, parts, fmt in sorted(entries, key=lambda e: e[1]):
        parent = root
        for i in range(1, len(parts)):
            key = parts[:i]
            if key not in dirs:
                d = {"type": "dir", "title": parts[i - 1], "path": "/".join(key), "summary": "", "nodes": []}
                parent["nodes"].append(d)
                dirs[key] = d
            parent = dirs[key]
        doc_id = store.new_doc_id()
        ext = Path(parts[-1]).suffix.lower()
        dst = config.FILES_DIR / f"{doc_id}{ext}"
        with z.open(info) as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out)
        try:
            parse.check_file(str(dst), fmt)
            pdf_path = None
            if fmt == "image":
                pdf_path = str(config.FILES_DIR / f"{doc_id}.pdf")
                parse.image_to_pdf(str(dst), pdf_path)
        except Exception as e:
            dst.unlink(missing_ok=True)
            skipped.append({"path": "/".join(parts), "reason": str(e)})
            continue
        rel = "/".join(parts)
        store.create_document(doc_id, parts[-1], str(dst), {"collection": root_name, "path": rel}, options,
                              format=fmt, pdf_path=pdf_path, collection_id=cid, rel_path=rel)
        parent["nodes"].append({"type": "file", "title": parts[-1], "path": rel, "doc_id": doc_id})
        doc_ids.append(doc_id)
    if not doc_ids:
        raise ValueError("zip contains no readable files: " + "; ".join(f"{s['path']}: {s['reason']}" for s in skipped))
    _number(root)
    info = {"files": len(doc_ids), "dirs": len(dirs), "skipped": skipped}
    store.create_collection(cid, root_name, root, info)
    log.info("collection %s (%s): %d files, %d skipped", cid, root_name, len(doc_ids), len(skipped))
    return cid, doc_ids, info


def _number(root):
    """Directory node ids d0000, d0001, ... (file leaves are addressed by doc_id)."""
    for i, n in enumerate(n for n in walk(root) if n["type"] == "dir"):
        n["node_id"] = f"d{i:04d}"


def walk(node):
    yield node
    for c in node.get("nodes", []):
        yield from walk(c)


DIR_PROMPT = """Write a 1-2 sentence summary of the folder "{path}" in the collection "{coll}", so that someone browsing the folder tree can tell whether a document they need is inside.
Name the kinds of documents and the concrete topics, entities and periods they cover. No preamble.
Write in the language of the folder and file names.

Contents:
{listing}"""


async def summarize(cid: str):
    """Bottom-up directory summaries from file descriptions and sub-folder summaries."""
    coll = store.get_collection(cid)
    root = coll["tree"]
    docs = {d["id"]: d for d in store.list_documents(limit=100000, collection_id=cid)[0]}

    async def one(node):
        await asyncio.gather(*(one(c) for c in node["nodes"] if c["type"] == "dir"))
        lines = []
        for c in node["nodes"]:
            if c["type"] == "dir":
                lines.append(f"- folder {c['title']}/: {c.get('summary') or ''}")
            else:
                d = docs.get(c["doc_id"]) or {}
                lines.append(f"- file {c['title']}: {d.get('description') or '(' + str(d.get('status')) + ')'}")
        listing = "\n".join(lines)[:12000]
        s = await chat([{"role": "user", "content": DIR_PROMPT.format(path=node["path"] or node["title"],
                                                                     coll=root["title"], listing=listing)}],
                       max_tokens=300)
        node["summary"] = s.strip()

    await one(root)
    counts = store.collection_status_counts(cid)
    info = {**(coll["info"] or {}), "status_counts": counts}
    store.update_collection(cid, tree=root, info=info,
                            status="completed" if counts.get("completed") else "failed",
                            error=None if counts.get("completed") else "no file could be indexed")
