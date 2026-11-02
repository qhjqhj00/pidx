"""SQLite persistence: documents (+ tree JSON), pages (+ layout blocks), collections (zip uploads:
directory tree JSON) and node vectors (float32 blobs)."""

import json
import sqlite3
import threading
import time
import uuid

import numpy as np

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, page_count INTEGER, description TEXT,
  metadata TEXT, options TEXT, tree TEXT, info TEXT, embedding TEXT,
  created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS pages (
  doc_id TEXT NOT NULL, page INTEGER NOT NULL, width REAL, height REAL,
  text TEXT, blocks TEXT, PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS collections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
  tree TEXT, info TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS node_vectors (
  doc_id TEXT NOT NULL, node_id TEXT NOT NULL, vec BLOB NOT NULL, PRIMARY KEY (doc_id, node_id)
);
"""

# columns added after the first release: (table, column, type)
MIGRATIONS = [("documents", "format", "TEXT"), ("documents", "pdf_path", "TEXT"),
              ("documents", "collection_id", "TEXT"), ("documents", "rel_path", "TEXT"),
              ("pages", "label", "TEXT")]


def _migrate(con):
    for table, col, typ in MIGRATIONS:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.execute("CREATE INDEX IF NOT EXISTS documents_collection ON documents(collection_id)")
    con.commit()


def db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(config.DB_PATH, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        _migrate(con)
        _local.con = con
    return con


def new_doc_id() -> str:
    return "pi-" + uuid.uuid4().hex[:12]


def new_collection_id() -> str:
    return "pc-" + uuid.uuid4().hex[:12]


def create_document(doc_id, name, path, metadata=None, options=None, *, format="pdf", pdf_path=None,
                    collection_id=None, rel_path=None):
    now = time.time()
    with db() as con:
        con.execute("INSERT INTO documents (id, name, path, status, metadata, options, format, pdf_path, collection_id, "
                    "rel_path, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (doc_id, name, path, "queued", json.dumps(metadata or {}, ensure_ascii=False),
                     json.dumps(options or {}), format, pdf_path, collection_id, rel_path, now, now))


def update_document(doc_id, **fields):
    for k in ("tree", "info", "metadata", "options", "embedding"):
        if k in fields and not isinstance(fields[k], str) and fields[k] is not None:
            fields[k] = json.dumps(fields[k], ensure_ascii=False)
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with db() as con:
        con.execute(f"UPDATE documents SET {cols} WHERE id=?", (*fields.values(), doc_id))


def save_pages(doc_id, pages):
    with db() as con:
        con.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        con.executemany("INSERT INTO pages (doc_id, page, width, height, text, blocks, label) VALUES (?,?,?,?,?,?,?)",
                        [(doc_id, p["page"], p["width"], p["height"], p["text"], json.dumps(p["blocks"]),
                          p.get("label")) for p in pages])


def _doc(row, full=False):
    if row is None:
        return None
    d = dict(row)
    for k in ("metadata", "options", "info", "tree", "embedding"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    d["format"] = d.get("format") or "pdf"
    if d["format"] == "pdf" and not d.get("pdf_path"):
        d["pdf_path"] = d["path"]
    if not full:
        d.pop("tree", None)
        d.pop("embedding", None)
    return d


def get_document(doc_id, full=False):
    return _doc(db().execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone(), full)


def find_by_name(name):
    return [_doc(r) for r in db().execute("SELECT * FROM documents WHERE name=? ORDER BY created_at", (name,))]


def list_documents(limit=50, offset=0, status=None, full=False, collection_id=None):
    where, args = [], []
    if status:
        where.append("status=?")
        args.append(status)
    if collection_id:
        where.append("collection_id=?")
        args.append(collection_id)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db().execute(f"SELECT * FROM documents{w} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (*args, limit, offset)).fetchall()
    total = db().execute(f"SELECT COUNT(*) FROM documents{w}", args).fetchone()[0]
    return [_doc(r, full) for r in rows], total


def get_pages(doc_id, page_nums=None):
    if page_nums:
        marks = ",".join("?" * len(page_nums))
        rows = db().execute(f"SELECT * FROM pages WHERE doc_id=? AND page IN ({marks}) ORDER BY page",
                            (doc_id, *page_nums)).fetchall()
    else:
        rows = db().execute("SELECT * FROM pages WHERE doc_id=? ORDER BY page", (doc_id,)).fetchall()
    return [{"page": r["page"], "width": r["width"], "height": r["height"], "text": r["text"],
             "label": r["label"], "blocks": json.loads(r["blocks"])} for r in rows]


def delete_document(doc_id):
    with db() as con:
        con.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM node_vectors WHERE doc_id=?", (doc_id,))
        n = con.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
    return n > 0


# --------------------------------------------------------------------------- collections

def create_collection(cid, name, tree, info):
    now = time.time()
    with db() as con:
        con.execute("INSERT INTO collections (id, name, status, tree, info, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (cid, name, "processing", json.dumps(tree, ensure_ascii=False), json.dumps(info, ensure_ascii=False),
                     now, now))


def update_collection(cid, **fields):
    for k in ("tree", "info"):
        if k in fields and not isinstance(fields[k], str) and fields[k] is not None:
            fields[k] = json.dumps(fields[k], ensure_ascii=False)
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with db() as con:
        con.execute(f"UPDATE collections SET {cols} WHERE id=?", (*fields.values(), cid))


def _coll(row):
    if row is None:
        return None
    d = dict(row)
    for k in ("tree", "info"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def get_collection(cid):
    return _coll(db().execute("SELECT * FROM collections WHERE id=?", (cid,)).fetchone())


def list_collections():
    return [_coll(r) for r in db().execute("SELECT * FROM collections ORDER BY created_at DESC")]


def collection_status_counts(cid) -> dict:
    return {r[0]: r[1] for r in db().execute(
        "SELECT status, COUNT(*) FROM documents WHERE collection_id=? GROUP BY status", (cid,))}


def delete_collection(cid):
    with db() as con:
        return con.execute("DELETE FROM collections WHERE id=?", (cid,)).rowcount > 0


# --------------------------------------------------------------------------- node vectors

def save_node_vectors(doc_id, items):
    """items: [(node_id, vector)]; replaces the document's vectors."""
    with db() as con:
        con.execute("DELETE FROM node_vectors WHERE doc_id=?", (doc_id,))
        con.executemany("INSERT INTO node_vectors VALUES (?,?,?)",
                        [(doc_id, nid, np.asarray(v, dtype=np.float32).tobytes()) for nid, v in items])


def load_node_vectors(doc_id) -> tuple[list[str], np.ndarray | None]:
    rows = db().execute("SELECT node_id, vec FROM node_vectors WHERE doc_id=?", (doc_id,)).fetchall()
    if not rows:
        return [], None
    return [r[0] for r in rows], np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])


def docs_without_vectors() -> list[str]:
    return [r[0] for r in db().execute(
        "SELECT id FROM documents WHERE status='completed' AND tree IS NOT NULL AND id NOT IN "
        "(SELECT DISTINCT doc_id FROM node_vectors)")]
