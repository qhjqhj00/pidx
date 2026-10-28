"""SQLite persistence: documents (+ tree JSON) and pages (+ layout blocks)."""

import json
import sqlite3
import threading
import time
import uuid

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
"""


def db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(config.DB_PATH, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        _local.con = con
    return con


def new_doc_id() -> str:
    return "pi-" + uuid.uuid4().hex[:12]


def create_document(doc_id, name, path, metadata=None, options=None):
    now = time.time()
    with db() as con:
        con.execute("INSERT INTO documents (id, name, path, status, metadata, options, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (doc_id, name, path, "queued", json.dumps(metadata or {}), json.dumps(options or {}), now, now))


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
        con.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?)",
                        [(doc_id, p["page"], p["width"], p["height"], p["text"], json.dumps(p["blocks"]))
                         for p in pages])


def _doc(row, full=False):
    if row is None:
        return None
    d = dict(row)
    for k in ("metadata", "options", "info", "tree", "embedding"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    if not full:
        d.pop("tree", None)
        d.pop("embedding", None)
    return d


def get_document(doc_id, full=False):
    return _doc(db().execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone(), full)


def find_by_name(name):
    return [_doc(r) for r in db().execute("SELECT * FROM documents WHERE name=? ORDER BY created_at", (name,))]


def list_documents(limit=50, offset=0, status=None, full=False):
    q, args = "SELECT * FROM documents", []
    if status:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    rows = db().execute(q, (*args, limit, offset)).fetchall()
    total = db().execute("SELECT COUNT(*) FROM documents" + (" WHERE status=?" if status else ""), args).fetchone()[0]
    return [_doc(r, full) for r in rows], total


def get_pages(doc_id, page_nums=None):
    if page_nums:
        marks = ",".join("?" * len(page_nums))
        rows = db().execute(f"SELECT * FROM pages WHERE doc_id=? AND page IN ({marks}) ORDER BY page",
                            (doc_id, *page_nums)).fetchall()
    else:
        rows = db().execute("SELECT * FROM pages WHERE doc_id=? ORDER BY page", (doc_id,)).fetchall()
    return [{"page": r["page"], "width": r["width"], "height": r["height"], "text": r["text"],
             "blocks": json.loads(r["blocks"])} for r in rows]


def delete_document(doc_id):
    with db() as con:
        con.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        n = con.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
    return n > 0
