"""MCP server (stdio) exposing the pidx HTTP API as tools.

  claude mcp add pidx -- /path/to/.venv/bin/python -m pidx.mcp_server
Set PIDX_API (default http://127.0.0.1:8890) to point at the API server.
"""

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

API = os.environ.get("PIDX_API", "http://127.0.0.1:8890")
mcp = FastMCP("pidx")
http = httpx.Client(base_url=API, timeout=600, trust_env=False)


def _get(path, **params):
    r = http.get(path, params=params)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def list_documents() -> str:
    """List indexed documents: id, name, status, page count, description."""
    docs = _get("/v1/documents", limit=500)["documents"]
    return json.dumps([{k: d[k] for k in ("id", "name", "status", "pageNum", "description")} for d in docs],
                      ensure_ascii=False, indent=1)


@mcp.tool()
def get_tree(doc_id: str, node_summary: bool = True) -> str:
    """Tree index (table of contents with node ids, page ranges and summaries) of a document."""
    return json.dumps(_get(f"/v1/documents/{doc_id}/tree", node_summary=node_summary), ensure_ascii=False)


@mcp.tool()
def get_pages(doc_id: str, pages: str) -> str:
    """Text of specific pages, e.g. pages="5-7" or "3,8"."""
    return json.dumps(_get(f"/v1/documents/{doc_id}/pages", pages=pages), ensure_ascii=False)


@mcp.tool()
def search(query: str, doc_ids: list[str] | None = None, answer: bool = True, top_k: int = 5) -> str:
    """Reasoning-based retrieval over the document trees. Returns the relevant sections (with text),
    and if answer=True an answer with block-level citations (doc, page, bbox)."""
    r = http.post("/v1/search", json={"query": query, "doc_ids": doc_ids, "answer": answer, "top_k": top_k})
    r.raise_for_status()
    return json.dumps(r.json(), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
