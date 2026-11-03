"""MCP server (stdio) exposing the pidx HTTP API as tools.

  claude mcp add pidx -e PIDX_API=http://127.0.0.1:8890 -- /path/to/pidx/.venv/bin/python -m pidx.mcp_server
Set PIDX_API (default http://127.0.0.1:8890) to point at the API server.
"""

import json
import os

import httpx
try:                                   # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

API = os.environ.get("PIDX_API", "http://127.0.0.1:8890")
mcp = MCPServer("pidx")
http = httpx.Client(base_url=API, timeout=600, trust_env=False)


def _get(path, **params):
    r = http.get(path, params=params)
    r.raise_for_status()
    return r.json()


def _post(path, body) -> str:
    r = http.post(path, json=body)
    r.raise_for_status()
    return json.dumps(r.json(), ensure_ascii=False)


@mcp.tool()
def list_documents(collection_id: str | None = None) -> str:
    """List indexed documents (PDF, images, Office files): id, name, format, status, page count, description,
    and for files from a zip collection its collection id and path inside the zip."""
    params = {"limit": 1000, **({"collection_id": collection_id} if collection_id else {})}
    docs = _get("/v1/documents", **params)["documents"]
    keys = ("id", "name", "format", "status", "pageNum", "description", "collectionId", "relPath")
    return json.dumps([{k: d.get(k) for k in keys} for d in docs], ensure_ascii=False, indent=1)


@mcp.tool()
def list_collections() -> str:
    """List zip collections (folder trees of documents): id, name, status, file counts."""
    return json.dumps(_get("/v1/collections")["collections"], ensure_ascii=False, indent=1)


@mcp.tool()
def get_collection(collection_id: str) -> str:
    """Folder tree of a zip collection, with folder summaries and each file's doc id, status and description."""
    return json.dumps(_get(f"/v1/collections/{collection_id}"), ensure_ascii=False)


@mcp.tool()
def get_tree(doc_id: str, node_summary: bool = True) -> str:
    """Tree index (table of contents with node ids, page ranges and summaries) of a document."""
    return json.dumps(_get(f"/v1/documents/{doc_id}/tree", node_summary=node_summary), ensure_ascii=False)


@mcp.tool()
def get_pages(doc_id: str, pages: str) -> str:
    """Text of specific pages, e.g. pages="5-7" or "3,8"."""
    return json.dumps(_get(f"/v1/documents/{doc_id}/pages", pages=pages), ensure_ascii=False)


@mcp.tool()
def search_pages(query: str, doc_ids: list[str] | None = None, collection_id: str | None = None, top_k: int = 10) -> str:
    """Fast page ranking by embeddings (no LLM): returns the pages most likely to answer `query`, each with
    a score, its title path and a snippet. Read them next with get_pages(doc_id, "N,M")."""
    return _post("/v1/pages/search", {"query": query, "doc_ids": doc_ids, "collection_id": collection_id, "top_k": top_k})


@mcp.tool()
def search(query: str, doc_ids: list[str] | None = None, collection_id: str | None = None, answer: bool = True,
           top_k: int = 5) -> str:
    """Retrieval over the document trees (vector recall of subtrees + LLM reasoning). Scope: doc_ids, or one
    zip collection, or the whole library. Returns the relevant sections (with text), and if answer=True an
    answer with block-level citations (doc, page, bbox; bbox is null for Office files)."""
    r = http.post("/v1/search", json={"query": query, "doc_ids": doc_ids, "collection_id": collection_id,
                                      "answer": answer, "top_k": top_k})
    r.raise_for_status()
    return json.dumps(r.json(), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
