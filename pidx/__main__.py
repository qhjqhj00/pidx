"""CLI.

  python -m pidx serve [--port 8890]
  python -m pidx index FILE.pdf [--mode auto|text|ocr]     (talks to a running server)
  python -m pidx tree DOC_ID
  python -m pidx ask "question" [--doc DOC_ID ...] [--no-answer]
"""

import argparse
import json
import os
import sys
import time

import httpx

API = os.environ.get("PIDX_API", "http://127.0.0.1:8890")


def main():
    ap = argparse.ArgumentParser(prog="pidx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8890)
    s = sub.add_parser("index")
    s.add_argument("file")
    s.add_argument("--mode", default="auto")
    s.add_argument("--no-wait", action="store_true")
    s = sub.add_parser("tree")
    s.add_argument("doc_id")
    s = sub.add_parser("ask")
    s.add_argument("query")
    s.add_argument("--doc", action="append")
    s.add_argument("--no-answer", action="store_true")
    a = ap.parse_args()

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("pidx.api:app", host=a.host, port=a.port)
        return

    http = httpx.Client(base_url=API, timeout=900, trust_env=False)
    if a.cmd == "index":
        with open(a.file, "rb") as f:
            r = http.post("/v1/documents", files={"file": (os.path.basename(a.file), f, "application/pdf")},
                          data={"mode": a.mode}).json()
        print(json.dumps(r, ensure_ascii=False))
        while not a.no_wait:
            d = http.get(f"/v1/documents/{r['doc_id']}").json()
            if d["status"] in ("completed", "failed"):
                print(json.dumps(d, ensure_ascii=False, indent=1))
                break
            time.sleep(3)
    elif a.cmd == "tree":
        t = http.get(f"/v1/documents/{a.doc_id}/tree", params={"node_summary": True}).json()

        def show(nodes, d=0):
            for n in nodes:
                print("  " * d + f"[{n['node_id']}] {n['title']}  (p.{n['start_index']}-{n['end_index']})")
                show(n.get("nodes", []), d + 1)
        show(t.get("result", []))
    elif a.cmd == "ask":
        r = http.post("/v1/search", json={"query": a.query, "doc_ids": a.doc, "answer": not a.no_answer,
                                          "include_text": False}).json()
        if "answer" in r:
            print(r["answer"], "\n")
            for c in r["citations"]:
                print(f"  [{c['tag']}] {c['doc_name']} p.{c['page']} bbox={c['bbox']}  {c['text'][:100]!r}")
        print("\nnodes:", ", ".join(f"{n['node_id']} {n['title']} (p.{n['start_index']}-{n['end_index']})" for n in r["nodes"]))
        print(f"{r['seconds']}s, llm {r['usage']}")


if __name__ == "__main__":
    sys.exit(main())
