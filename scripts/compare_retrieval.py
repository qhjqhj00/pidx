"""对比两种召回方式（hybrid：向量召回子树 + LLM 并行读；tree：LLM 读整棵树）在推荐问题上的表现。

  PIDX_API=http://127.0.0.1:8890 .venv/bin/python scripts/compare_retrieval.py [--llm deepseek]
跑 pidx/web/suggestions.json 里的问题（需要先导入演示文档），打印每题两种方式的耗时、LLM 调用、
输入 token、命中节点，结果（含答案全文，便于人工核对）写到 logs/compare_retrieval.json。
"""

import argparse
import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("PIDX_API", "http://127.0.0.1:8890"))
    ap.add_argument("--llm")
    ap.add_argument("--modes", default="hybrid,tree")
    ap.add_argument("--out", default=str(ROOT / "logs" / "compare_retrieval.json"))
    a = ap.parse_args()
    http = httpx.Client(base_url=a.api, timeout=900, trust_env=False)
    docs = {d["name"]: d["id"] for d in http.get("/v1/documents", params={"limit": 1000}).json()["documents"]}
    sugg = json.load(open(ROOT / "pidx" / "web" / "suggestions.json"))
    out = []
    for name, items in sugg.items():
        if name.startswith("_") or name.startswith("coll:") or (name != "all" and name not in docs):
            continue
        for it in items:
            row = {"doc": name, "tag": it["tag"], "q": it["q"]}
            for mode in a.modes.split(","):
                body = {"query": it["q"], "answer": True, "include_text": False, "retrieval": mode, "llm": a.llm,
                        "doc_ids": None if name == "all" else [docs[name]]}
                r = http.post("/v1/search", json=body).json()
                calls = r["timing"]["calls"]
                row[mode] = {"seconds": r["seconds"], "calls": len(calls),
                             "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                             "tree_phase_s": sum(p["seconds"] for p in r["timing"]["phases"] if p["name"].startswith("树检索")),
                             "nodes": [f"{n['doc_name'][:12]}:{n['node_id']} {n['title'][:40]}" for n in r["nodes"]],
                             "sufficient": r.get("sufficient"), "answer": r.get("answer", "")}
            out.append(row)
            print(f"\n[{it['tag']}] {name[:24]} — {it['q'][:70]}")
            for mode in a.modes.split(","):
                m = row[mode]
                print(f"  {mode:6s} {m['seconds']:5.1f}s (树检索 {m['tree_phase_s']:4.1f}s) 调用 {m['calls']:2d} 输入 {m['prompt_tokens']:6d} tok "
                      f"{'' if m['sufficient'] else '⚠证据不足 '}节点 {m['nodes'][:4]}")
    print()
    for mode in a.modes.split(","):
        n = len(out)
        print(f"{mode:6s} 平均 {sum(r[mode]['seconds'] for r in out) / n:.1f}s · 树检索平均 "
              f"{sum(r[mode]['tree_phase_s'] for r in out) / n:.1f}s · 平均输入 {sum(r[mode]['prompt_tokens'] for r in out) / n:.0f} tok · "
              f"证据充分 {sum(bool(r[mode]['sufficient']) for r in out)}/{n}")
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
