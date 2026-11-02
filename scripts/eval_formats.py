"""多格式 + zip 资料包 + 两种召回方式的端到端评测。

  .venv/bin/python scripts/eval_formats.py                    # 上传 samples/xinghe.zip，等索引完，跑全部问题
  .venv/bin/python scripts/eval_formats.py --collection pc-xxxx --modes hybrid,tree
需要服务已经启动；先跑 scripts/make_samples.py 生成数据。PIDX_API 指定服务地址。
每道题检查：答案里有没有标准答案（scripts/multiformat_cases.json）、选中的文档对不对、引用是否来自那份文件。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def wait_collection(http, cid):
    t0 = time.time()
    while True:
        c = http.get(f"/v1/collections/{cid}").json()
        if c["status"] != "processing":
            print(f"资料包 {c['status']}，{time.time() - t0:.0f}s，{c['counts']}")
            return c
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("PIDX_API", "http://127.0.0.1:8890"))
    ap.add_argument("--collection")
    ap.add_argument("--zip", default=str(ROOT / "samples" / "xinghe.zip"))
    ap.add_argument("--modes", default="hybrid,tree")
    ap.add_argument("--llm")
    ap.add_argument("--only", help="只跑 tag 包含这个字符串的题")
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_formats.json"))
    a = ap.parse_args()
    http = httpx.Client(base_url=a.api, timeout=900, trust_env=False)

    cid = a.collection
    if not cid:
        with open(a.zip, "rb") as f:
            r = http.post("/v1/collections", files={"file": (os.path.basename(a.zip), f, "application/zip")})
        r.raise_for_status()
        r = r.json()
        cid = r["collection_id"]
        print("上传资料包", cid, f"{r['files']} 个文件，跳过 {r['skipped']}")
    wait_collection(http, cid)
    docs = {d["id"]: d for d in http.get("/v1/documents", params={"collection_id": cid, "limit": 1000}).json()["documents"]}
    for d in docs.values():
        if d["status"] != "completed":
            print("  未完成：", d["name"], d["status"], d.get("error"))

    cases = json.load(open(ROOT / "scripts" / "multiformat_cases.json"))["cases"]
    if a.only:
        cases = [c for c in cases if a.only in c["tag"]]
    rows = []
    for mode in a.modes.split(","):
        for c in cases:
            body = {"query": c["q"], "collection_id": cid, "answer": True, "include_text": False,
                    "retrieval": mode, "llm": a.llm}
            r = http.post("/v1/search", json=body).json()
            ans = r.get("answer", "")
            ok = all(any(norm(x) in norm(ans) for x in e.split("|")) for e in c["expect"])
            picked = [x["name"] for x in r.get("documents", [])]
            cited = {x["doc_name"] for x in r.get("citations", [])}
            calls = r.get("timing", {}).get("calls", [])
            row = {"mode": r.get("retrieval", mode), "tag": c["tag"], "q": c["q"], "ok": ok,
                   "doc_ok": c["file"] in picked, "cite_ok": c["file"] in cited, "seconds": r.get("seconds"),
                   "llm_calls": len(calls), "prompt_tokens": sum(x["prompt_tokens"] for x in calls),
                   "picked": picked, "answer": ans}
            rows.append(row)
            mark = "✅" if ok else "❌"
            print(f"{mark} [{row['mode']:6s}] {c['tag']:10s} {row['seconds']:5.1f}s  调用 {row['llm_calls']:2d}  "
                  f"输入 {row['prompt_tokens']:6d} tok  选文档{'✓' if row['doc_ok'] else '✗'} 引用{'✓' if row['cite_ok'] else '✗'}  "
                  f"{c['q'][:30]}")
            if not ok:
                print("     答案：", ans[:300].replace("\n", " "))
    print()
    for mode in a.modes.split(","):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        print(f"{mode:7s} 答对 {sum(r['ok'] for r in rs)}/{n} · 选对文档 {sum(r['doc_ok'] for r in rs)}/{n} · "
              f"引用对 {sum(r['cite_ok'] for r in rs)}/{n} · 平均 {sum(r['seconds'] for r in rs) / n:.1f}s · "
              f"平均输入 {sum(r['prompt_tokens'] for r in rs) / n:.0f} tok")
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
