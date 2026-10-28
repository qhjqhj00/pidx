"""冒烟测试：上传一份 PDF，等索引完成，问一个问题，检查答案里有预期的事实和引用。

  .venv/bin/python scripts/smoke.py                      # 默认用 samples/q1-fy25-earnings.pdf
  PIDX_API=http://127.0.0.1:8890 .venv/bin/python scripts/smoke.py
需要服务已经启动（scripts/serve.sh start），并且先跑过 scripts/fetch_samples.sh。
"""

import os
import sys
import time

import httpx

API = os.environ.get("PIDX_API", "http://127.0.0.1:8890")
PDF = "samples/q1-fy25-earnings.pdf"
QUESTION = "What guidance did Disney give for full-year fiscal 2025 adjusted EPS?"
EXPECT = "high-single digit"

http = httpx.Client(base_url=API, timeout=600, trust_env=False)
print("可用模型：", http.get("/v1/llm").json())

with open(PDF, "rb") as f:
    doc = http.post("/v1/documents", files={"file": (os.path.basename(PDF), f, "application/pdf")}).json()
print("已上传：", doc)
t0 = time.time()
while True:
    d = http.get(f"/v1/documents/{doc['doc_id']}").json()
    if d["status"] in ("completed", "failed"):
        break
    time.sleep(3)
print(f"索引 {d['status']}，用时 {time.time() - t0:.0f}s：", d.get("info"), d.get("error") or "")
if d["status"] != "completed":
    sys.exit(1)

r = http.post("/v1/search", json={"query": QUESTION, "doc_ids": [doc["doc_id"]], "answer": True,
                                  "include_text": False}).json()
print(f"\n问：{QUESTION}\n答（{r['seconds']}s，{r['llm']}）：{r['answer']}\n引用：",
      [(c["page"], c["block_id"]) for c in r["citations"]])
ok = EXPECT in r["answer"].lower() and r["citations"]
print("\n冒烟测试", "通过 ✅" if ok else "未通过 ❌")
sys.exit(0 if ok else 1)
