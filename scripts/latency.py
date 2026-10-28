"""Run showcase queries through /v1/search and print where the time goes."""
import json, os, sys, httpx
API = "http://127.0.0.1:8890"
http = httpx.Client(base_url=API, timeout=900, trust_env=False)
docs = {d["name"]: d["id"] for d in http.get("/v1/documents").json()["documents"]}
sugg = json.load(open("pidx/web/suggestions.json"))
runs = []
for name, items in sugg.items():
    if name.startswith("_"):
        continue
    for it in items[: int(sys.argv[1]) if len(sys.argv) > 1 else 99]:
        body = {"query": it["q"], "answer": True, "include_text": False,
                "doc_ids": None if name == "all" else [docs[name]], "llm": os.environ.get("LLM")}
        r = http.post("/v1/search", json=body).json()
        t = r["timing"]
        runs.append(r)
        print(f"\n### [{it['tag']}] total {t['total_s']}s")
        for p in t["phases"]:
            print(f"   {p['name']:<16} {p['seconds']:6.1f}s")
        for c in t["calls"]:
            print(f"     llm {c['label'][:28]:<28} in {c['prompt_tokens']:>6} out {c['completion_tokens']:>5}  "
                  f"wait {c['wait_ms']/1000:5.1f}s  ttft {c['ttft_ms']/1000:5.1f}s  decode {c['decode_ms']/1000:5.1f}s  ({c['tok_per_s']} tok/s)")
json.dump(runs, open(sys.argv[2] if len(sys.argv) > 2 else "logs/latency.json", "w"), ensure_ascii=False, indent=1)
