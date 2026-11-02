"""EmbeddingGemma 2 path, end to end without an LLM key.

  PIDX_GEMMA_EMBED_URL=http://127.0.0.1:18300/v1 .venv/bin/python scripts/check_gemma.py

Builds a PDF with bookmarks (so the tree needs no LLM), real text sections and four pictures (bar
chart, pie chart, a red stop sign, a tiny logo that must be skipped), indexes it in a throw-away
data dir with the LLM summary steps stubbed, then checks:
  1. image blocks are cropped, sized-filtered and mapped to the right tree node;
  2. text -> image retrieval ranks the right picture first;
  3. text -> node retrieval (with Gemma task prefixes) ranks the right section first;
  4. the crop is served by GET /v1/documents/{id}/images/{image_id};
  5. vectors from another model (a fake 1024-d bge-m3 doc) are ignored by search and recomputed.
Exit code 0 when every check passes.
"""

import asyncio
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIDX_GEMMA_EMBED_URL", "http://127.0.0.1:18300/v1")
os.environ["PIDX_DATA_DIR"] = tempfile.mkdtemp(prefix="pidx-gemma-")

import numpy as np                                       # noqa: E402
import pymupdf                                           # noqa: E402
from PIL import Image, ImageDraw                         # noqa: E402

from pidx import config, indexer, llm, search, store, tree  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("✅ " if ok else "❌ ") + name + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def png(draw, size=(600, 400)):
    im = Image.new("RGB", size, "white")
    draw(ImageDraw.Draw(im), im.size)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def bars(d, s):
    for i, h in enumerate([90, 170, 240, 320]):
        d.rectangle([60 + i * 120, s[1] - 40 - h, 140 + i * 120, s[1] - 40], fill=(40, 90, 170))
    d.line([40, s[1] - 40, s[0] - 20, s[1] - 40], fill="black", width=3)


def pie(d, s):
    box = [150, 50, 450, 350]
    for a0, a1, c in [(0, 140, "orange"), (140, 250, "seagreen"), (250, 360, "slateblue")]:
        d.pieslice(box, a0, a1, fill=c)


def stop(d, s):
    cx, cy, r = s[0] / 2, s[1] / 2, 170
    pts = [(cx + r * np.cos(np.pi / 8 + i * np.pi / 4), cy + r * np.sin(np.pi / 8 + i * np.pi / 4)) for i in range(8)]
    d.polygon(pts, fill=(200, 20, 20))
    d.text((cx - 40, cy - 10), "STOP", fill="white")


def make_pdf(path):
    pdf = pymupdf.open()
    sections = [
        ("储能项目经营概况", "本季度储能电站运行稳定，发电量和收益均实现增长。下图为各季度收益变化。", bars),
        ("客户结构", "工商业客户占比最大，其次为港口物流和数据中心用户。下图为客户结构占比。", pie),
        ("安全管理", "所有电站执行严格的现场安全规程，动火作业必须办理一级动火证，并设置警示标识。", stop),
        ("消防与热管理", "电池舱采用液冷热管理方案，电芯最大温差控制在 3 摄氏度以内；可燃气体探测器报警阈值为 50 ppm。", None),
    ]
    font = pymupdf.Font("china-s")
    toc = []
    for i, (title, body, pic) in enumerate(sections):
        pg = pdf.new_page(width=595, height=842)
        pg.insert_font(fontname="cs", fontbuffer=font.buffer)
        pg.insert_text((60, 80), title, fontname="cs", fontsize=20)
        pg.insert_textbox(pymupdf.Rect(60, 100, 535, 260), (body + " ") * 4, fontname="cs", fontsize=11)
        if pic:
            pg.insert_image(pymupdf.Rect(90, 290, 505, 567), stream=png(pic))
        pg.insert_image(pymupdf.Rect(520, 20, 545, 45), stream=png(bars, (60, 60)))   # logo: must be skipped
        toc.append([1, title, i + 1])
    pdf.set_toc(toc)
    pdf.save(path)


async def main():
    print("embedding:", config.EMBED_URL, config.EMBED_MODEL, "multimodal" if config.EMBED_MULTIMODAL else "TEXT ONLY")
    check("Gemma path is on", config.EMBED_MULTIMODAL and llm.embed_model() == "embeddinggemma-2+mm")

    async def no_summary(nodes, pages, name):
        for n in tree.walk(nodes):
            n["summary"] = ""

    async def fixed_description(nodes, pages, name):
        return "星河储能项目季度报告：经营概况、客户结构、安全管理、消防与热管理。"

    tree.summarize, tree.describe = no_summary, fixed_description

    config.FILES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = str(config.FILES_DIR / "report.pdf")
    make_pdf(pdf_path)
    doc_id = store.new_doc_id()
    store.create_document(doc_id, "储能季度报告.pdf", pdf_path, format="pdf", options={"mode": "text"})
    await indexer.index_document(doc_id)
    d = store.get_document(doc_id, full=True)
    info = d["info"]
    check("indexed", d["status"] == "completed", f"method={info['method']} nodes={info['nodes']}")
    check("embed_model recorded", d["embed_model"] == "embeddinggemma-2+mm", d["embed_model"])
    check("doc vector is 768-d", d["embedding"] is not None and len(d["embedding"]) == 768)

    meta, m = store.load_image_vectors(doc_id)
    check("3 images embedded, logo skipped", len(meta) == 3, f"{len(meta)} images: {[x['image_id'] for x in meta]}")
    titles = {n["node_id"]: n["title"] for n in tree.walk(d["tree"])}
    mapped = {x["page"]: titles.get(x["node_id"]) for x in meta}
    check("images mapped to their sections", mapped == {1: "储能项目经营概况", 2: "客户结构", 3: "安全管理"}, str(mapped))
    crops = sorted(p.name for p in (config.CROPS_DIR / doc_id).glob("*.png"))
    check("crops on disk", len(crops) == 3, str(crops))

    async def qvec(q):
        v = await llm.embed([q], kind="query")
        return search._unit(v[0])

    page_of = {x["image_id"]: x["page"] for x in meta}
    img_cases = [("收益逐季增长的柱状图", 1), ("客户结构占比饼图", 2), ("红色的停车警示标志", 3),
                 ("a bar chart of quarterly revenue", 1), ("pie chart", 2)]
    for q, want in img_cases:
        hits = search.image_hits([d], [await qvec(q)], k=3)
        got = [(page_of[h["image_id"]], h["score"]) for h in hits]
        check(f"image search: {q}", bool(hits) and page_of[hits[0]["image_id"]] == want, f"(page, score) = {got}")

    for q in ["电芯最大温差控制在多少度？", "项目总投资多少钱"]:
        hits = search.image_hits([d], [await qvec(q)])
        check(f"no image for a text-only question: {q}", hits == [], str([(h['image_id'], h['score']) for h in hits]))

    text_cases = [("电芯最大温差控制在多少度？", "消防与热管理"), ("动火作业要办什么手续？", "安全管理"),
                  ("哪类客户占比最大？", "客户结构")]
    for q, want in text_cases:
        sc = search.node_scores(d, [await qvec(q)])
        best = max(sc, key=sc.get)
        check(f"node search: {q}", titles[best] == want, f"top={titles[best]} {sc[best]:.3f}")

    from fastapi.testclient import TestClient
    from pidx import api
    c = TestClient(api.app)          # no `with`: lifespan (background workers) not started
    first = meta[0]["image_id"]
    r = c.get(f"/v1/documents/{doc_id}/images/{first}")
    check("crop served", r.status_code == 200 and r.content[:4] == b"\x89PNG", f"{r.status_code} {len(r.content)} bytes")
    check("bad image id rejected", c.get(f"/v1/documents/{doc_id}/images/..%2Fx").status_code in (400, 404))
    emb = c.get("/v1/llm").json()["embedding"]
    check("/v1/llm reports Gemma", emb["multimodal"] is True, str(emb))

    # a document left over from the bge-m3 era: 1024-d vectors, embed_model NULL
    old = store.new_doc_id()
    store.create_document(old, "旧文档.pdf", pdf_path, format="pdf")
    store.update_document(old, status="completed", tree=d["tree"], description="旧文档", embedding=[0.1] * 1024)
    store.save_pages(old, store.get_pages(doc_id))
    store.save_node_vectors(old, [(n["node_id"], np.ones(1024)) for n in tree.walk(d["tree"])])
    od = store.get_document(old, full=True)
    qv = await qvec("温差")
    try:
        s1, s2 = search.node_scores(od, [qv]), search.doc_score(od, qv)
        check("stale bge vectors ignored, no crash", s1 == {} and s2 == 0.0, f"node_scores={len(s1)} doc_score={s2}")
    except Exception as e:
        check("stale bge vectors ignored, no crash", False, repr(e))
    check("stale doc queued for backfill", old in store.docs_needing_vectors(llm.embed_model()))
    await indexer._backfill_vectors()
    od = store.get_document(old, full=True)
    _, m2 = store.load_node_vectors(old)
    check("backfill recomputed under Gemma", od["embed_model"] == "embeddinggemma-2+mm" and m2.shape[1] == 768,
          f"{od['embed_model']} {m2.shape}")

    print(f"\ndata dir: {config.DATA_DIR}")
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
