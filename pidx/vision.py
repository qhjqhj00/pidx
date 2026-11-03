"""Image understanding at index time.

A vision model looks at each image and figure and writes what it shows: the text and numbers in it,
then a short description. That description is inserted into the page text as a layout block of type
"figure" carrying the figure's bbox, at the figure's place in reading order. Everything downstream
(tree, summaries, node and page vectors, answers, citations with highlighting) then sees it like any
other block.

Regions:
- image files: the whole picture (the one-page PDF wrapper)
- PDFs: image blocks, plus clusters of vector drawings (charts, diagrams) with enough paths to be a
  figure rather than table rules; at least VISION_MIN_AREA of the page, largest first, at most
  VISION_MAX_PER_DOC per document.
Office and text files are skipped (no page images).
"""

import asyncio
import base64
import logging

import pymupdf

from . import config, llm, progress

log = logging.getLogger("pidx.vision")

PROMPT = """This image is {what} from page {page} of the document "{doc}".
1. Transcribe the text and numbers that matter in it: titles, labels, axis ticks and values, legend entries, table cells, annotations. Keep numbers exactly as written.
2. Then describe in 2-4 sentences what it shows: the kind of image, the main content, and any trends, comparisons or relationships a reader would take from it.
Write in the language of the text in the image (English if there is none). Be factual, no preamble, at most 350 words."""

MARK = "[图像识别] "
_sem: asyncio.Semaphore | None = None


async def describe(png: bytes, what: str, page: int, doc_name: str) -> tuple[str | None, dict]:
    global _sem
    _sem = _sem or asyncio.Semaphore(config.VISION_CONCURRENCY)
    _, client = llm._get()
    body = {"model": config.VISION_MODEL, "max_tokens": 1200, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": PROMPT.format(what=what, page=page, doc=doc_name)}]}]}
    if "qwen3" in config.VISION_MODEL.lower():
        body["chat_template_kwargs"] = {"enable_thinking": False}
    async with _sem:
        try:
            r = await client.post(f"{config.VISION_URL}/chat/completions", json=body, timeout=180,
                                  headers={"Authorization": f"Bearer {config.VISION_API_KEY}"})
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            log.warning("vision call failed (%s p.%d): %r", doc_name, page, e)
            return None, {}
    text = (j["choices"][0]["message"].get("content") or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("markdown").strip()
    return text or None, j.get("usage") or {}


def _regions(pdf: pymupdf.Document, pages: list[dict], whole: bool) -> list[tuple[int, pymupdf.Rect, int | None]]:
    """(page number, rect in PDF points, index of an existing image block or None), largest first."""
    out = []
    for p in pages:
        pg = pdf[p["page"] - 1]
        area = pg.rect.width * pg.rect.height
        if whole:
            out.append((p["page"], pg.rect, None))
            continue
        found = []
        for i, b in enumerate(p["blocks"]):
            if b["type"] in ("image", "chart") and b.get("bbox"):
                found.append((pymupdf.Rect(b["bbox"]), i))
        try:
            drawings = pg.get_drawings()
            for r in pg.cluster_drawings(drawings=drawings):
                if sum(1 for d in drawings if d["rect"] in r + (-1, -1, 1, 1)) >= 15:
                    found.append((pymupdf.Rect(r), None))
        except Exception:
            pass
        merged = []                                   # union overlapping regions
        for r, i in sorted(found, key=lambda x: -x[0].get_area()):
            for m in merged:
                if m[0].intersects(r):
                    m[0] |= r
                    m[1] = m[1] if m[1] is not None else i
                    break
            else:
                merged.append([pymupdf.Rect(r), i])
        for r, i in merged:
            r &= pg.rect
            if r.get_area() >= config.VISION_MIN_AREA * area and r.width >= 60 and r.height >= 60:
                out.append((p["page"], r, i))
    out.sort(key=lambda x: -x[1].get_area())
    return out[:config.VISION_MAX_PER_DOC]


def _insert_at(page: dict, rect: pymupdf.Rect, block_idx: int | None) -> int:
    """Offset in the page text where a figure's description belongs: at its own image block, else just
    before the first block below it in the same column (usually its caption), else the page end."""
    if block_idx is not None:
        return page["blocks"][block_idx]["start"]
    best = None
    for b in page["blocks"]:
        if b["end"] <= b["start"] or not b.get("bbox"):
            continue
        x0, y0, x1, y1 = b["bbox"]
        if min(x1, rect.x1) - max(x0, rect.x0) <= 0.3 * min(x1 - x0, rect.width):
            continue                                  # different column
        if y0 >= rect.y1 - 5 and (best is None or y0 < best[0]):
            best = (y0, b["start"])
    return best[1] if best else len(page["text"])


def _insert(page: dict, lines: list[dict], at: int, text: str, rect, block_idx) -> None:
    """Insert `text` as a new block at offset `at`, shifting later blocks and lines on this page."""
    sep_before = "\n\n" if at > 0 and not page["text"][:at].endswith("\n\n") else ""
    chunk = sep_before + text + ("\n\n" if at < len(page["text"]) else "")
    shift = len(chunk)
    for b in page["blocks"]:
        if b["start"] >= at and not (block_idx is not None and b is page["blocks"][block_idx]):
            b["start"] += shift
            b["end"] += shift
    for l in lines:
        if l["page"] == page["page"] and l["start"] >= at:
            l["start"] += shift
    page["text"] = page["text"][:at] + chunk + page["text"][at:]
    start = at + len(sep_before)
    bbox = [round(v, 1) for v in rect]
    if block_idx is not None:                         # the image block itself now holds its description
        page["blocks"][block_idx].update(type="figure", start=start, end=start + len(text), bbox=bbox)
    else:                                             # new ids go at the end so existing ids never change
        page["blocks"].append({"id": f"p{page['page']}_b{len(page['blocks'])}", "type": "figure",
                               "bbox": bbox, "start": start, "end": start + len(text)})


async def enrich(doc: dict, parsed: dict) -> dict:
    """Describe the images / figures of a parsed PDF or image document in place. Returns stats."""
    if not config.VISION or doc["format"] not in ("pdf", "image"):
        return {}
    src = doc.get("pdf_path") or doc["path"]
    pdf = pymupdf.open(src)
    pages = parsed["pages"]
    regions = _regions(pdf, pages, whole=doc["format"] == "image")
    if not regions:
        return {"figures": 0}
    progress.stage("vision", {"done": 0, "total": len(regions)})
    done = [0]
    what = "an image" if doc["format"] == "image" else "a figure"

    async def one(pno, rect):
        zoom = min(4.0, max(1.0, 1400 / max(rect.width, rect.height)))
        png = pdf[pno - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect).tobytes("png")
        try:
            return await describe(png, what, pno, doc.get("rel_path") or doc["name"])
        finally:
            done[0] += 1
            progress.detail(done=done[0])

    results = await asyncio.gather(*(one(pno, rect) for pno, rect, _ in regions))
    by_page = {p["page"]: p for p in pages}
    added, tokens = 0, 0
    # insert from the bottom of each page upward so earlier offsets stay valid
    order = sorted(zip(regions, results), key=lambda x: (x[0][0], -_insert_at(by_page[x[0][0]], x[0][1], x[0][2])))
    for (pno, rect, bidx), (text, usage) in order:
        tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if not text:
            continue
        page = by_page[pno]
        _insert(page, parsed["lines"], _insert_at(page, rect, bidx), MARK + text, rect, bidx)
        added += 1
    log.info("vision %s: %d/%d figures described", doc["name"], added, len(regions))
    return {"figures": len(regions), "described": added, "tokens": tokens, "model": config.VISION_MODEL}
