"""PDF -> pages (text, layout blocks with bbox, styled lines).

Text PDFs go through PyMuPDF. Scanned / image-only PDFs go through MinerU (OCR + layout):
the mineru.net cloud API by default, or a self-hosted server if PIDX_MINERU_URL is set.

Page dict (persisted):   {page, width, height, text, blocks:[{id, type, bbox, start, end}]}
Line dict (build only):  {page, text, size, bold, start, block, y0, top, bottom, level?}
`start`/`end` are character offsets into the page text; bbox is in PDF points.
"""

import collections
import io
import json
import logging
import re
import time
import zipfile

import httpx
import pymupdf

from . import config

log = logging.getLogger("pidx.parse")

_ROMAN = re.compile(r"^[ivxlcdm]+$", re.I)


def _norm_furniture(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _is_page_number(text: str) -> bool:
    t = text.strip().strip("-–— ").lower()
    return bool(t) and len(t) <= 6 and (t.isdigit() or bool(_ROMAN.match(t)) or t.startswith("page "))


def parse_pdf(path: str, mode: str = "auto") -> dict:
    """Returns {"pages": [...], "lines": [...], "parser": "pymupdf"|"mineru", "toc": [...]}."""
    doc = pymupdf.open(path)
    toc = [[lvl, title.strip(), page] for lvl, title, page in doc.get_toc() if title.strip()]
    if mode != "ocr":
        res = _parse_pymupdf(doc)
        chars = sum(len(p["text"]) for p in res["pages"])
        if mode == "text" or chars / max(doc.page_count, 1) >= 80:
            res["toc"] = toc
            return res
        log.info("%s looks scanned (%.0f chars/page), using MinerU", path, chars / max(doc.page_count, 1))
    res = _parse_mineru(path, doc)
    res["toc"] = toc
    return res


def _parse_pymupdf(doc) -> dict:
    raw = []  # per page: list of blocks, each with list of lines
    for pno, page in enumerate(doc, start=1):
        d = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_PRESERVE_IMAGES)
        blocks = []
        for b in d["blocks"]:
            if b["type"] == 1:
                blocks.append({"type": "image", "bbox": list(b["bbox"]), "lines": []})
                continue
            lines = [_line_from_spans(row) for row in _rows(b)]
            if lines:
                blocks.append({"type": "text", "bbox": list(b["bbox"]), "lines": lines})
        raw.append((pno, page.rect.width, page.rect.height, blocks))

    # Running headers/footers: short lines near the top/bottom edge that recur on many pages.
    edge_counts = collections.Counter()
    for pno, w, h, blocks in raw:
        seen = set()
        for b in blocks:
            for l in b["lines"]:
                if l["bbox"][1] < 0.08 * h or l["bbox"][3] > 0.92 * h:
                    seen.add(_norm_furniture(l["text"]))
        edge_counts.update(seen)
    n = len(raw)
    furniture = {k for k, c in edge_counts.items() if n >= 4 and c >= max(3, 0.25 * n)}

    pages, all_lines = [], []
    for pno, w, h, blocks in raw:
        parts, out_blocks, off = [], [], 0
        for b in blocks:
            if b["type"] == "image":
                out_blocks.append({"id": f"p{pno}_b{len(out_blocks)}", "type": "image",
                                   "bbox": _r(b["bbox"]), "start": off, "end": off})
                continue
            keep = []
            for l in b["lines"]:
                at_edge = l["bbox"][1] < 0.08 * h or l["bbox"][3] > 0.92 * h
                if at_edge and (_norm_furniture(l["text"]) in furniture or _is_page_number(l["text"])):
                    continue
                keep.append(l)
            if not keep:
                continue
            bidx = len(out_blocks)
            if parts:
                off += 2
            bstart = off
            for i, l in enumerate(keep):
                if i:
                    off += 1
                all_lines.append({"page": pno, "text": l["text"], "size": l["size"], "bold": l["bold"],
                                  "start": off, "block": bidx, "y0": l["bbox"][1],
                                  "top": l["bbox"][1] < 0.15 * h})
                off += len(l["text"])
            parts.append("\n".join(l["text"] for l in keep))
            out_blocks.append({"id": f"p{pno}_b{bidx}", "type": "text", "bbox": _r(b["bbox"]),
                               "start": bstart, "end": off})
        pages.append({"page": pno, "width": round(w, 1), "height": round(h, 1),
                      "text": "\n\n".join(parts), "blocks": out_blocks})
    return {"pages": pages, "lines": all_lines, "parser": "pymupdf"}


def _span_size(s) -> float:
    # Some (e.g. Type3 / TeX) PDFs report a tiny font size and scale via the matrix.
    return s["size"] if s["size"] >= 2 else s["bbox"][3] - s["bbox"][1]


def _rows(block):
    """Regroup a block's spans into visual rows by vertical position.

    PyMuPDF sometimes emits one "line" per word fragment; rows by baseline fix that."""
    spans = [s for l in block["lines"] for s in l["spans"] if s["text"].strip()]
    spans.sort(key=lambda s: ((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0]))
    rows = []
    for s in spans:
        yc, hgt = (s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][3] - s["bbox"][1]
        if rows and abs(yc - rows[-1]["yc"]) <= 0.5 * max(hgt, rows[-1]["h"]):
            rows[-1]["spans"].append(s)
        else:
            rows.append({"yc": yc, "h": hgt, "spans": [s]})
    return [sorted(r["spans"], key=lambda s: s["bbox"][0]) for r in rows]


def _line_from_spans(spans):
    text, prev = "", None
    for s in spans:
        t = s["text"]
        if prev is not None:
            gap = s["bbox"][0] - prev["bbox"][2]
            if gap > 0.15 * _span_size(s) and not text.endswith(" ") and not t.startswith(" "):
                text += " "
        text += t
        prev = s
    # dominant style = style of the most characters
    big = max(spans, key=lambda s: len(s["text"].strip()))
    n = sum(len(s["text"]) for s in spans)
    bold = sum(len(s["text"]) for s in spans if s["flags"] & 16 or "bold" in s["font"].lower())
    bbox = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
            max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
    return {"text": re.sub(r"\s+", " ", text).strip(), "size": round(_span_size(big), 1),
            "bold": bold >= 0.6 * n, "bbox": bbox}


def _r(b):
    return [round(x, 1) for x in b]


def _parse_mineru(path: str, doc) -> dict:
    """OCR + layout through MinerU; content_list items carry page_idx and bbox in 0-1000 page units."""
    if config.MINERU_API_KEY and not config.MINERU_URL:
        items = mineru_cloud(path, doc.page_count)
    else:
        items = _mineru_selfhosted(path)

    by_page = collections.defaultdict(list)
    for it in items:
        by_page[it.get("page_idx", 0) + 1].append(it)

    pages, all_lines = [], []
    for pno in range(1, doc.page_count + 1):
        w, h = doc[pno - 1].rect.width, doc[pno - 1].rect.height
        parts, out_blocks, off = [], [], 0
        for it in by_page.get(pno, []):
            typ = it.get("type", "text")
            if typ in ("header", "footer", "page_number", "page_footnote"):
                continue
            if typ == "table":
                text = "\n".join(filter(None, [" ".join(it.get("table_caption") or []),
                                               it.get("table_body", ""), " ".join(it.get("table_footnote") or [])]))
            elif typ in ("image", "chart"):
                text = "\n".join(filter(None, [" ".join(it.get(f"{typ}_caption") or it.get("img_caption") or []),
                                               it.get("content", ""), " ".join(it.get(f"{typ}_footnote") or [])]))
            else:
                text = it.get("text", "")
            text = text.strip()
            bb = it.get("bbox") or [0, 0, 1000, 1000]
            bbox = _r([bb[0] * w / 1000, bb[1] * h / 1000, bb[2] * w / 1000, bb[3] * h / 1000])
            bidx = len(out_blocks)
            if not text:
                out_blocks.append({"id": f"p{pno}_b{bidx}", "type": typ, "bbox": bbox, "start": off, "end": off})
                continue
            if parts:
                off += 2
            level = it.get("text_level")
            all_lines.append({"page": pno, "text": text.split("\n")[0][:300], "size": 0, "bold": bool(level),
                              "level": level, "start": off, "block": bidx, "y0": bbox[1],
                              "top": bbox[1] < 0.15 * h})
            out_blocks.append({"id": f"p{pno}_b{bidx}", "type": "title" if level else typ, "bbox": bbox,
                               "start": off, "end": off + len(text)})
            parts.append(text)
            off += len(text)
        pages.append({"page": pno, "width": round(w, 1), "height": round(h, 1),
                      "text": "\n\n".join(parts), "blocks": out_blocks})
    return {"pages": pages, "lines": all_lines, "parser": "mineru"}


def _mineru_selfhosted(path: str) -> list[dict]:
    with open(path, "rb") as f:
        r = httpx.post(f"{config.MINERU_URL}/file_parse",
                       files={"files": (path.rsplit("/", 1)[-1], f, "application/pdf")},
                       data={"return_md": "false", "return_content_list": "true", "lang_list": "ch",
                             "backend": "hybrid-auto-engine", "parse_method": "auto"},
                       timeout=3600, trust_env=False)
    r.raise_for_status()
    if r.headers.get("content-type", "").startswith("application/zip"):
        return _content_list_from_zip(r.content)
    items = next(iter(r.json()["results"].values()))["content_list"]
    return json.loads(items) if isinstance(items, str) else items


def _content_list_from_zip(data: bytes) -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(data))
    name = next(n for n in z.namelist() if n.endswith("content_list.json"))
    return json.loads(z.read(name))


MINERU_MAX_PAGES = 200   # per-file limit of the cloud API


def mineru_cloud(path: str, page_count: int, timeout=3600) -> list[dict]:
    """mineru.net batch API: request upload URLs, PUT the file(s), poll, download each result zip.

    Files over 200 pages are split locally into 200-page parts; page_idx is shifted back afterwards."""
    parts = []   # (name, bytes, first page index)
    src = pymupdf.open(path)
    for start in range(0, page_count, MINERU_MAX_PAGES):
        end = min(start + MINERU_MAX_PAGES, page_count) - 1
        if start == 0 and end == page_count - 1:
            data = open(path, "rb").read()
        else:
            part = pymupdf.open()
            part.insert_pdf(src, from_page=start, to_page=end)
            data = part.tobytes()
        parts.append((f"part{start // MINERU_MAX_PAGES:03d}.pdf", data, start))

    http = httpx.Client(timeout=300, trust_env=False)
    auth = {"Authorization": f"Bearer {config.MINERU_API_KEY}"}
    r = http.post(f"{config.MINERU_API}/file-urls/batch", headers=auth, json={
        "files": [{"name": n, "is_ocr": True, "data_id": n[:-4]} for n, _, _ in parts],
        "model_version": config.MINERU_MODEL, "language": "ch", "enable_formula": True, "enable_table": True})
    r.raise_for_status()
    js = r.json()
    if js.get("code") != 0:
        raise RuntimeError(f"MinerU batch request failed: {js.get('code')} {js.get('msg')}")
    batch_id, urls = js["data"]["batch_id"], js["data"]["file_urls"]
    for (n, data, _), url in zip(parts, urls):
        http.put(url, content=data, timeout=600).raise_for_status()
    log.info("MinerU batch %s: uploaded %d part(s)", batch_id, len(parts))

    deadline = time.time() + timeout
    while True:
        r = http.get(f"{config.MINERU_API}/extract-results/batch/{batch_id}", headers=auth)
        r.raise_for_status()
        results = {x.get("data_id") or x["file_name"][:-4]: x for x in r.json()["data"]["extract_result"]}
        states = [results.get(n[:-4], {}).get("state", "waiting-file") for n, _, _ in parts]
        if any(s == "failed" for s in states):
            errs = [x.get("err_msg") for x in results.values() if x.get("state") == "failed"]
            raise RuntimeError(f"MinerU failed: {errs}")
        if all(s == "done" for s in states):
            break
        if time.time() > deadline:
            raise TimeoutError(f"MinerU batch {batch_id} not done after {timeout}s: {states}")
        time.sleep(5)

    items = []
    for n, _, offset in parts:
        z = http.get(results[n[:-4]]["full_zip_url"], timeout=600)
        z.raise_for_status()
        for it in _content_list_from_zip(z.content):
            it["page_idx"] = it.get("page_idx", 0) + offset
            items.append(it)
    return items
