"""Files -> pages (text, layout blocks with bbox, styled lines).

Text PDFs go through PyMuPDF. Scanned / image-only PDFs go through MinerU (OCR + layout):
the mineru.net cloud API by default, or a self-hosted server if PIDX_MINERU_URL is set.
Images are wrapped into a one-page PDF at upload time and take the scanned-PDF path.
Office files (doc/docx/ppt/pptx/xls/xlsx) are parsed by MinerU natively: there are no layout
coordinates, so blocks have bbox=None and the text is cut into "virtual pages" (one per slide /
sheet, and long documents or sheets split by size).

Page dict (persisted):   {page, width, height, text, label?, blocks:[{id, type, bbox, start, end}]}
Line dict (build only):  {page, text, size, bold, start, block, y0, top, bottom, level?}
`start`/`end` are character offsets into the page text; bbox is in PDF points (None for Office).
"""

import collections
import html.parser
import io
import json
import logging
import mimetypes
import re
import time
import zipfile
from pathlib import Path

import httpx
import pymupdf

from . import config, progress
from .llm import approx_tokens

log = logging.getLogger("pidx.parse")

_ROMAN = re.compile(r"^[ivxlcdm]+$", re.I)


def _norm_furniture(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _is_page_number(text: str) -> bool:
    t = text.strip().strip("-–— ").lower()
    return bool(t) and len(t) <= 6 and (t.isdigit() or bool(_ROMAN.match(t)) or t.startswith("page "))


def file_format(name: str) -> str | None:
    """"pdf" | "image" | "office" | "text" | None (unsupported), by extension."""
    ext = Path(name).suffix.lower()
    if ext in config.PDF_EXTS:
        return "pdf"
    if ext in config.IMAGE_EXTS:
        return "image"
    if ext in config.OFFICE_EXTS:
        return "office"
    if ext in config.TEXT_EXTS:
        return "text"
    return None


def check_file(path: str, fmt: str) -> None:
    """Raise ValueError if the file is not what its extension says."""
    head = open(path, "rb").read(8)
    if fmt == "pdf":
        try:
            pymupdf.open(path).close()
        except Exception:
            raise ValueError("not a readable PDF")
    elif fmt == "image":
        from PIL import Image
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            raise ValueError("not a readable image")
    elif fmt == "office":
        ext = Path(path).suffix.lower()
        ok = head.startswith(b"PK\x03\x04") if ext.endswith("x") else head.startswith(b"\xd0\xcf\x11\xe0")
        if not ok:
            raise ValueError(f"not a valid {ext} file")
    elif fmt == "text":
        read_text(path)


def image_to_pdf(src: str, dst: str) -> None:
    """Wrap an image (first frame for gif) into a one-page PDF; page size follows the pixel size at 150 dpi."""
    from PIL import Image
    with Image.open(src) as im:
        im.seek(0)
        rgb = im.convert("RGB")
    rgb.save(dst, "PDF", resolution=150.0)


def parse_file(path: str, fmt: str, mode: str = "auto", pdf_path: str | None = None) -> dict:
    """Any supported file -> parsed pages. Images are parsed from their PDF wrapper (`pdf_path`)."""
    if fmt == "pdf":
        return parse_pdf(path, mode)
    if fmt == "image":
        return parse_pdf(pdf_path or path, "ocr")
    if fmt == "office":
        return parse_office(path)
    if fmt == "text":
        return parse_text(path)
    raise ValueError(f"unsupported format {fmt!r}")


def _need_mineru():
    if not (config.MINERU_API_KEY or config.MINERU_URL):
        raise RuntimeError("this file needs MinerU (scanned PDF / image / Office): set MINERU_API_KEY or PIDX_MINERU_URL")


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


def mineru(path: str, page_count: int | None = None) -> list[dict]:
    """MinerU content_list for any file it accepts (cloud API, or self-hosted if PIDX_MINERU_URL is set)."""
    _need_mineru()
    progress.stage("mineru", {"state": "uploading"})
    if config.MINERU_API_KEY and not config.MINERU_URL:
        return mineru_cloud(path, page_count)
    return _mineru_selfhosted(path)


def _item_text(it: dict) -> str:
    typ = it.get("type", "text")
    if typ == "table":
        body = it.get("table_body", "")
        if "<table" in body:
            body = html_table_to_md(body)
        return "\n".join(filter(None, [" ".join(it.get("table_caption") or []), body,
                                       " ".join(it.get("table_footnote") or [])])).strip()
    if typ in ("image", "chart"):
        return "\n".join(filter(None, [" ".join(it.get(f"{typ}_caption") or it.get("img_caption") or []),
                                       it.get("content", ""), " ".join(it.get(f"{typ}_footnote") or [])])).strip()
    if typ == "list":
        return "\n".join(it.get("list_items") or []).strip()
    return (it.get("text") or "").strip()


def _parse_mineru(path: str, doc) -> dict:
    """OCR + layout through MinerU; content_list items carry page_idx and bbox in 0-1000 page units."""
    items = mineru(path, doc.page_count)

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
            text = _item_text(it)
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
                       files={"files": (path.rsplit("/", 1)[-1], f,
                                        mimetypes.guess_type(path)[0] or "application/octet-stream")},
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


_STATE_ORDER = ["waiting-file", "pending", "running", "converting", "done", "failed"]
MINERU_MAX_PAGES = 200   # per-file limit of the cloud API


def mineru_cloud(path: str, page_count: int | None = None, timeout=3600) -> list[dict]:
    """mineru.net batch API: request upload URLs, PUT the file(s), poll, download each result zip.

    PDFs over 200 pages are split locally into 200-page parts; page_idx is shifted back afterwards.
    Other files (Office) are sent as they are."""
    parts = []   # (name, bytes, first page index)
    ext = Path(path).suffix.lower()
    if ext != ".pdf":
        parts.append((f"file{ext}", open(path, "rb").read(), 0))
    else:
        src = pymupdf.open(path)
        page_count = page_count or src.page_count
        for start in range(0, page_count, MINERU_MAX_PAGES):
            end = min(start + MINERU_MAX_PAGES, page_count) - 1
            if start == 0 and end == page_count - 1:
                data = open(path, "rb").read()
            else:
                part = pymupdf.open()
                part.insert_pdf(src, from_page=start, to_page=end)
                data = part.tobytes()
            parts.append((f"part{start // MINERU_MAX_PAGES:03d}.pdf", data, start))

    def stem(n):
        return n.rsplit(".", 1)[0]

    http = httpx.Client(timeout=300, trust_env=False)
    auth = {"Authorization": f"Bearer {config.MINERU_API_KEY}"}
    r = http.post(f"{config.MINERU_API}/file-urls/batch", headers=auth, json={
        "files": [{"name": n, "is_ocr": n.endswith(".pdf"), "data_id": stem(n)} for n, _, _ in parts],
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
    errors = 0
    while True:
        try:            # a poll can fail transiently (403 / 429 / 5xx / timeout); only a run of failures counts
            r = http.get(f"{config.MINERU_API}/extract-results/batch/{batch_id}", headers=auth)
            r.raise_for_status()
            errors = 0
        except httpx.HTTPError as e:
            errors += 1
            log.warning("MinerU poll %s failed (%d in a row): %r", batch_id, errors, e)
            if errors >= 5 or time.time() > deadline:
                raise
            time.sleep(5 * errors)
            continue
        results = {x.get("data_id") or stem(x["file_name"]): x for x in r.json()["data"]["extract_result"]}
        states = [results.get(stem(n), {}).get("state", "waiting-file") for n, _, _ in parts]
        prog = [results.get(stem(n), {}).get("extract_progress") or {} for n, _, _ in parts]
        progress.detail(state=min(states, key=_STATE_ORDER.index) if all(s in _STATE_ORDER for s in states) else states[0],
                        pages_done=sum(int(p.get("extracted_pages") or 0) for p in prog) or None,
                        pages_total=sum(int(p.get("total_pages") or 0) for p in prog) or page_count, parts=len(parts))
        if any(s == "failed" for s in states):
            errs = [x.get("err_msg") for x in results.values() if x.get("state") == "failed"]
            raise RuntimeError(f"MinerU failed: {errs}")
        if all(s == "done" for s in states):
            break
        if time.time() > deadline:
            raise TimeoutError(f"MinerU batch {batch_id} not done after {timeout}s: {states}")
        time.sleep(5)

    progress.detail(state="downloading")
    items = []
    for n, _, offset in parts:
        z = http.get(results[stem(n)]["full_zip_url"], timeout=600)
        z.raise_for_status()
        for it in _content_list_from_zip(z.content):
            it["page_idx"] = it.get("page_idx", 0) + offset
            items.append(it)
    return items


# --------------------------------------------------------------------------- Office (MinerU native)

_KIND = {".ppt": "幻灯片", ".pptx": "幻灯片", ".xls": "工作表", ".xlsx": "工作表"}
_INLINE_TAG = re.compile(r"</?(strong|b|u|em|i|span|sup|sub|font|s)(\s[^>]*)?>", re.I)


def parse_office(path: str) -> dict:
    """doc/docx/ppt/pptx/xls/xlsx through MinerU. Its page_idx is the slide (ppt) or sheet (xls);
    Word documents come back as one page. Every source page is cut into virtual pages of about
    VIRTUAL_PAGE_TOKENS, preferring to break before a heading; big tables are split by rows with
    the header row repeated."""
    items = mineru(path)
    return office_pages(items, Path(path).suffix.lower())


def read_text(path: str) -> str:
    raw = open(path, "rb").read()
    if b"\x00" in raw[:4096]:
        raise ValueError("not a text file")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("text file is neither UTF-8 nor GBK")


def parse_text(path: str) -> dict:
    """Markdown / plain text, parsed locally (no MinerU): ATX headings become titled items, pipe
    tables table items, fenced code and paragraphs text items; then the Office virtual-page path."""
    progress.stage("parse")
    items, para, fence = [], [], None

    def flush():
        if para:
            text = "\n".join(para).strip()
            if text:
                table = all(l.lstrip().startswith("|") for l in para if l.strip())
                items.append({"type": "table", "table_body": text} if table else {"type": "text", "text": text})
            para.clear()

    for line in read_text(path).splitlines():
        if fence is not None:
            para.append(line)
            if line.strip().startswith(fence):
                fence = None
                flush()
            continue
        st = line.strip()
        if st.startswith(("```", "~~~")):
            flush()
            fence = st[:3]
            para.append(line)
            continue
        m = re.match(r"(#{1,6})\s+(.+?)\s*#*\s*$", st)
        if m:
            flush()
            items.append({"type": "text", "text": m.group(2), "text_level": len(m.group(1))})
        elif not st:
            flush()
        else:
            if para and (st.startswith("|") != para[-1].lstrip().startswith("|")):
                flush()
            para.append(line)
    flush()
    res = office_pages(items, Path(path).suffix.lower())
    res["parser"] = "markdown" if Path(path).suffix.lower() in (".md", ".markdown") else "text"
    return res


def office_pages(items: list[dict], ext: str) -> dict:
    budget = config.VIRTUAL_PAGE_TOKENS
    groups = collections.defaultdict(list)
    for it in items:
        if it.get("type") in ("header", "footer", "page_number", "page_footnote"):
            continue
        groups[it.get("page_idx", 0)].append(it)

    pages, all_lines = [], []
    kind = _KIND.get(ext)
    for src_idx in sorted(groups):
        units = []      # (type, text, level)
        for it in groups[src_idx]:
            typ, level = it.get("type", "text"), it.get("text_level")
            text = _item_text(it)
            if not text:
                continue
            if typ != "table":     # .doc output carries inline HTML (<strong><u>..</u></strong>)
                whole_bold = bool(re.fullmatch(r"(<(strong|b)>)(.+)(</\2>)", text, re.S))
                text = _INLINE_TAG.sub("", text).strip()
                if whole_bold:
                    text = f"**{text}**"
            m = re.fullmatch(r"\*\*(.+?)\*\*", text, re.S)   # whole-paragraph bold (docx headings)
            bold = bool(m) or bool(level)
            if m:
                text = m.group(1).strip()
            if typ == "table" and approx_tokens(text) > budget:
                units.extend(("table", t, None, False) for t in _split_table(text, budget))
            else:
                units.append((typ, text, level, bold))
        if not units:
            continue
        first_title = next((u[1] for u in units if u[2]), "")
        chunks, cur, tok = [], [], 0
        for u in units:
            t = approx_tokens(u[1])
            heading = bool(u[2])
            if cur and (tok + t > budget or heading and tok > 0.6 * budget):
                chunks.append(cur)
                cur, tok = [], 0
            cur.append(u)
            tok += t
        if cur:
            chunks.append(cur)
        for ci, chunk in enumerate(chunks):
            pno = len(pages) + 1
            parts, blocks, off = [], [], 0
            for typ, text, level, bold in chunk:
                if parts:
                    off += 2
                bidx = len(blocks)
                all_lines.append({"page": pno, "text": text.split("\n")[0][:300], "size": 0, "bold": bold,
                                  "level": level, "start": off, "block": bidx, "y0": 0, "top": bidx == 0})
                blocks.append({"id": f"p{pno}_b{bidx}", "type": "title" if level else typ, "bbox": None,
                               "start": off, "end": off + len(text)})
                parts.append(text)
                off += len(text)
            label = None
            if kind:
                label = f"{kind} {src_idx + 1}" + (f" · {first_title[:40]}" if kind == "工作表" and first_title else "")
                if len(chunks) > 1:
                    label += f"（{ci + 1}/{len(chunks)}）"
            pages.append({"page": pno, "width": None, "height": None, "text": "\n\n".join(parts),
                          "label": label, "blocks": blocks})
    if not pages:
        pages = [{"page": 1, "width": None, "height": None, "text": "", "label": None, "blocks": []}]
    return {"pages": pages, "lines": all_lines, "parser": "mineru-office"}


def _split_table(md: str, budget: int) -> list[str]:
    """Markdown table -> pieces under `budget` tokens, each repeating the header and separator rows."""
    rows = md.split("\n")
    if len(rows) < 3 or not rows[1].startswith("|"):
        n = budget * 3
        return [md[i:i + n] for i in range(0, len(md), n)]
    head, out, cur, tok = rows[:2], [], [], 0
    htok = approx_tokens("\n".join(head))
    for r in rows[2:]:
        t = approx_tokens(r)
        if cur and htok + tok + t > budget:
            out.append("\n".join(head + cur))
            cur, tok = [], 0
        cur.append(r)
        tok += t
    if cur:
        out.append("\n".join(head + cur))
    return out


class _TableParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.span = [], None, None, (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            self.span = (int(a.get("colspan") or 1), int(a.get("rowspan") or 1))
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append((re.sub(r"\s+", " ", "".join(self.cell)).strip(), *self.span))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def html_table_to_md(html_text: str) -> str:
    """MinerU table HTML -> markdown table. colspan/rowspan cells are repeated so columns line up."""
    p = _TableParser()
    try:
        p.feed(html_text)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_text)
    grid, carry = [], {}      # carry: col -> (text, rows left) from rowspans above
    for row in p.rows:
        out, col, cells = [], 0, list(row)
        while cells or col in carry:
            if col in carry:
                text, left = carry[col]
                out.append(text)
                if left > 1:
                    carry[col] = (text, left - 1)
                else:
                    del carry[col]
                col += 1
                continue
            text, cs, rs = cells.pop(0)
            for _ in range(max(cs, 1)):
                out.append(text)
                if rs > 1:
                    carry[col] = (text, rs - 1)
                col += 1
        grid.append(out)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    fmt = lambda r: "| " + " | ".join(c.replace("|", "\\|") for c in r) + " |"
    return "\n".join([fmt(grid[0]), "|" + "---|" * width] + [fmt(r) for r in grid[1:]])
