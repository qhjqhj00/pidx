"""Image blocks -> cropped PNGs (EmbeddingGemma path).

Both parsers already record where images are: PyMuPDF keeps the bbox of every embedded image, and
MinerU's content_list gives image / chart items a bbox (pidx converts it to page points). So the crop
is taken from the document's own PDF at that bbox, which works the same for text PDFs, scanned
PDFs and single images, and is sharper than MinerU's own thumbnails. Office files have no page
geometry and are skipped.

Crops live at CROPS_DIR/<doc_id>/<image_id>.png; image_id is the block id ("p3_b5"), or "p1_full"
for a whole uploaded image.
"""

import logging
import re
import shutil
from pathlib import Path

import pymupdf

from . import config, tree

log = logging.getLogger("pidx.images")

IMAGE_TYPES = ("image", "chart")
IMAGE_ID = re.compile(r"^p\d+_(b\d+|full)$")


def crop_path(doc_id: str, image_id: str) -> Path:
    return config.CROPS_DIR / doc_id / f"{image_id}.png"


def remove_crops(doc_id: str) -> None:
    shutil.rmtree(config.CROPS_DIR / doc_id, ignore_errors=True)


def node_at(nodes, page: int, off: int):
    """Deepest tree node whose range contains (page, offset), or None."""
    pos = (page, off)
    best = None
    for n in tree.walk(nodes):
        start = tuple(n["start"])
        end = tuple(n["end"]) if n.get("end") else None
        if start <= pos and (end is None or pos < end):
            best = n          # walk is pre-order, so a later match is deeper
    return best


def _big_enough(bbox, w, h) -> bool:
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return (bw >= config.IMAGE_MIN_PT and bh >= config.IMAGE_MIN_PT
            and bw * bh >= config.IMAGE_MIN_AREA * w * h)


def _render(page, rect) -> bytes:
    longest = max(rect.width, rect.height) or 1
    zoom = max(0.5, min(3.0, config.IMAGE_MAX_PX / longest))
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect).tobytes("png")


def extract(doc: dict, pages: list[dict], nodes: list) -> list[dict]:
    """Crop every image block worth embedding. Returns [dict(image_id, page, bbox, node_id, caption,
    png)] and writes each crop to disk. bbox is in page points, like the blocks."""
    pdf_path = doc.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return []
    out = []
    remove_crops(doc["id"])
    (config.CROPS_DIR / doc["id"]).mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(pdf_path)
    try:
        for p in pages:
            if p["page"] > pdf.page_count or not p.get("width"):
                continue
            w, h = p["width"], p["height"]
            pg = pdf[p["page"] - 1]
            found = []
            for b in p["blocks"]:
                if b.get("type") not in IMAGE_TYPES or not b.get("bbox") or not _big_enough(b["bbox"], w, h):
                    continue
                found.append((b["id"], b["bbox"], b["start"], p["text"][b["start"]:b["end"]].strip()))
            # an uploaded photo / screenshot is itself the image, whatever MinerU found inside it
            if doc.get("format") == "image" and p["page"] == 1:
                covered = sum((x[1][2] - x[1][0]) * (x[1][3] - x[1][1]) for x in found)
                if covered < 0.6 * w * h:
                    found.insert(0, ("p1_full", [0, 0, w, h], 0, p["text"][:300].strip()))
            for image_id, bbox, start, caption in found:
                if len(out) >= config.IMAGE_MAX_PER_DOC:
                    log.info("%s: image cap %d reached", doc["id"], config.IMAGE_MAX_PER_DOC)
                    return out
                try:
                    png = _render(pg, pymupdf.Rect(bbox) & pg.rect)
                except Exception as e:
                    log.warning("%s %s: crop failed: %r", doc["id"], image_id, e)
                    continue
                crop_path(doc["id"], image_id).write_bytes(png)
                n = node_at(nodes, p["page"], start)
                out.append({"image_id": image_id, "page": p["page"], "bbox": bbox,
                            "node_id": n["node_id"] if n else None, "caption": caption[:300], "png": png})
    finally:
        pdf.close()
    return out
