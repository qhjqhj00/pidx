"""Settings, read from the environment (and ./.env if present)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


DATA_DIR = Path(os.environ.get("PIDX_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "pidx.db"
FILES_DIR = DATA_DIR / "files"

# LLM (OpenAI-compatible). Backend "default" does all indexing (many calls, nobody waiting): it is
# PIDX_LLM_*, falling back to the team's Atria gateway (ATRIA_*, free, tailnet only).
LLM_BASE_URL = os.environ.get("PIDX_LLM_BASE_URL", os.environ.get("ATRIA_BASE_URL", "http://100.104.77.119:8100/v1"))
LLM_API_KEY = os.environ.get("PIDX_LLM_API_KEY", os.environ.get("ATRIA_API_KEY", ""))
LLM_MODEL = os.environ.get("PIDX_LLM_MODEL", os.environ.get("ATRIA_MODEL", "Atria-Dawn-Preview"))
# Models known to take image input (OpenAI-style image_url parts). PIDX_VISION_MODELS adds more.
VISION_MODELS = {"deepseek-flash", "deepseek-v4-flash",
                 *filter(None, os.environ.get("PIDX_VISION_MODELS", "").split(","))}
LLM_CONCURRENCY = _int("PIDX_LLM_CONCURRENCY", 24)   # Atria: <=100 per project


def _no_thinking(model: str) -> dict:
    # Atria and DeepSeek V4 think by default, and the thinking tokens eat max_tokens and time.
    # Other providers may reject the unknown field, so only send it where it is known to work.
    return {"thinking": {"type": "disabled"}} if any(k in model.lower() for k in ("atria", "deepseek")) else {}


# Named backends. Search can pick one per request (`llm` in the request body).
LLM_BACKENDS = {
    "default": {"base_url": LLM_BASE_URL, "api_key": LLM_API_KEY, "model": LLM_MODEL,
                "extra": _no_thinking(LLM_MODEL), "label": f"{LLM_MODEL}（建索引用的模型）"},
    "deepseek": {"base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                 "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
                 "model": os.environ.get("PIDX_DEEPSEEK_MODEL", "deepseek-flash"),
                 "extra": _no_thinking("deepseek"), "label": "DeepSeek V4.1 Flash"},
}
for _b in LLM_BACKENDS.values():
    _b["vision"] = _b["model"] in VISION_MODELS
LLM_BACKENDS = {k: v for k, v in LLM_BACKENDS.items() if v["api_key"] or k == "default"}
SEARCH_BACKEND = os.environ.get("PIDX_SEARCH_BACKEND", "deepseek" if "deepseek" in LLM_BACKENDS else "default")
if SEARCH_BACKEND not in LLM_BACKENDS:
    SEARCH_BACKEND = "default"
LLM_TIMEOUT = _int("PIDX_LLM_TIMEOUT", 300)

# Optional services
# Embeddings: SiliconFlow (OpenAI-compatible). Same bge-m3 model as before, so stored vectors stay valid.
EMBED_URL = os.environ.get("PIDX_EMBED_URL", "https://api.siliconflow.cn/v1")
EMBED_API_KEY = os.environ.get("PIDX_EMBED_API_KEY", os.environ.get("SILICON_FLOW_API_KEY", ""))
EMBED_MODEL = os.environ.get("PIDX_EMBED_MODEL", "BAAI/bge-m3")

# EmbeddingGemma 2 (text + image in one 768-d space). Setting PIDX_GEMMA_EMBED_URL switches the
# embedding path to it: task prefixes on text, image blocks cropped from the page and embedded too,
# and search returns matching images. Unset -> the bge-m3 path above, unchanged.
GEMMA_EMBED_URL = os.environ.get("PIDX_GEMMA_EMBED_URL", "").rstrip("/")
EMBED_MULTIMODAL = bool(GEMMA_EMBED_URL)
if EMBED_MULTIMODAL:
    EMBED_URL = GEMMA_EMBED_URL
    EMBED_API_KEY = os.environ.get("PIDX_GEMMA_EMBED_API_KEY", "")
    EMBED_MODEL = os.environ.get("PIDX_GEMMA_EMBED_MODEL", "embeddinggemma-2")
CROPS_DIR = DATA_DIR / "crops"                                   # cropped image blocks, one PNG each
IMAGE_MIN_PT = _int("PIDX_IMAGE_MIN_PT", 48)                     # skip icons / logos: both sides >= this (pt)
IMAGE_MIN_AREA = float(os.environ.get("PIDX_IMAGE_MIN_AREA", 0.01))   # ... and >= this share of the page
IMAGE_MAX_PER_DOC = _int("PIDX_IMAGE_MAX_PER_DOC", 300)
IMAGE_MAX_PX = _int("PIDX_IMAGE_MAX_PX", 1024)                   # longest side of a crop
IMAGE_TOP_K = _int("PIDX_IMAGE_TOP_K", 6)                        # images returned per search
# Text->image cosines are compressed: unrelated pictures land around 0.50-0.56, a matching one at
# 0.63-0.75 with a clear gap to the next. Keep hits above the floor and within the margin of the best.
IMAGE_MIN_SCORE = float(os.environ.get("PIDX_IMAGE_MIN_SCORE", 0.6))
IMAGE_REL_MARGIN = float(os.environ.get("PIDX_IMAGE_REL_MARGIN", 0.05))
# Answering with a vision model: up to this many cropped images (matched images first, then the
# figures inside the sections being read) go into the answer prompt. 0 = text only.
ANSWER_IMAGES = _int("PIDX_ANSWER_IMAGES", 4)

# OCR / layout for scanned PDFs: MinerU cloud API (mineru.net). Setting PIDX_MINERU_URL instead
# points at a self-hosted MinerU server (`mineru-api`) and needs no key.
MINERU_API_KEY = os.environ.get("MINERU_API_KEY", "")
MINERU_API = os.environ.get("PIDX_MINERU_API", "https://mineru.net/api/v4")
MINERU_MODEL = os.environ.get("PIDX_MINERU_MODEL", "pipeline")      # pipeline | vlm
MINERU_URL = os.environ.get("PIDX_MINERU_URL", "")

# Input formats. PDFs and images get page images and bbox highlighting; Office files are parsed by
# MinerU natively (no layout coordinates) and cut into "virtual pages" of about this many tokens.
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}
OFFICE_EXTS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
SUPPORTED_EXTS = PDF_EXTS | IMAGE_EXTS | OFFICE_EXTS
VIRTUAL_PAGE_TOKENS = _int("PIDX_VIRTUAL_PAGE_TOKENS", 1500)

# Zip upload -> collection (directory tree + one document per supported file)
ZIP_MAX_FILES = _int("PIDX_ZIP_MAX_FILES", 1000)
ZIP_MAX_FILE_MB = _int("PIDX_ZIP_MAX_FILE_MB", 200)
ZIP_MAX_TOTAL_MB = _int("PIDX_ZIP_MAX_TOTAL_MB", 2048)

# Retrieval. "hybrid": node embeddings recall candidate subtrees (anchors, each small enough for one
# prompt), then the LLM reads those subtrees in parallel. "tree": pure LLM reading of the whole tree
# (layered expansion when it is too big). Hybrid falls back to tree when embeddings are unavailable.
RETRIEVAL = os.environ.get("PIDX_RETRIEVAL", "hybrid")
ANCHOR_TOKENS = _int("PIDX_ANCHOR_TOKENS", 3000)      # skeleton size of one anchor subtree
ANCHORS_PER_DOC = _int("PIDX_ANCHORS_PER_DOC", 3)     # subtrees read in parallel per document
DOC_SHORTLIST = _int("PIDX_DOC_SHORTLIST", 30)        # documents shown to the LLM when picking documents

# Tree shape (same defaults as open-source PageIndex)
MAX_PAGES_PER_NODE = _int("PIDX_MAX_PAGES_PER_NODE", 10)
MAX_TOKENS_PER_NODE = _int("PIDX_MAX_TOKENS_PER_NODE", 20000)
MAX_SPLIT_DEPTH = _int("PIDX_MAX_SPLIT_DEPTH", 3)
INDEX_WORKERS = _int("PIDX_INDEX_WORKERS", 2)

# Requests arriving through the public reverse proxy (they carry X-Forwarded-For) must present this
# token (?token=, cookie, or Bearer). Direct access (localhost / LAN / tailnet) is not gated.
PUBLIC_TOKEN = os.environ.get("PIDX_PUBLIC_TOKEN", "")
