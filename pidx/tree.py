"""Build a PageIndex-style tree from parsed pages.

Strategy, in order:
  1. PDF bookmarks, when present and sane.
  2. Layout heading candidates (font size / bold / numbering; MinerU title levels),
     organised into a hierarchy by the LLM. The LLM only picks candidate ids, so
     every heading is anchored to a real line on a real page.
  3. LLM reads page text directly and names the headings; each one is verified
     by locating it on its page.
Then any leaf longer than MAX_PAGES_PER_NODE / MAX_TOKENS_PER_NODE is split by
re-running 2 (then 3) inside its range, recursively.

Positions are (page, char offset in page text); node text is the slice between
a node's start and the next node's start.
"""

import asyncio
import collections
import difflib
import logging
import re

from . import config
from .llm import approx_tokens, chat, chat_json

log = logging.getLogger("pidx.tree")

_DOT_LEADER = re.compile(r"(\.{3,}|…{2,}|\s{3,})\s*[\divxlc]+\s*$", re.I)
_NUMBERED = re.compile(r"^((\d+(\.\d+)*\.?)|([A-Z]\.)|([IVX]+\.)|(item\s+\d+[a-z]?\.?)|(chapter|section|part|appendix)\b"
                       r"|(第[一二三四五六七八九十百\d]+[章节部分篇条])|([一二三四五六七八九十]+、))\s*", re.I)


# --------------------------------------------------------------------------- text helpers

def norm(s: str) -> str:
    s = re.sub(r"[\s ]+", " ", s).strip().lower()
    return re.sub(r"[^\w　-鿿 ]", "", s)


def text_between(pages: list[dict], start, end) -> str:
    """Text from position start=(page, off) up to end=(page, off), exclusive. end=None -> doc end."""
    (sp, so) = start
    (ep, eo) = end if end else (len(pages), len(pages[-1]["text"]))
    if sp == ep:
        return pages[sp - 1]["text"][so:eo]
    out = [pages[sp - 1]["text"][so:]]
    for p in range(sp + 1, ep):
        out.append(pages[p - 1]["text"])
    out.append(pages[ep - 1]["text"][:eo])
    return "\n\n".join(x for x in out if x)


def end_page(pages, start, end) -> int:
    """Last page that holds content of a range ending at `end` (exclusive)."""
    if end is None:
        return len(pages)
    ep, eo = end
    if eo == 0 or not pages[ep - 1]["text"][:eo].strip():
        ep -= 1
    return max(ep, start[0])


def locate(title: str, page: int, lines_by_page: dict, pages: list[dict]) -> int | None:
    """Offset of the line(s) on `page` that best match `title`, or None."""
    t = norm(title)
    if not t:
        return None
    best, best_off = 0.0, None
    lines = lines_by_page.get(page, [])
    for i, l in enumerate(lines):
        joined = ""
        for j in range(i, min(i + 4, len(lines))):   # headings can wrap over a few lines
            joined = (joined + " " + lines[j]["text"]).strip()
            n = norm(joined)
            if not n:
                continue
            if n == t or (len(t) >= 12 and n.startswith(t)) or (len(n) >= 12 and t.startswith(n) and j == i + 3):
                return l["start"]
            r = difflib.SequenceMatcher(None, n, t).ratio()
            if r > best:
                best, best_off = r, l["start"]
            if len(n) > len(t) + 20:
                break
    if best >= 0.8:
        return best_off
    idx = norm(pages[page - 1]["text"]).find(t) if len(t) >= 8 else -1
    if idx >= 0:   # rough fallback: map normalised index to the nearest line start
        before = [l for l in lines if norm(pages[page - 1]["text"][:l["start"]]).__len__() <= idx]
        return before[-1]["start"] if before else 0
    return None


# --------------------------------------------------------------------------- strategy 1: bookmarks

def from_bookmarks(toc, pages, lines_by_page):
    n = len(pages)
    entries = [(lvl, title, p) for lvl, title, p in toc if 1 <= p <= n]
    if len(entries) < 3:
        return None
    ordered = sum(1 for a, b in zip(entries, entries[1:]) if b[2] >= a[2])
    if ordered < 0.9 * (len(entries) - 1):
        return None
    flat = []
    for lvl, title, p in entries:
        off = locate(title, p, lines_by_page, pages)
        flat.append({"title": re.sub(r"\s+", " ", title).strip(), "level": lvl, "start": (p, off or 0)})
    return flat


# --------------------------------------------------------------------------- strategy 2: layout candidates

def body_size(lines) -> float:
    c = collections.Counter()
    for l in lines:
        if l["size"]:
            c[l["size"]] += len(l["text"])
    return c.most_common(1)[0][0] if c else 0.0


def candidates(lines, body: float, relaxed=False):
    """Heading-looking lines; consecutive lines of one heading are merged."""
    out = []
    prev_block = None
    for l in lines:
        text = l["text"].strip()
        first_in_block = (l["page"], l["block"]) != prev_block
        prev_block = (l["page"], l["block"])
        if not text or len(text) > 160 or not re.search(r"[A-Za-z　-鿿]", text) and not text.strip().isdigit():
            continue
        if _DOT_LEADER.search(text):
            continue
        if l.get("level"):                       # MinerU title
            ok = True
        else:
            bigger = l["size"] >= body + (0.5 if relaxed else 0.9)
            bold_head = l["bold"] and l["size"] >= body - 0.6 and len(text) <= 120 and first_in_block
            numbered = relaxed and first_in_block and len(text) <= 100 and bool(_NUMBERED.match(text))
            ok = bigger or bold_head or numbered
        if not ok:
            continue
        c = {"page": l["page"], "start": l["start"], "size": l["size"], "bold": l["bold"],
             "level": l.get("level"), "block": l["block"], "text": text, "y0": l["y0"]}
        p = out[-1] if out else None
        if p and p["page"] == c["page"] and p["size"] == c["size"] and p["bold"] == c["bold"] and \
                (p["block"] == c["block"] or re.fullmatch(r"[\dA-Z.]{1,4}", p["text"])) and len(p["text"]) < 160:
            p["text"] = f"{p['text']} {text}"
            continue
        out.append(c)
    # drop lines that repeat on many pages (residual running heads)
    cnt = collections.Counter(norm(c["text"]) for c in out)
    return [c for c in out if cnt[norm(c["text"])] <= 3 or len(out) < 10]


def toc_hint(pages, lines) -> str:
    """Text of table-of-contents pages near the front, if any."""
    out = []
    for p in pages[:20]:
        t = p["text"]
        pl = [l for l in lines if l["page"] == p["page"]]
        leaders = sum(1 for l in pl if _DOT_LEADER.search(l["text"]))
        if leaders >= 5 or re.search(r"^\s*(table of )?contents\s*$|^\s*目\s*录\s*$", t[:300], re.I | re.M) and leaders >= 2:
            out.append(f"[page {p['page']}]\n{t}")
    s = "\n\n".join(out)
    return s[:24000]


SELECT_PROMPT = """You are reconstructing the table of contents of a document from heading candidates found by layout analysis.
Each candidate line is: id | page | font size | bold | text.  Body text font size is {body}.
Candidates include false positives: figure/table/chart titles and labels, table cells, emphasized phrases, names, running headers.
{scope}
{hint}
Pick the candidates that are real section headings and give each a hierarchy level (1 = top level{level_note}).
Headings with the same styling (size, bold, numbering scheme) are normally the same level.
Keep figure, table and chart titles out. Clean up the title text (fix spacing, drop trailing page numbers), but do not paraphrase.
Return JSON {{"headings": [{{"id": <int>, "level": <int>, "title": "<title>"}}]}} in document order.

Candidates:
{cands}"""


async def select_headings(cands, body, *, scope_title=None, hint="", chunk=400):
    if not cands:
        return []
    scope = f"All candidates come from inside the section \"{scope_title}\"; return only its subsections." if scope_title else ""
    level_note = ", i.e. a direct subsection of that section" if scope_title else ""
    results = []
    for i in range(0, len(cands), chunk):
        part = cands[i:i + chunk]
        listing = "\n".join(f"{i + k} | {c['page']} | {c['size'] or ('L' + str(c['level']))} | "
                            f"{'B' if c['bold'] else '-'} | {c['text']}" for k, c in enumerate(part))
        prev = ""
        if results:
            prev = "Headings already chosen before this batch (for level consistency):\n" + "\n".join(
                f"level {h['level']}: {h['title']}" for h in results[-15:])
        hint_txt = (f"Table of contents found in the document (use it to decide levels):\n{hint}\n" if hint else "") + prev
        js = await chat_json(SELECT_PROMPT.format(body=body, scope=scope, hint=hint_txt, level_note=level_note,
                                                  cands=listing), max_tokens=16000)
        for h in js.get("headings", []):
            try:
                idx = int(h["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if i <= idx < i + len(part):
                c = cands[idx]
                results.append({"title": (h.get("title") or c["text"]).strip(), "level": max(1, int(h.get("level", 1))),
                                "start": (c["page"], c["start"]), "_cid": idx})
    results.sort(key=lambda h: h["start"])
    dedup, seen = [], set()
    for h in results:
        if h["start"] not in seen:
            seen.add(h["start"])
            dedup.append(h)
    return dedup


# --------------------------------------------------------------------------- strategy 3: LLM reads pages

PAGE_PROMPT = """Below is the text of pages {p0}-{p1} of a document{scope}. Each page is wrapped in <page_N> tags.
List the section headings that START on these pages, in order, with the page where each starts and a hierarchy level (1 = top level).
Copy each title exactly as it appears in the text. Ignore figure/table captions and running headers.
If these pages have no headings, return an empty list.
{prev}
Return JSON {{"headings": [{{"title": "...", "page": <int>, "level": <int>}}]}}

{text}"""


async def from_pages_llm(pages, lines_by_page, p_from, p_to, scope_title=None, budget=12000):
    groups, cur, tok = [], [], 0
    for p in range(p_from, p_to + 1):
        t = pages[p - 1]["text"]
        k = approx_tokens(t)
        if cur and tok + k > budget:
            groups.append(cur)
            cur, tok = [], 0
        cur.append(p)
        tok += k
    if cur:
        groups.append(cur)
    out = []
    for g in groups:   # sequential: each group sees the headings found so far
        text = "\n".join(f"<page_{p}>\n{pages[p - 1]['text']}\n</page_{p}>" for p in g)
        prev = ("Headings found on earlier pages:\n" + "\n".join(f"level {h['level']}: {h['title']}" for h in out[-10:])) if out else ""
        scope = f", inside the section \"{scope_title}\" (level 1 = its direct subsections)" if scope_title else ""
        js = await chat_json(PAGE_PROMPT.format(p0=g[0], p1=g[-1], scope=scope, prev=prev, text=text), max_tokens=8000)
        for h in js.get("headings", []):
            try:
                p = int(h["page"])
            except (KeyError, ValueError, TypeError):
                continue
            title = str(h.get("title", "")).strip()
            off = None
            for q in (p, p - 1, p + 1):
                if g[0] <= q <= g[-1]:
                    off = locate(title, q, lines_by_page, pages)
                    if off is not None:
                        p = q
                        break
            if off is None:
                log.info("unverified heading dropped: %r p%s", title, p)
                continue
            out.append({"title": title, "level": max(1, int(h.get("level", 1))), "start": (p, off)})
    out.sort(key=lambda h: h["start"])
    return out


# --------------------------------------------------------------------------- assembling

def nest(flat, pages, end=None, base_level=0):
    """Flat ordered headings -> nested nodes with start/end positions."""
    flat = sorted(flat, key=lambda h: h["start"])
    # normalise levels so the shallowest is 1, and children are at most parent+1
    nodes, stack = [], []
    for i, h in enumerate(flat):
        node = {"title": h["title"], "start": h["start"], "nodes": [], "_level": h["level"]}
        while stack and stack[-1]["_level"] >= h["level"]:
            stack.pop()
        (stack[-1]["nodes"] if stack else nodes).append(node)
        stack.append(node)

    def fill(siblings, parent_end):
        for i, n in enumerate(siblings):
            n["end"] = siblings[i + 1]["start"] if i + 1 < len(siblings) else parent_end
            n["text_end"] = n["nodes"][0]["start"] if n["nodes"] else n["end"]
            fill(n["nodes"], n["end"])

    fill(nodes, end)
    return nodes


def add_preface(nodes, pages, start=(1, 0)):
    if not nodes:
        return nodes
    first = nodes[0]["start"]
    if first > start and len(text_between(pages, start, first).strip()) > 200:
        nodes.insert(0, {"title": "Preface", "start": start, "end": first, "text_end": first, "nodes": []})
    return nodes


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n["nodes"])


async def refine(node, pages, lines, lines_by_page, body, depth=0):
    """Split a leaf that is too long into subsections."""
    if node["nodes"] or depth >= config.MAX_SPLIT_DEPTH:
        return
    sp = node["start"][0]
    ep = end_page(pages, node["start"], node["end"])
    text = text_between(pages, node["start"], node["end"])
    if ep - sp + 1 <= config.MAX_PAGES_PER_NODE and approx_tokens(text) <= config.MAX_TOKENS_PER_NODE:
        return
    end = node["end"] or (len(pages), len(pages[-1]["text"]))
    inside = [l for l in lines if node["start"] < (l["page"], l["start"]) < end]
    cands = candidates(inside, body, relaxed=True)
    title_n = norm(node["title"])
    cands = [c for c in cands if norm(c["text"]) != title_n]
    flat = await select_headings(cands, body, scope_title=node["title"]) if cands else []
    if len(flat) < 2:
        flat = await from_pages_llm(pages, lines_by_page, sp, ep, scope_title=node["title"])
        flat = [h for h in flat if node["start"] < h["start"] < end and norm(h["title"]) != title_n]
    if len(flat) < 2:
        return
    log.info("split %r (p%d-%d) into %d subsections", node["title"], sp, ep, len(flat))
    node["nodes"] = nest(flat, pages, end=node["end"])
    node["text_end"] = node["nodes"][0]["start"]
    await asyncio.gather(*(refine(c, pages, lines, lines_by_page, body, depth + 1) for c in walk_leaves(node["nodes"])))


def walk_leaves(nodes):
    return [n for n in walk(nodes) if not n["nodes"]]


async def build_tree(parsed) -> tuple[list, dict]:
    pages, lines = parsed["pages"], parsed["lines"]
    lines_by_page = collections.defaultdict(list)
    for l in lines:
        lines_by_page[l["page"]].append(l)
    body = body_size(lines)
    info = {"body_size": body}

    flat = from_bookmarks(parsed.get("toc") or [], pages, lines_by_page)
    info["method"] = "bookmarks"
    if not flat:
        cands = candidates(lines, body)
        info["candidates"] = len(cands)
        flat = await select_headings(cands, body, hint=toc_hint(pages, lines)) if cands else []
        info["method"] = "layout+llm"
        if len(flat) < 2:
            flat = await from_pages_llm(pages, lines_by_page, 1, len(pages))
            info["method"] = "llm-pages"
    if not flat:   # no structure at all: one node per page group
        flat = [{"title": f"Pages {p}-{min(p + config.MAX_PAGES_PER_NODE - 1, len(pages))}", "level": 1, "start": (p, 0)}
                for p in range(1, len(pages) + 1, config.MAX_PAGES_PER_NODE)]
        info["method"] = "page-groups"

    nodes = add_preface(nest(flat, pages), pages)
    await asyncio.gather(*(refine(n, pages, lines, lines_by_page, body) for n in walk_leaves(nodes)))
    finalize(nodes, pages)
    info["nodes"] = sum(1 for _ in walk(nodes))
    return nodes, info


def finalize(nodes, pages):
    """node_id, start_index/end_index (1-based pages), drop helper keys."""
    for i, n in enumerate(walk(nodes)):
        n["node_id"] = f"{i:04d}"
        n["start_index"] = n["start"][0]
        n["end_index"] = end_page(pages, n["start"], n["end"])
        n.pop("_level", None)
        n["start"] = list(n["start"])
        n["end"] = list(n["end"]) if n["end"] else None
        n["text_end"] = list(n["text_end"]) if n["text_end"] else None


# --------------------------------------------------------------------------- summaries

SUMMARY_PROMPT = """Write a {length} summary of this section of the document "{doc}" so that someone scanning the table of contents can tell whether the answer to a question is in it.
Mention the concrete topics, entities, metrics, tables and time periods it covers. No preamble.

Section: {title}
{children}
Text:
{text}"""


def _clip(text, max_tok=6000):
    if approx_tokens(text) <= max_tok:
        return text
    n = max_tok * 3
    return text[:n] + "\n...\n" + text[-n // 3:]


async def summarize(nodes, pages, doc_name):
    async def one(n):
        own = text_between(pages, tuple(n["start"]), tuple(n["text_end"]) if n["text_end"] else None)
        if n["nodes"]:
            kids = "Subsections: " + "; ".join(c["title"] for c in n["nodes"])
            if len(own.strip()) < 150:
                n["prefix_summary"] = ""
                return
            s = await chat([{"role": "user", "content": SUMMARY_PROMPT.format(
                length="1-2 sentence", doc=doc_name, title=n["title"], children=kids, text=_clip(own, 3000))}], max_tokens=300)
            n["prefix_summary"] = s.strip()
        else:
            if len(own.strip()) < 40:
                n["summary"] = own.strip()
                return
            s = await chat([{"role": "user", "content": SUMMARY_PROMPT.format(
                length="2-4 sentence", doc=doc_name, title=n["title"], children="", text=_clip(own))}], max_tokens=400)
            n["summary"] = s.strip()

    await asyncio.gather(*(one(n) for n in walk(nodes)))


async def describe(nodes, pages, doc_name) -> str:
    outline = "\n".join("  " * d + n["title"] for n, d in walk_depth(nodes) if d < 2)
    first = _clip(pages[0]["text"] + "\n" + (pages[1]["text"] if len(pages) > 1 else ""), 1500)
    s = await chat([{"role": "user", "content":
                     f"Describe in 2-3 sentences what the document \"{doc_name}\" is (type, issuer, period, main contents), "
                     f"for picking the right document out of a library. No preamble.\n\nOutline:\n{outline}\n\nFirst pages:\n{first}"}],
                   max_tokens=300)
    return s.strip()


def walk_depth(nodes, d=0):
    for n in nodes:
        yield n, d
        yield from walk_depth(n["nodes"], d + 1)
