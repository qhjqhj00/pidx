"""Minimal async OpenAI-compatible chat client with retries and JSON parsing."""

import asyncio
import contextvars
import json
import logging
import re
import time

import httpx

from . import config

log = logging.getLogger("pidx.llm")

_sem: asyncio.Semaphore | None = None
_client: httpx.AsyncClient | None = None
usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
# Per-job counter: set a fresh dict in a task's context; child tasks share it.
scope_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("scope_usage", default=None)


def new_scope() -> dict:
    u = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    scope_usage.set(u)
    return u


def _get():
    global _sem, _client
    if _client is None:
        _sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
        _client = httpx.AsyncClient(timeout=config.LLM_TIMEOUT, trust_env=False)  # all endpoints are tailnet/local
    return _sem, _client


def approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars/token for latin text, ~1.5 for CJK."""
    cjk = len(re.findall(r"[　-鿿가-힯]", text))
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1


class Tracer:
    """Collects per-call timings for one request; optionally forwards events (e.g. to an SSE stream)."""

    def __init__(self, emit=None):
        self.t0 = time.time()
        self.calls = []
        self.emit = emit

    async def event(self, ev: dict):
        ev.setdefault("t", round(time.time() - self.t0, 2))
        if self.emit:
            await self.emit(ev)


tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("tracer", default=None)
backend: contextvars.ContextVar[str] = contextvars.ContextVar("backend", default="default")


def _account(u: dict):
    for acc in filter(None, (usage, scope_usage.get())):
        acc["calls"] += 1
        acc["prompt_tokens"] += u.get("prompt_tokens", 0)
        acc["completion_tokens"] += u.get("completion_tokens", 0)


async def chat(messages, *, max_tokens=2048, temperature=0.0, json_mode=False, model=None,
               label="llm", on_delta=None) -> str:
    """One chat completion. Streams (to measure time-to-first-token and to feed `on_delta`)
    whenever a Tracer is active or a delta callback is given."""
    sem, client = _get()
    tr = tracer.get()
    stream = tr is not None or on_delta is not None
    be = config.LLM_BACKENDS[backend.get()]
    body = {
        "model": model or be["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **be.get("extra", {}),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {be['api_key']}"} if be["api_key"] else {}
    url = f"{be['base_url']}/chat/completions"
    delay = 5.0
    t_start = time.time()
    for attempt in range(6):
        try:
            async with sem:
                t_sent = time.time()
                if not stream:
                    r = await client.post(url, json=body, headers=headers)
                    if r.status_code in (429, 502, 503, 504):
                        raise _Retry(r.headers.get("Retry-After"), r.status_code)
                    r.raise_for_status()
                    data = r.json()
                    _account(data.get("usage") or {})
                    return data["choices"][0]["message"]["content"] or ""
                parts, u, t_first = [], {}, None
                async with client.stream("POST", url, json=body, headers=headers) as r:
                    if r.status_code in (429, 502, 503, 504):
                        await r.aread()
                        raise _Retry(r.headers.get("Retry-After"), r.status_code)
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        j = json.loads(data)
                        u = j.get("usage") or u
                        for ch in j.get("choices") or []:
                            d = (ch.get("delta") or {}).get("content")
                            if d:
                                if t_first is None:
                                    t_first = time.time()
                                parts.append(d)
                                if on_delta:
                                    await on_delta(d)
            t_end = time.time()
            _account(u)
            if tr:
                t_first = t_first or t_end
                n_out = u.get("completion_tokens", 0)
                rec = {"label": label, "backend": backend.get(), "start": round(t_start - tr.t0, 2), "end": round(t_end - tr.t0, 2),
                       "wait_ms": round((t_sent - t_start) * 1000),          # retries + local semaphore
                       "ttft_ms": round((t_first - t_sent) * 1000),          # upstream queue + prefill
                       "decode_ms": round((t_end - t_first) * 1000),
                       "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": n_out,
                       "tok_per_s": round(n_out / max(t_end - t_first, 1e-3), 1)}
                tr.calls.append(rec)
                await tr.event({"type": "llm", **rec})
            return "".join(parts)
        except _Retry as e:
            wait = float(e.retry_after or delay)
            log.warning("llm %s, retry in %.0fs", e.status, wait)
            await asyncio.sleep(wait)
            delay = min(delay * 2, 60)
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 5:
                raise
            log.warning("llm error %r, retry in %.0fs", e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("LLM retries exhausted")


class _Retry(Exception):
    def __init__(self, retry_after, status):
        self.retry_after, self.status = retry_after, status


def parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


async def chat_json(prompt: str, *, system: str | None = None, max_tokens=4096, label="llm") -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for attempt in range(3):
        out = await chat(msgs, max_tokens=max_tokens, json_mode=True, label=label)
        try:
            return parse_json(out)
        except (json.JSONDecodeError, ValueError):
            log.warning("bad JSON from llm (attempt %d): %.200s", attempt, out)
    raise ValueError("LLM did not return valid JSON")


def embed_model() -> str:
    """Identifies the vector space; vectors made under another value are stale and get recomputed."""
    return config.EMBED_MODEL + ("+mm" if config.EMBED_MULTIMODAL else "")


def _embed_text(t, kind: str) -> str:
    """`t` is a string or a (title, text) pair. EmbeddingGemma wants task prefixes; bge-m3 takes the
    plain text (title and text joined by a newline, as before, so existing bge vectors stay valid)."""
    title, text = t if isinstance(t, tuple) else (None, t)
    if not config.EMBED_MULTIMODAL:
        return "\n".join(x for x in (title, text) if x)
    if kind == "query":
        return f"task: search result | query: {text}"
    return f"title: {(title or 'none').replace(chr(10), ' / ')} | text: {text}"


async def _embed_raw(inputs: list, timeout=60) -> list[list[float]] | None:
    _, client = _get()
    headers = {"Authorization": f"Bearer {config.EMBED_API_KEY}"} if config.EMBED_API_KEY else {}
    try:
        r = await client.post(f"{config.EMBED_URL}/embeddings", headers=headers,
                              json={"model": config.EMBED_MODEL, "input": inputs}, timeout=timeout)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    except Exception as e:  # optional dependency
        log.warning("embedding unavailable: %r", e)
        return None


async def embed(texts: list, kind: str = "doc") -> list[list[float]] | None:
    """Text embeddings; kind is "doc" (indexed content) or "query". Items are strings or
    (title, text) pairs. Returns None if the service is unavailable."""
    return await _embed_raw([_embed_text(t, kind)[:20000] for t in texts])


async def embed_images(images: list[bytes]) -> list[list[float]] | None:
    """Image embeddings (EmbeddingGemma path only), PNG bytes in, same space as text."""
    if not config.EMBED_MULTIMODAL or not images:
        return None
    import base64
    return await _embed_raw([{"image": "data:image/png;base64," + base64.b64encode(b).decode()} for b in images],
                            timeout=180)
