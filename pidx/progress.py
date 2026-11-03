"""Live indexing progress: the stage each document (or collection) is in right now.

Kept in memory while a job runs; the finished stage list is folded into the document's `info["stages"]`
so the timeline can still be drawn afterwards. A job that dies with the process simply restarts its
stages when the queue resumes.

Stages of a document: parse (PyMuPDF text layer) -> mineru (OCR / Office via MinerU) -> tree -> summary
-> embed. A collection adds `dirs` (bottom-up folder summaries) once all its files are done.
"""

import contextvars
import time

_live: dict[str, dict] = {}
_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("progress_job", default=None)


def begin(job_id: str, usage: dict | None = None) -> None:
    """Start tracking `job_id` in the current task (and the threads / child tasks it spawns)."""
    _live[job_id] = {"stage": None, "stages": [], "detail": None, "usage": usage}
    _current.set(job_id)


def stage(name: str, detail: dict | None = None) -> None:
    p = _live.get(_current.get() or "")
    if p is None:
        return
    now = time.time()
    if p["stages"] and p["stages"][-1][2] is None:
        p["stages"][-1][2] = now
    p["stages"].append([name, now, None])
    p["stage"], p["detail"] = name, detail


def detail(**kw) -> None:
    """Update what the current stage is doing (e.g. MinerU state and pages extracted)."""
    p = _live.get(_current.get() or "")
    if p is not None:
        p["detail"] = {**(p["detail"] or {}), **kw}


def end(job_id: str) -> list:
    """Stop tracking; returns [[stage, start, end], ...] with absolute timestamps."""
    p = _live.pop(job_id, None)
    if not p:
        return []
    if p["stages"] and p["stages"][-1][2] is None:
        p["stages"][-1][2] = time.time()
    return [[n, round(a, 2), round(b, 2)] for n, a, b in p["stages"]]


def get(job_id: str) -> dict | None:
    p = _live.get(job_id)
    if p is None:
        return None
    u = p.get("usage") or {}
    return {"stage": p["stage"], "detail": p["detail"],
            "stages": [[n, round(a, 2), b and round(b, 2)] for n, a, b in p["stages"]],
            "llm": {k: u.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens")}}
