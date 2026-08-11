"""Public packet upload: validation, safe save, and Sage notification.

Kept separate from server.py so the safety-critical logic is unit-testable
without loading the ML model. server.py adds the GET/POST /upload routes.
"""
import collections
import json
import os
import re
import time
import urllib.request
from pathlib import Path

UPLOAD_DIR = Path(os.environ.get("PACKET_UPLOAD_DIR", "/opt/packet-uploads"))
MAX_BYTES = 30 * 1024 * 1024              # per-file cap
DIR_CAP_BYTES = 2 * 1024 * 1024 * 1024    # total staging cap — a public endpoint must not fill the disk
SAGE_NOTIFY_URL = os.environ.get("SAGE_NOTIFY_URL", "http://localhost:7779/notify")
SAGE_KEY = os.environ.get("SAGE_CONSOLE_KEY", "")

ALLOWED = {".pdf", ".docx"}
# PDF starts "%PDF"; DOCX is an OOXML zip, so its first bytes are the local-file header "PK\x03\x04".
MAGIC = {".pdf": b"%PDF", ".docx": b"PK\x03\x04"}


class UploadError(ValueError):
    """Rejected upload. The message is safe to show the submitter."""


RATE_MAX = 30          # uploads per IP...
RATE_WINDOW = 3600     # ...per hour — enough to submit a couple tournaments; blocks floods
_hits: dict[str, list[float]] = collections.defaultdict(list)


def check_rate(ip: str, now: float | None = None) -> None:
    """Per-IP sliding window. ponytail: in-process dict, single uvicorn worker —
    move to a shared store only if LeBot ever runs multiple workers/hosts."""
    now = time.time() if now is None else now
    cutoff = now - RATE_WINDOW
    recent = [t for t in _hits[ip] if t > cutoff]
    if len(recent) >= RATE_MAX:
        _hits[ip] = recent
        raise UploadError("Too many uploads from your network recently. Please try again later.")
    recent.append(now)
    _hits[ip] = recent  # pruned each call, so idle IPs keep at most their last-hour entries


def safe_name(raw: str) -> str:
    """basename + whitelist → no path traversal, no surprise extensions."""
    base = os.path.basename(raw or "")
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED:
        raise UploadError("Only .pdf and .docx files are accepted.")
    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", stem).strip(" .") or "packet"
    return stem[:120] + ext


def validate_magic(name: str, data: bytes) -> None:
    ext = os.path.splitext(name)[1].lower()
    if not data.startswith(MAGIC[ext]):
        raise UploadError(f"That file does not look like a real {ext[1:].upper()}.")


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*") if f.is_file())


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def save_upload(filename: str, data: bytes, tournament: str) -> Path:
    if not tournament.strip():
        raise UploadError("Tournament name is required.")
    if not data:
        raise UploadError("Empty file.")
    if len(data) > MAX_BYTES:
        raise UploadError("File too large (max 30 MB).")
    name = safe_name(filename)
    validate_magic(name, data)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if _dir_size(UPLOAD_DIR) + len(data) > DIR_CAP_BYTES:
        raise UploadError("Upload storage is temporarily full; try again later.")
    dest = _unique(UPLOAD_DIR / name)
    dest.write_bytes(data)
    dest.chmod(0o644)
    return dest


def notify_sage(dest: Path, size: int, tournament: str, submitter: str) -> None:
    if not SAGE_KEY:
        return  # notifications disabled (e.g. local dev); the upload still succeeds
    body = (f"Tournament: {tournament}\n"
            f"File: {dest.name} ({size / 1024 / 1024:.1f} MB)\n"
            f"From: {submitter.strip() or 'anonymous'}\n"
            f"Saved: {dest}")
    payload = json.dumps({"level": "info", "title": "New packet upload", "body": body}).encode()
    req = urllib.request.Request(
        SAGE_NOTIFY_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {SAGE_KEY}"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # ponytail: best-effort — a Sage hiccup must never lose an already-saved file
