"""Read safetensors headers and tensor samples from the Hub with HTTP range requests.

A safetensors header gives every tensor's exact byte range, so a model can be
compared with a few kilobytes per tensor instead of a full download. Responses
are cached on disk (PEDIGREE_CACHE/ranges), so reruns cost nothing and a run
interrupted by the network resumes. Adapted from Stowaway's survey sampler (MIT).

Rate limits (HTTP 429) and transient server errors are retried with backoff that
respects Retry-After; a run reports any model it could not read rather than
dropping it silently.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

from .hub import UA, cache_root

RETRY_STATUS = (429, 500, 502, 503, 504)
RETRIES = 8
HEADER_MAX = 100 * 1024 * 1024


def resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _cache_path(url: str, start: int, end: int) -> Path:
    key = hashlib.sha256(f"{url}|{start}|{end}".encode()).hexdigest()
    return cache_root() / "ranges" / key[:2] / key


def fetch_range(url: str, start: int, end: int, retries: int = RETRIES) -> bytes:
    """Bytes [start, end] of url (inclusive), cached; retries 429/5xx and dropped
    connections with backoff (at least the server's Retry-After)."""
    path = _cache_path(url, start, end)
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                status = r.status
            if status != 206 and start > 0:
                raise IOError(f"server ignored the range request (status {status})")
            data = data[: end - start + 1]
            if len(data) != end - start + 1:
                raise IOError(f"short read: {len(data)} of {end - start + 1} bytes")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return data
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == retries - 1:
                raise
            try:
                wait = float(e.headers.get("Retry-After", 0))
            except (TypeError, ValueError):
                wait = 0.0
            time.sleep(min(max(wait, 5 * 2 ** attempt), 300))
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, IOError):
            if attempt == retries - 1:
                raise
            time.sleep(min(5 * 2 ** attempt, 120))
    raise AssertionError("unreachable")


def read_header(url: str) -> tuple[int, dict]:
    """(byte offset where tensor data starts, parsed header) of a remote safetensors file."""
    (n,) = struct.unpack("<Q", fetch_range(url, 0, 7))
    if n > HEADER_MAX:
        raise ValueError(f"header length {n} is implausible")
    return 8 + n, json.loads(fetch_range(url, 8, 8 + n - 1))


def tensor_bytes(url: str, data_start: int, spec: dict, max_bytes: int) -> bytes:
    """The first max_bytes (rounded down to whole elements) of one tensor."""
    b, e = spec["data_offsets"]
    itemsize = ITEMSIZE[spec["dtype"]]
    length = min(e - b, max_bytes) // itemsize * itemsize
    if length <= 0:
        return b""
    return fetch_range(url, data_start + b, data_start + b + length - 1)


ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U64": 8, "U32": 4, "U16": 2,
            "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
