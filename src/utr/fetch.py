"""Automated download of NOAA IGRA data with integrity checks and delivery to a shared folder.

Designed to be run on a schedule (cron / launchd / Task Scheduler). It only downloads when
NOAA's copy changed, verifies the download, and publishes it to a 'shared' folder atomically
so downstream software never sees a half-written file.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

from . import config
from .quality import check_archive

UA = {"User-Agent": "under-the-radar-mvp/0.1 (UCSD i4NS student project)"}


def _head(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                "length": int(r.headers.get("Content-Length") or -1)}


def _download(url: str, dest: Path, expected_len: int, timeout: int = 300) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    size = part.stat().st_size
    if expected_len >= 0 and size != expected_len:
        part.unlink(missing_ok=True)
        raise IOError(f"incomplete download: got {size} bytes, expected {expected_len}")
    part.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_station(station: str, raw_dir=config.RAW_DIR, shared_dir=None,
                  base_url: str = config.IGRA_BASE, force: bool = False,
                  retries: int = 3) -> dict:
    """Download one station archive if NOAA's copy changed. Returns a status dict."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = f"{station}-data.txt.zip"
    url = f"{base_url}/{name}"
    dest = raw_dir / name
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    prev = manifest.get(station, {})

    result = {"station": station, "url": url, "downloaded": False, "problems": []}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            head = _head(url)
            unchanged = (dest.exists() and prev.get("etag") == head["etag"]
                         and prev.get("last_modified") == head["last_modified"] and not force)
            if unchanged and not check_archive(dest):
                result["note"] = "up to date; no download needed"
            else:
                _download(url, dest, head["length"])
                result["downloaded"] = True
            result["remote"] = head
            break
        except Exception as exc:  # network errors, truncated files
            last_err = exc
            time.sleep(min(2 ** attempt, 10))
    else:
        result["problems"].append(f"download failed after {retries} attempts: {last_err}")
        return result

    result["problems"] += check_archive(dest)
    if result["problems"]:
        return result

    entry = {"etag": result["remote"]["etag"], "last_modified": result["remote"]["last_modified"],
             "bytes": dest.stat().st_size, "sha256": _sha256(dest),
             "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest[station] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result["manifest"] = entry

    if shared_dir:
        shared = Path(shared_dir)
        shared.mkdir(parents=True, exist_ok=True)
        tmp = shared / (name + ".part")
        shutil.copyfile(dest, tmp)
        tmp.replace(shared / name)
        result["shared_path"] = str(shared / name)
    return result
