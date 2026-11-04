"""Startup update check against the GitHub releases of Deep-Boost-Cam.

Runs once in a background thread; any network error is ignored silently, so
the program starts normally offline. Only the release tag is compared — the
user installs an update themselves from the page that opens.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from typing import Callable, Optional, Tuple

from modules import metadata

API_URL = f"https://api.github.com/repos/{metadata.repo}/releases/latest"
RELEASES_URL = f"https://github.com/{metadata.repo}/releases/latest"


def _parse(version: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", version or "")
    return tuple(int(n) for n in nums[:3]) or (0,)


def is_newer(remote: str, local: str) -> bool:
    return _parse(remote) > _parse(local)


def latest_release(timeout: float = 6.0) -> Optional[dict]:
    """{'tag': 'v1.2.0', 'name': ..., 'url': ...} or None on any error."""
    try:
        req = urllib.request.Request(API_URL, headers={
            "User-Agent": f"{metadata.name}/{metadata.version}",
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return {"tag": data.get("tag_name", ""), "name": data.get("name") or data.get("tag_name", ""),
                "url": data.get("html_url") or RELEASES_URL, "notes": data.get("body") or ""}
    except Exception:
        return None


def check_async(on_update: Callable[[dict], None]) -> None:
    """Call on_update(release) from a worker thread if a newer release exists."""
    def run():
        rel = latest_release()
        if rel and is_newer(rel["tag"], metadata.version):
            on_update(rel)
    threading.Thread(target=run, name="update-check", daemon=True).start()
