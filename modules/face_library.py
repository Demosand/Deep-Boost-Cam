"""Face library: named sets of source photos in <project>/face_library/<name>/.

Saving copies the photos into the library (the originals may move or vanish);
loading makes the first photo the main source and the rest the extra photos.
"""

from __future__ import annotations

import os
import shutil
from typing import List

from modules.paths import ROOT_DIR

LIB_DIR = os.path.join(ROOT_DIR, "face_library")
_IMG = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def safe_name(name: str) -> str:
    return "".join(c for c in name.strip() if c.isalnum() or c in " -_").strip()


def list_sets() -> List[str]:
    if not os.path.isdir(LIB_DIR):
        return []
    return sorted(d for d in os.listdir(LIB_DIR) if os.path.isdir(os.path.join(LIB_DIR, d)))


def photos(name: str) -> List[str]:
    folder = os.path.join(LIB_DIR, name)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(_IMG)]


def save_set(name: str, paths: List[str]) -> str:
    """Copy paths into the set (replacing it); returns the stored name."""
    name = safe_name(name)
    if not name:
        raise ValueError("empty name")
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        raise ValueError("no photos")
    folder = os.path.join(LIB_DIR, name)
    tmp = folder + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for i, p in enumerate(paths):
        shutil.copy2(p, os.path.join(tmp, f"{i:02d}{os.path.splitext(p)[1].lower()}"))
    shutil.rmtree(folder, ignore_errors=True)   # replace only after the copy succeeded
    os.replace(tmp, folder)
    return name


def delete_set(name: str) -> None:
    shutil.rmtree(os.path.join(LIB_DIR, name), ignore_errors=True)
