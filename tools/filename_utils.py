from __future__ import annotations

import re


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def safe_filename(value: str, max_length: int = 120) -> str:
    normalized = re.sub(r"\s+", "_", value.strip())
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip(" ._")
    if not normalized:
        normalized = "untitled"
    if normalized.upper() in WINDOWS_RESERVED_NAMES:
        normalized = f"{normalized}_file"
    return normalized[:max_length].rstrip(" ._") or "untitled"
