"""Small text utilities used by the lightweight reference encoder."""

from __future__ import annotations

import hashlib
import re

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|\d+|==|!=|<=|>=|&&|\|\||[^\s]")


def tokenize(text: str) -> list[str]:
    """Tokenize code-like text without language-specific dependencies."""

    return TOKEN_RE.findall(text or "")


def stable_hash(token: str, modulo: int) -> int:
    """Return a deterministic positive bucket id for a token."""

    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def compact_text(*parts: object) -> str:
    """Join non-empty fields into a compact evidence text string."""

    return " ".join(str(part).strip() for part in parts if part not in (None, ""))

