"""Thin re-export of the canonical brand styling helper.

Single source of truth lives at:
    /home/jiwei/arcadia/superstack/skills/_shared/mpl_style.py

Keep project-local plot scripts importing from `src.styling` (or directly
`styling` when running from `src/`) so the brand surface stays uniform; this
module just forwards to the canonical helper.
"""
from __future__ import annotations

import sys

_CANONICAL = "/home/jiwei/arcadia/superstack/skills/_shared"
if _CANONICAL not in sys.path:
    sys.path.insert(0, _CANONICAL)

from mpl_style import (  # noqa: E402,F401  -- re-export
    AMBER,
    BLUEVIOLET,
    DEEPPINK,
    FONT_BODY,
    FONT_TITLE,
    GOLD,
    PALETTE_FILLS,
    PALETTE_LINES,
    TURQUOISE,
    apply_style,
    palette,
    title,
)

__all__ = [
    "AMBER", "BLUEVIOLET", "DEEPPINK", "GOLD", "TURQUOISE",
    "FONT_BODY", "FONT_TITLE",
    "PALETTE_FILLS", "PALETTE_LINES",
    "apply_style", "palette", "title",
]
