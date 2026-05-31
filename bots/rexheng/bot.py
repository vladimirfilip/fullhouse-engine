"""Single-file submission entrypoint.

Engine drivers expect `bot.py` with a top-level `decide` function. We import
from the package and re-export. Engine sandbox should be able to load this as
a path-based module since the directory is on the import path.
"""
from __future__ import annotations
import sys
import os

# Ensure the parent directory is on path so `bot.*` imports resolve when this
# file is loaded by spec-from-file-location.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from bot.decide import decide, BOT_NAME, BOT_AVATAR  # noqa: E402,F401

__all__ = ["decide", "BOT_NAME", "BOT_AVATAR"]
