"""Allow legacy script execution without requiring an editable install first."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_package_root() -> None:
    package_root = str(Path(__file__).resolve().parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
