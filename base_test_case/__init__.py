"""Compatibility package that re-exports the canonical BaseTestCase.

Placing this small package at the repository root makes imports like
``from base_test_case import BaseTestCase`` succeed regardless of current
working directory during pytest collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure tests/ is on sys.path so the real implementation can be imported
_repo_root = Path(__file__).resolve().parents[1]
_tests_dir = _repo_root / "tests"
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from tests.base_test_case import BaseTestCase  # noqa: E402, F401
