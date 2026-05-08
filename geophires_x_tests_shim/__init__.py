"""Shim package to make geophires_x_tests.* imports resolve to tests/geophires_x_tests.

This module inserts the real tests/geophires_x_tests directory onto the
package's __path__ so imports like ``import geophires_x_tests.test_options_list``
work during pytest collection.
"""

from __future__ import annotations

from pathlib import Path

__all__ = []

_here = Path(__file__).resolve().parent
_repo_root = _here.parent
_candidate = _repo_root / "tests" / "geophires_x_tests"
if _candidate.is_dir():
    __path__.insert(0, str(_candidate))
