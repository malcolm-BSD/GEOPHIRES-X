"""Compatibility package to expose tests/geophires_x_tests as a top-level package.

This small shim adjusts the package search path so imports like
``import geophires_x_tests.test_options_list`` succeed during test
collection. It does not contain test logic itself.
"""

from __future__ import annotations

from pathlib import Path

# Determine repository root and append the real tests package path to this
# package's __path__ so Python can import the real test modules.
_here = Path(__file__).resolve().parent
_repo_root = _here.parent
_candidate = _repo_root / "tests" / "geophires_x_tests"
if _candidate.is_dir():
    __path__.insert(0, str(_candidate))
