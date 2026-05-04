"""Pytest site-wide conftest to ensure repository root is on sys.path.

Some historical tests import helper modules using bare module names
(`from base_test_case import BaseTestCase`). When pytest runs it sets the
first entry of sys.path to the test directory, which can prevent the repo
root from being discovered. Adding the repo root to sys.path here ensures
those imports succeed during collection.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
