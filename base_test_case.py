"""Compatibility shim to make tests that import `base_test_case` succeed

This file simply re-exports the canonical test helper located under the
``tests/`` directory so that both ``from tests.base_test_case import ...``
and ``from base_test_case import ...`` work during test collection.
"""

from tests.base_test_case import BaseTestCase  # noqa: F401
