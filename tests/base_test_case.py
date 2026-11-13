from __future__ import annotations

import atexit
import gc
import inspect
import math
import numbers
import os.path
import shutil
import subprocess
import sys
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pint.facets.plain import PlainQuantity

from geophires_x.GeoPHIRESUtils import sig_figs
from geophires_x_client import GeophiresInputParameters
from geophires_x_client import GeophiresXResult

# noinspection PyProtectedMember
from geophires_x_client import _get_logger


class _QuietTestAssetHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        pass


class BaseTestCase(unittest.TestCase):
    maxDiff = None
    _test_asset_server: ThreadingHTTPServer | None = None
    _test_asset_server_thread: threading.Thread | None = None
    _test_asset_server_lock = threading.Lock()

    @classmethod
    def _repo_root(cls) -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _tests_root(cls) -> Path:
        return cls._repo_root() / "tests"

    @classmethod
    def _ignored_test_artifacts(cls) -> set[Path]:
        git_executable = shutil.which("git")
        if git_executable is None:
            return set()

        try:
            result = subprocess.run(  # noqa: S603
                [
                    git_executable,
                    "-C",
                    str(cls._repo_root()),
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "-z",
                    "--",
                    "tests",
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return set()

        artifacts = set()
        for relative_path in result.stdout.decode("utf-8").split("\0"):
            if relative_path:
                artifacts.add((cls._repo_root() / relative_path).resolve())
        return artifacts

    @classmethod
    def _remove_test_artifact(cls, artifact_path: Path) -> None:
        artifact_path = artifact_path.resolve()
        tests_root = cls._tests_root().resolve()
        if not cls._path_is_relative_to(artifact_path, tests_root) or not artifact_path.exists():
            return

        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
        else:
            artifact_path.unlink()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Snapshot once per class instead of launching Git before every test. This preserves pre-existing ignored
        # files while avoiding hundreds of redundant subprocesses in a full suite run.
        cls._ignored_test_artifacts_before_class = cls._ignored_test_artifacts()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            new_artifacts = cls._ignored_test_artifacts() - getattr(
                cls,
                "_ignored_test_artifacts_before_class",
                set(),
            )
            for artifact_path in sorted(new_artifacts, key=lambda path: len(path.parts), reverse=True):
                cls._remove_test_artifact(artifact_path)
        finally:
            cls._clear_simulation_caches()
            super().tearDownClass()

    @staticmethod
    def _clear_simulation_caches() -> None:
        # Cached reservoir calculations retain entire Models, including large CLGS data arrays.
        # Keep caching within each test class, but release simulations before the next class runs.
        from geophires_x.AGSWellBores import data
        from geophires_x.CylindricalReservoir import CylindricalReservoir
        from geophires_x.Reservoir import Reservoir
        from geophires_x.SBTReservoir import SBTReservoir

        for calculation in (
            Reservoir.Calculate,
            CylindricalReservoir.Calculate,
            SBTReservoir.Calculate_Coaxial,
            SBTReservoir.Calculate_Uloop,
        ):
            calculation.cache_clear()
        data._cache.clear()
        gc.collect()

    @classmethod
    def _test_asset_url(cls, repository_relative_path: str | Path) -> str:
        repo_root = cls._repo_root().resolve()
        asset_path = (repo_root / repository_relative_path).resolve()
        if not cls._path_is_relative_to(asset_path, repo_root) or not asset_path.is_file():
            raise ValueError(f"Test asset must be an existing file inside the repository: {asset_path}")

        with BaseTestCase._test_asset_server_lock:
            if BaseTestCase._test_asset_server is None:
                handler = partial(_QuietTestAssetHandler, directory=str(repo_root))
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                BaseTestCase._test_asset_server = server
                BaseTestCase._test_asset_server_thread = server_thread

        relative_url_path = quote(asset_path.relative_to(repo_root).as_posix())
        server_port = BaseTestCase._test_asset_server.server_address[1]
        return f"http://127.0.0.1:{server_port}/{relative_url_path}"

    @staticmethod
    def _path_is_relative_to(path: Path, other: Path) -> bool:
        try:
            path.relative_to(other)
            return True
        except ValueError:
            return False

    def _get_test_file_path(self, test_file_name) -> str:
        return os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(self.__class__))), test_file_name)

    def _get_test_file_content(self, test_file_name, **open_kw_args) -> str:
        with open(self._get_test_file_path(test_file_name), **open_kw_args) as f:
            return f.readlines()

    def _list_test_files_dir(self, test_files_dir: str):
        return os.listdir(self._get_test_file_path(test_files_dir))  # noqa: PTH208

    # noinspection PyMethodMayBeStatic
    def _sanitize_nan(self, r: GeophiresXResult) -> GeophiresXResult:
        """
        Workaround for float('nan') != float('nan')
        See https://stackoverflow.com/questions/51728427/unittest-how-to-assert-if-the-two-possibly-nan-values-are-equal

        TODO generalize beyond After-tax IRR

        Mutates passed-in result object.
        """

        irr_key = "After-tax IRR"
        if irr_key in r.result["ECONOMIC PARAMETERS"]:
            try:
                if math.isnan(r.result["ECONOMIC PARAMETERS"][irr_key]["value"]):
                    r.result["ECONOMIC PARAMETERS"][irr_key]["value"] = "NaN"
            except TypeError:
                pass

        return r

    # noinspection PyMethodMayBeStatic
    def _strip_metadata(self, geophires_result: GeophiresXResult) -> GeophiresXResult:
        """
        Useful for comparing results from different runs.

        Mutates passed-in result object.
        """

        for key in ["metadata", "Simulation Metadata"]:
            if key in geophires_result.result:
                del geophires_result.result[key]

        return geophires_result

    def assertAlmostEqualWithinPercentage(self, expected, actual, msg: str | None = None, percent=5):
        if msg is not None and not isinstance(msg, str):
            raise ValueError(f"msg must be a string (you may have meant to pass percent={msg})")

        if isinstance(expected, numbers.Real):
            try:
                self.assertAlmostEqual(expected, actual, msg=msg, delta=abs(percent / 100.0 * expected))
            except AssertionError as ae:
                try:
                    difference_percent = abs(100.0 * (actual - expected) / expected)
                except ZeroDivisionError:
                    difference_percent = float("nan")
                raise AssertionError(f"{actual} != {expected} within {percent}% ({difference_percent:.2f}%)") from ae
        else:
            if isinstance(expected, list) and isinstance(actual, list):
                suggest = f"self.assertListAlmostEqual({expected}, {actual}, msg={msg}, percent={percent})"
                suggest = f"Got 2 lists, you probably meant to call:\n\t{suggest}"
                log = _get_logger(__name__)
                log.warning(suggest)

            self.assertEqual(expected, actual, msg)

    def assertDictAlmostEqual(self, expected, actual, msg=None, places=7, percent=None):
        """
        https://stackoverflow.com/a/53081544/21380804
        """

        if percent is not None:
            places = None

        # check if both inputs are dicts
        self.assertIsInstance(expected, dict, "First argument is not a dictionary")
        self.assertIsInstance(actual, dict, "Second argument is not a dictionary")

        # check if both inputs have the same keys
        self.assertEqual(expected.keys(), actual.keys())

        # check each key
        for key, value in expected.items():
            if isinstance(value, dict):
                self.assertDictAlmostEqual(expected[key], actual[key], msg=msg, places=places, percent=percent)
            elif isinstance(value, list):
                self.assertListAlmostEqual(expected[key], actual[key], msg=msg, places=places, percent=percent)
            else:
                if places is not None:
                    self.assertAlmostEqual(expected[key], actual[key], places=places, msg=msg)
                else:
                    self.assertAlmostEqualWithinPercentage(expected[key], actual[key], percent=percent, msg=msg)

    def assertListAlmostEqual(self, expected, actual, msg=None, places=7, percent=None):
        if percent is not None:
            places = None

        # check if both inputs are dicts
        self.assertIsInstance(expected, list, "First argument is not a list")
        self.assertIsInstance(actual, list, "Second argument is not a list")

        # check if both inputs have the same keys
        self.assertEqual(len(expected), len(actual))

        # check each key
        for i in range(len(expected)):
            v1 = expected[i]
            v2 = actual[i]
            if isinstance(v1, dict):
                self.assertDictAlmostEqual(v1, v2, msg=msg, places=places, percent=percent)
            elif isinstance(v1, list):
                self.assertListAlmostEqual(v1, v2, msg=msg, places=places, percent=percent)
            else:
                if places is not None:
                    self.assertAlmostEqual(v1, v2, places=places, msg=msg)
                else:
                    self.assertAlmostEqualWithinPercentage(v1, v2, percent=percent, msg=msg)

    def assertFileContentsEqual(self, expected, actual):
        with open(
            expected, newline=None
        ) as f1_o:  # newline=None enables universal line endings which is required by Windows
            with open(actual, newline=None) as f2_o:
                f1_lines = f1_o.readlines()
                f2_lines = f2_o.readlines()
                self.assertListEqual(f1_lines, f2_lines, msg=f"{expected}, {actual}")

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def assertHasLogRecordWithMessage(self, logs_, message, treat_substring_match_as_match: bool = False):
        messages = [record.message for record in logs_.records]
        assert any(it == message or (treat_substring_match_as_match and message in it) for it in messages)

    def assertAlmostEqualWithinSigFigs(self, expected: float | int, actual: float | int, num_sig_figs: int = 3):
        self.assertEqual(
            sig_figs(expected, num_sig_figs),
            sig_figs(actual, num_sig_figs),
        )

    # noinspection PyMethodMayBeStatic
    def _is_github_actions(self):
        return "CI" in os.environ or "TOXPYTHON" in os.environ

    def _handle_assert_logs_failure(self, ae: AssertionError):
        if sys.version_info[:2] == (3, 8) and self._is_github_actions():
            # FIXME - see
            #  https://github.com/softwareengineerprogrammer/GEOPHIRES/actions/runs/19646240874/job/56262028512#step:5:344
            print(
                f"WARNING: Skipping logs assertion in GitHub Actions "
                f"for Python {sys.version_info.major}.{sys.version_info.minor}"
            )
        else:
            raise ae

    @staticmethod
    def get_input_parameter(params: GeophiresInputParameters, param_name: str) -> float | str | None:
        """
        TODO refactor into generic utility method
        TODO should return quantity
        """

        for line in reversed(params.as_text().split("\n")):
            parts = line.strip().split(",")
            if parts[0].strip() == param_name:
                ret = parts[1].strip()
                try:
                    return float(ret)
                except ValueError:
                    pass

                return str(ret)

        return None

    @staticmethod
    def value_unit_as_quantity(v_u_entry: dict[str, Any]) -> PlainQuantity:
        return PlainQuantity(v_u_entry["value"], v_u_entry["unit"])


def _shutdown_test_asset_server() -> None:
    server = BaseTestCase._test_asset_server
    server_thread = BaseTestCase._test_asset_server_thread
    if server is None:
        return
    server.shutdown()
    server.server_close()
    if server_thread is not None:
        server_thread.join(timeout=5)


atexit.register(_shutdown_test_asset_server)
