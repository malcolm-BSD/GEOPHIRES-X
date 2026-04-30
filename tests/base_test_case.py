from __future__ import annotations

import inspect
import numbers
import os.path
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from geophires_x.GeoPHIRESUtils import sig_figs
from geophires_x_client import GeophiresInputParameters

# noinspection PyProtectedMember
from geophires_x_client import _get_logger


class BaseTestCase(unittest.TestCase):
    maxDiff = None

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
        if not artifact_path.is_relative_to(tests_root) or not artifact_path.exists():
            return

        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
        else:
            artifact_path.unlink()

    def setUp(self) -> None:
        super().setUp()
        self._ignored_test_artifacts_before = self._ignored_test_artifacts()

    def tearDown(self) -> None:
        try:
            new_artifacts = self._ignored_test_artifacts() - getattr(
                self,
                "_ignored_test_artifacts_before",
                set(),
            )
            for artifact_path in sorted(new_artifacts, key=lambda path: len(path.parts), reverse=True):
                self._remove_test_artifact(artifact_path)
        finally:
            super().tearDown()

    def _get_test_file_path(self, test_file_name) -> str:
        return os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(self.__class__))), test_file_name)

    def _get_test_file_content(self, test_file_name, **open_kw_args) -> str:
        with open(self._get_test_file_path(test_file_name), **open_kw_args) as f:
            return f.readlines()

    def _list_test_files_dir(self, test_files_dir: str):
        return os.listdir(self._get_test_file_path(test_files_dir))  # noqa: PTH208

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
