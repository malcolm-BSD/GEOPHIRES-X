"""Exercise the example comparison harness without running reservoir simulations."""

from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests import test_geophires_x as examples


def _result(power: float):
    return SimpleNamespace(
        result={
            "metadata": {"output_file_path": "unused.out"},
            "Simulation Metadata": {"time": "unused"},
            "ECONOMIC PARAMETERS": {"After-tax IRR": {"value": float("nan")}},
            "SUMMARY OF RESULTS": {"power": power},
        }
    )


@pytest.mark.parametrize("calculated_power", [10.0, 20.0])
def test_example_harness_runs_once_and_checks_results(calculated_power):
    case = examples.GeophiresXTestCase(methodName="test_geophires_examples")
    files = ["example13.txt", "example13.out", "example13.png", "example13.html", "example6.txt"]
    expected = _result(10.0)
    actual = _result(calculated_power)

    with ExitStack() as stack:
        stack.enter_context(patch.object(case, "_list_test_files_dir", return_value=files))
        stack.enter_context(patch.object(case, "_is_github_actions", return_value=False))
        input_parameters = stack.enter_context(patch.object(examples, "GeophiresInputParameters"))
        client = stack.enter_context(patch.object(examples, "GeophiresXClient"))
        stack.enter_context(patch.object(examples, "GeophiresXResult", return_value=deepcopy(expected)))
        client.return_value.get_geophires_result.return_value = actual
        if calculated_power == 10.0:
            case.test_geophires_examples()
        else:
            with pytest.raises(AssertionError):
                case.test_geophires_examples()

        client.return_value.get_geophires_result.assert_called_once()
        input_parameters.assert_called_once()
        assert input_parameters.call_args.kwargs["from_file_path"].endswith("example13.txt")
        assert actual.result["ECONOMIC PARAMETERS"]["After-tax IRR"]["value"] == "NaN"
        assert "metadata" not in actual.result
        assert "Simulation Metadata" not in actual.result
