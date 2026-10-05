"""Regression checks for upstream fixes crossing fork-specific refactoring boundaries."""

import logging
import sys
from unittest.mock import Mock

import numpy as np

from geophires_x import MatplotlibUtils
from geophires_x.GeoPHIRESUtils import read_input_file
from geophires_x.OutputsProfiles import shorten_array_to_annual
from hip_ra_x.hip_ra_x import HIP_RA_X


def test_input_reader_returns_parameters_with_upstream_warning_suppression(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Reservoir Temperature, 150\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hip-ra-x", str(input_file)])
    logger = Mock(spec=logging.Logger)

    parameters = read_input_file(logger=logger, suppress_input_sys_argv_warnings=True)

    assert parameters["Reservoir Temperature"].sValue == "150"
    logger.warning.assert_not_called()


def test_hip_ra_reads_parameters_through_fork_reader(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Reservoir Temperature, 150\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hip-ra-x", str(input_file)])
    model = HIP_RA_X(enable_hip_ra_logging_config=False)

    model.read_parameters(suppress_input_sys_argv_warnings=True)

    assert model.InputParameters["Reservoir Temperature"].sValue == "150"
    assert model.ParameterDict["Reservoir Temperature"].value == 150


def test_one_year_profile_uses_start_sample_without_overflow():
    np.testing.assert_array_equal(shorten_array_to_annual(np.array([12.0, 11.0]), 1, 1), [12.0])


def test_multi_year_profile_keeps_annual_sampling():
    np.testing.assert_array_equal(shorten_array_to_annual(np.arange(9), 3, 3), [0, 3, 6])


def test_noninteractive_ci_does_not_show_plot(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(MatplotlibUtils.matplotlib, "get_backend", lambda: "Agg")
    show = Mock()
    monkeypatch.setattr(MatplotlibUtils.plt, "show", show)

    MatplotlibUtils.plt_show()

    show.assert_not_called()


def test_interactive_tcl_error_retains_fork_backend_fallback(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    tcl_error = type("TclError", (Exception,), {})
    show = Mock(side_effect=[tcl_error("test backend unavailable"), None])
    switch_backend = Mock()
    monkeypatch.setattr(MatplotlibUtils.plt, "show", show)
    monkeypatch.setattr(MatplotlibUtils.plt, "switch_backend", switch_backend)

    MatplotlibUtils.plt_show()

    switch_backend.assert_called_once_with("Agg")
    assert show.call_count == 2
