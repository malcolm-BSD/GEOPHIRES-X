"""Display operations must respect the backend, on desktops as well as in CI."""

import io
import warnings
from unittest.mock import Mock

import pytest

from geophires_x import MatplotlibUtils


@pytest.mark.parametrize("backend", ["Agg", "PDF", "svg"])
@pytest.mark.parametrize("ci", [False, True])
def test_file_backend_skips_display(backend, ci, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if ci:
        monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(MatplotlibUtils.matplotlib, "get_backend", lambda: backend)
    show = Mock(side_effect=AssertionError("A file backend cannot display a window"))
    monkeypatch.setattr(MatplotlibUtils.plt, "show", show)

    MatplotlibUtils.plt_show(block=False)

    show.assert_not_called()


def test_interactive_backend_receives_show_arguments(monkeypatch):
    monkeypatch.setattr(MatplotlibUtils.matplotlib, "get_backend", lambda: "QtAgg")
    show = Mock()
    monkeypatch.setattr(MatplotlibUtils.plt, "show", show)

    MatplotlibUtils.plt_show(block=False)

    show.assert_called_once_with(block=False)


@pytest.mark.parametrize("error", [RuntimeError("plot failed"), UserWarning("unrelated plot warning")])
def test_unrelated_display_errors_are_not_hidden(error, monkeypatch):
    monkeypatch.setattr(MatplotlibUtils.matplotlib, "get_backend", lambda: "QtAgg")
    monkeypatch.setattr(MatplotlibUtils.plt, "show", Mock(side_effect=error))
    with pytest.raises(type(error), match=str(error)):
        MatplotlibUtils.plt_show()


def test_agg_can_still_save_after_show_with_warnings_as_errors(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    # Linux normally suppresses the non-GUI warning when DISPLAY is absent.
    # Force the warning-producing condition seen on desktop Windows.
    monkeypatch.setenv("DISPLAY", ":0")
    plt = MatplotlibUtils.plt
    previous_backend = MatplotlibUtils.matplotlib.get_backend()
    fig = None
    try:
        plt.switch_backend("Agg")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plt.figure()
            fig.add_subplot().plot([0, 1], [0, 1])
            MatplotlibUtils.plt_show(block=False)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
        assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        if fig is not None:
            plt.close(fig)
        plt.switch_backend(previous_backend)
