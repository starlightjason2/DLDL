import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def _test_case_description(item: pytest.Item) -> str:
    """Return the one-line test case description from the test docstring."""
    obj = getattr(item, "obj", None)
    doc = getattr(obj, "__doc__", "") or ""
    first_line = doc.strip().splitlines()[0] if doc.strip() else ""
    return first_line or item.nodeid


def _write_status_line(config: pytest.Config, message: str) -> None:
    """Write test status output through pytest's terminal reporter when available."""
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(message)
    else:
        print(message, flush=True)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Print the test case description before each test runs."""
    _write_status_line(item.config, f"TEST CASE: {_test_case_description(item)}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Print whether each test case succeeded, failed, or was skipped."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    description = _test_case_description(item)
    if report.passed:
        _write_status_line(item.config, f"SUCCESS: {description}")
    elif report.failed:
        _write_status_line(item.config, f"FAILED: {description}")
    elif report.skipped:
        _write_status_line(item.config, f"SKIPPED: {description}")
