import contextlib

import pytest
from _pytest.runner import runtestprotocol


def pytest_addoption(parser):
    parser.addoption(
        "--run-unity", action="store_true", default=False, help="run tests using unity environment"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "stochastic(sample_size, max_samples=None): mark a test as stochastic running it `sample_size` times"
                   " and providing access to statistical analysis of the runs via the injected `stochastic_run` "
                   "fixture. Optionally specify `max_samples`. By default it is equal to `sample_size`. If it is "
                   "bigger, then additional samples are drawn if the test fails after taking `sample_size` samples, up"
                   "to the specified `max_samples`. If it is smaller than `max_samples` it is capped to `sample_size`")
    config.addinivalue_line("markers", "unity: mark test as to run only within a unity environment")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-unity"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-unity option to run")
    for item in items:
        if "unity" in item.keywords:
            item.add_marker(skip_slow)


class StochasticRunRecorder:
    def __init__(self):
        self._results = list()

    @contextlib.contextmanager
    def current_run(self):
        self._results.clear()
        yield

    def record(self, result):
        self._results.append(result)

    def average(self):
        return sum(self._results) / len(self._results)

    def __len__(self):
        return len(self._results)


_RECORDER = StochasticRunRecorder()


def pytest_runtest_protocol(item, nextitem):
    m = _get_marker(item)
    if m is None:
        return None

    reports = None
    with _RECORDER.current_run():
        n, max_n = _get_sample_range(m)
        s = 0
        while s < n:
            item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
            reports = runtestprotocol(item, nextitem=nextitem, log=False)
            s += 1
            if s == n and _has_failed(reports):
                n = min(n + 1, max_n)

    _report_last_run(item, reports)
    return True


def _get_marker(item):
    try:
        return item.get_closest_marker("stochastic")
    except AttributeError:
        return item.get_marker("stochastic")


def _get_sample_range(m):
    min_n = m.kwargs.get('sample_size')
    if min_n is None:
        min_n = m.args[0]
    max_n = max(m.kwargs.get('max_samples', min_n), min_n)
    return min_n, max_n


def _has_failed(reports):
    return any(r.failed for r in reports if r.when == 'call')


def _report_last_run(item, reports):
    for r in reports:
        item.ihook.pytest_runtest_logreport(report=r)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


@pytest.fixture
def stochastic_run():
    return _RECORDER
