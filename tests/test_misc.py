import random

import pytest

import lm_eval.api.metrics as metrics


def test_bootstrapping():
    random.seed(42)
    arr = [random.random() for _ in range(1000)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)


def test_bootstrap_stderr_runs_requested_iterations(monkeypatch):
    monkeypatch.setenv("DISABLE_MULTIPROC", "1")
    calls = 0

    def statistic(xs):
        nonlocal calls
        calls += 1
        return sum(xs) / len(xs)

    metrics.bootstrap_stderr(statistic, [0, 1, 1, 0], iters=1500)

    assert calls == 1500
