"""Tests for scoring and aggregate metrics."""

from __future__ import annotations

import pytest

from habla_bench.checks import CheckResult
from habla_bench.metrics import CaseOutcome, aggregate_model, case_score


def _check(passed: bool, kind: str = "is_spanish") -> CheckResult:
    return CheckResult(kind, passed, "detail")


def _outcome(
    *,
    case_id: str = "case-1",
    category: str = "neutral_spanish",
    weight: float = 1.0,
    checks: list[CheckResult] | None = None,
    response: str = "Claro, te ayudo con eso.",
    latency_ms: float = 100.0,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 20,
    cost_usd: float | None = 0.0001,
    error: str | None = None,
) -> CaseOutcome:
    checks = checks if checks is not None else [_check(True)]
    return CaseOutcome(
        case_id=case_id,
        category=category,
        weight=weight,
        response=response,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        checks=checks,
        score=case_score(checks),
        error=error,
    )


class TestCaseScore:
    def test_all_pass(self):
        assert case_score([_check(True), _check(True)]) == 100.0

    def test_half_pass(self):
        assert case_score([_check(True), _check(False)]) == 50.0

    def test_none_pass(self):
        assert case_score([_check(False)]) == 0.0

    def test_empty_is_zero(self):
        assert case_score([]) == 0.0


class TestAggregate:
    def test_overall_is_weighted_mean(self):
        outcomes = [
            _outcome(case_id="a", weight=1.0, checks=[_check(True)]),
            _outcome(case_id="b", weight=3.0, checks=[_check(False)]),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.total_score == 25.0
        assert metrics.cases == 2
        assert metrics.fully_passed_cases == 1
        assert metrics.failed_cases == 1

    def test_category_scores(self):
        outcomes = [
            _outcome(case_id="a", category="neutral_spanish", checks=[_check(True)]),
            _outcome(case_id="b", category="neutral_spanish", checks=[_check(False)]),
            _outcome(case_id="c", category="language_lock", checks=[_check(True)]),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.category_scores == {
            "neutral_spanish": 50.0,
            "language_lock": 100.0,
        }

    def test_voseo_rate(self):
        outcomes = [
            _outcome(case_id="a", response="Mirá, vos podés."),
            _outcome(case_id="b", response="Revisa los datos."),
            _outcome(case_id="c", response="Decime si te sirve."),
            _outcome(case_id="d", response="Te ayudo."),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.voseo_rate == 50.0
        assert metrics.spain_rate == 0.0

    def test_spain_rate(self):
        outcomes = [
            _outcome(case_id="a", response="Vale, vosotros podéis."),
            _outcome(case_id="b", response="Ustedes pueden."),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.spain_rate == 50.0

    def test_mean_latency(self):
        outcomes = [
            _outcome(case_id="a", latency_ms=100.0),
            _outcome(case_id="b", latency_ms=300.0),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.mean_latency_ms == 200.0

    def test_token_totals(self):
        outcomes = [
            _outcome(case_id="a", prompt_tokens=10, completion_tokens=5),
            _outcome(case_id="b", prompt_tokens=20, completion_tokens=7),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.prompt_tokens == 30
        assert metrics.completion_tokens == 12
        assert metrics.cost_usd == pytest.approx(0.0002, abs=1e-9)
        assert metrics.cost_complete

    def test_missing_usage_marks_cost_incomplete(self):
        outcomes = [
            _outcome(case_id="a", prompt_tokens=10, completion_tokens=5),
            _outcome(case_id="b", prompt_tokens=None, completion_tokens=None, cost_usd=None),
        ]
        metrics = aggregate_model("mock-x", outcomes)
        assert metrics.prompt_tokens == 10
        assert metrics.cost_usd is None
        assert not metrics.cost_complete

    def test_example_price_flag_is_forwarded(self):
        metrics = aggregate_model("mock-x", [_outcome()], price_is_example=True)
        assert metrics.price_is_example
        assert metrics.to_dict()["price_is_example"] is True

    def test_error_outcome_has_zero_score_and_failed(self):
        outcome = _outcome(checks=[], error="boom")
        assert outcome.score == 0.0
        assert not outcome.passed
        assert outcome.failed_checks == []
