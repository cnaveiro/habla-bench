"""Scoring and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .checks import CheckResult, find_spain_forms, find_voseo


def case_score(checks: Sequence[CheckResult]) -> float:
    """Score one case from 0 to 100 as the share of checks that passed."""

    if not checks:
        return 0.0
    passed = sum(1 for check in checks if check.passed)
    return round(100.0 * passed / len(checks), 2)


@dataclass
class CaseOutcome:
    case_id: str
    category: str
    weight: float
    response: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    checks: list[CheckResult]
    score: float
    error: str | None = None
    system: str = ""
    user: str = ""
    context: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[str]:
        return [check.type for check in self.checks if not check.passed]


@dataclass
class ModelMetrics:
    model: str
    total_score: float
    category_scores: dict[str, float]
    voseo_rate: float
    spain_rate: float
    mean_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    cost_complete: bool
    price_is_example: bool
    cases: int
    fully_passed_cases: int
    failed_cases: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "total_score": self.total_score,
            "category_scores": self.category_scores,
            "voseo_rate": self.voseo_rate,
            "spain_rate": self.spain_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "cost_complete": self.cost_complete,
            "price_is_example": self.price_is_example,
            "cases": self.cases,
            "fully_passed_cases": self.fully_passed_cases,
            "failed_cases": self.failed_cases,
        }


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    pairs = [(value, weight) for value, weight in values if weight > 0]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return round(sum(value * weight for value, weight in pairs) / total_weight, 2)


def aggregate_model(
    model: str,
    outcomes: Sequence[CaseOutcome],
    *,
    price_is_example: bool = False,
) -> ModelMetrics:
    """Aggregate case outcomes into report-ready metrics."""

    categories: dict[str, float] = {}
    for category in dict.fromkeys(outcome.category for outcome in outcomes):
        categories[category] = _weighted_mean(
            (outcome.score, outcome.weight)
            for outcome in outcomes
            if outcome.category == category
        )

    total = _weighted_mean(
        (outcome.score, outcome.weight) for outcome in outcomes
    )
    count = len(outcomes) or 1
    voseo_cases = sum(
        1 for outcome in outcomes if find_voseo(outcome.response)
    )
    spain_cases = sum(
        1 for outcome in outcomes if find_spain_forms(outcome.response)
    )
    with_latency = [outcome.latency_ms for outcome in outcomes]
    prompt_tokens = sum(
        outcome.prompt_tokens or 0 for outcome in outcomes
    )
    completion_tokens = sum(
        outcome.completion_tokens or 0 for outcome in outcomes
    )
    usage_complete = all(
        outcome.prompt_tokens is not None
        and outcome.completion_tokens is not None
        for outcome in outcomes
    ) and bool(outcomes)
    cost_values = [
        outcome.cost_usd for outcome in outcomes if outcome.cost_usd is not None
    ]
    cost_complete = usage_complete and len(cost_values) == len(outcomes)

    return ModelMetrics(
        model=model,
        total_score=total,
        category_scores=categories,
        voseo_rate=round(100.0 * voseo_cases / count, 2),
        spain_rate=round(100.0 * spain_cases / count, 2),
        mean_latency_ms=round(
            sum(with_latency) / len(with_latency), 1
        )
        if with_latency
        else 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(sum(cost_values), 6) if cost_complete else None,
        cost_complete=cost_complete,
        price_is_example=price_is_example,
        cases=len(outcomes),
        fully_passed_cases=sum(1 for outcome in outcomes if outcome.passed),
        failed_cases=sum(1 for outcome in outcomes if not outcome.passed),
    )


def categories_in_order(outcomes: Sequence[CaseOutcome]) -> list[str]:
    """Return categories in first-seen order."""

    return list(dict.fromkeys(outcome.category for outcome in outcomes))
