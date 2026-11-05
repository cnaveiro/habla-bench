"""Benchmark execution and result serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .checks import run_checks
from .config import PriceTable, estimate_cost
from .metrics import (
    CaseOutcome,
    aggregate_model,
    case_score,
)
from .providers import Provider, ProviderError
from .suite import Suite

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class RunResult:
    payload: dict[str, object]
    path: Path


def _relative_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def run_benchmark(
    suite: Suite,
    models: Sequence[str],
    provider: Provider,
    prices: PriceTable,
    out_dir: str | Path,
    *,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Run every case against every model and write ``results.json``."""

    if not models:
        raise ValueError("at least one model id is required")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    model_payloads: dict[str, object] = {}
    summaries: dict[str, object] = {}
    total = len(suite.cases)

    for model in models:
        price = prices.for_model(model)
        outcomes: list[CaseOutcome] = []
        for index, case in enumerate(suite.cases, start=1):
            if progress is not None:
                progress(model, index, total, case.id)

            context = "\n".join(
                part for part in (case.context, case.system, case.user) if part
            )
            error: str | None = None
            response = ""
            latency_ms = 0.0
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            simulated = False

            try:
                completion = provider.complete(
                    model, case.system, case.user, case=case
                )
                response = completion.text or ""
                latency_ms = completion.latency_ms
                prompt_tokens = completion.prompt_tokens
                completion_tokens = completion.completion_tokens
                simulated = completion.simulated
            except ProviderError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 - keep the run going
                error = f"{type(exc).__name__}: {exc}"

            checks = (
                run_checks(case.checks, response, context)
                if error is None
                else []
            )
            score = 0.0 if error is not None else case_score(checks)
            cost = (
                None
                if error is not None
                else estimate_cost(price, prompt_tokens, completion_tokens)
            )

            outcomes.append(
                CaseOutcome(
                    case_id=case.id,
                    category=case.category,
                    weight=case.weight,
                    response=response,
                    latency_ms=round(latency_ms, 1),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    checks=checks,
                    score=score,
                    error=error,
                    system=case.system,
                    user=case.user,
                    context=case.context,
                )
            )

        metrics = aggregate_model(
            model, outcomes, price_is_example=price.is_example
        )
        summaries[model] = {
            **metrics.to_dict(),
            "simulated": dry_run,
        }
        model_payloads[model] = {
            "summary": summaries[model],
            "cases": [_outcome_to_dict(outcome) for outcome in outcomes],
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "provider": provider.name,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "path": _relative_or_none(suite.path),
            "sha256": suite.sha256,
            "cases": len(suite.cases),
            "categories": suite.categories,
        },
        "prices": {
            "currency": prices.currency,
            "unit": "per_million_tokens",
            "path": _relative_or_none(prices.path),
            "note": (
                "Example values from prices.yaml, not live provider quotes."
                if any(price.is_example for price in prices.models.values())
                or prices.default.is_example
                else "Values supplied by prices.yaml."
            ),
        },
        "models": model_payloads,
    }

    result_path = destination / "results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return RunResult(payload=payload, path=result_path)


def _outcome_to_dict(outcome: CaseOutcome) -> dict[str, object]:
    return {
        "id": outcome.case_id,
        "category": outcome.category,
        "weight": outcome.weight,
        "system": outcome.system,
        "user": outcome.user,
        "context": outcome.context,
        "response": outcome.response,
        "latency_ms": outcome.latency_ms,
        "usage": (
            {
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
            }
            if outcome.prompt_tokens is not None
            or outcome.completion_tokens is not None
            else None
        ),
        "cost_usd": outcome.cost_usd,
        "checks": [
            {
                "type": check.type,
                "passed": check.passed,
                "detail": check.detail,
            }
            for check in outcome.checks
        ],
        "score": outcome.score,
        "passed": outcome.passed,
        "error": outcome.error,
    }


def load_results(path: str | Path) -> tuple[dict[str, object], Path]:
    """Load a ``results.json`` file or find one inside a directory."""

    target = Path(path)
    if target.is_dir():
        candidate = target / "results.json"
        if candidate.is_file():
            target = candidate
        else:
            json_files = sorted(target.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(
                    f"no results.json found in {target}"
                )
            target = json_files[0]
    if not target.is_file():
        raise FileNotFoundError(f"results file not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {target}: {exc}") from exc
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError(f"{target} does not look like a habla-bench result")
    return payload, target
