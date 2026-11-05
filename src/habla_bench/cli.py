"""Command-line interface for habla-bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import ConfigError, load_prices, load_settings
from .providers import MockProvider, OpenAICompatibleProvider, ProviderError
from .report import build_report
from .runner import run_benchmark
from .suite import SuiteError, resolve_suite


def _split_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ConfigError("--models must contain at least one model id")
    deduped: list[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def _print_summary(payload: dict[str, object]) -> None:
    models = payload.get("models") or {}
    print()
    print(f"{'model':<24} {'score':>7} {'voseo':>7} {'latency':>10} {'cost':>12}")
    print("-" * 64)
    for model, data in models.items():
        summary = (data or {}).get("summary", {})
        score = float(summary.get("total_score") or 0)
        voseo = float(summary.get("voseo_rate") or 0)
        latency = float(summary.get("mean_latency_ms") or 0)
        cost = summary.get("cost_usd")
        cost_text = "n/a" if cost is None else f"${float(cost):.6f}"
        print(
            f"{model:<24} {score:>7.1f} {voseo:>6.1f}% "
            f"{latency:>8.0f}ms {cost_text:>12}"
        )
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="habla-bench",
        description=(
            "Benchmark low-cost LLMs on neutral Latin American Spanish, "
            "instruction following, and customer-service replies."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="run a suite against one or more models"
    )
    run_parser.add_argument(
        "--models",
        required=True,
        help="comma-separated model ids, for example mock-a,mock-b",
    )
    run_parser.add_argument(
        "--suite",
        default="core",
        help="suite name or path to a YAML file (default: core)",
    )
    run_parser.add_argument(
        "--out",
        default="results",
        help="directory for results.json (default: results)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the deterministic mock provider; no network calls",
    )
    run_parser.add_argument(
        "--prices",
        default="prices.yaml",
        help="price table in YAML (default: prices.yaml if present)",
    )
    run_parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: HABLA_BASE_URL or OpenRouter)",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="request timeout in seconds (default: 60)",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="maximum completion tokens (default: 512)",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print one line per case",
    )

    report_parser = subparsers.add_parser(
        "report", help="build report.md, report.csv, and report.html"
    )
    report_parser.add_argument(
        "results",
        help="results directory or path to results.json",
    )
    report_parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: next to results.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            outputs = build_report(args.results, args.out)
            print("Wrote:")
            for label, path in outputs.items():
                print(f"  {label}: {path}")
            return 0
    except (ConfigError, SuiteError, ProviderError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


def _run(args: argparse.Namespace) -> int:
    models = _split_models(args.models)
    suite = resolve_suite(args.suite)
    settings = load_settings(
        dry_run=args.dry_run,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    prices_path = Path(args.prices)
    prices = (
        load_prices(prices_path)
        if prices_path.is_file()
        else load_prices(None)
    )
    if not prices_path.is_file() and not args.dry_run:
        print(
            f"warning: {prices_path} not found; cost will not be estimated",
            file=sys.stderr,
        )

    if args.max_tokens <= 0:
        raise ConfigError("--max-tokens must be greater than zero")
    if not 0 <= args.temperature <= 2:
        raise ConfigError("--temperature must be between 0 and 2")

    if args.dry_run:
        provider = MockProvider()
    else:
        provider = OpenAICompatibleProvider(
            settings.base_url,
            settings.api_key or "",
            timeout=settings.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    print(
        f"habla-bench {__version__} | suite {suite.name!r} | "
        f"{len(suite.cases)} cases | models: {', '.join(models)}"
    )
    if args.dry_run:
        print("dry run: deterministic mock provider, no network calls")
    else:
        print(f"provider: {settings.base_url}")

    def progress(model: str, index: int, total: int, case_id: str) -> None:
        if args.verbose:
            print(f"[{index:>3}/{total}] {model:<20} {case_id}")

    result = run_benchmark(
        suite,
        models,
        provider,
        prices,
        args.out,
        dry_run=args.dry_run,
        progress=progress if args.verbose else None,
    )
    payload = result.payload
    _print_summary(payload)
    print(f"results: {result.path}")
    print(f"next:    habla-bench report {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
