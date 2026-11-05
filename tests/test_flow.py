"""End-to-end flow tests: run with the mock, then build reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from habla_bench.cli import main
from habla_bench.config import load_prices
from habla_bench.providers import MockProvider, ProviderError
from habla_bench.report import SAMPLE_NOTICE, build_report, render_html, render_markdown
from habla_bench.runner import load_results, run_benchmark
from habla_bench.suite import load_suite, parse_suite

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "suites" / "core.yaml"
PRICES = ROOT / "prices.yaml"


@pytest.fixture(scope="module")
def core_suite():
    return load_suite(CORE)


@pytest.fixture()
def run_results(tmp_path, core_suite):
    prices = load_prices(PRICES)
    return run_benchmark(
        core_suite,
        ["mock-a", "mock-b"],
        MockProvider(),
        prices,
        tmp_path / "results",
        dry_run=True,
    )


def test_run_writes_results_json(run_results):
    path = run_results.path
    assert path.name == "results.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["provider"] == "mock"
    assert payload["suite"]["name"] == "core"
    assert payload["suite"]["cases"] == 36
    assert set(payload["models"]) == {"mock-a", "mock-b"}
    for model in ("mock-a", "mock-b"):
        cases = payload["models"][model]["cases"]
        assert len(cases) == 36
        assert {case["category"] for case in cases} == {
            "instruction_following",
            "neutral_spanish",
            "language_lock",
            "customer_service",
        }
        assert all("checks" in case for case in cases)
        assert all(case["response"] for case in cases)


def test_mock_a_scores_higher_than_mock_b(run_results):
    models = run_results.payload["models"]
    assert (
        models["mock-a"]["summary"]["total_score"]
        > models["mock-b"]["summary"]["total_score"]
    )
    assert models["mock-a"]["summary"]["cost_usd"] is not None
    assert models["mock-a"]["summary"]["prompt_tokens"] > 0


def test_dry_run_is_deterministic(tmp_path, core_suite):
    prices = load_prices(PRICES)
    first = run_benchmark(
        core_suite, ["mock-a"], MockProvider(), prices, tmp_path / "one", dry_run=True
    )
    second = run_benchmark(
        core_suite, ["mock-a"], MockProvider(), prices, tmp_path / "two", dry_run=True
    )

    def comparable(payload):
        cases = payload["models"]["mock-a"]["cases"]
        return [
            (
                case["id"],
                case["response"],
                case["score"],
                case["latency_ms"],
                case["usage"],
            )
            for case in cases
        ]

    assert comparable(first.payload) == comparable(second.payload)


def test_progress_callback(tmp_path, run_results):
    seen: list[tuple[str, int, int, str]] = []
    suite = parse_suite(
        {
            "name": "tiny",
            "cases": [
                {
                    "id": "tiny-1",
                    "category": "neutral_spanish",
                    "system": "Responde en español.",
                    "user": "Di hola.",
                    "checks": [{"type": "is_spanish"}],
                    "weight": 1,
                }
            ],
        }
    )
    run_benchmark(
        suite,
        ["mock-a"],
        MockProvider(),
        load_prices(None),
        tmp_path / "progress",
        dry_run=True,
        progress=lambda *args: seen.append(args),
    )
    assert seen == [("mock-a", 1, 1, "tiny-1")]


def test_report_files(run_results):
    outputs = build_report(run_results.path)
    assert set(outputs) == {"markdown", "csv", "cases_csv", "html"}
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_markdown_labels_mock_output(run_results):
    outputs = build_report(run_results.path, run_results.path.parent / "md")
    text = outputs["markdown"].read_text(encoding="utf-8")
    assert SAMPLE_NOTICE in text
    assert "## Summary" in text
    assert "## Failing checks" in text
    assert "## Method" in text
    assert "mock-a" in text and "mock-b" in text
    assert "no_invented_numbers" in text


def test_html_is_self_contained_and_sortable(run_results):
    html = render_html(run_results.payload)
    assert "<!doctype html>" in html
    assert "data-sortable" in html
    assert "sortTable" in html
    assert "prefers-color-scheme" in html
    assert "localStorage" in html
    assert "<script src" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert SAMPLE_NOTICE in html


def test_html_escapes_script_payload(tmp_path):
    suite = parse_suite(
        {
            "name": "<script>alert(1)</script>",
            "cases": [
                {
                    "id": "xss-1",
                    "category": "neutral_spanish",
                    "system": "Responde en español.",
                    "user": "Di hola.",
                    "checks": [{"type": "is_spanish"}],
                    "weight": 1,
                }
            ],
        }
    )
    result = run_benchmark(
        suite,
        ["<img src=x onerror=alert(1)>"],
        MockProvider(),
        load_prices(None),
        tmp_path / "xss",
        dry_run=True,
    )
    html = render_html(result.payload)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img src=x" not in html


def test_summary_csv_has_one_row_per_model(run_results):
    outputs = build_report(run_results.path)
    with outputs["csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 2
    assert rows[0][:2] == ["model", "overall_score"]
    assert {row[0] for row in rows[1:]} == {"mock-a", "mock-b"}
    assert all("e-" not in row[11] for row in rows[1:])


def test_cases_csv_has_one_row_per_case(run_results):
    outputs = build_report(run_results.path)
    with outputs["cases_csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 36 * 2
    assert rows[0][:4] == ["model", "case_id", "category", "score"]
    assert {row[0] for row in rows[1:]} == {"mock-a", "mock-b"}
    assert all("e-" not in row[9] for row in rows[1:])


def test_report_accepts_custom_output_directory(run_results, tmp_path):
    destination = tmp_path / "custom"
    outputs = build_report(run_results.path, destination)
    assert all(path.parent == destination for path in outputs.values())


def test_report_from_json_path(tmp_path, run_results):
    copied = tmp_path / "copy.json"
    copied.write_bytes(run_results.path.read_bytes())
    outputs = build_report(copied)
    assert outputs["html"].parent == tmp_path


def test_load_results_rejects_non_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_results(path)


def test_load_results_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_provider_error_is_recorded(tmp_path):
    class FailingProvider:
        name = "failing"

        def complete(self, model, system, user, *, case=None):
            raise ProviderError("simulated outage")

    suite = parse_suite(
        {
            "name": "tiny",
            "cases": [
                {
                    "id": "tiny-1",
                    "category": "neutral_spanish",
                    "system": "Responde en español.",
                    "user": "Di hola.",
                    "checks": [{"type": "is_spanish"}],
                    "weight": 1,
                }
            ],
        }
    )
    result = run_benchmark(
        suite,
        ["mock-a"],
        FailingProvider(),
        load_prices(None),
        tmp_path / "out",
        dry_run=False,
    )
    case = result.payload["models"]["mock-a"]["cases"][0]
    assert case["error"] == "simulated outage"
    assert case["score"] == 0
    assert case["checks"] == []
    markdown = render_markdown(result.payload)
    assert "simulated outage" in markdown
    assert "simulated outage" in render_html(result.payload)


def test_cli_run_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out_dir = tmp_path / "cli-results"
    code = main(
        [
            "run",
            "--dry-run",
            "--models",
            "mock-a,mock-b",
            "--suite",
            "core",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "results.json").is_file()
    output = capsys.readouterr().out
    assert "mock-a" in output
    assert "dry run" in output

    code = main(["report", str(out_dir)])
    assert code == 0
    for name in ("report.md", "report.csv", "cases.csv", "report.html"):
        assert (out_dir / name).is_file()


def test_cli_live_run_without_key_fails_before_network(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("HABLA_API_KEY", raising=False)
    code = main(
        [
            "run",
            "--models",
            "some/model",
            "--suite",
            "core",
            "--out",
            str(tmp_path / "live"),
        ]
    )
    assert code == 2
    assert "HABLA_API_KEY" in capsys.readouterr().err


def test_cli_missing_results_returns_error(tmp_path, capsys):
    code = main(["report", str(tmp_path / "nothing")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_empty_models(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    code = main(
        ["run", "--dry-run", "--models", " , ", "--suite", "core", "--out", str(tmp_path)]
    )
    assert code == 2
    assert "at least one model" in capsys.readouterr().err
