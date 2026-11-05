"""Report generation: Markdown, CSV, and a self-contained HTML file."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SAMPLE_NOTICE = (
    "Sample output from the mock provider, not real benchmark results."
)


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_latency(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_cost(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _models(results: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    raw = results.get("models") or {}
    items = [
        (str(model), payload)
        for model, payload in raw.items()
        if isinstance(payload, Mapping)
    ]
    items.sort(
        key=lambda item: float(
            (item[1].get("summary") or {}).get("total_score") or 0
        ),
        reverse=True,
    )
    return items


def _categories(results: Mapping[str, Any]) -> list[str]:
    suite = results.get("suite") or {}
    categories = suite.get("categories")
    if isinstance(categories, list) and categories:
        return [str(category) for category in categories]
    found: list[str] = []
    for _, payload in _models(results):
        for case in payload.get("cases") or []:
            category = str(case.get("category", ""))
            if category and category not in found:
                found.append(category)
    return found


def _all_cases(
    results: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for model, payload in _models(results):
        for case in payload.get("cases") or []:
            if isinstance(case, Mapping):
                rows.append((model, case))
    return rows


def _failed_check_names(case: Mapping[str, Any]) -> list[str]:
    if case.get("error"):
        return ["provider_error"]
    failed: list[str] = []
    for check in case.get("checks") or []:
        if isinstance(check, Mapping) and not check.get("passed"):
            failed.append(str(check.get("type", "unknown")))
    return failed


def _summary_table_md(results: Mapping[str, Any]) -> str:
    categories = _categories(results)
    header = [
        "Model",
        "Overall",
        *categories,
        "Voseo rate",
        "Spain rate",
        "Avg latency",
        "Tokens in",
        "Tokens out",
        "Est. cost",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for model, payload in _models(results):
        summary = payload.get("summary") or {}
        category_scores = summary.get("category_scores") or {}
        row = [
            f"`{model}`",
            f"**{_fmt_score(summary.get('total_score'))}**",
            *[_fmt_score(category_scores.get(category)) for category in categories],
            f"{_fmt_score(summary.get('voseo_rate'))}%",
            f"{_fmt_score(summary.get('spain_rate'))}%",
            _fmt_latency(summary.get("mean_latency_ms")),
            _fmt_int(summary.get("prompt_tokens")),
            _fmt_int(summary.get("completion_tokens")),
            _fmt_cost(summary.get("cost_usd")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _case_table_md(results: Mapping[str, Any]) -> str:
    lines = [
        "| Model | Case | Category | Score | Checks | Latency | Est. cost | Failed checks |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for model, case in _all_cases(results):
        checks = case.get("checks") or []
        passed = sum(
            1
            for check in checks
            if isinstance(check, Mapping) and check.get("passed")
        )
        failed = ", ".join(_failed_check_names(case)) or "—"
        if case.get("error"):
            failed = f"provider error: {case.get('error')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model}`",
                    f"`{case.get('id', '')}`",
                    str(case.get("category", "")),
                    _fmt_score(case.get("score")),
                    f"{passed}/{len(checks)}",
                    _fmt_latency(case.get("latency_ms")),
                    _fmt_cost(case.get("cost_usd")),
                    str(failed).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown(results: Mapping[str, Any]) -> str:
    suite = results.get("suite") or {}
    prices = results.get("prices") or {}
    dry_run = bool(results.get("dry_run"))
    generated = str(results.get("generated_at", "unknown"))

    parts: list[str] = ["# habla-bench report", ""]
    if dry_run:
        parts.extend([f"> **{SAMPLE_NOTICE}**", ""])
    parts.extend(
        [
            f"- Generated: {generated}",
            f"- Suite: `{suite.get('name', 'unknown')}` "
            f"({suite.get('cases', 0)} cases, SHA-256 "
            f"`{str(suite.get('sha256', ''))[:12]}`)",
            f"- Provider: `{results.get('provider', 'unknown')}`"
            + (" (dry run, simulated latency and tokens)" if dry_run else ""),
            f"- Prices: {prices.get('note', '')}",
            "",
            "Scores are 0–100. Category scores are weighted by case weight. "
            "The overall score is the weighted mean of every case.",
            "",
            "## Summary",
            "",
            _summary_table_md(results),
            "",
        ]
    )
    if prices.get("note"):
        parts.extend(
            [
                "Costs are estimates based on the token usage returned by the "
                "provider and the example prices in `prices.yaml`.",
                "",
            ]
        )

    failures: list[tuple[str, Mapping[str, Any]]] = [
        (model, case)
        for model, case in _all_cases(results)
        if _failed_check_names(case)
    ]
    parts.extend(["## Failing checks", ""])
    if failures:
        parts.extend(
            [
                "| Model | Case | Check | Detail |",
                "| --- | --- | --- | --- |",
            ]
        )
        for model, case in failures:
            if case.get("error"):
                parts.append(
                    f"| `{model}` | `{case.get('id', '')}` | provider_error | "
                    f"{str(case.get('error')).replace('|', chr(92) + '|')} |"
                )
                continue
            for check in case.get("checks") or []:
                if isinstance(check, Mapping) and not check.get("passed"):
                    detail = str(check.get("detail", "")).replace("|", "\\|")
                    parts.append(
                        f"| `{model}` | `{case.get('id', '')}` | "
                        f"`{check.get('type', '')}` | {detail} |"
                    )
    else:
        parts.append("No failing checks in this run.")
    parts.append("")

    parts.extend(
        [
            "## Per-case results",
            "",
            _case_table_md(results),
            "",
            "## Method",
            "",
            "Each response is scored by deterministic checks only: JSON shape, "
            "word limits, bullet counts, forbidden words, voseo markers, "
            "peninsular Spanish markers, a Spanish-language heuristic, and "
            "numbers that do not appear in the provided context. No model is "
            "used as a judge.",
            "",
            "## Limitations",
            "",
            "- The language check is a word-frequency heuristic, not a full "
            "language detector.",
            "- `no_invented_numbers` covers digits only; spelled-out numbers "
            "are outside its scope.",
            "- Template-style checks reward compliance with a format, not the "
            "truthfulness or usefulness of the content.",
            "- Cost estimates depend on the `usage` block returned by the "
            "provider and on the example prices in `prices.yaml`.",
            "",
            "<sub>Generated by habla-bench. No external assets.</sub>",
            "",
        ]
    )
    return "\n".join(parts)


def _write_summary_csv(results: Mapping[str, Any], path: Path) -> None:
    categories = _categories(results)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "overall_score",
                *categories,
                "voseo_rate",
                "spain_rate",
                "mean_latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "cost_complete",
                "price_is_example",
                "cases",
                "fully_passed_cases",
                "failed_cases",
                "simulated",
            ]
        )
        for model, payload in _models(results):
            summary = payload.get("summary") or {}
            scores = summary.get("category_scores") or {}
            cost = summary.get("cost_usd")
            writer.writerow(
                [
                    model,
                    summary.get("total_score", ""),
                    *[scores.get(category, "") for category in categories],
                    summary.get("voseo_rate", ""),
                    summary.get("spain_rate", ""),
                    summary.get("mean_latency_ms", ""),
                    summary.get("prompt_tokens", ""),
                    summary.get("completion_tokens", ""),
                    "" if cost is None else f"{float(cost):.8f}",
                    "yes" if summary.get("cost_complete") else "no",
                    "yes" if summary.get("price_is_example") else "no",
                    summary.get("cases", ""),
                    summary.get("fully_passed_cases", ""),
                    summary.get("failed_cases", ""),
                    "yes" if summary.get("simulated") else "no",
                ]
            )


def _write_cases_csv(results: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "case_id",
                "category",
                "score",
                "checks_passed",
                "checks_total",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "passed",
                "failed_checks",
                "response",
            ]
        )
        for model, case in _all_cases(results):
            checks = case.get("checks") or []
            passed = sum(
                1
                for check in checks
                if isinstance(check, Mapping) and check.get("passed")
            )
            usage = case.get("usage") or {}
            cost = case.get("cost_usd")
            cost_text = "" if cost is None else f"{float(cost):.8f}"
            writer.writerow(
                [
                    model,
                    case.get("id", ""),
                    case.get("category", ""),
                    case.get("score", ""),
                    passed,
                    len(checks),
                    case.get("latency_ms", ""),
                    usage.get("prompt_tokens", ""),
                    usage.get("completion_tokens", ""),
                    cost_text,
                    "yes" if case.get("passed") else "no",
                    ";".join(_failed_check_names(case)),
                    case.get("response", ""),
                ]
            )


def _html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _summary_table_html(results: Mapping[str, Any]) -> str:
    categories = _categories(results)
    head = "".join(
        [
            '<th data-sort="text">Model</th>',
            '<th data-sort="number">Overall</th>',
            *[
                f'<th data-sort="number">{_html_escape(category)}</th>'
                for category in categories
            ],
            '<th data-sort="number">Voseo rate</th>',
            '<th data-sort="number">Spain rate</th>',
            '<th data-sort="number">Avg latency</th>',
            '<th data-sort="number">Tokens in</th>',
            '<th data-sort="number">Tokens out</th>',
            '<th data-sort="number">Est. cost</th>',
        ]
    )
    rows: list[str] = []
    for model, payload in _models(results):
        summary = payload.get("summary") or {}
        category_scores = summary.get("category_scores") or {}
        rows.append(
            "<tr>"
            f"<td><code>{_html_escape(model)}</code></td>"
            f'<td data-sort-value="{_fmt_score(summary.get("total_score"))}">'
            f'<strong>{_fmt_score(summary.get("total_score"))}</strong></td>'
            + "".join(
                f'<td data-sort-value="{_fmt_score(category_scores.get(category))}">'
                f"{_fmt_score(category_scores.get(category))}</td>"
                for category in categories
            )
            + f'<td data-sort-value="{_fmt_score(summary.get("voseo_rate"))}">'
            f'{_fmt_score(summary.get("voseo_rate"))}%</td>'
            f'<td data-sort-value="{_fmt_score(summary.get("spain_rate"))}">'
            f'{_fmt_score(summary.get("spain_rate"))}%</td>'
            f'<td data-sort-value="{float(summary.get("mean_latency_ms") or 0):.1f}">'
            f'{_fmt_latency(summary.get("mean_latency_ms"))}</td>'
            f'<td data-sort-value="{int(summary.get("prompt_tokens") or 0)}">'
            f'{_fmt_int(summary.get("prompt_tokens"))}</td>'
            f'<td data-sort-value="{int(summary.get("completion_tokens") or 0)}">'
            f'{_fmt_int(summary.get("completion_tokens"))}</td>'
            f'<td data-sort-value="{float(summary.get("cost_usd") or 0):.6f}">'
            f'{_fmt_cost(summary.get("cost_usd"))}</td>'
            "</tr>"
        )
    return (
        '<table class="sortable" data-sortable>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _case_table_html(results: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for model, case in _all_cases(results):
        checks = case.get("checks") or []
        passed = sum(
            1
            for check in checks
            if isinstance(check, Mapping) and check.get("passed")
        )
        failed = ", ".join(_failed_check_names(case)) or "—"
        if case.get("error"):
            failed = f"provider error: {case.get('error')}"
        score = float(case.get("score") or 0)
        status = "pass" if case.get("passed") else "fail"
        rows.append(
            "<tr>"
            f"<td><code>{_html_escape(model)}</code></td>"
            f"<td><code>{_html_escape(case.get('id', ''))}</code></td>"
            f"<td>{_html_escape(case.get('category', ''))}</td>"
            f'<td data-sort-value="{score:.2f}">{_fmt_score(score)}</td>'
            f'<td data-sort-value="{passed}">{passed}/{len(checks)}</td>'
            f'<td data-sort-value="{float(case.get("latency_ms") or 0):.1f}">'
            f'{_fmt_latency(case.get("latency_ms"))}</td>'
            f'<td data-sort-value="{float(case.get("cost_usd") or 0):.6f}">'
            f'{_fmt_cost(case.get("cost_usd"))}</td>'
            f'<td><span class="tag {status}">{status}</span></td>'
            f"<td>{_html_escape(failed)}</td>"
            "</tr>"
        )
    head = "".join(
        [
            '<th data-sort="text">Model</th>',
            '<th data-sort="text">Case</th>',
            '<th data-sort="text">Category</th>',
            '<th data-sort="number">Score</th>',
            '<th data-sort="number">Checks</th>',
            '<th data-sort="number">Latency</th>',
            '<th data-sort="number">Est. cost</th>',
            '<th data-sort="text">Status</th>',
            '<th data-sort="text">Failed checks</th>',
        ]
    )
    return (
        '<table class="sortable" data-sortable>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _failure_details_html(results: Mapping[str, Any]) -> str:
    details: list[str] = []
    for model, payload in _models(results):
        cases = [
            case
            for case in payload.get("cases") or []
            if isinstance(case, Mapping) and _failed_check_names(case)
        ]
        if not cases:
            continue
        items: list[str] = []
        for case in cases:
            if case.get("error"):
                items.append(
                    f"<li><code>{_html_escape(case.get('id'))}</code>: "
                    f"provider error — {_html_escape(case.get('error'))}</li>"
                )
                continue
            checks = [
                check
                for check in case.get("checks") or []
                if isinstance(check, Mapping) and not check.get("passed")
            ]
            rendered = "; ".join(
                f"<code>{_html_escape(check.get('type'))}</code> "
                f"({_html_escape(check.get('detail'))})"
                for check in checks
            )
            items.append(
                f"<li><code>{_html_escape(case.get('id'))}</code>: {rendered}</li>"
            )
        details.append(
            f"<details><summary><code>{_html_escape(model)}</code> "
            f"({len(cases)} case(s) with failures)</summary>"
            f"<ul>{''.join(items)}</ul></details>"
        )
    if not details:
        return "<p>No failing checks in this run.</p>"
    return "".join(details)


def render_html(results: Mapping[str, Any]) -> str:
    suite = results.get("suite") or {}
    prices = results.get("prices") or {}
    dry_run = bool(results.get("dry_run"))
    notice = (
        f'<div class="notice" role="note"><strong>{SAMPLE_NOTICE}</strong> '
        "Latency and token usage are simulated.</div>"
        if dry_run
        else ""
    )
    footer_note = (
        "Costs use example prices from <code>prices.yaml</code>."
        if "example" in str(prices.get("note", "")).lower()
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>habla-bench report · {_html_escape(suite.get("name", "suite"))}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f7f8fa;
  --surface: #ffffff;
  --surface-2: #eef1f5;
  --text: #17202b;
  --muted: #5b6878;
  --border: #d8dee7;
  --accent: #0b6b53;
  --accent-2: #0b6b53;
  --pass: #0b6b53;
  --fail: #a33a2b;
  --shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0c1118;
    --surface: #121a24;
    --surface-2: #192331;
    --text: #e8edf4;
    --muted: #a3b0c2;
    --border: #2a3748;
    --accent: #6fd7b6;
    --accent-2: #6fd7b6;
    --pass: #6fd7b6;
    --fail: #ff9b8a;
    --shadow: none;
  }}
}}
html[data-theme="light"] {{
  color-scheme: light;
  --bg: #f7f8fa; --surface: #ffffff; --surface-2: #eef1f5;
  --text: #17202b; --muted: #5b6878; --border: #d8dee7;
  --accent: #0b6b53; --accent-2: #0b6b53;
  --pass: #0b6b53; --fail: #a33a2b; --shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}}
html[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #0c1118; --surface: #121a24; --surface-2: #192331;
  --text: #e8edf4; --muted: #a3b0c2; --border: #2a3748;
  --accent: #6fd7b6; --accent-2: #6fd7b6;
  --pass: #6fd7b6; --fail: #ff9b8a; --shadow: none;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
}}
header {{
  padding: 2.4rem 1.25rem 1.4rem; border-bottom: 1px solid var(--border);
  background: var(--surface);
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
.brand {{ display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap; }}
h1 {{ font-size: clamp(1.7rem, 4vw, 2.5rem); margin: 0; letter-spacing: -0.03em; }}
.sub {{ color: var(--muted); margin: 0.4rem 0 0; }}
.meta {{ display: flex; gap: 0.55rem; flex-wrap: wrap; margin-top: 1rem; }}
.pill {{
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 999px; padding: 0.2rem 0.65rem; color: var(--muted);
  font-size: 0.83rem;
}}
main {{ max-width: 1180px; margin: 0 auto; padding: 1.4rem 1.25rem 3rem; }}
section {{ margin-top: 2rem; }}
h2 {{ font-size: 1.18rem; margin: 0 0 0.7rem; letter-spacing: -0.01em; }}
p {{ margin: 0.5rem 0; }}
.notice {{
  margin-top: 1rem; padding: 0.8rem 1rem; border: 1px solid var(--border);
  border-left: 4px solid var(--accent); background: var(--surface-2);
  border-radius: 8px;
}}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
caption {{ text-align: left; padding: 0.55rem 0.8rem; color: var(--muted); }}
th, td {{ padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
th {{ background: var(--surface-2); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
th[data-sort] {{ cursor: pointer; user-select: none; }}
th[data-sort]:hover {{ color: var(--text); }}
th[data-sort]::after {{ content: " ↕"; opacity: 0.55; font-size: 0.8em; }}
th[data-dir="asc"]::after {{ content: " ↑"; opacity: 1; }}
th[data-dir="desc"]::after {{ content: " ↓"; opacity: 1; }}
tbody tr:hover {{ background: var(--surface-2); }}
tbody tr:last-child td {{ border-bottom: 0; }}
code {{ font: 0.88em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.tag {{ display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: 0.78rem; border: 1px solid currentColor; }}
.tag.pass {{ color: var(--pass); }}
.tag.fail {{ color: var(--fail); }}
details {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.7rem 0.9rem; margin: 0.55rem 0; }}
summary {{ cursor: pointer; }}
footer {{ border-top: 1px solid var(--border); padding: 1.1rem 1.25rem; color: var(--muted); font-size: 0.85rem; }}
.toolbar {{ margin-left: auto; }}
button {{
  background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.4rem 0.75rem; cursor: pointer; font: inherit;
}}
button:hover {{ border-color: var(--accent); }}
ul {{ margin: 0.4rem 0 0.2rem 1.1rem; padding: 0; }}
@media print {{
  header, main, footer {{ max-width: none; }}
  .toolbar, button {{ display: none; }}
  table {{ font-size: 10px; }}
}}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="brand">
      <h1>habla-bench report</h1>
      <span class="toolbar"><button id="theme-toggle" type="button" aria-label="Toggle theme">Light / dark</button></span>
    </div>
    <p class="sub">Neutral Latin American Spanish · deterministic checks · no model judge</p>
    <div class="meta">
      <span class="pill">Suite: {_html_escape(suite.get("name", "unknown"))}</span>
      <span class="pill">{_html_escape(suite.get("cases", 0))} cases</span>
      <span class="pill">Provider: {_html_escape(results.get("provider", "unknown"))}</span>
      <span class="pill">{_html_escape(results.get("generated_at", "unknown"))}</span>
    </div>
    {notice}
  </div>
</header>
<main>
  <section>
    <h2>Summary</h2>
    <p>Scores are 0–100. Category scores are weighted by case weight; the overall score is the weighted mean of every case.</p>
    <div class="table-wrap">{_summary_table_html(results)}</div>
    <p>{_html_escape(prices.get("note", ""))} {footer_note}</p>
  </section>
  <section>
    <h2>Failing checks</h2>
    {_failure_details_html(results)}
  </section>
  <section>
    <h2>Per-case results</h2>
    <div class="table-wrap">{_case_table_html(results)}</div>
  </section>
  <section>
    <h2>Method</h2>
    <p>Every response is scored by deterministic checks only: JSON shape, word limits, bullet counts, forbidden words, voseo markers, peninsular Spanish markers, a Spanish-language heuristic, and numbers that do not appear in the provided context. No model is used as a judge.</p>
    <h2>Limitations</h2>
    <ul>
      <li>The language check is a word-frequency heuristic, not a full language detector.</li>
      <li><code>no_invented_numbers</code> covers digits only; spelled-out numbers are outside its scope.</li>
      <li>Template-style checks reward format compliance, not truthfulness or usefulness.</li>
      <li>Cost is an estimate based on provider token usage and the example prices in <code>prices.yaml</code>.</li>
    </ul>
  </section>
</main>
<footer>
  <div class="wrap">Generated by habla-bench · self-contained file · no external assets</div>
</footer>
<script>
(function () {{
  var root = document.documentElement;
  var saved = null;
  try {{ saved = localStorage.getItem("habla-bench-theme"); }} catch (err) {{}}
  if (saved === "light" || saved === "dark") {{ root.setAttribute("data-theme", saved); }}
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {{
    toggle.addEventListener("click", function () {{
      var current = root.getAttribute("data-theme");
      if (!current) {{
        current = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }}
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try {{ localStorage.setItem("habla-bench-theme", next); }} catch (err) {{}}
    }});
  }}
  function valueOf(cell) {{
    if (cell.hasAttribute("data-sort-value")) {{ return cell.getAttribute("data-sort-value"); }}
    return cell.textContent.trim();
  }}
  function sortTable(table, index, header) {{
    var tbody = table.tBodies[0];
    if (!tbody) {{ return; }}
    var rows = Array.prototype.slice.call(tbody.rows);
    var numeric = header.getAttribute("data-sort") === "number";
    var direction = header.getAttribute("data-dir") === "asc" ? "desc" : "asc";
    Array.prototype.forEach.call(table.tHead.rows[0].cells, function (cell) {{
      cell.removeAttribute("data-dir");
      cell.removeAttribute("aria-sort");
    }});
    header.setAttribute("data-dir", direction);
    header.setAttribute("aria-sort", direction === "asc" ? "ascending" : "descending");
    rows.sort(function (a, b) {{
      var av = valueOf(a.cells[index]);
      var bv = valueOf(b.cells[index]);
      var result;
      if (numeric) {{
        result = (parseFloat(av) || 0) - (parseFloat(bv) || 0);
      }} else {{
        result = av.toLocaleLowerCase("es").localeCompare(bv.toLocaleLowerCase("es"));
      }}
      return direction === "asc" ? result : -result;
    }});
    rows.forEach(function (row) {{ tbody.appendChild(row); }});
  }}
  Array.prototype.forEach.call(document.querySelectorAll("table[data-sortable]"), function (table) {{
    if (!table.tHead || !table.tHead.rows.length) {{ return; }}
    Array.prototype.forEach.call(table.tHead.rows[0].cells, function (header, index) {{
      if (!header.hasAttribute("data-sort")) {{ return; }}
      header.tabIndex = 0;
      header.setAttribute("role", "button");
      header.addEventListener("click", function () {{ sortTable(table, index, header); }});
      header.addEventListener("keydown", function (event) {{
        if (event.key === "Enter" || event.key === " ") {{
          event.preventDefault();
          sortTable(table, index, header);
        }}
      }});
    }});
  }});
}})();
</script>
</body>
</html>
"""


def build_report(
    results_path: str | Path, out_dir: str | Path | None = None
) -> dict[str, Path]:
    """Read a results file and write ``report.md``, ``report.csv``, ``report.html``."""

    payload, source = _load(results_path)
    destination = (
        Path(out_dir) if out_dir is not None else source.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    md_path = destination / "report.md"
    csv_path = destination / "report.csv"
    cases_csv_path = destination / "cases.csv"
    html_path = destination / "report.html"

    md_path.write_text(render_markdown(payload), encoding="utf-8")
    _write_summary_csv(payload, csv_path)
    _write_cases_csv(payload, cases_csv_path)
    html_path.write_text(render_html(payload), encoding="utf-8")
    return {
        "markdown": md_path,
        "csv": csv_path,
        "cases_csv": cases_csv_path,
        "html": html_path,
    }


def _load(path: str | Path) -> tuple[dict[str, Any], Path]:
    # Kept local to avoid a circular import at module import time.
    from .runner import load_results

    return load_results(path)
