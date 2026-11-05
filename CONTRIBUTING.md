# Contributing

Thanks for helping make model selection less guessy.

## Setup

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the tests before opening a pull request:

```bash
pytest -q
```

## What makes a good contribution

- **A new deterministic check.** A check must be a pure function of the response and the context. No network, no clocks, no randomness, and no second model as a judge. Include unit tests for the passing and failing cases, plus a short entry in the checks table in `README.md`.
- **A new suite.** Add cases that are realistic, short, and clearly fictional. Every case needs a unique `id`, `category`, `system`, `user`, `checks`, and a positive `weight`. Run the suite with `--dry-run` before submitting. The bundled `core` suite lives in both `suites/core.yaml` and `src/habla_bench/suites/core.yaml` (package data); keep the two files byte-identical, since a test compares them.
- **A bug fix.** Add a regression test that fails before the fix and passes after it.

## Rules for data

- Use fictional businesses, people, prices, addresses, and contact details.
- Do not include keys, tokens, personal data, or content from private customer conversations.
- Do not publish real provider prices without saying when they were checked.

## Pull requests

Keep changes focused. Explain what the change measures, how you tested it, and any limitation a reviewer should know about. If a check has a known false positive, document it in the check description and in the report limitations.

For a new check, open an issue first with the template in `.github/ISSUE_TEMPLATE/new-check.md`.
