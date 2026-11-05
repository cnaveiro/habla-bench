"""Tests for suite loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from habla_bench.checks import CHECK_TYPES
from habla_bench.suite import SuiteError, load_suite, parse_suite, resolve_suite

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "suites" / "core.yaml"


def _case(**overrides):
    payload = {
        "id": "case-1",
        "category": "instruction_following",
        "system": "Responde en español.",
        "user": "Di hola.",
        "checks": [{"type": "is_spanish"}],
        "weight": 1.0,
    }
    payload.update(overrides)
    return payload


def test_core_suite_loads():
    suite = load_suite(CORE)
    assert suite.name == "core"
    assert len(suite.cases) >= 30
    assert suite.path == CORE.resolve()
    assert len(suite.sha256) == 64


def test_core_suite_has_required_categories():
    suite = load_suite(CORE)
    assert suite.categories == [
        "instruction_following",
        "neutral_spanish",
        "language_lock",
        "customer_service",
    ]
    for category in suite.categories:
        assert sum(1 for case in suite.cases if case.category == category) >= 8


def test_core_suite_cases_are_well_formed():
    suite = load_suite(CORE)
    ids = [case.id for case in suite.cases]
    assert len(ids) == len(set(ids))
    for case in suite.cases:
        assert case.system.strip()
        assert case.user.strip()
        assert case.weight > 0
        assert case.checks
        for spec in case.checks:
            assert spec["type"] in CHECK_TYPES
        assert case.prompt_words > 0


def test_core_has_examples_of_every_check_type():
    suite = load_suite(CORE)
    used = {spec["type"] for case in suite.cases for spec in case.checks}
    assert used == set(CHECK_TYPES)


def test_resolve_suite_by_name():
    suite = resolve_suite("core")
    assert suite.name == "core"
    assert len(suite.cases) >= 30


def test_resolve_suite_by_path():
    suite = resolve_suite(str(CORE))
    assert suite.path == CORE.resolve()


def test_packaged_suite_matches_repo_suite():
    packaged = ROOT / "src" / "habla_bench" / "suites" / "core.yaml"
    assert packaged.read_bytes() == CORE.read_bytes()


def test_resolve_suite_from_unrelated_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    suite = resolve_suite("core")
    assert suite.name == "core"
    assert len(suite.cases) == 36


def test_resolve_missing_suite_raises():
    with pytest.raises(SuiteError):
        resolve_suite("does-not-exist")


def test_load_missing_file_raises():
    with pytest.raises(SuiteError):
        load_suite(ROOT / "suites" / "missing.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: broken\ncases: [\n", encoding="utf-8")
    with pytest.raises(SuiteError):
        load_suite(path)


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ([], "mapping"),
        ({"name": "x", "cases": []}, "non-empty list"),
        ({"name": "x", "cases": [_case(id="")]}, "'id'"),
        ({"name": "x", "cases": [_case(weight=0)]}, "greater than zero"),
        ({"name": "x", "cases": [_case(weight="heavy")]}, "'weight'"),
        ({"name": "x", "cases": [_case(checks=[])]}, "non-empty list"),
        ({"name": "x", "cases": [_case(checks=[{"type": "nope"}])]}, "unknown check"),
        (
            {"name": "x", "cases": [_case(checks=[{"type": "json_keys"}])]},
            "keys",
        ),
        (
            {"name": "x", "cases": [_case(checks=[{"type": "max_words"}])]},
            "max",
        ),
        (
            {"name": "x", "cases": [_case(checks=[{"type": "bullet_count"}])]},
            "count",
        ),
        (
            {"name": "x", "cases": [_case(checks=[{"type": "forbidden_words"}])]},
            "words",
        ),
        (
            {"name": "x", "cases": [_case(checks=[{"type": "is_spanish", "max": 3}])]},
            "unexpected parameter",
        ),
    ],
)
def test_parse_rejects_invalid_payloads(payload, fragment):
    with pytest.raises(SuiteError) as excinfo:
        parse_suite(payload)
    assert fragment in str(excinfo.value)


def test_duplicate_ids_are_rejected():
    payload = {"name": "x", "cases": [_case(), _case()]}
    with pytest.raises(SuiteError, match="duplicate"):
        parse_suite(payload)


def test_context_and_notes_are_optional():
    suite = parse_suite({"name": "mini", "cases": [_case(context=None, notes=None)]})
    assert suite.cases[0].context == ""
    assert suite.cases[0].notes == ""


def test_parse_accepts_string_description():
    suite = parse_suite({"name": "mini", "description": "demo", "cases": [_case()]})
    assert suite.description == "demo"
    assert suite.path is None


def test_parse_defaults_name_to_fallback():
    suite = parse_suite({"cases": [_case()]}, fallback_name="fallback")
    assert suite.name == "fallback"


def test_load_suite_preserves_utf8(tmp_path):
    path = tmp_path / "mini.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: mini
            cases:
              - id: uno
                category: neutral_spanish
                system: Responde en español neutro.
                user: ¿Cómo estás?
                checks:
                  - type: is_spanish
                weight: 1.5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.cases[0].user == "¿Cómo estás?"
    assert suite.cases[0].weight == 1.5
