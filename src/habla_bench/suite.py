"""Suite loading and validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .checks import CHECK_PARAMETERS, CHECK_TYPES


class SuiteError(ValueError):
    """Raised when a suite file is missing or malformed."""


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    system: str
    user: str
    checks: tuple[Mapping[str, Any], ...]
    weight: float
    context: str = ""
    notes: str = ""

    @property
    def prompt_words(self) -> int:
        return len((self.system + " " + self.user).split())


@dataclass(frozen=True)
class Suite:
    name: str
    cases: tuple[Case, ...]
    path: Path | None = None
    description: str = ""

    @property
    def categories(self) -> list[str]:
        seen: list[str] = []
        for case in self.cases:
            if case.category not in seen:
                seen.append(case.category)
        return seen

    @property
    def sha256(self) -> str:
        if self.path is None or not self.path.exists():
            payload = repr(self.cases).encode("utf-8")
        else:
            payload = self.path.read_bytes()
        return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteError(f"case {index}: '{field}' must be a non-empty string")
    return value.strip()


def _validate_check(spec: Any, case_id: str) -> Mapping[str, Any]:
    if not isinstance(spec, Mapping):
        raise SuiteError(f"case {case_id}: every check must be a mapping")
    kind = spec.get("type")
    if kind not in CHECK_TYPES:
        raise SuiteError(
            f"case {case_id}: unknown check type {kind!r}; "
            f"expected one of {', '.join(CHECK_TYPES)}"
        )
    unexpected = set(spec) - CHECK_PARAMETERS[kind]
    if unexpected:
        raise SuiteError(
            f"case {case_id}: unexpected parameter(s) for {kind}: "
            f"{', '.join(sorted(unexpected))}"
        )
    if kind == "json_keys":
        keys = spec.get("keys")
        if not isinstance(keys, (list, tuple)) or not keys:
            raise SuiteError(
                f"case {case_id}: json_keys needs a non-empty 'keys' list"
            )
    elif kind == "max_words":
        maximum = spec.get("max")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise SuiteError(
                f"case {case_id}: max_words needs a non-negative integer 'max'"
            )
    elif kind == "bullet_count":
        count = spec.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SuiteError(
                f"case {case_id}: bullet_count needs a non-negative integer 'count'"
            )
    elif kind == "forbidden_words":
        words = spec.get("words")
        if not isinstance(words, (list, tuple)) or not words:
            raise SuiteError(
                f"case {case_id}: forbidden_words needs a non-empty 'words' list"
            )
    return spec


def parse_suite(payload: Any, fallback_name: str = "suite") -> Suite:
    """Build a :class:`Suite` from already-parsed YAML data."""

    if not isinstance(payload, Mapping):
        raise SuiteError("suite file must contain a mapping at the top level")

    name = payload.get("name", fallback_name)
    if not isinstance(name, str) or not name.strip():
        raise SuiteError("'name' must be a non-empty string")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError("'cases' must be a non-empty list")

    cases: list[Case] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, Mapping):
            raise SuiteError(f"case {index}: must be a mapping")

        case_id = _require_text(raw.get("id"), "id", index)
        if case_id in seen_ids:
            raise SuiteError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        category = _require_text(raw.get("category"), "category", index)
        system = _require_text(raw.get("system"), "system", index)
        user = _require_text(raw.get("user"), "user", index)

        raw_checks = raw.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            raise SuiteError(f"case {case_id}: 'checks' must be a non-empty list")
        checks = tuple(_validate_check(spec, case_id) for spec in raw_checks)

        weight = raw.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise SuiteError(f"case {case_id}: 'weight' must be a number")
        if weight <= 0:
            raise SuiteError(f"case {case_id}: 'weight' must be greater than zero")

        context = raw.get("context", "")
        if context is None:
            context = ""
        if not isinstance(context, str):
            raise SuiteError(f"case {case_id}: 'context' must be a string")
        notes = raw.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise SuiteError(f"case {case_id}: 'notes' must be a string")

        cases.append(
            Case(
                id=case_id,
                category=category,
                system=system,
                user=user,
                checks=checks,
                weight=float(weight),
                context=context.strip(),
                notes=notes.strip(),
            )
        )

    description = payload.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise SuiteError("'description' must be a string")

    return Suite(
        name=name.strip(),
        cases=tuple(cases),
        description=description.strip(),
    )


def load_suite(path: str | Path) -> Suite:
    """Load a suite from a YAML file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise SuiteError(f"suite file not found: {file_path}")
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(f"invalid YAML in {file_path}: {exc}") from exc
    suite = parse_suite(payload, fallback_name=file_path.stem)
    return Suite(
        name=suite.name,
        cases=suite.cases,
        path=file_path.resolve(),
        description=suite.description,
    )


def _candidate_paths(name: str) -> Iterable[Path]:
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_file():
        yield candidate
        return

    file_name = f"{name}.yaml"
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        yield directory / "suites" / file_name

    package_dir = Path(__file__).resolve().parent
    yield package_dir / "suites" / file_name
    yield package_dir.parent.parent / "suites" / file_name


def resolve_suite(name: str) -> Suite:
    """Resolve a suite name or path and load it.

    ``core`` is looked up as ``suites/core.yaml`` first in the current
    directory, then in each parent directory, and finally next to the
    installed package. A path ending in ``.yaml``/``.yml`` is used directly.
    """

    tried: list[str] = []
    for candidate in _candidate_paths(name):
        tried.append(str(candidate))
        if candidate.is_file():
            return load_suite(candidate)
    raise SuiteError(
        f"could not find suite {name!r}; tried: " + ", ".join(tried)
    )
