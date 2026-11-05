"""Deterministic response checks.

Every check is a pure function of the response text and, where relevant, the
conversation context. There are no model calls, network requests, random
numbers, or clocks in this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

CHECK_TYPES = (
    "json_keys",
    "max_words",
    "bullet_count",
    "forbidden_words",
    "no_voseo",
    "no_spain_forms",
    "is_spanish",
    "no_invented_numbers",
)

CHECK_PARAMETERS = {
    "json_keys": {"type", "keys", "exact"},
    "max_words": {"type", "max"},
    "bullet_count": {"type", "count"},
    "forbidden_words": {"type", "words"},
    "no_voseo": {"type"},
    "no_spain_forms": {"type"},
    "is_spanish": {"type"},
    "no_invented_numbers": {"type"},
}

_WORD_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*",
    re.UNICODE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Markers of voseo (Río de la Plata and related varieties). Only accented
# forms are listed for second-person singular verbs that would otherwise be
# indistinguishable from the tú form.
VOSEO_TERMS = frozenset(
    {
        "vos",
        "sos",
        "tenés",
        "podés",
        "querés",
        "sabés",
        "decime",
        "decíme",
        "mirá",
        "andá",
        "vení",
        "hacé",
        "poné",
        "dejá",
        "tomá",
        "fijate",
        "fijáte",
        "contame",
        "contáme",
        "escuchá",
        "hablá",
        "comprá",
        "probá",
        "elegí",
        "escribí",
        "abrí",
        "cerrá",
        "sentate",
        "acercate",
        "llevá",
        "mandá",
        "pasá",
        "esperá",
        "seguí",
        "empezá",
        "terminá",
        "usá",
        "apretá",
        "tocá",
        "buscá",
        "pedí",
        "decí",
        "salí",
        "dale",
    }
)

# Markers of peninsular Spanish: second-person plural pronouns and verb forms,
# plus "vale" used as a filler word.
SPAIN_TERMS = frozenset(
    {
        "vosotros",
        "vosotras",
        "os",
        "habéis",
        "habeis",
        "sois",
        "estáis",
        "estais",
        "hacéis",
        "haceis",
        "podéis",
        "podeis",
        "queréis",
        "quereis",
        "sabéis",
        "sabeis",
        "tenéis",
        "teneis",
        "vais",
    }
)

# "vale" is only a peninsular filler when it stands on its own; the same word
# is a common verb ("vale 25.000 pesos"). It is handled separately from the
# token set above.
_VALE_FILLER_RE = re.compile(
    r"(?<![\wáéíóúüñ])vale\s*(?:[,.;:!?]|$)", re.IGNORECASE | re.UNICODE
)

# Frequent Spanish function words plus a small set of everyday content words.
# The list intentionally mixes both so short customer-service replies can be
# classified without an external language detector.
SPANISH_COMMON = frozenset(
    """
    de la que el en y a los se del las un por con no una su para es al lo
    como más pero sus le ya o este sí porque esta entre cuando muy sin sobre
    también me hasta hay donde quien desde todo nos durante todos uno les ni
    contra otros ese eso ante ellos esto mí antes algunos qué unos yo otro
    otras otra él tanto esa estos mucho quienes nada muchos cual poco ella
    estar estas algunas algo nosotros mi mis tú te ti tu tus ellas nosotras
    mío mía tuyo tuya suyo suya nuestro nuestra nuestros nuestras esos esas
    estoy estás está estamos están sería serían fue fueron ser soy eres somos
    son era eran he has ha han había habían hay hubo tengo tienes tiene
    tenemos tienen tenía podía pueden puede podemos puedo puedes quiero
    quieres necesita necesitas hacer hace hacen hice dicho dijo dicen ver voy
    va van vamos ir hola gracias buenos buenas días tardes noches favor claro
    perfecto seguro cómo dónde cuándo cuánto qué quién cuál revisa revisar
    confirma confirmar información datos detalle detalles ayuda ayudarte
    ayudar podemos necesitas gracias saludos cordialmente mensaje correo
    respuesta precio valor cuesta costo horario atención consulta curso
    semanas incluye garantía servicio equipo cliente clientes reunión
    """.split()
)

ENGLISH_COMMON = frozenset(
    """
    the and of to in is are you your for with that this on as be at or from by
    it not have has can will would should please what how hello thanks about
    we i my me our us they them their he she his her was were been do does did
    there here all any but if then than so just also more most some such only
    other into over after before under again very when where why who whom
    which while
    """.split()
)


class CheckError(ValueError):
    """Raised when a check specification is malformed."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one deterministic check."""

    type: str
    passed: bool
    detail: str


def tokens(text: str) -> list[str]:
    """Return lowercase word/number tokens without punctuation."""

    return [token.lower() for token in _WORD_TOKEN.findall(text or "")]


def word_count(text: str) -> int:
    """Count word-like tokens, including numbers with separators."""

    return len(tokens(text))


def _evidence(found: Iterable[str], limit: int = 6) -> str:
    items = sorted(dict.fromkeys(found))
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f", +{len(items) - limit} more"
    return shown


def find_voseo(text: str) -> list[str]:
    """Return voseo markers present in *text*."""

    return sorted({token for token in tokens(text) if token in VOSEO_TERMS})


def find_spain_forms(text: str) -> list[str]:
    """Return peninsular Spanish markers present in *text*."""

    found = {token for token in tokens(text) if token in SPAIN_TERMS}
    if _VALE_FILLER_RE.search(text or ""):
        found.add("vale")
    return sorted(found)


def check_json_keys(
    text: str, keys: Sequence[str], exact: bool = False
) -> CheckResult:
    """Require a valid JSON object containing *keys*.

    Markdown code fences around the object are tolerated. When ``exact`` is
    true the object must contain exactly the requested keys.
    """

    if not keys:
        raise CheckError("json_keys requires a non-empty 'keys' list")

    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        if start != -1:
            try:
                json.loads(raw)
            except json.JSONDecodeError as exc:
                return CheckResult("json_keys", False, f"invalid JSON: {exc.msg}")
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return CheckResult("json_keys", False, "no JSON object found")
            if not isinstance(value, dict):
                return CheckResult(
                    "json_keys", False, "JSON value is not an object"
                )
        return CheckResult("json_keys", False, "no JSON object found")
    candidate = raw[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return CheckResult("json_keys", False, f"invalid JSON: {exc.msg}")
    if not isinstance(value, dict):
        return CheckResult("json_keys", False, "JSON value is not an object")

    expected = [str(key) for key in keys]
    missing = [key for key in expected if key not in value]
    extra = [key for key in value if key not in expected]
    if missing:
        return CheckResult(
            "json_keys", False, f"missing keys: {', '.join(missing)}"
        )
    if exact and extra:
        return CheckResult(
            "json_keys", False, f"unexpected keys: {', '.join(extra)}"
        )
    return CheckResult(
        "json_keys",
        True,
        f"valid JSON object with {len(expected)} required key(s)",
    )


def check_max_words(text: str, maximum: int) -> CheckResult:
    """Require at most *maximum* word-like tokens."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise CheckError("max_words requires a non-negative integer 'max'")
    count = word_count(text)
    if count <= maximum:
        return CheckResult("max_words", True, f"{count} words (limit {maximum})")
    return CheckResult("max_words", False, f"{count} words (limit {maximum})")


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def count_bullets(text: str) -> int:
    """Count lines that start with a bullet marker or a numbered marker."""

    return sum(1 for line in (text or "").splitlines() if _BULLET_RE.match(line))


def check_bullet_count(text: str, count: int) -> CheckResult:
    """Require exactly *count* bullet lines."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CheckError("bullet_count requires a non-negative integer 'count'")
    found = count_bullets(text)
    if found == count:
        return CheckResult(
            "bullet_count", True, f"{found} bullet line(s) as required"
        )
    return CheckResult(
        "bullet_count", False, f"{found} bullet line(s), expected {count}"
    )


def _word_pattern(word: str) -> re.Pattern[str]:
    escaped = re.escape(word.strip())
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE | re.UNICODE)


def check_forbidden_words(
    text: str, words: Sequence[str]
) -> CheckResult:
    """Fail when any forbidden word or phrase appears in *text*."""

    if not isinstance(words, (list, tuple)) or not words:
        raise CheckError("forbidden_words requires a non-empty 'words' list")
    found = [
        word
        for word in words
        if word and _word_pattern(str(word)).search(text or "")
    ]
    if found:
        return CheckResult(
            "forbidden_words", False, f"found forbidden: {_evidence(found)}"
        )
    return CheckResult(
        "forbidden_words", True, f"none of {len(words)} forbidden word(s) present"
    )


def check_no_voseo(text: str) -> CheckResult:
    """Fail when voseo markers are present."""

    found = find_voseo(text)
    if found:
        return CheckResult(
            "no_voseo", False, f"voseo markers: {_evidence(found)}"
        )
    return CheckResult("no_voseo", True, "no voseo markers found")


def check_no_spain_forms(text: str) -> CheckResult:
    """Fail when peninsular Spanish markers are present."""

    found = find_spain_forms(text)
    if found:
        return CheckResult(
            "no_spain_forms",
            False,
            f"peninsular markers: {_evidence(found)}",
        )
    return CheckResult(
        "no_spain_forms", True, "no peninsular markers found"
    )


def looks_spanish(text: str) -> bool:
    """Heuristic language check based on frequent Spanish words.

    The heuristic is deliberately simple and documented as approximate. It
    handles one-word answers such as "sí" and "no", and otherwise compares
    Spanish and English function-word hits. Accented characters are treated
    as strong positive evidence.
    """

    clean = (text or "").strip()
    if not clean:
        return False
    lowered = clean.lower().strip(" .,!¡?¿;:()[]{}'\"")
    if lowered in {"sí", "si", "no", "hola", "gracias"}:
        return True

    toks = tokens(clean)
    if not toks:
        return False

    spanish_hits = sum(1 for token in toks if token in SPANISH_COMMON)
    english_hits = sum(1 for token in toks if token in ENGLISH_COMMON)
    has_spanish_marks = bool(re.search(r"[áéíóúüñ¿¡]", lowered))

    if spanish_hits == 0:
        return False
    ratio = spanish_hits / len(toks)
    if has_spanish_marks and spanish_hits >= 1:
        return True
    if ratio >= 0.15 and spanish_hits >= 2 and spanish_hits >= english_hits:
        return True
    if len(toks) <= 3 and english_hits == 0:
        return True
    return False


def check_is_spanish(text: str) -> CheckResult:
    """Pass when the heuristic considers the response Spanish."""

    if looks_spanish(text):
        hits = sum(1 for token in tokens(text) if token in SPANISH_COMMON)
        return CheckResult(
            "is_spanish", True, f"Spanish heuristic passed ({hits} familiar words)"
        )
    return CheckResult(
        "is_spanish", False, "response does not look like Spanish"
    )


def _number_variants(raw: str) -> set[str]:
    raw = raw.strip().strip(".,")
    if not raw:
        return set()
    variants = {raw}
    digits = re.sub(r"\D", "", raw)
    if digits:
        variants.add(digits)
        variants.add(digits.lstrip("0") or "0")
    no_thousands = raw.replace(".", "").replace(",", "")
    variants.add(no_thousands)
    variants.add(no_thousands.lstrip("0") or "0")
    variants.add(raw.replace(",", "."))
    variants.add(raw.replace(".", ","))
    return {variant for variant in variants if variant}


def find_invented_numbers(response: str, context: str) -> list[str]:
    """Return numeric tokens in *response* that do not appear in *context*.

    Numbers are compared by several normalizations so ``25.000``, ``25000``
    and ``25,000`` are treated as the same value. The check only covers
    digits; spelled-out numbers are outside its scope.
    """

    allowed: set[str] = set()
    for match in _NUMBER_RE.finditer(context or ""):
        allowed |= _number_variants(match.group())

    invented: list[str] = []
    for match in _NUMBER_RE.finditer(response or ""):
        raw = match.group()
        if _number_variants(raw) & allowed:
            continue
        invented.append(raw)
    return invented


def check_no_invented_numbers(
    text: str, context: str = ""
) -> CheckResult:
    """Fail when the response contains digits absent from the context."""

    invented = find_invented_numbers(text, context)
    if invented:
        return CheckResult(
            "no_invented_numbers",
            False,
            f"numbers not in context: {_evidence(invented)}",
        )
    return CheckResult(
        "no_invented_numbers", True, "all numbers are grounded in the context"
    )


def _require_mapping(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise CheckError("each check must be a mapping with a 'type' key")


def run_check(
    spec: Mapping[str, Any], response: str, context: str = ""
) -> CheckResult:
    """Dispatch one check specification against a response."""

    _require_mapping(spec)
    kind = spec.get("type")
    if not isinstance(kind, str) or kind not in CHECK_TYPES:
        raise CheckError(
            f"unknown check type {kind!r}; expected one of {', '.join(CHECK_TYPES)}"
        )

    allowed_params = CHECK_PARAMETERS[kind]
    unexpected = set(spec) - allowed_params
    if unexpected:
        raise CheckError(
            f"unexpected parameter(s) for {kind}: {', '.join(sorted(unexpected))}"
        )

    if kind == "json_keys":
        keys = spec.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, (list, tuple)):
            raise CheckError("json_keys requires a 'keys' list")
        return check_json_keys(
            response, [str(key) for key in keys], bool(spec.get("exact", False))
        )
    if kind == "max_words":
        return check_max_words(response, spec.get("max"))  # type: ignore[arg-type]
    if kind == "bullet_count":
        return check_bullet_count(
            response, spec.get("count")  # type: ignore[arg-type]
        )
    if kind == "forbidden_words":
        return check_forbidden_words(
            response, spec.get("words")  # type: ignore[arg-type]
        )
    if kind == "no_voseo":
        return check_no_voseo(response)
    if kind == "no_spain_forms":
        return check_no_spain_forms(response)
    if kind == "is_spanish":
        return check_is_spanish(response)
    return check_no_invented_numbers(response, context)


def run_checks(
    specs: Sequence[Mapping[str, Any]], response: str, context: str = ""
) -> list[CheckResult]:
    """Run every check in *specs* against *response*."""

    return [run_check(spec, response, context) for spec in specs]
