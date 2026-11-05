"""OpenAI-compatible provider and a deterministic mock provider."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .suite import Case


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a usable completion."""


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider: str = "unknown"
    simulated: bool = False


class Provider(Protocol):
    name: str

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        case: Case | None = None,
    ) -> Completion: ...


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class OpenAICompatibleProvider:
    """Minimal client for the OpenAI Chat Completions wire format."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if not api_key:
            raise ProviderError("an API key is required for a live run")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        case: Case | None = None,
    ) -> Completion:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "habla-bench/0.1",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover - defensive
                pass
            raise ProviderError(
                f"HTTP {exc.code} from {self.base_url}: {detail}".strip()
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"could not reach {self.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                f"request to {self.base_url} timed out after {self.timeout}s"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("provider returned a non-JSON body") from exc

        try:
            choice = data["choices"][0]
            text = _extract_text(choice.get("message", {}).get("content"))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"provider response is missing choices[0].message.content: {data!r}"
            ) from exc

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = None
        if not isinstance(completion_tokens, int):
            completion_tokens = None

        return Completion(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

_MOCK_PROFILES: dict[str, tuple[float, float, float]] = {
    # model: (compliance, min latency ms, max latency ms)
    "mock-a": (0.90, 700.0, 1500.0),
    "mock-b": (0.55, 350.0, 1000.0),
}
_DEFAULT_PROFILE = (0.72, 500.0, 1200.0)

_JSON_VALUES = {
    "estado": "listo",
    "mensaje": "todo en orden",
    "ciudad": "una ciudad",
    "clima": "templado",
    "color": "verde",
    "hex": "#3a7d44",
    "nombre": "ejemplo",
    "respuesta": "sí",
    "tema": "resumen",
    "confianza": "alta",
    "prioridad": "media",
    "fecha": "pendiente",
}

_BULLETS = [
    "Revisa los datos antes de empezar.",
    "Confirma la información con tu equipo.",
    "Cierra con una conclusión breve.",
    "Anota los acuerdos importantes.",
    "Escucha las preguntas con atención.",
    "Define el siguiente paso.",
]

_NEUTRAL_TEXT = (
    "Claro. Te explico de forma breve: revisa los pasos, confirma los datos "
    "y avanza con calma."
)
_LANGUAGE_TEXT = (
    "Entiendo tu mensaje. Respondo en español como pediste: conviene comparar "
    "opciones y confirmar los datos antes de decidir."
)
_CUSTOMER_TEXT = (
    "Hola, gracias por escribir. Revisa la información disponible y con gusto "
    "te confirmamos los detalles."
)
_INSTRUCTION_TEXT = "Listo. La respuesta cumple con la instrucción solicitada."
_COMPLAINT_TEXT = (
    "Lamento la molestia. Vamos a revisar lo ocurrido y te contactamos para "
    "resolverlo."
)

_LONG_TEXT = " ".join(
    [
        "Claro, revisa los datos con calma y confirma la información antes de",
        "avanzar; luego anota los acuerdos importantes y comparte el resumen",
        "con tu equipo para que todos tengan el mismo contexto y puedan",
        "preparar el siguiente paso sin dudas ni retrasos innecesarios en el",
        "proceso completo.",
    ]
    * 5
)

_AMOUNT_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _checks_by_type(case: Case | None) -> dict[str, Mapping[str, Any]]:
    if case is None:
        return {}
    return {str(spec["type"]): spec for spec in case.checks}


def _wants_no(case: Case) -> bool:
    lowered = (case.system + " " + case.user).lower()
    return "solo \"no\"" in lowered or "solamente no" in lowered or "solo no" in lowered


def _json_response(keys: list[str]) -> str:
    payload = {
        key: _JSON_VALUES.get(key, "ejemplo")
        for key in keys
    }
    return json.dumps(payload, ensure_ascii=False)


def _price_question(case: Case) -> bool:
    lowered = case.user.lower()
    return any(
        word in lowered
        for word in ("cuánto", "cuanto", "precio", "cuesta", "costo", "valor")
    )


def _amount_from_context(case: Case) -> str | None:
    amounts = _AMOUNT_RE.findall(case.context)
    if not amounts:
        return None
    return amounts[-1]


def _good_response(case: Case) -> str:
    checks = _checks_by_type(case)
    if "json_keys" in checks:
        keys = [
            str(key)
            for key in checks["json_keys"].get("keys", [])  # type: ignore[union-attr]
        ]
        return _json_response(keys)

    if "bullet_count" in checks:
        count = int(checks["bullet_count"].get("count", 3))  # type: ignore[union-attr]
        selected = (_BULLETS * ((count // len(_BULLETS)) + 1))[:count]
        return "\n".join(f"- {item}" for item in selected)

    if "max_words" in checks:
        maximum = int(checks["max_words"].get("max", 30))  # type: ignore[union-attr]
        if maximum <= 2:
            return "No." if _wants_no(case) else "Sí."
        if maximum <= 10:
            return "Recibí tu mensaje. Gracias."
        if maximum <= 14:
            return "Hola, gracias por escribir. Cuéntame en qué puedo ayudarte."

    if "no_invented_numbers" in checks and _price_question(case):
        amount = _amount_from_context(case)
        if amount:
            return (
                "Hola, gracias por escribir. Según la información disponible, "
                f"el valor es {amount} pesos. Si necesitas otro detalle, con "
                "gusto te ayudamos."
            )

    if case.category == "neutral_spanish":
        return _NEUTRAL_TEXT
    if case.category == "language_lock":
        return _LANGUAGE_TEXT
    if case.category == "customer_service":
        if "reclamo" in case.id or "complaint" in case.id:
            return _COMPLAINT_TEXT
        return _CUSTOMER_TEXT
    if case.category == "instruction_following":
        return _INSTRUCTION_TEXT
    return _LANGUAGE_TEXT


def _bad_response(case: Case, spec: Mapping[str, Any], rng: random.Random) -> str:
    kind = spec["type"]
    if kind == "json_keys":
        return '{"respuesta": "texto"}'
    if kind == "max_words":
        return _LONG_TEXT
    if kind == "bullet_count":
        count = int(spec.get("count", 3))
        return "\n".join(f"- Punto {index}" for index in range(1, count + 2))
    if kind == "forbidden_words":
        words = [str(word) for word in spec.get("words", [])]
        word = rng.choice(words) if words else "prohibido"
        return f"{word}. Revisa los datos y confirma la información."
    if kind == "no_voseo":
        return "Mirá, vos podés revisar los datos y decime si te sirve."
    if kind == "no_spain_forms":
        return "Vale, vosotros podéis revisar los datos y habéis avanzado bien."
    if kind == "is_spanish":
        return "Sure, I can help you with that. Please check the details and let me know."
    return (
        "El valor es 999.999 pesos. Revisa los demás detalles en la "
        "información disponible."
    )


def mock_response(case: Case | None, model: str) -> str:
    """Return a deterministic mock response for a case and model id."""

    if case is None:
        return _LANGUAGE_TEXT

    compliance, _, _ = _MOCK_PROFILES.get(model, _DEFAULT_PROFILE)
    rng = random.Random(_seed(model, case.id))
    if rng.random() > compliance:
        specs = list(case.checks)
        spec = rng.choice(specs)
        return _bad_response(case, spec, rng)
    return _good_response(case)


def simulated_latency_ms(model: str, case_id: str) -> float:
    _, low, high = _MOCK_PROFILES.get(model, _DEFAULT_PROFILE)
    rng = random.Random(_seed("latency", model, case_id))
    return round(rng.uniform(low, high), 1)


def simulated_usage(
    system: str, user: str, response: str
) -> tuple[int, int]:
    prompt_tokens = max(1, round(len(system + user) / 4) + 16)
    completion_tokens = max(1, round(len(response) / 4) + 1)
    return prompt_tokens, completion_tokens


class MockProvider:
    """Deterministic, offline provider used by ``--dry-run``."""

    name = "mock"

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        case: Case | None = None,
    ) -> Completion:
        text = mock_response(case, model)
        prompt_tokens, completion_tokens = simulated_usage(system, user, text)
        latency = simulated_latency_ms(model, case.id if case else "generic")
        return Completion(
            text=text,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=self.name,
            simulated=True,
        )
