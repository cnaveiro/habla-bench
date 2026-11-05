"""Environment, settings, and price-table handling.

API keys are read from the environment at call time and are never written to
disk. The CLI never loads a ``.env`` file implicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class ConfigError(ValueError):
    """Raised for missing or invalid configuration."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str | None
    timeout: float = 60.0
    dry_run: bool = False


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float
    note: str = "example"

    @property
    def is_example(self) -> bool:
        return self.note.strip().lower() == "example"


@dataclass(frozen=True)
class PriceTable:
    models: Mapping[str, Price]
    default: Price
    currency: str = "USD"
    path: Path | None = None

    def for_model(self, model: str) -> Price:
        if model in self.models:
            return self.models[model]
        for prefix, price in self.models.items():
            if model.startswith(prefix):
                return price
        return self.default


def load_settings(
    *,
    dry_run: bool,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> Settings:
    """Read settings from arguments and environment variables."""

    resolved_base = (
        base_url
        or os.environ.get("HABLA_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    if not resolved_base.startswith(("http://", "https://")):
        raise ConfigError("HABLA_BASE_URL must start with http:// or https://")

    resolved_key = (api_key if api_key is not None else os.environ.get("HABLA_API_KEY"))
    if resolved_key is not None:
        resolved_key = resolved_key.strip() or None

    if not dry_run and not resolved_key:
        raise ConfigError(
            "HABLA_API_KEY is not set. Set it in the environment or use --dry-run."
        )

    if timeout <= 0:
        raise ConfigError("timeout must be greater than zero")

    return Settings(
        base_url=resolved_base.rstrip("/"),
        api_key=None if dry_run else resolved_key,
        timeout=timeout,
        dry_run=dry_run,
    )


def _read_price(
    raw: Any, field: str, default_note: str = "example"
) -> Price:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    try:
        input_price = float(raw["input"])
        output_price = float(raw["output"])
    except KeyError as exc:
        raise ConfigError(f"{field} is missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} prices must be numbers") from exc
    if input_price < 0 or output_price < 0:
        raise ConfigError(f"{field} prices cannot be negative")
    note = str(raw.get("note", default_note))
    return Price(input_price, output_price, note)


def load_prices(path: str | Path | None = None) -> PriceTable:
    """Load the optional price table.

    When the file is absent, zero-cost example values are used so reports do
    not invent a cost. The bundled ``prices.yaml`` documents that its values
    are examples, not live quotes.
    """

    zero = Price(0.0, 0.0, note="example")
    if path is None:
        return PriceTable(models={}, default=zero)

    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"price file not found: {file_path}")
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file_path}: {exc}") from exc
    if payload is None:
        return PriceTable(models={}, default=zero, path=file_path.resolve())
    if not isinstance(payload, Mapping):
        raise ConfigError("prices file must contain a mapping")

    currency = str(payload.get("currency", "USD"))
    models_raw = payload.get("models", {})
    if models_raw is None:
        models_raw = {}
    if not isinstance(models_raw, Mapping):
        raise ConfigError("'models' must be a mapping of model id to prices")
    models = {
        str(model): _read_price(raw, f"models.{model}")
        for model, raw in models_raw.items()
    }
    default = _read_price(payload.get("default", {"input": 0, "output": 0}), "default")
    return PriceTable(
        models=models,
        default=default,
        currency=currency,
        path=file_path.resolve(),
    )


def estimate_cost(
    price: Price,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Estimate cost in the price table's currency.

    Returns ``None`` when the provider did not return token usage.
    """

    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        (prompt_tokens / 1_000_000) * price.input_per_million
        + (completion_tokens / 1_000_000) * price.output_per_million
    )
