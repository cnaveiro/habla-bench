"""Tests for environment settings and price handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from habla_bench.config import (
    ConfigError,
    estimate_cost,
    load_prices,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_does_not_require_key(monkeypatch):
    monkeypatch.delenv("HABLA_API_KEY", raising=False)
    settings = load_settings(dry_run=True)
    assert settings.api_key is None
    assert settings.base_url.startswith("https://")


def test_live_run_requires_key(monkeypatch):
    monkeypatch.delenv("HABLA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="HABLA_API_KEY"):
        load_settings(dry_run=False)


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("HABLA_API_KEY", "secret-value")
    settings = load_settings(dry_run=False)
    assert settings.api_key == "secret-value"


def test_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("HABLA_BASE_URL", "http://localhost:8080/v1/")
    settings = load_settings(dry_run=True)
    assert settings.base_url == "http://localhost:8080/v1"


def test_invalid_base_url(monkeypatch):
    monkeypatch.setenv("HABLA_BASE_URL", "localhost:8080")
    with pytest.raises(ConfigError, match="http"):
        load_settings(dry_run=True)


def test_non_positive_timeout(monkeypatch):
    monkeypatch.delenv("HABLA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="timeout"):
        load_settings(dry_run=True, timeout=0)


def test_bundled_prices_are_marked_example():
    prices = load_prices(ROOT / "prices.yaml")
    assert prices.currency == "USD"
    assert prices.models
    assert all(price.is_example for price in prices.models.values())
    assert prices.default.is_example
    assert prices.for_model("unknown-model") is prices.default


def test_no_prices_file_returns_zero_costs():
    prices = load_prices(None)
    assert prices.for_model("anything").input_per_million == 0


def test_missing_explicit_price_file_raises():
    with pytest.raises(ConfigError):
        load_prices("/definitely/not/a/file.yaml")


def test_invalid_price_payload(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("models:\n  x:\n    input: nope\n    output: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_prices(path)


def test_negative_price_rejected(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("models:\n  x:\n    input: -1\n    output: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_prices(path)


def test_estimate_cost_per_million():
    price = load_prices(ROOT / "prices.yaml").for_model("mock-a")
    cost = estimate_cost(price, 1_000_000, 1_000_000)
    assert cost == pytest.approx(2.00)


def test_estimate_cost_without_usage_is_none():
    price = load_prices(ROOT / "prices.yaml").for_model("mock-a")
    assert estimate_cost(price, None, 100) is None
    assert estimate_cost(price, 100, None) is None


def test_price_prefix_match():
    prices = load_prices(ROOT / "prices.yaml")
    price = prices.for_model("mock-a-2026")
    assert price.input_per_million == 0.40
