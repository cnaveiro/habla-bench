"""Tests for the OpenAI-compatible client and the mock provider."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from habla_bench import providers
from habla_bench.checks import run_checks
from habla_bench.providers import (
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    _good_response,
)
from habla_bench.suite import Case, load_suite

ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _case(**overrides) -> Case:
    payload = {
        "id": "case-1",
        "category": "instruction_following",
        "system": "Responde en español.",
        "user": "Di hola.",
        "checks": ({"type": "is_spanish"},),
        "weight": 1.0,
    }
    payload.update(overrides)
    return Case(**payload)


class TestOpenAICompatibleProvider:
    def test_request_shape_and_usage(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(
                json.dumps(
                    {
                        "choices": [{"message": {"content": "Hola, te ayudo."}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider(
            "https://example.invalid/v1",
            "test-key",
            timeout=12,
            temperature=0.2,
            max_tokens=99,
        )
        completion = provider.complete("test/model", "sistema", "usuario")

        assert completion.text == "Hola, te ayudo."
        assert completion.prompt_tokens == 11
        assert completion.completion_tokens == 7
        assert completion.provider == "openai-compatible"
        assert not completion.simulated
        assert completion.latency_ms >= 0

        request = captured["request"]
        assert request.full_url == "https://example.invalid/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-key"
        assert captured["timeout"] == 12
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "test/model"
        assert payload["messages"] == [
            {"role": "system", "content": "sistema"},
            {"role": "user", "content": "usuario"},
        ]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 99

    def test_content_as_parts(self, monkeypatch):
        body = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "Hola"}, {"text": " mundo"}]}}
            ]
        }
        monkeypatch.setattr(
            providers.urllib.request,
            "urlopen",
            lambda request, timeout: FakeResponse(json.dumps(body).encode()),
        )
        provider = OpenAICompatibleProvider("https://example.invalid", "k")
        assert provider.complete("m", "s", "u").text == "Hola mundo"

    def test_missing_usage_is_none(self, monkeypatch):
        body = {"choices": [{"message": {"content": "Hola"}}]}
        monkeypatch.setattr(
            providers.urllib.request,
            "urlopen",
            lambda request, timeout: FakeResponse(json.dumps(body).encode()),
        )
        completion = OpenAICompatibleProvider(
            "https://example.invalid", "k"
        ).complete("m", "s", "u")
        assert completion.prompt_tokens is None
        assert completion.completion_tokens is None

    def test_http_error_becomes_provider_error(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error": "rate limit"}'),
            )

        monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider("https://example.invalid", "k")
        with pytest.raises(ProviderError, match="429"):
            provider.complete("m", "s", "u")

    def test_network_error_becomes_provider_error(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider("https://example.invalid", "k")
        with pytest.raises(ProviderError, match="could not reach"):
            provider.complete("m", "s", "u")

    def test_non_json_body_raises(self, monkeypatch):
        monkeypatch.setattr(
            providers.urllib.request,
            "urlopen",
            lambda request, timeout: FakeResponse(b"<html>oops</html>"),
        )
        provider = OpenAICompatibleProvider("https://example.invalid", "k")
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.complete("m", "s", "u")

    def test_missing_choices_raises(self, monkeypatch):
        monkeypatch.setattr(
            providers.urllib.request,
            "urlopen",
            lambda request, timeout: FakeResponse(b'{"choices": []}'),
        )
        provider = OpenAICompatibleProvider("https://example.invalid", "k")
        with pytest.raises(ProviderError, match="choices"):
            provider.complete("m", "s", "u")

    def test_empty_key_is_rejected(self):
        with pytest.raises(ProviderError):
            OpenAICompatibleProvider("https://example.invalid", "")


class TestMockProvider:
    def test_is_deterministic(self):
        case = _case()
        first = MockProvider().complete("mock-a", case.system, case.user, case=case)
        second = MockProvider().complete("mock-a", case.system, case.user, case=case)
        assert first == second

    def test_no_network_calls(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("network call attempted")

        monkeypatch.setattr(providers.urllib.request, "urlopen", boom)
        completion = MockProvider().complete(
            "mock-a", "sistema", "usuario", case=_case()
        )
        assert completion.text
        assert completion.simulated
        assert completion.prompt_tokens and completion.prompt_tokens > 0
        assert completion.completion_tokens and completion.completion_tokens > 0

    def test_models_differ_on_core_suite(self):
        suite = load_suite(ROOT / "suites" / "core.yaml")
        provider = MockProvider()
        responses_a = [
            provider.complete("mock-a", case.system, case.user, case=case).text
            for case in suite.cases
        ]
        responses_b = [
            provider.complete("mock-b", case.system, case.user, case=case).text
            for case in suite.cases
        ]
        assert responses_a != responses_b

    def test_good_mock_response_passes_every_check(self):
        suite = load_suite(ROOT / "suites" / "core.yaml")
        for case in suite.cases:
            response = _good_response(case)
            context = "\n".join(
                part for part in (case.context, case.system, case.user) if part
            )
            results = run_checks(case.checks, response, context)
            failures = [
                f"{check.type}: {check.detail}"
                for check in results
                if not check.passed
            ]
            assert not failures, f"{case.id} -> {failures} | {response!r}"

    def test_mock_ids_are_stable(self):
        case = _case()
        assert (
            MockProvider().complete("mock-a", "s", "u", case=case).text
            == MockProvider().complete("mock-a", "s", "u", case=case).text
        )
