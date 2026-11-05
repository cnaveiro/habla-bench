"""Unit tests for the deterministic response checks."""

from __future__ import annotations

import pytest

from habla_bench.checks import (
    CheckError,
    check_bullet_count,
    check_forbidden_words,
    check_is_spanish,
    check_json_keys,
    check_max_words,
    check_no_invented_numbers,
    check_no_spain_forms,
    check_no_voseo,
    count_bullets,
    find_invented_numbers,
    find_spain_forms,
    find_voseo,
    looks_spanish,
    run_check,
    run_checks,
    word_count,
)


class TestJsonKeys:
    def test_valid_object(self):
        result = check_json_keys('{"estado": "listo", "mensaje": "ok"}', ["estado", "mensaje"])
        assert result.passed
        assert result.type == "json_keys"

    def test_missing_key_fails(self):
        result = check_json_keys('{"estado": "listo"}', ["estado", "mensaje"])
        assert not result.passed
        assert "mensaje" in result.detail

    def test_exact_rejects_extra_keys(self):
        result = check_json_keys(
            '{"estado": "listo", "mensaje": "ok", "extra": 1}',
            ["estado", "mensaje"],
            exact=True,
        )
        assert not result.passed
        assert "extra" in result.detail

    def test_non_exact_allows_extra_keys(self):
        result = check_json_keys(
            '{"estado": "listo", "mensaje": "ok", "extra": 1}',
            ["estado", "mensaje"],
        )
        assert result.passed

    def test_code_fence_is_tolerated(self):
        result = check_json_keys(
            '```json\n{"ciudad": "Nueva", "clima": "templado"}\n```',
            ["ciudad", "clima"],
        )
        assert result.passed

    def test_invalid_json_fails(self):
        result = check_json_keys('{"estado": ', ["estado"])
        assert not result.passed
        assert "invalid JSON" in result.detail

    def test_array_is_not_an_object(self):
        result = check_json_keys('["estado"]', ["estado"])
        assert not result.passed
        assert "not an object" in result.detail

    def test_no_object_fails(self):
        result = check_json_keys("no hay json aquí", ["estado"])
        assert not result.passed

    def test_empty_keys_raises(self):
        with pytest.raises(CheckError):
            check_json_keys("{}", [])


class TestMaxWords:
    def test_boundary_passes(self):
        assert check_max_words("uno dos tres", 3).passed

    def test_over_limit_fails(self):
        result = check_max_words("uno dos tres cuatro", 3)
        assert not result.passed
        assert "4" in result.detail

    def test_numbers_count_as_one_token(self):
        assert word_count("25.000 pesos") == 2

    def test_empty_response_passes_zero_limit(self):
        assert check_max_words("", 0).passed

    def test_negative_limit_raises(self):
        with pytest.raises(CheckError):
            check_max_words("hola", -1)


class TestBulletCount:
    def test_dash_bullets(self):
        text = "- uno\n- dos\n- tres"
        assert count_bullets(text) == 3
        assert check_bullet_count(text, 3).passed

    def test_star_and_numbered_bullets(self):
        text = "* uno\n1. dos\n2) tres"
        assert count_bullets(text) == 3

    def test_wrong_count_fails(self):
        result = check_bullet_count("- uno\n- dos", 3)
        assert not result.passed
        assert "2" in result.detail

    def test_paragraph_text_has_no_bullets(self):
        assert count_bullets("Uno. Dos. Tres.") == 0

    def test_negative_count_raises(self):
        with pytest.raises(CheckError):
            check_bullet_count("- uno", -1)


class TestForbiddenWords:
    def test_absent_words_pass(self):
        assert check_forbidden_words("Una respuesta breve.", ["gratis"]).passed

    def test_case_insensitive(self):
        result = check_forbidden_words("Es GRATIS.", ["gratis"])
        assert not result.passed

    def test_word_boundary(self):
        assert check_forbidden_words("La gratitud importa.", ["gratis"]).passed

    def test_phrase(self):
        result = check_forbidden_words("Sin duda, funciona.", ["sin duda"])
        assert not result.passed
        assert "sin duda" in result.detail

    def test_empty_list_raises(self):
        with pytest.raises(CheckError):
            check_forbidden_words("hola", [])


class TestVoseo:
    @pytest.mark.parametrize(
        "text",
        [
            "Mirá, vos podés hacerlo.",
            "Decime si querés avanzar.",
            "Tenés que revisar los datos.",
            "Andá y fijate bien.",
        ],
    )
    def test_detects_markers(self, text):
        assert find_voseo(text)
        assert not check_no_voseo(text).passed

    @pytest.mark.parametrize(
        "text",
        [
            "Revisa los datos y confirma la información.",
            "Puedes avanzar con calma.",
            "Te explico de forma breve.",
            "Ustedes pueden revisar el formulario.",
        ],
    )
    def test_neutral_text_passes(self, text):
        assert check_no_voseo(text).passed

    def test_terms_are_reported(self):
        result = check_no_voseo("Vos podés y mirá.")
        assert not result.passed
        assert "vos" in result.detail


class TestSpainForms:
    @pytest.mark.parametrize(
        "text",
        [
            "Vosotros podéis revisar el formulario.",
            "Habéis avanzado bien.",
            "Vale, lo revisamos.",
            "Os enviamos el documento.",
        ],
    )
    def test_detects_peninsular_forms(self, text):
        assert find_spain_forms(text)
        assert not check_no_spain_forms(text).passed

    @pytest.mark.parametrize(
        "text",
        [
            "Ustedes pueden revisar el formulario.",
            "Revisamos el documento y te avisamos.",
            "El paquete vale 25.000 pesos.",
        ],
    )
    def test_neutral_text_passes(self, text):
        assert check_no_spain_forms(text).passed


class TestIsSpanish:
    @pytest.mark.parametrize(
        "text",
        [
            "Hola, gracias por escribir. Cuéntame en qué puedo ayudarte.",
            "El precio es 25.000 pesos.",
            "Sí.",
            "No.",
            "Revisa los datos.",
        ],
    )
    def test_spanish_passes(self, text):
        assert looks_spanish(text)
        assert check_is_spanish(text).passed

    @pytest.mark.parametrize(
        "text",
        [
            "Sure, I can help you with that. Please check the details.",
            "This is an English answer to your question.",
            "",
            "JSON",
            "Hello!",
        ],
    )
    def test_english_or_empty_fails(self, text):
        assert not looks_spanish(text)
        assert not check_is_spanish(text).passed


class TestNoInventedNumbers:
    def test_context_number_passes(self):
        result = check_no_invented_numbers(
            "La limpieza cuesta 40.000 pesos.", "Limpieza: 40.000 pesos."
        )
        assert result.passed

    def test_unformatted_variant_passes(self):
        result = check_no_invented_numbers(
            "Cuesta 40000 pesos.", "Limpieza: 40.000 pesos."
        )
        assert result.passed

    def test_other_separator_passes(self):
        result = check_no_invented_numbers("Mide 1,5 metros.", "Mide 1.5 metros.")
        assert result.passed

    def test_invented_number_fails(self):
        result = check_no_invented_numbers(
            "Cuesta 99.999 pesos.", "Limpieza: 40.000 pesos."
        )
        assert not result.passed
        assert "99.999" in result.detail

    def test_different_number_fails(self):
        result = check_no_invented_numbers(
            "Cuesta 40 pesos.", "Limpieza: 40.000 pesos."
        )
        assert not result.passed

    def test_no_numbers_passes(self):
        assert check_no_invented_numbers("Te confirmamos pronto.", "").passed

    def test_number_missing_from_context_fails(self):
        assert find_invented_numbers("Son 2026 unidades.", "") == ["2026"]


class TestDispatcher:
    def test_runs_each_type(self):
        specs = [
            {"type": "max_words", "max": 10},
            {"type": "no_voseo"},
            {"type": "is_spanish"},
        ]
        results = run_checks(specs, "Hola, te ayudo con eso.")
        assert len(results) == 3
        assert all(result.passed for result in results)

    def test_unknown_type_raises(self):
        with pytest.raises(CheckError):
            run_check({"type": "nope"}, "texto")

    def test_unexpected_parameter_raises(self):
        with pytest.raises(CheckError):
            run_check({"type": "max_words", "max": 5, "extra": 1}, "texto")

    def test_missing_parameter_raises(self):
        with pytest.raises(CheckError):
            run_check({"type": "max_words"}, "texto")

    def test_non_mapping_raises(self):
        with pytest.raises(CheckError):
            run_check("max_words", "texto")  # type: ignore[arg-type]
