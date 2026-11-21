"""Language detection tests (S2). Need no data or models."""
import pytest

from src.triage import detect_language_with_prob, language_text, triage

GERMAN = "Sehr geehrtes Support-Team, unser VPN funktioniert seit heute Morgen nicht mehr."
ENGLISH = "Dear support team, our VPN has been down since this morning."


def test_full_sentences():
    assert detect_language_with_prob(GERMAN)[0] == "de"
    assert detect_language_with_prob(ENGLISH)[0] == "en"


@pytest.mark.parametrize("text", [
    "VPN ausgefallen", "Drucker kaputt", "Passwort vergessen", "Kein Internet",
    "Laptop startet nicht", "Mein Konto ist gesperrt",
])
def test_short_german(text):
    assert detect_language_with_prob(text)[0] == "de"


@pytest.mark.parametrize("text", [
    "VPN down", "Forgot password", "Login not working", "Refund request",
    "My account is locked", "Laptop won't start",
])
def test_short_english(text):
    assert detect_language_with_prob(text)[0] == "en"


@pytest.mark.xfail(reason="known limit: two-word English with no telltale words can read as German")
@pytest.mark.parametrize("text", ["Printer broken", "No internet"])
def test_two_word_english_known_misses(text):
    assert detect_language_with_prob(text)[0] == "en"


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!! ???", None])
def test_no_letters_returns_flagged_guess(text):
    assert detect_language_with_prob(text) == ("en", 0.0)


def test_same_answer_every_run():
    answers = {detect_language_with_prob("Drucker druckt nicht") for _ in range(10)}
    assert len(answers) == 1


def test_ignores_placeholder_and_literal_newlines():
    body = "Hallo,\\n\\nich habe ein Problem mit dem Drucker."  # raw data stores \n as two characters
    text = language_text(None, body)
    assert "[no subject]" not in text and "\\n" not in text
    assert triage(None, body)["language"] == "de"
    assert triage(float("nan"), None)["language"] == "en"  # no crash on missing values
