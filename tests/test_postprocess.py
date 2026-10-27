from __future__ import annotations

from voicetyper.config import Config
from voicetyper.postprocess import clean_transcript, matches_exclusion, preview_text


def test_clean_transcript_normalizes_spaces_and_punctuation() -> None:
    assert clean_transcript("  привет ,  мир ! ", Config(capitalize=True)) == "Привет, мир!"
    assert clean_transcript("hello   world", Config(capitalize=False)) == "hello world"


def test_fillers_are_removed_only_when_enabled() -> None:
    text = "Ну, это самое, привет"
    assert "Ну" in clean_transcript(text, Config(remove_fillers=False, capitalize=False))
    cleaned = clean_transcript(text, Config(remove_fillers=True, capitalize=False))
    assert "Ну" not in cleaned
    assert "это самое" not in cleaned
    assert "привет" in cleaned


def test_exclusion_matches_whole_phrase_only() -> None:
    phrases = ["Спасибо за просмотр.", "Thank you."]
    assert matches_exclusion("Спасибо за просмотр.", phrases) == "Спасибо за просмотр."
    assert matches_exclusion("Спасибо за просмотр! Спасибо за просмотр.", phrases) == "Спасибо за просмотр."
    assert matches_exclusion("Спасибо за просмотр и подписку", phrases) is None
    assert matches_exclusion("Thank you very much", phrases) is None


def test_preview_is_truncated() -> None:
    text = "x" * 200
    assert len(preview_text(text, 20)) <= 20
    assert preview_text("short", 20) == "short"
