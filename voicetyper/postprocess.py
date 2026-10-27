"""Transcript cleanup and exclusion rules."""

from __future__ import annotations

import re
import unicodedata

from .config import Config

_WHITESPACE = re.compile(r"\s+")
_HESITATIONS = re.compile(
    r"(?<!\w)(?:э-э+|ээ+|эм+|мм+|uh+|um+|erm+)(?!\w)",
    re.IGNORECASE,
)
# Discourse fillers are only removed where punctuation makes them unambiguous:
# a leading "Ну," or a comma-delimited "это самое". A bare "ну" can carry
# meaning ("ну да"), so it is left alone.
_LEADING_FILLER = re.compile(r"(^|[.!?…]\s+)(?:ну)\s*,\s*", re.IGNORECASE)
_INLINE_FILLER = re.compile(
    r"(^|,\s*)(?:это самое|как бы|так сказать|собственно говоря)\s*(?:,\s*|$)",
    re.IGNORECASE,
)
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)([a-zа-яё])")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_TRAILING_PUNCT = " \t\r\n.,!?;:…—–-\"'«»"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    return text


def remove_fillers(text: str) -> str:
    cleaned = _HESITATIONS.sub("", text)
    cleaned = _LEADING_FILLER.sub(r"\1", cleaned)
    cleaned = _INLINE_FILLER.sub(lambda match: match.group(1), cleaned)
    cleaned = re.sub(r"\s+([,.;:!?…])", r"\1", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.strip(" ,;")


def capitalize_sentences(text: str) -> str:
    def upper(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()

    return _SENTENCE_START.sub(upper, text)


def clean_transcript(text: str, cfg: Config) -> str:
    result = normalize_text(text)
    if cfg.remove_fillers:
        result = remove_fillers(result)
    if cfg.capitalize:
        result = capitalize_sentences(result)
    return normalize_text(result)


def _exclusion_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "").casefold()
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return normalized.strip(_TRAILING_PUNCT)


def matches_exclusion(text: str, phrases: list[str]) -> str | None:
    """Return the phrase that should cancel insertion, if any.

    A phrase cancels insertion only when the transcript consists of that phrase,
    possibly repeated with punctuation in between. A phrase that merely appears
    inside a longer dictation ("thank you very much") must not discard the rest
    of the sentence.
    """
    normalized = _exclusion_key(text)
    if not normalized:
        return None
    for phrase in phrases:
        key = _exclusion_key(phrase)
        if not key:
            continue
        if normalized == key:
            return phrase
        pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
        if not pattern.search(normalized):
            continue
        remainder = pattern.sub("", normalized)
        if not _NON_WORD.sub("", remainder):
            return phrase
    return None


def preview_text(text: str, limit: int = 48) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"
