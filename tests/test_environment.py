"""Tests for the process-wide environment defaults."""

from __future__ import annotations

import os

from voicetyper import environment


def test_defaults_are_applied_to_an_empty_mapping() -> None:
    environ: dict[str, str] = {}

    applied = environment.prepare(environ)

    assert set(applied) == set(environment.DEFAULTS)
    assert environ == dict(environment.DEFAULTS)


def test_explicit_user_values_are_kept() -> None:
    environ = {"TOKENIZERS_PARALLELISM": "true"}

    applied = environment.prepare(environ)

    assert "TOKENIZERS_PARALLELISM" not in applied
    assert environ["TOKENIZERS_PARALLELISM"] == "true"
    assert environ["HF_HUB_DISABLE_XET"] == "1"


def test_prepare_is_idempotent() -> None:
    environ: dict[str, str] = {}

    assert environment.prepare(environ)
    assert environment.prepare(environ) == []


def test_importing_the_package_sets_the_defaults() -> None:
    # The variables must be in place before huggingface_hub is imported, and
    # every supported entry point imports the package first.
    for name in environment.DEFAULTS:
        assert os.environ.get(name)
