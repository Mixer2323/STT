from __future__ import annotations

from voicetyper.native_audio import (
    _AUTOCONVERTPCM,
    CaptureError,
    _attempt_details,
    _detect_sample_format,
    _hresult_text,
    format_candidates,
    parse_mix_format,
)

FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")
PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


def _format(tag: int, bits: int, subformat: bytes | None = None) -> bytes:
    raw = bytearray(40)
    raw[0:2] = tag.to_bytes(2, "little")
    raw[2:4] = (1).to_bytes(2, "little")
    raw[4:8] = (44100).to_bytes(4, "little")
    raw[8:12] = (44100 * (bits // 8)).to_bytes(4, "little")
    raw[12:14] = (bits // 8).to_bytes(2, "little")
    raw[14:16] = bits.to_bytes(2, "little")
    if subformat is not None:
        raw[16:18] = (22).to_bytes(2, "little")
        raw[18:20] = bits.to_bytes(2, "little")
        raw[20:24] = (0).to_bytes(4, "little")
        raw[24:40] = subformat
    return bytes(raw)


def test_plain_ieee_float_mix_format() -> None:
    assert _detect_sample_format(_format(0x0003, 32)) == "float32"


def test_plain_pcm16_mix_format() -> None:
    assert _detect_sample_format(_format(0x0001, 16)) == "pcm16"


def test_extensible_float_mix_format() -> None:
    assert _detect_sample_format(_format(0xFFFE, 32, FLOAT_GUID)) == "float32"


def test_extensible_pcm24_mix_format() -> None:
    assert _detect_sample_format(_format(0xFFFE, 24, PCM_GUID)) == "pcm24"


def test_parse_mix_format_reads_extensible_fields() -> None:
    mix = parse_mix_format(_format(0xFFFE, 32, FLOAT_GUID))

    assert mix.tag == 0xFFFE
    assert mix.channels == 1
    assert mix.rate == 44100
    assert mix.bits == 32
    assert mix.valid_bits == 32
    assert mix.sample_format == "float32"
    assert "44100 Гц" in mix.describe()
    assert "IEEE float" not in mix.describe()


def test_parse_mix_format_reads_plain_pcm() -> None:
    mix = parse_mix_format(_format(0x0001, 16))

    assert mix.tag == 0x0001
    assert mix.sample_format == "pcm16"
    assert mix.cb_size == 0
    assert "PCM" in mix.describe()


def test_format_candidates_prefers_device_format_then_autoconvert() -> None:
    mix = parse_mix_format(_format(0xFFFE, 32, FLOAT_GUID))

    candidates = format_candidates(mix, preferred_rate=16000)

    assert candidates[0].raw == mix.raw
    assert candidates[0].rate == 44100
    assert candidates[1].raw == mix.raw
    assert candidates[2].flags & _AUTOCONVERTPCM
    assert any(
        candidate.rate == 16000
        and candidate.channels == 1
        and candidate.flags & _AUTOCONVERTPCM
        for candidate in candidates
    )


def test_format_candidates_work_without_mix_format() -> None:
    candidates = format_candidates(None, preferred_rate=16000)

    assert candidates
    assert all(candidate.raw is None for candidate in candidates)
    assert any(candidate.flags & _AUTOCONVERTPCM for candidate in candidates)


def test_hresult_text_names_invalid_argument() -> None:
    assert "0x80070057" in _hresult_text(0x80070057)
    assert "E_INVALIDARG" in _hresult_text(0x80070057)


def test_attempt_details_keeps_every_variant() -> None:
    errors = [
        CaptureError(
            "Initialize [формат устройства, буфер 0: 44100 Гц]: E_INVALIDARG",
            step="Initialize",
            hresult=0x80070057,
            variant="формат устройства, буфер 0",
        ),
        CaptureError(
            "Initialize [PCM16 16 кГц, моно: 16000 Гц]: E_INVALIDARG",
            step="Initialize",
            hresult=0x80070057,
            variant="PCM16 16 кГц, моно",
        ),
        CaptureError(
            "GetService: AUDCLNT_E_UNSUPPORTED_FORMAT",
            step="GetService",
            hresult=0x88890008,
            variant="формат устройства",
        ),
        RuntimeError("Activate: доступ отказал"),
    ]

    details = _attempt_details(errors)

    assert "формат устройства, буфер 0, PCM16 16 кГц, моно" in details
    assert "(AUDCLNT_E_UNSUPPORTED_FORMAT) → формат устройства" in details
    assert "Activate: доступ отказал" in details
