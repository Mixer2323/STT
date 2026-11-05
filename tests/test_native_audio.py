from __future__ import annotations

from voicetyper.native_audio import _detect_sample_format


def _format(tag: int, bits: int, subformat: bytes | None = None) -> bytes:
    raw = bytearray(40)
    raw[0:2] = tag.to_bytes(2, "little")
    raw[2:4] = (1).to_bytes(2, "little")
    raw[12:14] = (bits // 8).to_bytes(2, "little")
    raw[14:16] = bits.to_bytes(2, "little")
    if subformat is not None:
        raw[16:18] = (22).to_bytes(2, "little")
        raw[24:40] = subformat
    return bytes(raw)


FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")
PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


def test_plain_ieee_float_mix_format() -> None:
    assert _detect_sample_format(_format(0x0003, 32)) == "float32"


def test_plain_pcm16_mix_format() -> None:
    assert _detect_sample_format(_format(0x0001, 16)) == "pcm16"


def test_extensible_float_mix_format() -> None:
    assert _detect_sample_format(_format(0xFFFE, 32, FLOAT_GUID)) == "float32"


def test_extensible_pcm24_mix_format() -> None:
    assert _detect_sample_format(_format(0xFFFE, 24, PCM_GUID)) == "pcm24"
