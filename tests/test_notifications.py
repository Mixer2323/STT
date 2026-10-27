from __future__ import annotations

from voicetyper.notifications import Notifier


class _FakeIcon:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def notify(self, message: str, title: str) -> None:
        self.calls.append((message, title))


def test_long_notifications_are_truncated() -> None:
    icon = _FakeIcon()
    notifier = Notifier()
    notifier.set_icon(icon)

    assert notifier.notify("T" * 500, "M" * 500)

    message, title = icon.calls[0]
    assert len(message) <= 256
    assert len(title) <= 256
    assert message.endswith("…")
    assert title.endswith("…")