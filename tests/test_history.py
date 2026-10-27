from __future__ import annotations

from pathlib import Path

from voicetyper.history import HistoryStore


def test_history_add_list_search_and_limit(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db", limit=3)
    for index in range(5):
        store.add(f"text {index}", engine="gigaam", duration=1.5, app="Editor")
    assert store.count() == 3
    assert [entry.text for entry in store.list()] == ["text 4", "text 3", "text 2"]
    assert [entry.text for entry in store.list(query="text 3")] == ["text 3"]
    assert store.list(limit=1)[0].text == "text 4"


def test_history_can_be_disabled(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db", enabled=False)
    assert store.add("ignored") is None
    assert store.count() == 0


def test_export_and_clear(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    store.add("привет; мир", engine="test")
    destination = store.export_csv(tmp_path / "history.csv")
    content = destination.read_text(encoding="utf-8-sig")
    assert "привет; мир" in content
    assert content.count(";") >= 1
    assert store.clear() is True
    assert store.count() == 0


def test_delete_entry(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    entry_id = store.add("delete me")
    assert entry_id is not None
    assert store.delete(entry_id) is True
    assert store.count() == 0
    assert store.delete(entry_id) is False
