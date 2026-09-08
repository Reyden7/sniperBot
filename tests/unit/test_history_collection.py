from datetime import UTC, datetime
from types import SimpleNamespace

from sniper.data import history
from sniper.data.history import collect_history
from sniper.data.parquet_store import ParquetStore
from sniper.data.time import CollectionConfig, TimePolicy


def test_history_collection_paginates_by_utc_day_and_resumes(monkeypatch, tmp_path):
    calls = []

    def fake_collect(client, store, start, end, config, *, symbol):
        calls.append((start, end, symbol))
        return SimpleNamespace(accepted_ticks=7)

    monkeypatch.setattr(history, "collect_ticks", fake_collect)
    start = datetime(2026, 9, 1, 12, tzinfo=UTC)
    end = datetime(2026, 9, 3, 12, tzinfo=UTC)
    config = CollectionConfig(time=TimePolicy(source_basis="UTC"))
    store = ParquetStore(tmp_path)
    first = collect_history(object(), store, start, end, config)
    assert [(left.hour, right.hour) for left, right, _ in calls] == [(12, 0), (0, 0), (0, 12)]
    assert first.downloaded_chunks == 3
    assert first.resumed_chunks == 0
    assert first.accepted_ticks_downloaded == 21
    calls.clear()
    second = collect_history(object(), store, start, end, config)
    assert calls == []
    assert second.downloaded_chunks == 0
    assert second.resumed_chunks == 3
    assert len(list(tmp_path.glob("manifests/history/**/*.json"))) == 3
