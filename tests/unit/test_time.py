from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sniper.data.time import (
    ClockMonitor,
    TimePolicy,
    compare_tick_clock,
    epoch_ms,
    from_epoch_ms,
    localize_wall,
    parse_instant,
)


def test_iso_offset_is_normalized_to_canonical_utc():
    assert parse_instant("2026-09-08T12:30:00+02:00") == datetime(2026, 9, 8, 10, 30, tzinfo=UTC)
    assert parse_instant("2026-09-08T10:30:00Z").tzinfo is UTC


def test_naive_time_is_rejected():
    with pytest.raises(ValueError, match="timezone required"):
        parse_instant("2026-09-08T10:30:00")


def test_epoch_milliseconds_round_trip_exactly():
    instant = datetime(2026, 9, 8, 10, 30, 12, 345000, tzinfo=UTC)
    assert from_epoch_ms(epoch_ms(instant)) == instant


def test_server_local_and_utc_are_distinct_representations_of_one_instant():
    policy = TimePolicy(server_timezone="Europe/Helsinki", local_timezone="Europe/Paris")
    utc = datetime(2026, 1, 8, 10, tzinfo=UTC)
    server, local = policy.representations(utc)
    assert server is not None
    assert server.hour == 12
    assert local.hour == 11
    assert server.astimezone(UTC) == local.astimezone(UTC) == utc


def test_ambiguous_and_nonexistent_server_wall_times_are_rejected():
    paris = ZoneInfo("Europe/Paris")
    with pytest.raises(ValueError, match="ambiguous"):
        localize_wall(datetime(2026, 10, 25, 2, 30), paris)
    with pytest.raises(ValueError, match="nonexistent"):
        localize_wall(datetime(2026, 3, 29, 2, 30), paris)


def test_tick_offset_never_claims_clock_desynchronization_by_itself():
    observed = datetime(2026, 9, 8, 10, tzinfo=UTC)
    result = compare_tick_clock(observed + timedelta(hours=3), observed, tolerance=5)
    assert result.status == "TIME_REFERENCE_UNVERIFIED"
    assert result.observed_offset_seconds == 10800


def test_monotonic_comparison_can_prove_a_local_clock_jump():
    monitor = ClockMonitor(tolerance_seconds=1)
    start = datetime(2026, 9, 8, 10, tzinfo=UTC)
    assert monitor.observe(start, 1_000_000_000).status == "CONSISTENT"
    result = monitor.observe(start + timedelta(seconds=10), 2_000_000_000)
    assert result.status == "CLOCK_DESYNCHRONIZATION"
    assert result.observed_offset_seconds == 9
