"""UTC is canonical. Server wall time and local display time are never guessed."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, model_validator

from sniper.config import Model

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def as_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timezone required: use ISO 8601 with Z or an explicit UTC offset")
    return value.astimezone(UTC)


def epoch_ms(value: datetime) -> int:
    delta = as_utc(value) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def from_epoch_ms(value: int) -> datetime:
    if value < 0:
        raise ValueError("negative MT5 epoch")
    return EPOCH + timedelta(milliseconds=value)


def parse_instant(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    return as_utc(datetime.fromisoformat(normalized))


def localize_wall(wall: datetime, zone: ZoneInfo) -> datetime:
    """Reject ambiguous/nonexistent server wall times instead of choosing a DST fold."""
    if wall.tzinfo is not None:
        raise ValueError("wall time must be naive before explicit localization")
    candidates = set()
    for fold in (0, 1):
        instant = wall.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if instant.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.add(instant)
    if len(candidates) != 1:
        raise ValueError("ambiguous or nonexistent server wall time (DST)")
    return candidates.pop()


class TimePolicy(Model):
    source_basis: Literal["UTC", "SERVER_WALL"] = "UTC"
    server_timezone: str | None = None
    local_timezone: str = "Europe/Paris"
    basis_evidence: str = "MT5 Python API documented UTC contract"

    @model_validator(mode="after")
    def validate_zones(self) -> Self:
        try:
            ZoneInfo(self.local_timezone)
            if self.server_timezone:
                ZoneInfo(self.server_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown IANA timezone") from exc
        if self.source_basis == "SERVER_WALL" and (
            not self.server_timezone
            or not self.basis_evidence.strip()
            or self.basis_evidence == "MT5 Python API documented UTC contract"
        ):
            raise ValueError("SERVER_WALL requires a verified server timezone and basis_evidence")
        return self

    def decode(self, raw_time_msc: int) -> datetime:
        raw = from_epoch_ms(raw_time_msc)
        if self.source_basis == "UTC":
            return raw
        assert self.server_timezone is not None
        return localize_wall(raw.replace(tzinfo=None), ZoneInfo(self.server_timezone))

    def encode_query(self, instant: datetime) -> datetime:
        utc = as_utc(instant)
        if self.source_basis == "UTC":
            return utc
        assert self.server_timezone is not None
        wall = utc.astimezone(ZoneInfo(self.server_timezone)).replace(tzinfo=None)
        # Check that the inverse mapping is unambiguous before querying the SDK.
        if localize_wall(wall, ZoneInfo(self.server_timezone)) != utc:
            raise ValueError("server time conversion is not reversible")
        return wall.replace(tzinfo=UTC)

    def representations(self, utc: datetime) -> tuple[datetime | None, datetime]:
        utc = as_utc(utc)
        server = utc.astimezone(ZoneInfo(self.server_timezone)) if self.server_timezone else None
        return server, utc.astimezone(ZoneInfo(self.local_timezone))


class TimeDiagnostic(Model):
    status: Literal["CONSISTENT", "TIME_REFERENCE_UNVERIFIED", "CLOCK_DESYNCHRONIZATION"]
    reason: str
    observed_offset_seconds: float | None = None


def compare_tick_clock(
    tick_utc: datetime, observed_utc: datetime, tolerance: float
) -> TimeDiagnostic:
    offset = (as_utc(tick_utc) - as_utc(observed_utc)).total_seconds()
    return TimeDiagnostic(
        status="TIME_REFERENCE_UNVERIFIED" if abs(offset) > tolerance else "CONSISTENT",
        reason="Tick/host offset alone cannot prove which clock or source convention is wrong"
        if abs(offset) > tolerance
        else "Tick and observed UTC are within tolerance",
        observed_offset_seconds=offset,
    )


@dataclass
class ClockMonitor:
    """Prove local wall-clock jumps against an independent monotonic elapsed clock."""

    tolerance_seconds: float = 2.0
    previous_utc: datetime | None = None
    previous_monotonic_ns: int | None = None

    def observe(self, utc: datetime, monotonic_ns: int) -> TimeDiagnostic:
        utc = as_utc(utc)
        difference = 0.0
        if self.previous_utc is not None and self.previous_monotonic_ns is not None:
            elapsed = (monotonic_ns - self.previous_monotonic_ns) / 1_000_000_000
            if elapsed < 0:
                raise ValueError("monotonic clock moved backwards")
            difference = (utc - self.previous_utc).total_seconds() - elapsed
        self.previous_utc, self.previous_monotonic_ns = utc, monotonic_ns
        return TimeDiagnostic(
            status="CLOCK_DESYNCHRONIZATION"
            if abs(difference) > self.tolerance_seconds
            else "CONSISTENT",
            reason="Local UTC elapsed time compared with monotonic elapsed time",
            observed_offset_seconds=difference,
        )


class CollectionConfig(Model):
    time: TimePolicy = Field(default_factory=TimePolicy)
    chunk_minutes: int = Field(60, ge=1, le=1440)
    gap_threshold_seconds: float = Field(60, gt=0, allow_inf_nan=False)
    max_ticks_per_chunk: int = Field(2_000_000, ge=1)
    source_id: str = Field("mt5-development", pattern=r"^[A-Za-z0-9_-]{1,64}$")
