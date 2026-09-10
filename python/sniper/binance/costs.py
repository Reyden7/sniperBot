"""Explicit Binance Spot cost model with account fee provenance."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from sniper.binance.models import CostEstimate, FeeSchedule
from sniper.binance.settings import BinanceSettings

ONE_HUNDRED = Decimal(100)
TEN_THOUSAND = Decimal(10000)

ExecutionMode = Literal["TAKER_ENTRY_TAKER_EXIT", "MAKER_ENTRY_TAKER_EXIT"]


@dataclass(frozen=True)
class PairExecutionCosts:
    taker_taker: CostEstimate
    maker_taker: CostEstimate


def fallback_fee_schedule(symbol: str, settings: BinanceSettings) -> FeeSchedule:
    return FeeSchedule(
        symbol=symbol,
        maker_rate=settings.fallback_maker_fee_rate,
        taker_rate=settings.fallback_taker_fee_rate,
        source="CONFIGURED_FALLBACK",
        is_account_specific=False,
    )


def fee_schedules_from_trade_fee(payload: list[dict[str, Any]]) -> dict[str, FeeSchedule]:
    result = {}
    for item in payload:
        schedule = FeeSchedule(
            symbol=str(item["symbol"]),
            maker_rate=Decimal(str(item["makerCommission"])),
            taker_rate=Decimal(str(item["takerCommission"])),
            source="BINANCE_ACCOUNT_API",
            is_account_specific=True,
        )
        result[schedule.symbol] = schedule
    return result


def fee_schedule_from_account(symbol: str, payload: dict[str, Any]) -> FeeSchedule:
    rates = payload.get("commissionRates", {})
    return FeeSchedule(
        symbol=symbol,
        maker_rate=Decimal(str(rates["maker"])),
        taker_rate=Decimal(str(rates["taker"])),
        source="ACCOUNT_COMMISSION_RATE",
        is_account_specific=True,
    )


class BinanceCostModel:
    """Estimate both supported round-trip Spot execution modes."""

    def __init__(
        self,
        fee: FeeSchedule,
        *,
        slippage_bps_per_side: Decimal,
        uncertainty_buffer_bps: Decimal,
    ):
        self.fee = fee
        self.slippage_bps_per_side = slippage_bps_per_side
        self.uncertainty_buffer_bps = uncertainty_buffer_bps

    def estimate(
        self,
        *,
        expected_gross_move_pct: Decimal,
        bid: Decimal,
        ask: Decimal,
        use_taker: bool | None = None,
        execution_mode: ExecutionMode = "TAKER_ENTRY_TAKER_EXIT",
        estimated_slippage_pct: Decimal | None = None,
        maker_adverse_selection_pct: Decimal = Decimal(0),
    ) -> CostEstimate:
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("invalid executable bid/ask")
        # Preserve the original boolean call while making its semantics explicit.
        if use_taker is not None:
            execution_mode = "TAKER_ENTRY_TAKER_EXIT" if use_taker else "MAKER_ENTRY_TAKER_EXIT"
        entry_rate = (
            self.fee.taker_rate
            if execution_mode == "TAKER_ENTRY_TAKER_EXIT"
            else self.fee.maker_rate
        )
        entry_fee_pct = entry_rate * ONE_HUNDRED
        exit_fee_pct = self.fee.taker_rate * ONE_HUNDRED
        commission_pct = entry_fee_pct + exit_fee_pct
        midpoint = (ask + bid) / 2
        full_spread_pct = (ask - bid) / midpoint * ONE_HUNDRED
        if execution_mode == "TAKER_ENTRY_TAKER_EXIT":
            spread_pct = full_spread_pct
            default_slippage_pct = 2 * self.slippage_bps_per_side / ONE_HUNDRED
            slippage_pct = (
                default_slippage_pct
                if estimated_slippage_pct is None
                else max(estimated_slippage_pct, default_slippage_pct)
            )
        else:
            # A resting entry avoids the entry crossing but the taker exit still crosses
            # half the quoted spread. Its fill is assessed separately and never assumed.
            spread_pct = full_spread_pct / 2
            default_slippage_pct = self.slippage_bps_per_side / ONE_HUNDRED
            measured_exit_slippage = (
                None if estimated_slippage_pct is None else estimated_slippage_pct / 2
            )
            slippage_pct = max(measured_exit_slippage or Decimal(0), default_slippage_pct)
            slippage_pct += maker_adverse_selection_pct
        buffer_pct = self.uncertainty_buffer_bps / ONE_HUNDRED
        estimated_cost_pct = spread_pct + slippage_pct + commission_pct
        expected_net = expected_gross_move_pct - estimated_cost_pct
        return CostEstimate(
            symbol=self.fee.symbol,
            expected_gross_move_pct=expected_gross_move_pct,
            entry_fee_pct=entry_fee_pct,
            exit_fee_pct=exit_fee_pct,
            spread_pct=spread_pct,
            expected_slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            estimated_cost_pct=estimated_cost_pct,
            uncertainty_buffer_pct=buffer_pct,
            expected_net_edge_pct=expected_net,
            acceptable=expected_net > buffer_pct,
            fee_source=self.fee.source,
        )

    def compare_execution_modes(
        self,
        *,
        expected_gross_move_pct: Decimal,
        bid: Decimal,
        ask: Decimal,
        estimated_slippage_pct: Decimal | None = None,
        maker_adverse_selection_pct: Decimal = Decimal(0),
    ) -> PairExecutionCosts:
        return PairExecutionCosts(
            taker_taker=self.estimate(
                expected_gross_move_pct=expected_gross_move_pct,
                bid=bid,
                ask=ask,
                execution_mode="TAKER_ENTRY_TAKER_EXIT",
                estimated_slippage_pct=estimated_slippage_pct,
            ),
            maker_taker=self.estimate(
                expected_gross_move_pct=expected_gross_move_pct,
                bid=bid,
                ask=ask,
                execution_mode="MAKER_ENTRY_TAKER_EXIT",
                estimated_slippage_pct=estimated_slippage_pct,
                maker_adverse_selection_pct=maker_adverse_selection_pct,
            ),
        )
