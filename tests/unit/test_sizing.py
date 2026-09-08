from decimal import Decimal as D

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sniper.domain.broker import VolumeConstraints
from sniper.risk.sizing import floor_volume, size_position


def constraints(minimum="0.0001", step="0.0001", maximum="1"):
    return VolumeConstraints(minimum=D(minimum), step=D(step), maximum=D(maximum))


def size(**overrides):
    args = dict(
        equity=D(10),
        target_risk_pct=D("0.25"),
        volumes=constraints(),
        loss_at_stop=lambda volume: volume * 50 + D("0.001"),
        margin_required=lambda volume: volume * 1000,
        free_margin=D(10),
    )
    return size_position(**(args | overrides))


def test_largest_volume_respects_costs_and_rounds_down():
    result = size()
    assert result.volume == D("0.0004")
    assert result.estimated_loss == D("0.021")
    assert result.required_margin == D("0.4")
    assert result.reason == "ACCEPT"


def test_no_round_up_to_broker_minimum():
    result = size(volumes=constraints(minimum="0.01", step="0.01"))
    assert result.volume == 0
    assert result.reason == "REJECT_INSUFFICIENT_GRANULARITY"


def test_margin_buffer_limits_size():
    assert size(free_margin=D("0.3"), minimum_margin_buffer=D("0.1")).volume == D("0.0002")
    assert size(free_margin=D("0.09")).reason == "REJECT_INSUFFICIENT_MARGIN"


def test_fixed_fee_can_prevent_any_position():
    assert size(loss_at_stop=lambda v: D("0.03") + v).volume == 0


def test_exact_risk_budget_is_allowed():
    result = size(loss_at_stop=lambda v: v * 50)
    assert result.volume == D("0.0005")
    assert result.estimated_loss == D("0.025")


@pytest.mark.parametrize(
    "overrides",
    [
        {"equity": D(0)},
        {"equity": D("NaN")},
        {"target_risk_pct": D(-1)},
        {"target_risk_pct": D("0.51")},
        {"hard_max_risk_pct": D(1)},
        {"minimum_margin_buffer": D(-1)},
        {"free_margin": D("Infinity")},
        {"loss_at_stop": lambda v: D("NaN")},
        {"loss_at_stop": lambda v: D(0)},
        {"loss_at_stop": lambda v: D(-1)},
        {"margin_required": lambda v: D(-1)},
    ],
)
def test_invalid_inputs_and_oracle_results_raise(overrides):
    with pytest.raises(ValueError):
        size(**overrides)


def test_volume_grid_can_be_offset_from_zero():
    grid = constraints(minimum="0.00015", step="0.0001", maximum="0.00048")
    assert floor_volume(D("0.000149"), grid) == 0
    assert floor_volume(D("0.0004"), grid) == D("0.00035")
    assert floor_volume(D(1), grid) == D("0.00045")
    assert size(volumes=grid, loss_at_stop=lambda v: v).volume == D("0.00045")


@given(
    step_units=st.integers(1, 100),
    min_units=st.integers(1, 100),
    steps=st.integers(0, 10000),
    loss_factor=st.integers(1, 1000),
    margin_factor=st.integers(1, 5000),
    fee_units=st.integers(0, 100),
    equity_units=st.integers(1, 10000),
)
def test_sizing_invariants_and_maximality(
    step_units, min_units, steps, loss_factor, margin_factor, fee_units, equity_units
):
    step = D(step_units) / 100000
    minimum = D(min_units) / 100000
    equity = D(equity_units) / 100
    grid = VolumeConstraints(minimum=minimum, step=step, maximum=minimum + steps * step)
    fee = D(fee_units) / 10000

    def loss(volume):
        return volume * loss_factor + fee

    def margin(volume):
        return volume * margin_factor

    result = size_position(
        equity=equity,
        target_risk_pct=D("0.25"),
        volumes=grid,
        loss_at_stop=loss,
        margin_required=margin,
        free_margin=equity,
    )
    budget = equity * D("0.0025")
    assert result.volume >= 0
    if result.volume:
        assert (result.volume - minimum) % step == 0
        assert minimum <= result.volume <= grid.maximum
        assert result.estimated_loss == loss(result.volume) <= budget
        assert result.estimated_loss <= equity * D("0.005")
        assert margin(result.volume) <= equity
        next_volume = result.volume + step
        assert (
            next_volume > grid.maximum or loss(next_volume) > budget or margin(next_volume) > equity
        )
    else:
        assert loss(minimum) > budget or margin(minimum) > equity


@given(requested=st.decimals(min_value=0, max_value=10, places=8, allow_nan=False))
def test_floor_volume_never_increases_requested_risk(requested):
    grid = constraints()
    rounded = floor_volume(requested, grid)
    assert 0 <= rounded <= min(requested, grid.maximum)
    if rounded:
        assert rounded >= grid.minimum
        assert (rounded - grid.minimum) % grid.step == 0


def test_rejects_estimate_that_changes_after_search():
    calls = 0

    def changed_loss(volume):
        nonlocal calls
        calls += 1
        return D("0.01") if calls == 1 else D(1)

    result = size(volumes=constraints(maximum="0.0001"), loss_at_stop=changed_loss)
    assert result.volume == 0
    assert result.reason == "REJECT_ESTIMATE_CHANGED"
