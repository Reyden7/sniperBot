from decimal import Decimal as D

import pytest
from sniper.backtest.execution_model import MinimumCommission, NoCommission, PerLotCommission
from sniper.backtest.slippage import FixedSlippage, NoSlippage, RandomSlippage
from sniper.domain.trade import Side


def test_commission_models():
    assert NoCommission().per_side(D("0.1")) == 0
    assert PerLotCommission(D(2)).per_side(D("0.1")) == D("0.2")
    assert MinimumCommission(D(2), D("0.5")).per_side(D("0.1")) == D("0.5")
    assert MinimumCommission(D(2), D("0.5")).per_side(D(1)) == D(2)


@pytest.mark.parametrize(
    "side,entry,expected",
    [
        (Side.BUY, True, D(101)),
        (Side.BUY, False, D(99)),
        (Side.SELL, True, D(99)),
        (Side.SELL, False, D(101)),
    ],
)
def test_fixed_adverse_slippage_direction(side, entry, expected):
    assert FixedSlippage(D(1)).apply(D(100), D(1), side, entry=entry) == expected
    assert NoSlippage().apply(D(100), D(1), side, entry=entry) == D(100)


def test_fixed_slippage_can_be_favorable():
    assert FixedSlippage(D(-1)).apply(D(100), D(1), Side.BUY, entry=True) == D(99)


def test_random_slippage_is_seeded_and_can_use_both_directions():
    first = RandomSlippage(3, seed=7)
    second = RandomSlippage(3, seed=7)
    values1 = [first.apply(D(100), D(1), Side.BUY, entry=True) for _ in range(30)]
    values2 = [second.apply(D(100), D(1), Side.BUY, entry=True) for _ in range(30)]
    assert values1 == values2
    assert min(values1) < 100 < max(values1)
