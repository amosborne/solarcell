from itertools import product

import numpy as np
import pytest

from solarcell import solarcell

azurspace = solarcell(
    isc=(0.5196, 0.00036),
    voc=(2.690, -0.0062),
    imp=(0.5029, 0.00024),
    vmp=(2.409, -0.0067),
    area=30.18,
    t=28,
)


def test_dark_cell():
    cell = azurspace.cell(t=28, g=0)
    assert cell.isc == 0 and cell.imp == 0  # no current in nominal range
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert np.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert np.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert np.isnan(cell.vi(-1))  # blocked current when injected


def test_cell():
    cell = azurspace.cell(t=40, g=1)
    assert cell.isc > azurspace.isc
    assert cell.imp > azurspace.imp
    assert cell.voc < azurspace.voc
    assert cell.vmp < azurspace.vmp
    assert pytest.approx(cell.pmp, rel=0.01) == cell.pv(cell.vmp)
    assert pytest.approx(cell.pmp, rel=0.01) == cell.pi(cell.imp)
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert np.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert np.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert np.isnan(cell.vi(-1))  # blocked current when injected


def test_solution():
    ts = [-100, 150]
    gs = [0.1, 1]
    for t, g in product(ts, gs):
        azurspace.cell(t, g)


def test_string():
    string = azurspace.string(t=[28, 28], g=[1, 1])
    assert string.isc == pytest.approx(azurspace.isc * 1, rel=0.01)
    assert string.voc == pytest.approx(azurspace.voc * 2, rel=0.01)
    assert string.imp == pytest.approx(azurspace.imp * 1, rel=0.01)
    assert string.vmp == pytest.approx(azurspace.vmp * 2, rel=0.01)
    assert string.pmp == pytest.approx(azurspace.pmp * 2, rel=0.01)


def test_array():
    # Fully illuminated.
    array = azurspace.array(t=np.full((3, 3), 28), g=np.full((3, 3), 1))
    assert array.pmp == pytest.approx(azurspace.pmp * 9, rel=0.01)
    assert array.isc == pytest.approx(azurspace.isc * 3, rel=0.01)
    assert array.voc == pytest.approx(azurspace.voc * 3, rel=0.01)

    # Half illuminated.
    array = azurspace.array(t=np.full((3, 3), 28), g=np.eye(3))
    assert array.pmp == pytest.approx(azurspace.pmp * 3, rel=0.01)
    assert array.isc == pytest.approx(azurspace.isc * 3, rel=0.01)
    assert array.voc == pytest.approx(azurspace.voc * 1, rel=0.01)
