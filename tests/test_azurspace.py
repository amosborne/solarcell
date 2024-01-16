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
    assert pytest.approx(cell.pmp, rel=1e-3) == cell.pv(cell.vmp)
    assert pytest.approx(cell.pmp, rel=1e-3) == cell.pi(cell.imp)
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
    assert pytest.approx(string.isc, rel=1e-3) == azurspace.isc
    assert pytest.approx(string.voc, rel=1e-3) == azurspace.voc * 2
    assert pytest.approx(string.imp, rel=1e-3) == azurspace.imp
    assert pytest.approx(string.vmp, rel=1e-3) == azurspace.vmp * 2
    assert pytest.approx(string.pmp, rel=1e-3) == azurspace.pmp * 2


def test_array():
    # Fully illuminated.
    array = azurspace.array(t=np.full((3, 3), 28), g=np.full((3, 3), 1))
    assert pytest.approx(array.pmp, rel=1e-3) == azurspace.pmp * 9
    assert pytest.approx(array.isc, rel=1e-3) == azurspace.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == azurspace.voc * 3

    # Half illuminated.
    array = azurspace.array(t=np.full((3, 3), 28), g=np.eye(3))
    assert pytest.approx(array.pmp, rel=1e-3) == azurspace.pmp * 3
    assert pytest.approx(array.isc, rel=1e-3) == azurspace.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == azurspace.voc * 1
