from itertools import product

import numpy as np
import pytest

from solarcell import solarcell

bluebird = solarcell(
    isc=(6.7291, 0.00403),
    voc=(0.6882, -0.0021),
    imp=(6.5276, 0.00403),
    vmp=(0.5580, -0.0024),
    area=120,
    t=28,
)


def test_dark_cell():
    cell = bluebird.cell(t=28, g=0)
    assert cell.isc == 0 and cell.imp == 0  # no current in nominal range
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert np.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert np.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert np.isnan(cell.vi(-1))  # blocked current when injected


def test_cell():
    cell = bluebird.cell(t=40, g=1)
    assert cell.isc > bluebird.isc
    assert cell.imp > bluebird.imp
    assert cell.voc < bluebird.voc
    assert cell.vmp < bluebird.vmp
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
        bluebird.cell(t, g)


def test_string():
    string = bluebird.string(t=[28, 28], g=[1, 1])
    assert pytest.approx(string.isc, rel=1e-3) == bluebird.isc
    assert pytest.approx(string.voc, rel=1e-3) == bluebird.voc * 2
    assert pytest.approx(string.imp, rel=1e-3) == bluebird.imp
    assert pytest.approx(string.vmp, rel=1e-3) == bluebird.vmp * 2
    assert pytest.approx(string.pmp, rel=1e-3) == bluebird.pmp * 2


def test_array():
    # Fully illuminated.
    array = bluebird.array(t=np.full((3, 3), 28), g=np.full((3, 3), 1))
    assert pytest.approx(array.pmp, rel=1e-3) == bluebird.pmp * 9
    assert pytest.approx(array.isc, rel=1e-3) == bluebird.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == bluebird.voc * 3

    # Half illuminated.
    array = bluebird.array(t=np.full((3, 3), 28), g=np.eye(3))
    assert pytest.approx(array.pmp, rel=1e-3) == bluebird.pmp * 3
    assert pytest.approx(array.isc, rel=1e-3) == bluebird.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == bluebird.voc * 1
