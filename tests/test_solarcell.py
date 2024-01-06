import numpy as np
import pytest

from solarcell import solarcell

azur3g30a = solarcell(
    isc=(0.5196, 0.00036),
    voc=(2.690, -0.0062),
    imp=(0.5029, 0.00024),
    vmp=(2.409, -0.0067),
    t=28,
)


def gtapprox(gt, lt, rel):
    return pytest.approx(gt, rel) == lt or gt > lt


def test_dark_cell():
    cell = azur3g30a.cell(t=28, g=0)
    assert cell.isc == 0 and cell.imp == 0  # no current in nominal range
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert np.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert np.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert np.isnan(cell.vi(-1))  # blocked current when injected


def test_cell():
    cell = azur3g30a.cell(t=azur3g30a.t + 20, g=1)
    assert gtapprox(cell.isc, azur3g30a.isc[0], rel=4e-2)
    assert gtapprox(cell.imp, azur3g30a.imp[0], rel=4e-2)
    assert gtapprox(azur3g30a.voc[0], cell.voc, rel=1e-3)
    assert gtapprox(azur3g30a.vmp[0], cell.vmp, rel=1e-3)
    assert pytest.approx(cell.pmp, rel=1e-3) == cell.pv(cell.vmp)
    assert pytest.approx(cell.pmp, rel=1e-3) == cell.vi(cell.imp) * cell.imp
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert np.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert np.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert np.isnan(cell.vi(-1))  # blocked current when injected


def test_solution():
    azur3g30a.cell(t=150, g=1)
    azur3g30a.cell(t=-100, g=1)
    azur3g30a.cell(t=150, g=0.1)
    azur3g30a.cell(t=-100, g=0.1)


def test_string():
    cell = azur3g30a.cell(t=28, g=1)
    string = azur3g30a.string(t=[28, 28], g=[1, 1])
    assert pytest.approx(string.isc, rel=1e-3) == cell.isc
    assert pytest.approx(string.voc, rel=1e-3) == cell.voc * 2
    assert pytest.approx(string.imp, rel=1e-3) == cell.imp
    assert pytest.approx(string.vmp, rel=1e-3) == cell.vmp * 2
    assert pytest.approx(string.pmp, rel=1e-3) == cell.pmp * 2


def test_array():
    cell = azur3g30a.cell(t=28, g=1)

    # Fully illuminated.
    array = azur3g30a.array(t=np.full((3, 3), 28), g=np.full((3, 3), 1))
    assert pytest.approx(array.pmp, rel=1e-3) == cell.pmp * 9
    assert pytest.approx(array.isc, rel=1e-3) == cell.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == cell.voc * 3

    # Half illuminated.
    array = azur3g30a.array(t=np.full((3, 3), 28), g=np.eye(3))
    assert pytest.approx(array.pmp, rel=1e-3) == cell.pmp * 3
    assert pytest.approx(array.isc, rel=1e-3) == cell.isc * 3
    assert pytest.approx(array.voc, rel=1e-3) == cell.voc * 1
