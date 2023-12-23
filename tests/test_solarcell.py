import numpy
import pytest

from solarcell import solarcell


@pytest.fixture
def azur3g30a():
    return solarcell(
        isc=(0.5196, 0.00036),
        voc=(2.690, -0.0062),
        imp=(0.5029, 0.00024),
        vmp=(2.409, -0.0067),
        t=28,
    )


def test_dark_cell(azur3g30a):
    cell = azur3g30a.cell(t=28, g=0)
    assert cell.isc == 0 and cell.imp == 0  # no current in nominal range
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert numpy.isnan(cell.vi(-1))  # blocked current when injected


def test_cell(azur3g30a):
    cell = azur3g30a.cell(t=azur3g30a.t + 10, g=1)
    assert cell.isc > azur3g30a.isc[0] and cell.imp > azur3g30a.imp[0]
    assert cell.voc < azur3g30a.voc[0] and cell.vmp < azur3g30a.vmp[0]
    assert pytest.approx(cell.pmp, rel=1e-4) == cell.pv(cell.vmp)
    assert pytest.approx(cell.pmp, rel=1e-4) == cell.vi(cell.imp) * cell.imp
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert numpy.isnan(cell.vi(-1))  # blocked current when injected


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_solution(azur3g30a):
    azur3g30a.cell(t=220, g=1)
    azur3g30a.cell(t=-180, g=1)
    azur3g30a.cell(t=220, g=0.1)
    azur3g30a.cell(t=-180, g=0.1)

def test_string(azur3g30a):
    cell = azur3g30a.cell(t=28, g=1)
    string = azur3g30a.string(t=28, g=1, ns=2)
    assert cell.isc == string.isc
    assert cell.voc * 2 == string.voc
    assert cell.imp == string.imp
    assert cell.vmp * 2 == string.vmp
    assert cell.pmp * 2 == string.pmp

def test_curve(azur3g30a):
    cell = azur3g30a.cell(t=60, g=1)

    # Fully illuminated.
    curve = azur3g30a.curve(t=60, g=1)
    assert pytest.approx(curve.pmp, rel=1e-3) == cell.pmp * azur3g30a.ns * azur3g30a.np
    assert pytest.approx(curve.isc, rel=1e-3) == cell.isc * azur3g30a.np
    assert pytest.approx(curve.voc, rel=1e-3) == cell.voc * azur3g30a.ns

    # Half illuminated.
    curve = azur3g30a.curve(t=60, g=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert pytest.approx(curve.pmp, rel=1e-3) == cell.pmp * azur3g30a.ns * azur3g30a.np / 2
    assert pytest.approx(curve.isc, rel=1e-3) == cell.isc * azur3g30a.np / 2
    assert pytest.approx(curve.voc, rel=1e-3) == cell.voc * azur3g30a.ns
