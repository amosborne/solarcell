import numpy
import pytest

import solarcell


@pytest.fixture
def array():
    return solarcell.array(
        isc=(0.5196, 0.00036),
        voc=(2.690, -0.0062),
        imp=(0.5029, 0.00024),
        vmp=(2.409, -0.0067),
        t=28,
        ns=24,
        np=12,
    )


def test_dark_cell(array):
    cell = array.cell(t=array.t, g=0)
    assert cell.isc == 0 and cell.imp == 0  # no current in nominal range
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert numpy.isnan(cell.vi(-1))  # blocked current when injected


def test_cell(array):
    cell = array.cell(t=array.t + 10, g=1)
    assert cell.isc > array.isc[0] and cell.imp > array.imp[0]
    assert cell.voc < array.voc[0] and cell.vmp < array.vmp[0]
    assert pytest.approx(cell.pmp, rel=1e-4) == cell.pv(cell.vmp)
    assert pytest.approx(cell.pmp, rel=1e-4) == cell.vi(cell.imp) * cell.imp
    assert cell.iv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isposinf(cell.iv(-1))  # bypass current when reversed
    assert cell.pv(cell.voc + 1) == 0  # no current when blocking
    assert numpy.isneginf(cell.pv(-1))  # bypass current when reversed
    assert cell.vi(cell.isc + 1) == 0  # no voltage when bypassed
    assert numpy.isnan(cell.vi(-1))  # blocked current when injected


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_solution(array):
    array.cell(t=220, g=1)
    array.cell(t=-180, g=1)
    array.cell(t=220, g=0.1)
    array.cell(t=-180, g=0.1)
