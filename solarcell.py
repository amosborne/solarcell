from collections import namedtuple
from warnings import warn

import numpy
from scipy import constants
from scipy.optimize import least_squares, minimize, root_scalar


def _warn_solution(name, target, result, threshold=0.02):
    err = numpy.abs((result - target) / target)
    if err > threshold:
        wmsg1 = "{:s} fit error of {:0.1f}% exceeds {:0.1f}% threshold."
        wmsg1 = wmsg1.format(name, err * 100, threshold * 100)
        wmsg2 = "Target={:0.3f}, Result={:0.3f}, Delta={:0.3f}"
        wmsg2 = wmsg2.format(target, result, result - target)
        msg = " ".join([wmsg1, wmsg2])
        warn(msg)


_metrics = namedtuple("metrics", "isc voc imp vmp pmp vi iv pv")


class array:
    def __init__(self, isc, voc, imp, vmp, t, ns, np):
        assert isc[0] > imp[0] and isc[1] > imp[1], "Isc must exceed Imp."
        assert voc[0] > vmp[0] and voc[1] > vmp[1], "Voc must exceed Vmp."
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert ns > 0, "There must be atleast one cell per string."
        assert np > 0, "There must be atleast one string per array."

        self.isc = isc
        self.voc = voc
        self.imp = imp
        self.vmp = vmp
        self.t = t
        self.ns = numpy.ceil(ns)
        self.np = numpy.ceil(np)

    def params(self, t, g):
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert g >= 0, "Intensity must be non-negative."

        # Compute the adjusted cell parameters.
        dt = t - self.t
        isc = (self.isc[0] + dt * self.isc[1]) * g
        voc = self.voc[0] + dt * self.voc[1]
        imp = (self.imp[0] + dt * self.imp[1]) * g
        vmp = self.vmp[0] + dt * self.vmp[1]

        return isc, voc, imp, vmp

    def cell(self, t, g):
        isc, voc, imp, vmp = self.params(t, g)
        pmp = imp * vmp

        if g == 0:
            xvi = lambda i: numpy.nan if i < 0 else 0  # Blocking diode.
            xiv = lambda v: numpy.inf if v < 0 else 0  # Bypass diode.
            xpv = lambda v: v * xiv(v)
            return _metrics(isc, voc, imp, vmp, pmp, xvi, xiv, xpv)

        q_kT = constants.e / (constants.k * (t + 273.15))

        def x2eqn(x):
            # Diode model is a logarithmic function, V=F(I).
            i0, rs, n = x  # Parameters to be solved for numerically.
            i0 = i0 * 1e-20  # Scale factor to assist solver.
            q_nkT = q_kT / n

            def v(i):
                with numpy.errstate(invalid="ignore"):
                    vraw = numpy.log((isc - i) / i0 + 1) / q_nkT - i * rs
                    vraw = numpy.nan_to_num(vraw)  # Bypass diode.
                    vraw = numpy.where(i < 0, numpy.nan, vraw)  # Blocking diode.
                    return vraw

            return v

        def x2params(x):
            # Numerically invert the diode model to solve for Isc/Imp.
            v = x2eqn(x)
            risc = root_scalar(f=v, x0=isc, bracket=(isc * 0.98, isc))
            rimp = minimize(
                fun=lambda ximp: 1 / (ximp * v(ximp)),
                x0=isc / 2,
                bounds=[(0, isc * 0.99)],
            )
            assert risc.converged and rimp.success

            # Return the cell parameters: (xisc, xvoc, ximp, xvmp).
            return risc.root, v(0), rimp.x[0], v(rimp.x[0])

        def minfun(x):
            # The IV-curve is optimized againt Voc, Vmp, and Pmp.
            _, xvoc, ximp, xvmp = x2params(x)
            return voc - xvoc, vmp - xvmp, pmp - (ximp * xvmp)

        with numpy.errstate(all="ignore"):
            result = least_squares(
                fun=minfun,
                x0=(1, 0.1, 2.5),
                bounds=((1e-3, 0, 0.1), (1e3, 1, 10)),
                xtol=None,
            )
        emsg = "Curve fit failed for cell(t={:0.1f}, g={:0.2f})."
        assert result.success and result.cost < 0.01, emsg.format(t, g)

        # Warn if the solution is a poor fit.
        xisc, xvoc, ximp, xvmp = x2params(result.x)
        _warn_solution("Isc", isc, xisc)
        _warn_solution("Voc", voc, xvoc)
        _warn_solution("Imp", imp, ximp)
        _warn_solution("Vmp", vmp, xvmp)

        # Interpolate over voltage.
        xvi = lambda i: x2eqn(result.x)(i)
        i = numpy.linspace(0, xisc, 1000)
        xv = numpy.linspace(0, xvoc, 1000)
        xi = numpy.interp(xv, xvi(i)[::-1], i[::-1])
        xiv = lambda v: numpy.interp(v, xv, xi, left=numpy.inf, right=0)
        xpv = lambda v: v * xiv(v)

        return _metrics(xisc, xvoc, ximp, xvmp, ximp * xvmp, xvi, xiv, xpv)

    def curve(self, t, g):
        pass
