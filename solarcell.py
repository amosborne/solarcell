from collections import namedtuple
from functools import partial
from warnings import warn

import numpy
from scipy import constants
from scipy.optimize import least_squares, minimize_scalar, root_scalar


def _warn_solution(name, target, result, threshold=0.02):
    err = numpy.abs((result - target) / target)
    if err > threshold:
        wmsg1 = "{:s} fit error of {:0.1f}% exceeds {:0.1f}% threshold."
        wmsg1 = wmsg1.format(name, err * 100, threshold * 100)
        wmsg2 = "Target={:0.3f}, Result={:0.3f}, Delta={:0.3f}"
        wmsg2 = wmsg2.format(target, result, result - target)
        warn(" ".join([wmsg1, wmsg2]))


def _metrics(i, v):
    # Given current increasing from 0 to Isc and corresponding voltages.
    # Returns namedtuple: (isc, voc, imp, vmp, pmp, vi, iv, pi, pv)
    m = namedtuple("_metrics", "isc voc imp vmp pmp vi iv pi pv")
    vi = partial(numpy.interp, xp=i, fp=v, left=numpy.nan, right=0)
    iv = partial(numpy.interp, xp=v[::-1], fp=i[::-1], left=numpy.inf, right=0)
    return m(
        isc=i[-1],  # Short-circuit current.
        voc=v[0],  # Open-circuit voltage.
        imp=i[numpy.argmax(i * v)],  # Max-power current.
        vmp=v[numpy.argmax(i * v)],  # Max-power voltage.
        pmp=numpy.max(i * v),  # Max-power power.
        vi=vi,  # Function, current to voltage.
        iv=iv,  # Function, voltage to current.
        pi=lambda i: i * vi(i),  # Function, current to power.
        pv=lambda v: v * iv(v),  # Function, voltage to power.
    )


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
        self.ns = int(numpy.ceil(ns))  # Round up to nearest integer.
        self.np = int(numpy.ceil(np))  # Round up to nearest integer.

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
        # Skip the curve fit altogether if the cell is dark.
        if g == 0:
            return _metrics(numpy.array([0]), numpy.array([0]))

        # Otherwise proceed with the curve fit to the following parameters.
        isc, voc, imp, vmp = self.params(t, g)

        def x2eqn(x):
            i0, rs, n = x  # Parameters to be solved for numerically.
            i0 = i0 * 1e-20  # Scale factor to assist solver.

            def v(i):
                # Diode model: voltage is a logarithmic function of current.
                with numpy.errstate(invalid="ignore"):
                    q_kT = constants.e / (constants.k * (t + 273.15))
                    v = numpy.log((isc - i) / i0 + 1) / (q_kT / n) - i * rs
                    v = numpy.nan_to_num(v)  # Bypass diode.
                    v = numpy.where(i < 0, numpy.nan, v)  # Blocking diode.
                    return v

            return v

        def x2params(x):
            # Numerically invert the diode model to solve for Isc/Imp.
            # Return the cell parameters: (xisc, xvoc, ximp, xvmp).
            v = x2eqn(x)
            pinv = lambda i: 1 / (i * v(i))
            risc = root_scalar(f=v, x0=isc, bracket=(isc * 0.98, isc))
            rimp = minimize_scalar(fun=pinv, bounds=(0, isc * 0.99))
            assert risc.converged and rimp.success
            return risc.root, v(0), rimp.x, v(rimp.x)

        def minfun(x):
            # The IV-curve is optimized againt Voc, Vmp, and Pmp.
            _, xvoc, ximp, xvmp = x2params(x)
            return voc - xvoc, vmp - xvmp, imp * vmp - ximp * xvmp

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

        xi = numpy.linspace(0, xisc, 1000)
        xv = x2eqn(result.x)(xi)
        return _metrics(xi, xv)

    def curve(self, t, g):
        # Create an array of size (ns, np) with elements (t, g).
        def repmat(arr):
            arr = numpy.array(arr)
            if arr.size == 1:
                arr = numpy.full((self.ns, self.np), arr)
            elif arr.shape == (self.np,):
                arr = numpy.tile(arr, (self.ns, 1))
            elif arr.shape == (self.ns * self.np,):
                arr = numpy.reshape(arr, (self.ns, self.np))
            elif arr.shape == (self.ns, self.np):
                pass
            else:
                emsg = "Input cannot be cast into (ns, np) shape."
                raise UserWarning(emsg)

            return arr

        # Compute cell-level metrics. Columns are strings.
        t, g = repmat(t), repmat(g)
        tgs = set(list(zip(t.flat, g.flat)))
        cell_metrics = dict()
        for tx, gx in tgs:
            cell_metrics[(tx, gx)] = self.cell(tx, gx)

        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        isc = [0] * self.np
        for s, p in numpy.ndindex((self.ns, self.np)):
            isc[p] = max(cell_metrics[(t[s, p], g[s, p])].isc, isc[p])

        string_metrics = [None] * self.np
        for p in range(self.np):
            i = numpy.linspace(0, isc[p], 1000)
            v = numpy.zeros(i.shape)
            for s in range(self.ns):
                v += cell_metrics[(t[s, p], g[s, p])].vi(i)

            string_metrics[p] = _metrics(i, v)

        # All parallel strings in an array have equal voltage.
        # The open-circuit voltage of the array is of the greatest string.
        # Compute the array current by interpolating over the array voltage.
        voc = max([string.voc for string in string_metrics])
        v = numpy.linspace(0, voc, 1000)
        i = numpy.zeros(v.shape)
        for p in range(self.np):
            i += string_metrics[p].iv(v)

        return _metrics(i[::-1], v[::-1])
