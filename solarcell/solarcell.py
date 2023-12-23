from collections import namedtuple
from functools import lru_cache, partial, wraps
from inspect import Signature
from warnings import warn

import numpy as np
from scipy import constants
from scipy.optimize import least_squares, minimize_scalar, root_scalar


def _warn_solution(name, target, result, threshold=0.02):
    err = np.abs((result - target) / target)
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
    vi = partial(np.interp, xp=i, fp=v, left=np.nan, right=0)
    iv = partial(np.interp, xp=v[::-1], fp=i[::-1], left=np.inf, right=0)
    return m(
        isc=i[-1],  # Short-circuit current.
        voc=v[0],  # Open-circuit voltage.
        imp=i[np.argmax(i * v)],  # Max-power current.
        vmp=v[np.argmax(i * v)],  # Max-power voltage.
        pmp=np.max(i * v),  # Max-power power.
        vi=vi,  # Function, current to voltage.
        iv=iv,  # Function, voltage to current.
        pi=lambda i: i * vi(i),  # Function, current to power.
        pv=lambda v: v * iv(v),  # Function, voltage to power.
    )


def _round_cell_inputs(ttol, gtol):
    # The numerical fitting takes time. A cache is implemented to prevent
    # duplicate work. Rounding of temperature and intensity is performed
    # prior to accessing the cache to increase hit likelihood.
    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            binding = Signature.from_callable(fun).bind(*args, **kwargs)
            t = binding.arguments["t"]
            g = binding.arguments["g"]
            return fun(args[0], round(t / ttol) * ttol, round(g / gtol) * gtol)

        return wrapper

    return decorator


class solarcell:
    def __init__(self, isc, voc, imp, vmp, t):
        assert isc[0] > imp[0] and isc[1] > imp[1], "Isc must exceed Imp."
        assert voc[0] > vmp[0] and voc[1] > vmp[1], "Voc must exceed Vmp."
        assert t > -273.15, "Temperature must exceed absolute zero."
        self.isc = isc
        self.voc = voc
        self.imp = imp
        self.vmp = vmp
        self.t = t

    @_round_cell_inputs(ttol=0.1, gtol=0.01)
    @lru_cache(maxsize=128)
    def cell(self, t, g):
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert g >= 0, "Intensity must be non-negative."

        # Skip the curve fit altogether if the cell is dark.
        if g == 0:
            return _metrics(np.array([0]), np.array([0]))

        # Otherwise proceed with the curve fit to the following parameters.
        # Compute the adjusted cell parameters.
        dt = t - self.t
        isc = (self.isc[0] + dt * self.isc[1]) * g
        voc = self.voc[0] + dt * self.voc[1]
        imp = (self.imp[0] + dt * self.imp[1]) * g
        vmp = self.vmp[0] + dt * self.vmp[1]

        def x2eqn(x):
            i0, rs, n = x  # Parameters to be solved for numerically.
            i0 = i0 * 1e-20  # Scale factor to assist solver.

            def v(i):
                # Diode model: voltage is a logarithmic function of current.
                with np.errstate(invalid="ignore"):
                    q_kT = constants.e / (constants.k * (t + 273.15))
                    v = np.log((isc - i) / i0 + 1) / (q_kT / n) - i * rs
                    v = np.nan_to_num(v)  # Bypass diode.
                    v = np.where(i < 0, np.nan, v)  # Blocking diode.
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

        with np.errstate(all="ignore"):
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

        xi = np.linspace(0, xisc, 1000)
        xv = x2eqn(result.x)(xi)
        return _metrics(xi, xv)

    def string(self, t, g):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        isc = max([self.cell(*tg).isc for tg in zip(t, g)])
        i = np.linspace(0, isc, 1000)
        v = np.sum([self.cell(*tg).vi(i) for tg in zip(t, g)], 0)
        return _metrics(i, v)

    def array(self, t, g):
        # All parallel strings in an array have equal voltage.
        # The open-circuit voltage of the array is of the greatest string.
        # Compute the array current by interpolating over the array voltage.
        voc = np.max([self.string(*tg).voc for tg in zip(t.T, g.T)])
        v = np.linspace(0, voc, 1000)
        i = np.sum([self.string(*tg).iv(v) for tg in zip(t.T, g.T)], 0)
        return _metrics(i[::-1], v[::-1])
