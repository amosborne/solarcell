from functools import lru_cache, partial, wraps
from inspect import Signature

import numpy as np
from scipy import constants
from scipy.optimize import brute, minimize, root_scalar, least_squares
from scipy.special import lambertw

from warnings import warn

class _metrics:
    def __init__(self, isc, voc, imp, vmp, iv, vi):
        self.isc = isc
        self.voc = voc
        self.imp = imp
        self.vmp = vmp
        self.iv = iv
        self.vi = vi
        self.pmp = imp * vmp
        self.pv = lambda v: v * iv(v)
        self.pi = lambda i: i + vi(i)

    def __repr__(self):
        fs = "isc={:0.4g}, voc={:0.4g}, imp={:0.4g}, vmp={:0.4g}, pmp={:0.4g}"
        return fs.format(self.isc, self.voc, self.imp, self.vmp, self.pmp)


def _round_cell_inputs(ttol, gtol):
    # The numerical fitting takes time. A cache is implemented to prevent
    # duplicate work. Rounding of temperature and intensity is performed
    # prior to accessing the cache to increase hit likelihood.
    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            binding = Signature.from_callable(fun).bind(*args, **kwargs)
            binding.arguments["t"] = round(binding.arguments["t"] / ttol) * ttol
            binding.arguments["g"] = round(binding.arguments["g"] / gtol) * gtol
            return fun(*binding.args, **binding.kwargs)

        return wrapper

    return decorator


def _model(iph, i0, rs, rsh, n, t, nsamp):
    # See the technical reference: "Exact analytical solutions of the parameters
    # of real solar cells using Lambert W-function," Amit Jain, Avinashi Kapoor (2003).

    nvth = n * constants.k * (t + 273.15) / constants.e
    rtot = rs + rsh
    itot = iph + i0

    @np.errstate(over="ignore")
    def iv(v):
        # equation #2 from the technical reference
        ex = rsh * (rs * itot + v) / nvth / rtot
        wf = rs * rsh * i0 * np.exp(ex) / nvth / rtot
        lm = np.real_if_close(lambertw(wf))
        ret = (rsh * itot - v) / rtot - lm * nvth / rs
        return ret

    # compute the open-circuit voltage numerically
    voc = root_scalar(lambda v: iv(v), bracket=(0, iph * rsh))
    voc = voc.root if voc.converged else np.nan

    # initialize the sampled data to be interpolated
    vx = np.linspace(0, voc, nsamp)
    ix = iv(vx)
    ix[-1] = 0  # set current at voc to exactly 0
    isc = ix[0]

    # define the interpolation functions with boundary conditions
    iv = lambda v: np.interp(v, vx, ix, left=np.inf, right=0)
    vi = lambda i: np.interp(i, ix[::-1], vx[::-1], left=np.nan, right=0)

    # locate the max power point by interpolation
    imp = ix[np.argmax(vx * ix)]
    vmp = vx[np.argmax(vx * ix)]

    return isc, voc, imp, vmp, iv, vi


class solarcell:
    rss = 0.2  # the RSS error threshold at which the fit fails
    nsamp = 1000  # number of samples used to interpolate curves

    def __init__(self, jsc, voc, jmp, vmp, area, rs=1, t=28):
        # jsc: (mA/cm2, uA/cm2/K), short circuit current density
        # voc: (V, mV/K), open circuit voltage
        # jmp: (mA/cm2, uA/cm2/K), max power point current density
        # vmp: (V, mV/K), max power point voltage
        # rs: ohm*cm2, area-normalized series resistance
        # t: C, reference temperature
        
        assert jsc[0] > jmp[0], "Isc must exceed Imp."
        assert voc[0] > vmp[0], "Voc must exceed Vmp."
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert 0.5 <= rs <= 1.3, "Rs is outside the typical range of 0.5 <= rs <= 1.3 ohm*cm2."
        
        self.isc = (jsc[0] * 1e-3 * area, jsc[1] * 1e-6 * area) # (A, A/K)
        self.voc = (voc[0], voc[1] * 1e-3) # (V, V/K)
        self.imp = (jmp[0] * 1e-3 * area, jmp[1] * 1e-6 * area) # (A, A/K)
        self.vmp = (vmp[0], vmp[1] * 1e-3) # (V, V/K)
        self.rs = rs / area # ohm
        self.area = area # cm2
        self.t = t # C

    @_round_cell_inputs(ttol=0.1, gtol=0.01)
    @lru_cache(maxsize=128)
    def cell(self, t, g, wi=1, wv=1):
        # Skip the curve fit altogether if the cell is dark.
        if g == 0:
            iv = lambda v: np.where(v < 0, np.inf, 0)
            vi = lambda i: np.where(i < 0, np.nan, 0)
            return _metrics(0, 0, 0, 0, iv, vi)

        # Otherwise proceed with the curve fit to the following parameters.
        dt = t - self.t
        isc = (self.isc[0] + dt * self.isc[1]) * g
        voc = self.voc[0] + dt * self.voc[1]
        imp = (self.imp[0] + dt * self.imp[1]) * g
        vmp = self.vmp[0] + dt * self.vmp[1]

        # Error checking.
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert g >= 0, "Intensity must be non-negative."
        assert isc > imp, "Isc must exceed Imp."
        assert voc > vmp, "Voc must exceed Vmp."

        # Use least-squares numeric optimized to solve for (i0, rs, rsh, n).
        def model(x):
            return _model(isc, 10**x[0], 1 / self.area, 10**x[1], x[2], t, self.nsamp)

        def resids(x, rss=False):
            xisc, xvoc, ximp, xvmp, _, _ = model(x)
            actual = (xisc * wi, xvoc * wv, ximp * wi, xvmp * wv)
            target = (isc * wi, voc * wv, imp * wi, vmp * wv)

            if rss:
                print("Fit Targets:", _metrics(*target, None, None))
                print("Fit Results:", _metrics(*actual, None, None))
            
            errors = np.subtract(target, actual)
            errors = np.nan_to_num(errors, nan=1e16, posinf=1e16, neginf=-1e16)
            relerr = np.sqrt(np.sum(np.divide(errors, target) ** 2))
            return relerr if rss else errors

        cost = lambda x: np.sum(resids(x)**2)
        lb = (-24, np.log10(1e3 / self.area), 0.4)
        ub = (-10, np.log10(1e6 / self.area), 8.0)
        x = brute(cost, tuple(zip(lb, ub)), Ns=24, finish=None)

        # lb = (-24, 0.5 / self.area, np.log10(1e3 / self.area), 0.4)
        # ub = (-10, 1.3 / self.area, np.log10(1e6 / self.area), 8.0)
        # x0 = (-18, 1.0 / self.area, np.log10(1e4 / self.area), 2.0)
        # result = least_squares(resids, x0, bounds=(lb, ub))
        # print(result)
        # assert result.success

        
        print("Fit Solution:", x)

        rss = resids(x, rss=True)
        print("Fit RSS:", rss)
        
        emsg = "Failed model fit. RSS error of {:0.1f}% exceeds {:0.1f}%."
        assert rss < self.rss, emsg.format(rss * 100, self.rss * 100)

        return _metrics(*model(x))

    def string(self, t, g, wi):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        wv = t.shape[0]
        isc = max([self.cell(*tg, wi, wv).isc for tg in zip(t, g)])
        i = np.linspace(0, isc, self.nsamp)
        v = np.sum([self.cell(*tg, wi, wv).vi(i) for tg in zip(t, g)], 0)
        v[-1] = 0  # guarantee the (isc, 0) point for interpolation

        voc = v[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]

        iv = partial(np.interp, xp=v[::-1], fp=i[::-1], left=np.inf, right=0)
        vi = partial(np.interp, xp=i, fp=v, left=np.nan, right=0)

        return _metrics(isc, voc, imp, vmp, iv, vi)

    def array(self, t, g):
        # All parallel strings in an array have equal voltage.
        # The open-circuit voltage of the array is of the greatest string.
        # Compute the array current by interpolating over the array voltage.
        wi = t.shape[1]
        voc = np.max([self.string(*tg, wi).voc for tg in zip(t.T, g.T)])
        v = np.linspace(0, voc, self.nsamp)
        i = np.sum([self.string(*tg, wi).iv(v) for tg in zip(t.T, g.T)], 0)
        i[-1] = 0  # guarantee the (0, voc) point for interpolation

        isc = i[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]

        iv = partial(np.interp, xp=v, fp=i, left=np.inf, right=0)
        vi = partial(np.interp, xp=i[::-1], fp=v[::-1], left=np.nan, right=0)

        return _metrics(isc, voc, imp, vmp, iv, vi)
