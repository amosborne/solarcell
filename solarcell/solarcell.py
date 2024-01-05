from functools import lru_cache, partial, wraps
from inspect import Signature
from itertools import starmap

import numpy as np
from scipy import constants
from scipy.optimize import least_squares, root_scalar, brute, minimize, direct, basinhopping
from scipy.special import lambertw


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
            t = binding.arguments["t"]
            g = binding.arguments["g"]
            return fun(args[0], round(t / ttol) * ttol, round(g / gtol) * gtol)

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
    voc = root_scalar(lambda v: iv(v), bracket=(0, itot * rsh))
    assert voc.converged
    voc = voc.root

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
    rss = 0.05  # the RSS error threshold at which the fit fails
    nsamp = 1000  # number of samples used to interpolate curves

    def __init__(self, isc, voc, imp, vmp, t):
        assert isc[0] > imp[0] and isc[1] >= imp[1], "Isc must exceed Imp."
        assert voc[0] > vmp[0] and voc[1] >= vmp[1], "Voc must exceed Vmp."
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
            iv = lambda v: np.where(v < 0, np.inf, 0)
            vi = lambda i: np.where(i < 0, np.nan, 0)
            return _metrics(0, 0, 0, 0, iv, vi)

        # Otherwise proceed with the curve fit to the following parameters.
        dt = t - self.t
        isc = (self.isc[0] + dt * self.isc[1]) * g
        voc = self.voc[0] + dt * self.voc[1]
        imp = (self.imp[0] + dt * self.imp[1]) * g
        vmp = self.vmp[0] + dt * self.vmp[1]
        target = (isc, voc, imp, vmp)

        def model(x):
            irelph, irel0, rsexp, rshexp, n = x
            return _model(
                iph=isc * (1 + 10 ** irelph),
                i0=isc * (10 ** irel0),
                rs=10 ** rsexp,
                rsh=10 ** rshexp,
                n=n,
                t=t,
                nsamp=self.nsamp,
            )
        
        def cost(x):
            errors = np.divide(np.subtract(target, model(x)[:4]), target)
            errors = np.nan_to_num(errors, nan=1e16, posinf=1e16, neginf=-1e16)
            return np.sqrt(np.sum(errors ** 2))

        rsexp0 = np.ceil(np.log10((voc - vmp) / imp))
        rshexp0 = np.floor(np.log10(vmp / (isc - imp)))

        lb = (-4, -32, rsexp0 - 2, rshexp0, 0.1)
        ub = (-1, -13, rsexp0, rshexp0 + 2, 10)
        ns = (5, 16, 5, 5, 16)
        ranges = [slice(lx, ux, complex(nx)) for lx, ux, nx in zip(lb, ub, ns)]

        x0 = brute(cost, ranges)

        print("Fit Targets:", _metrics(*target, None, None))
        print("Fit Results:", _metrics(*model(x0)))
        print("Fit RSS:", cost(x0))
        print("Fit UB:", ("{:6.2f} "*5).format(*ub))
        print("Fit X0:", ("{:6.2f} "*5).format(*x0))
        print("Fit LB:", ("{:6.2f} "*5).format(*lb))

        # rss = np.sqrt(2 * result.cost)
        # emsg = "Failed model fit. RSS error of {:0.1f}% exceeds {:0.1f}%."
        # assert rss < self.rss, emsg.format(rss * 100, self.rss * 100)

        return _metrics(*model(x0))

    def string(self, t, g):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        isc = max([self.cell(*tg).isc for tg in zip(t, g)])
        i = np.linspace(0, isc, self.nsamp)
        v = np.sum([self.cell(*tg).vi(i) for tg in zip(t, g)], 0)
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
        voc = np.max([self.string(*tg).voc for tg in zip(t.T, g.T)])
        v = np.linspace(0, voc, self.nsamp)
        i = np.sum([self.string(*tg).iv(v) for tg in zip(t.T, g.T)], 0)
        i[-1] = 0  # guarantee the (0, voc) point for interpolation

        isc = i[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]

        iv = partial(np.interp, xp=v, fp=i, left=np.inf, right=0)
        vi = partial(np.interp, xp=i[::-1], fp=v[::-1], left=np.nan, right=0)

        return _metrics(isc, voc, imp, vmp, iv, vi)
