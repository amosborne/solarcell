from functools import lru_cache, partial, wraps
from inspect import Signature

import numpy as np
from scipy import constants
from scipy.optimize import (
    brute,
    minimize,
    root_scalar,
    least_squares,
    minimize_scalar,
    dual_annealing,
    differential_evolution,
)
from scipy.special import lambertw

from warnings import warn


def _model(iph, i0, rs, rsh, n, t):
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

    def vi(i):
        result = least_squares(lambda v: i - iv(v), 0)
        assert result.success
        return np.squeeze(result.x)

    # compute the open-circuit voltage numerically
    voc = root_scalar(lambda v: iv(v), bracket=(0, iph * rsh))
    voc = voc.root if voc.converged else np.nan

    # compute the max-power voltage numerically
    vmp = minimize(
        lambda v: 1 / (v * iv(v)), voc / 2, bounds=[(0.01 * voc, 0.99 * voc)]
    )
    vmp = vmp.x[0] if vmp.success else np.nan

    return iv(0), voc, iv(vmp), vmp, iv, vi


def _fit(isc, voc, imp, vmp, area, t):
    def model(x):
        return _model(isc, 10 ** x[0], 10 ** x[1], 10 ** x[2], x[3], t)

    def errors(x):
        xisc, xvoc, ximp, xvmp, xiv, xvi = model(x)
        ierrors = np.array([xisc - isc, ximp - imp, xiv(voc)])
        verrors = np.array([xvoc - voc, xvmp - vmp, xvi(isc)])
        return *ierrors, *verrors

    # Perform a least-squares numeric fit to the diode model curve definition.
    # Try multiple loss functions and use that which produces the best result.
    lb = (-20 + np.log10(area), np.log10(0.5 / area), np.log10(1e3 / area), 0.5)
    ub = (-10 + np.log10(area), np.log10(1.3 / area), np.log10(1e7 / area), 3.5)
    x0 = (-15 + np.log10(area), np.log10(0.9 / area), np.log10(1e5 / area), 2.0)

    best = None
    for loss in ["linear", "soft_l1", "huber", "cauchy", "arctan"]:
        for f_scale in np.linspace(1, 2, 10):
            result = least_squares(
                errors, x0, bounds=(lb, ub), loss=loss, f_scale=f_scale, ftol=0.001
            )
            if result.success and (best is None or result.cost < best.cost):
                best = result
            if loss == "linear":
                break

    assert best is not None

    # Scale and shift the resulting curve to optimize for the critical points.
    xierr, xverr = np.reshape(errors(best.x), (2, 3))
    xisc, xvoc, ximp, xvmp, xiv, xvi = model(best.x)
    ipoly = np.poly1d(np.polyfit(x=(xisc, ximp, 0), y=(isc, imp, 0), deg=2))
    vpoly = np.poly1d(np.polyfit(x=(xvoc, xvmp, 0), y=(voc, vmp, 0), deg=2))
    iroot = np.vectorize(lambda i: (ipoly - i).roots[1])
    vroot = np.vectorize(lambda v: (vpoly - v).roots[1])

    yiv = lambda v: ipoly(xiv(vroot(v)))
    yvi = lambda i: vpoly(xvi(iroot(i)))

    yierr = np.vectorize(
        lambda dxi, yi: np.roots([ipoly[0], 2 * ipoly[0] * yi + ipoly[1], -dxi])[1],
        signature="(),()->()",
    )(xierr, [isc, imp, 0])
    yverr = np.vectorize(
        lambda dxv, yv: np.roots([vpoly[0], 2 * vpoly[0] * yv + vpoly[1], -dxv])[1],
        signature="(),()->()",
    )(xverr, [voc, vmp, 0])

    return yiv, yvi, yierr, yverr

def _transform():
    pass


class solarcurve:
    nsamp = 1000

    def __init__(self, isc, voc, imp, vmp, iv, vi, iunc, vunc):
        self.isc = isc  # A
        self.voc = voc  # V
        self.imp = imp  # A
        self.vmp = vmp  # V
        self.iv = iv  # A -> V
        self.vi = vi  # V -> A
        self.iunc = iunc  # A
        self.vunc = vunc  # V

    @property
    def pmp(self):
        return self.imp * self.vmp  # W

    @property
    def punc(self):
        unc = self.iunc * self.vmp, self.imp * self.vunc
        return np.sqrt(np.sum(np.square(unc)))  # W

    def pv(self, v):
        return v * self.iv(v)  # V -> W

    def pi(self, i):
        return i + self.vi(i)  # A -> W

    def __repr__(self):
        iL, iR = "{:0.4g}".format(self.isc).split(".")
        istr = " = {:" + str(len(iL)) + "." + str(len(iR)) + "f} A"
        isc = "Isc " + istr.format(self.isc)
        imp = "Imp " + istr.format(self.imp)
        iunc = "Iunc" + istr.format(self.iunc)

        vL, vR = "{:0.4g}".format(self.voc).split(".")
        vstr = " = {:" + str(len(vL)) + "." + str(len(vR)) + "f} V"
        voc = "Voc " + vstr.format(self.voc)
        vmp = "Vmp " + vstr.format(self.vmp)
        vunc = "Vunc" + vstr.format(self.vunc)

        pL, pR = "{:0.4g}".format(self.pmp).split(".")
        pstr = " = {:" + str(len(pL)) + "." + str(len(pR)) + "f} W"
        pmp = "Pmp " + pstr.format(self.pmp)
        punc = "Punc" + pstr.format(self.punc)

        fs1 = ", ".join([isc, voc])
        fs2 = ", ".join([imp, vmp, pmp])
        fs3 = ", ".join([iunc, vunc, punc])
        return "\n".join([fs1, fs2, fs3])


class solarcell(solarcurve):
    def __init__(self, isc, voc, imp, vmp, area, t=28):
        # isc: (A, A/K), short circuit current
        # voc: (V, V/K), open circuit voltage
        # imp: (A, A/K), max power point current
        # vmp: (V, V/K), max power point voltage
        # area: cm2, cell surface area
        # t: C, reference temperature

        assert isc[0] > imp[0], "Isc must exceed Imp."
        assert voc[0] > vmp[0], "Voc must exceed Vmp."

        # The original input parameters are saved for later extrapolation.
        self.isc = isc[0]
        self.voc = voc[0]
        self.imp = imp[0]
        self.vmp = vmp[0]
        self.area = area
        self.t = t

        self.disc = isc[1]
        self.dvoc = voc[1]
        self.dimp = imp[1]
        self.dvmp = vmp[1]

        # Numerically fit using the cell model at full illumination.
        self.iv, self.vi, self.ierr, self.verr = _fit(
            isc[0], voc[0], imp[0], vmp[0], area, t
        )

    @property
    def iunc(self):
        return np.sqrt(np.sum(self.ierr**2))

    @property
    def vunc(self):
        return np.sqrt(np.sum(self.verr**2))

    def cell(self, t, g):
        # Skip the curve fit altogether if the cell is dark.
        if g == 0:
            iv = lambda v: np.where(v < 0, np.inf, 0)
            vi = lambda i: np.where(i < 0, np.nan, 0)
            return solarcurve(0, 0, 0, 0, iv, vi, 0, 0)

        # Otherwise proceed with the curve fit to the following parameters.
        dt = t - self.t
        isc = (self.isc + dt * self.disc) * g
        voc = self.voc + dt * self.dvoc
        imp = (self.imp + dt * self.dimp) * g
        vmp = self.vmp + dt * self.dvmp

        # Error checking.
        assert g >= 0, "Intensity must be non-negative."
        assert isc > imp, "Isc must exceed Imp."
        assert voc > vmp, "Voc must exceed Vmp."

        # Scale and shift the underlying curve to optimize for the critical points.
        ipoly = np.poly1d(np.polyfit(x=(self.isc, self.imp, 0), y=(isc, imp, 0), deg=2))
        vpoly = np.poly1d(np.polyfit(x=(self.voc, self.vmp, 0), y=(voc, vmp, 0), deg=2))
        iroot = np.vectorize(lambda i: (ipoly - i).roots[1])
        vroot = np.vectorize(lambda v: (vpoly - v).roots[1])

        iv = lambda v: ipoly(self.iv(vroot(v)))
        vi = lambda i: vpoly(self.vi(iroot(i)))
        #ierr = self.ierr / (2 * ipoly[0] * np.array([isc, imp, 0]) + ipoly[1])
        #verr = self.verr / (2 * vpoly[0] * np.array([voc, vmp, 0]) + vpoly[1])

        ierr = np.vectorize(
            lambda di, i: np.roots([ipoly[0], 2 * ipoly[0] * i + ipoly[1], -di])[1],
            signature="(),()->()",
        )(self.ierr, [isc, imp, 0])
        verr = np.vectorize(
            lambda dv, v: np.roots([vpoly[0], 2 * vpoly[0] * v + vpoly[1], -dv])[1],
            signature="(),()->()",
        )(self.verr, [voc, vmp, 0])

        
        iunc = np.sqrt(np.sum(ierr**2))
        vunc = np.sqrt(np.sum(verr**2))

        return solarcurve(isc, voc, imp, vmp, iv, vi, iunc, vunc)

    def string(self, t, g):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        cells = [self.cell(*tg) for tg in zip(t, g)]
        isc = max([c.isc for c in cells])
        i = np.linspace(0, isc, self.nsamp)
        v = np.sum([c.vi(i) for c in cells])
        # print(i, v)
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
