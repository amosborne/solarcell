from functools import cache, partial

import numpy as np
from numpy.polynomial import Polynomial
from scipy import constants
from scipy.optimize import least_squares, minimize, root_scalar
from scipy.special import lambertw


def _model(iph, i0, rs, rsh, n, t, k):
    # See the technical reference: "Exact analytical solutions of the parameters
    # of real solar cells using Lambert W-function," Amit Jain, Avinashi Kapoor (2003).

    nvth = n * constants.k * (t + 273.15) / constants.e
    rtot = rs + rsh
    itot = iph + i0
    tol = 1e-10

    @np.errstate(over="ignore")
    def iv(v):
        # equation #2 from the technical reference
        ex = rsh * (rs * itot + v) / nvth / rtot
        wf = rs * rsh * i0 * np.exp(ex) / nvth / rtot
        lm = np.real_if_close(lambertw(wf, tol=tol))
        ret = (rsh * itot - v) / rtot - lm * nvth / rs
        return ret

    # compute the open-circuit voltage numerically
    voc = root_scalar(lambda v: iv(v), bracket=(0, iph * rsh), rtol=tol)
    voc = voc.root if voc.converged else np.nan

    # compute the max-power voltage numerically
    pinv = lambda v: 1 / (v * iv(v))
    vmp = minimize(pinv, voc / 2, bounds=[(0.01 * voc, 0.99 * voc)], tol=tol)
    vmp = vmp.x[0] if vmp.success else np.nan

    return iv(0), voc, iv(vmp), vmp, iv


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
        return i * self.vi(i)  # A -> W

    def __repr__(self):
        def fstring(quantities, unit):
            nL, nR = 0, 0
            for quantity in quantities:
                strL, strR = "{:#.4g}".format(quantity).split(".")
                nL, nR = max(len(strL), nL), max(len(strR), nR)

            return " = {:" + str(nL + nR + 1) + "." + str(nR) + "f} " + unit

        istr = fstring([self.isc, self.imp, self.iunc], "A")
        isc = "Isc " + istr.format(self.isc)
        imp = "Imp " + istr.format(self.imp)
        iunc = "Iunc" + istr.format(self.iunc)

        vstr = fstring([self.voc, self.vmp, self.vunc], "V")
        voc = "Voc " + vstr.format(self.voc)
        vmp = "Vmp " + vstr.format(self.vmp)
        vunc = "Vunc" + vstr.format(self.vunc)

        pstr = fstring([self.pmp, self.punc], "W")
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

        self._fit()  # Numerically fit using the cell model at full illumination.
        self.iv, self.vi, self.iunc, self.vunc = self._transform(self.isc, self.voc, self.imp, self.vmp)

    def _fit(self):
        # Perform a least-squares numeric fit to the diode model curve definition.
        # This is the underlying model which all other curves will be derived from.

        def model(x):
            return _model(self.isc, 10 ** x[0], 10 ** x[1], 10 ** x[2], x[3], self.t, self.nsamp)

        def relerrors(x):
            xisc, xvoc, ximp, xvmp, _ = model(x)
            ri = np.divide(np.subtract([xisc, ximp], [self.isc, self.imp]), [self.isc, self.imp])
            rv = np.divide(np.subtract([xvoc, xvmp], [self.voc, self.vmp]), [self.voc, self.vmp])
            return *ri, *rv

        lb = (-20 + np.log10(self.area), np.log10(0.5 / self.area), np.log10(1e3 / self.area), 0.5)
        ub = (-10 + np.log10(self.area), np.log10(1.3 / self.area), np.log10(1e7 / self.area), 3.5)
        x0 = (-15 + np.log10(self.area), np.log10(0.9 / self.area), np.log10(1e5 / self.area), 2.0)

        result = least_squares(relerrors, x0, bounds=(lb, ub))
        assert result.success
        self.xisc, self.xvoc, self.ximp, self.xvmp, self.xiv = model(result.x)

    def _transform(self, isc, voc, imp, vmp):
        # x is the underlying model to be transformed
        # y is the target parameters used to define the transformation
        xi = [self.xisc, self.ximp, 0]
        xv = [0, self.xvmp, self.xvoc]
        yi = [isc, imp, 0]
        yv = [0, vmp, voc]

        # compute the quadratic transformation
        iyx = Polynomial.fit(xi, yi, deg=2, domain=None)  # quadratic polynomial, xi -> yi
        vyx = Polynomial.fit(xv, yv, deg=2, domain=None)  # quadratic polynomial, xv -> yv
        vxy = Polynomial.fit(yv, xv, deg=2, domain=None)  # quadratic polynomial, yv -> xv
        yiv = np.vectorize(lambda v: iyx(self.xiv(vxy(v))))  # transformed equation, yv -> yi

        # generate the interpolation functions
        yv = np.linspace(0, voc, self.nsamp)
        yv = np.insert(yv, np.searchsorted(yv, vmp), vmp)
        yi = yiv(yv)
        yi[-1] = 0  # guarantee the (0, voc) point for interpolation
        yiv = partial(np.interp, xp=yv, fp=yi, left=np.inf, right=0)
        yvi = partial(np.interp, xp=yi[::-1], fp=yv[::-1], left=np.nan, right=0)

        # propagate the error
        eyi = [iyx(self.xisc) - iyx(self.isc), iyx(self.ximp) - iyx(self.imp)]
        eyv = [vyx(self.xvoc) - vyx(self.voc), vyx(self.xvmp) - vyx(self.vmp)]
        yiunc = np.sqrt(np.sum(np.square(eyi)))
        yvunc = np.sqrt(np.sum(np.square(eyv)))

        return yiv, yvi, yiunc, yvunc

    @cache
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

        # Transform the resulting curve to force onto the critical points.
        iv, vi, iunc, vunc = self._transform(isc, voc, imp, vmp)
        # TODO: after transformation, true imp/vmp must be relocated
        return solarcurve(isc, voc, imp, vmp, iv, vi, iunc, vunc)

    def string(self, t, g):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        cells = [self.cell(*tg) for tg in zip(t, g)]
        isc = max([c.isc for c in cells])
        imp = [c.imp for c in cells]
        i = np.linspace(0, isc, self.nsamp)
        i = np.insert(i, np.searchsorted(i, imp), imp)
        v = np.sum([c.vi(i) for c in cells], 0)
        v[-1] = 0  # guarantee the (isc, 0) point for interpolation
        
        voc = v[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]

        iv = partial(np.interp, xp=v[::-1], fp=i[::-1], left=np.inf, right=0)
        vi = partial(np.interp, xp=i, fp=v, left=np.nan, right=0)

        iunc = np.max([c.iunc for c in cells])
        vunc = np.sqrt(np.sum(np.square([c.vunc for c in cells])))

        return solarcurve(isc, voc, imp, vmp, iv, vi, iunc, vunc)

    def array(self, t, g):
        # All parallel strings in an array have equal voltage.
        # The open-circuit voltage of the array is of the greatest string.
        # Compute the array current by interpolating over the array voltage.

        strings = [self.string(*tg) for tg in zip(t.T, g.T)]
        voc = np.max([s.voc for s in strings])
        vmp = [s.vmp for s in strings]
        v = np.linspace(0, voc, self.nsamp)
        v = np.insert(v, np.searchsorted(v, vmp), vmp)
        i = np.sum([s.iv(v) for s in strings], 0)
        i[-1] = 0  # guarantee the (0, voc) point for interpolation

        isc = i[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]

        iv = partial(np.interp, xp=v, fp=i, left=np.inf, right=0)
        vi = partial(np.interp, xp=i[::-1], fp=v[::-1], left=np.nan, right=0)

        iunc = np.sqrt(np.sum(np.square([s.iunc for s in strings])))
        vunc = np.max([s.vunc for s in strings])

        return solarcurve(isc, voc, imp, vmp, iv, vi, iunc, vunc)
