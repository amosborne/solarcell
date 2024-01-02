from collections import namedtuple
from functools import lru_cache, partial, wraps
from inspect import Signature
from warnings import warn

import numpy as np
from scipy import constants
from scipy.optimize import least_squares, root_scalar
from scipy.special import lambertw


def _warn_solution(name, target, result, threshold=0.02):
    err = np.abs((result - target) / target)
    if err > threshold:
        wmsg1 = "{:s} fit error of {:0.1f}% exceeds {:0.1f}% threshold."
        wmsg1 = wmsg1.format(name, err * 100, threshold * 100)
        wmsg2 = "Target={:0.3f}, Result={:0.3f}, Delta={:0.3f}"
        wmsg2 = wmsg2.format(target, result, result - target)
        warn(" ".join([wmsg1, wmsg2]))


def _metrics(isc, voc, imp, vmp, iv, vi):
    m = namedtuple("_metrics", "isc voc imp vmp iv vi pmp pv pi")
    pmp = imp * vmp
    pv = lambda v: v * iv(v)
    pi = lambda i: i + vi(i)
    return m(isc, voc, imp, vmp, iv, vi, pmp, pv, pi)
    
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

def _model(iph, i0, rs, rsh, n, t):
    # See the technical reference: "Exact analytical solutions of the parameters
    # of real solar cells using Lambert W-function," Amit Jain, Avinashi Kapoor (2003).
    
    nvth = n * constants.k * (t + 273.15) / constants.e
    rtot = rs + rsh
    itot = iph + i0

    def iv(v):
        # equation #2 from the technical reference
        ex = rsh * (rs * itot + v) / nvth / rtot
        wf = rs * rsh * i0 * np.exp(ex) / nvth / rtot
        lm = np.real_if_close(lambertw(wf))
        return (rsh * itot - v) / rtot - lm * nvth / rs

    def vi(i):
        # equation #3 from the technical reference
        ex = rsh * (itot - i) / nvth
        wf = rsh * i0 * np.exp(ex) / nvth
        lm = np.real_if_close(lambertw(wf))
        return rsh * itot - rtot * i - lm * nvth

    def dpdi(i):
        # equation #10 from the technical reference
        ex = rsh * (itot - i) / nvth
        wf = rsh * i0 * np.exp(ex) / nvth
        lm = np.real_if_close(lambertw(wf))
        t1 = rsh * itot - rtot * i - lm * nvth
        t2 = i * (lm * rsh / (1 + lm) - rtot)
        return t1 + t2

    def dpdv(v):
        # equation #11 from the technical reference
        ex = rsh * (rs * itot + v) / nvth / rtot
        wf = rs * rsh * i0 * np.exp(ex) / nvth / rtot
        lm = np.real_if_close(lambertw(wf))
        t1 = (rsh * itot - v) / rtot - lm * nvth / rs
        t2 = v * (1 / rtot + lm * rsh / (1 + lm) / rtot / rs)
        return t1 - t2

    def root(f, bnd):
        try:
            result = root_scalar(f, bracket=(0, bnd))
        except ValueError:
            return np.nan
        else:
            return result.root if result.converged else np.nan

    imp = root(dpdi, isc:=iv(0))
    vmp = root(dpdv, voc:=vi(0))
    
    return isc, voc, imp, vmp, iv, vi

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

    def cell(self, t, g):
        assert t > -273.15, "Temperature must exceed absolute zero."
        assert g >= 0, "Intensity must be non-negative."

        # Skip the curve fit altogether if the cell is dark.
        if g == 0:
            null = lambda _: 0
            return _metrics(0,0,0,0,null,null)

        # Otherwise proceed with the curve fit to the following parameters.
        # Compute the adjusted cell parameters.
        dt = t - self.t
        isc = (self.isc[0] + dt * self.isc[1]) * g
        voc = self.voc[0] + dt * self.voc[1]
        imp = (self.imp[0] + dt * self.imp[1]) * g
        vmp = self.vmp[0] + dt * self.vmp[1]

        print("goal:", isc, voc, imp, vmp)

        def params(x):
            iph, irel0, rs, rsh, n = x
            with np.errstate(invalid="ignore"):
                return _model(iph, irel0 * 1e-20, rs, rsh, n, t)
            
        def params_relerror(x):
            xisc, xvoc, ximp, xvmp, _, _ = params(x)
            isc_relerror = (isc - xisc) / isc
            voc_relerror = (voc - xvoc) / voc
            imp_relerror = (imp - ximp) / imp
            vmp_relerror = (vmp - xvmp) / vmp
            return isc - xisc, voc - xvoc, imp - ximp, vmp - xvmp
            return isc_relerror, voc_relerror, imp_relerror, vmp_relerror

        rs0 = (voc - vmp) / imp
        rsh0 = vmp / (isc - imp)
        x0 = (isc, 1, rs0, rsh0, 2.5) # iph, irel0, rs, rsh, n
        lb = (isc * 0.95, 1e-3, 0, 10, 0.1)
        ub = (isc * 1.05, 1e3, 1, np.inf, 10)

        result = least_squares(params_relerror, x0, bounds=(lb, ub))
        print(result)
        assert result.success

        # Warn if the solution is a poor fit.
        xisc, xvoc, ximp, xvmp, _, _ = params(result.x)
        print("result:", xisc, xvoc, ximp, xvmp)
        _warn_solution("Isc", isc, xisc)
        _warn_solution("Voc", voc, xvoc)
        _warn_solution("Imp", imp, ximp)
        _warn_solution("Vmp", vmp, xvmp)

        return _metrics(*params(result.x))

    def string(self, t, g):
        # All series cells in a string have equal current.
        # The short-circuit current of the string is of the greatest cell.
        # Compute the string voltage by interpolating over string current.
        isc = max([self.cell(*tg).isc for tg in zip(t, g)])
        i = np.linspace(0, isc, 1000)
        v = np.sum([self.cell(*tg).vi(i) for tg in zip(t, g)], 0)
        
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
        v = np.linspace(0, voc, 1000)
        i = np.sum([self.string(*tg).iv(v) for tg in zip(t.T, g.T)], 0)

        isc = i[0]
        imp = i[np.argmax(i * v)]
        vmp = v[np.argmax(i * v)]
        
        iv = partial(np.interp, xp=v[::-1], fp=i[::-1], left=np.inf, right=0)
        vi = partial(np.interp, xp=i, fp=v, left=np.nan, right=0)
        
        return _metrics(isc, voc, imp, vmp, iv, vi)
