"""
meson_spectra.py -- vector and axial-vector mesons on the chiral background: spectral
functions on the real axis and quasinormal modes in the complex plane, both from the
same analytic function M(omega) (meson_core.mode_function).

    model = Model(lam1=7.438, gamma=-22.6, lam4=4.2, mq=9.0)          # mu_g = 440 MeV
    bg    = Background(T, mu, model)            # solves the chiral field (all branches)
    ax    = bg.channel('axial')                 # or 'vector'; branch='broken'/'restored'/'stable'
    ax.M(omega)                                 # complex omega in MeV
    ax.rho(omegas)                              # |Im(B/A)|, as axial_spectra.BA
    modes = ax.scan_modes(re_max=2500, im_min=-400)       # every mode in the rectangle
    peaks = ax.find_peaks(300, 2500)                      # refined peak positions and widths

Why M(omega)
------------
* M is analytic and its zeros are the quasinormal modes: a mode with omega = m - i Gamma/2
  is a state of mass m and width Gamma.  A narrow state cannot be missed in the complex
  plane the way it can on a real-omega grid: ``scan_modes`` counts the zeros in every cell
  of a grid by the argument principle and then polishes each with Newton's method.
* On the real axis |Im(B/A)| = 2 w e^{-b}/|M|^2, so the spectral function needs no
  numerical Wronskian (PROJECT_ASSESSMENT.md section 3.4b) and uses a horizon boundary
  condition of 10th order instead of leading order.
* At low temperature the widths are ~ e^{-2b} and far below double precision.  The mode
  search still returns the masses (the normal modes); the widths are then reported as
  unresolved when |Im omega| < WIDTH_FLOOR * |omega|.

Parameters of the chiral potential (see ``Model``): the chiral sector is invariant under
chi -> chi/s, lam3 -> s lam3, lam4 -> s^2 lam4, so the phase structure depends only on
lam1, gamma^2/lam4 and mq sqrt(lam4).  lam4 by itself only rescales chi, i.e. the strength
g5^2 chi^2 with which the axial channel feels the condensate.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import chiral_solve_complete as cs
import meson_core as mc

G5SQ = (2.0 * np.pi)**2

SETTINGS = {
    'z_uv': 1e-3,            # mu_g z of the UV starting point of psi_1
    'grid_points': 4001,     # u grid on which chi is stored (cubic Hermite in between)
    'order': 10,             # order of the infalling Frobenius series
    'series_tol': 1e-13,
    'rtol': 1e-10,
    'atol': 1e-200,
}
LAM4_SEARCH = 4.2           # lam4 at which the chiral root search is done (see chiral_branches)
WIDTH_FLOOR = 1e-8           # |Im omega|/|omega| below this is not resolved in double precision


@dataclass(frozen=True)
class Model:
    lam1: float = 7.438
    gamma: float = -22.6
    lam4: float = 4.2
    mq: float = 9.0
    mu_g: float = 440.0

    @property
    def lam3(self):
        return self.gamma / (6.0 * np.sqrt(2.0))

    @property
    def kappa(self):
        """gamma^2/lam4: the only combination of gamma and lam4 the phase structure sees."""
        return self.gamma**2 / self.lam4

    @property
    def mq_scaled(self):
        """mq sqrt(lam4): the quark mass in units in which chi is canonically scaled."""
        return self.mq * np.sqrt(self.lam4)

    @staticmethod
    def from_reduced(lam1, kappa, mq_scaled, lam4, mu_g=440.0):
        """Model from the independent combinations (lam1, gamma^2/lam4, mq sqrt(lam4)) and lam4."""
        return Model(lam1=lam1, gamma=-np.sqrt(kappa * lam4), lam4=lam4,
                     mq=mq_scaled / np.sqrt(lam4), mu_g=mu_g)


def chiral_branches(T, mu, model, d0_upper=None):
    """
    All solutions of the chiral equation with the model's quark mass: a list of dicts
    (d0, sigma), ordered by d0 (restored-like first, broken last), and the free-energy
    difference of each relative to the first, in units where dOmega/dm_q = -sigma
    (equal-area construction along the multivalued sigma(m_q) curve).
    """
    # The root search runs on the equivalent problem with lam4 = LAM4_SEARCH (chi -> chi/s,
    # lam3 -> s lam3, lam4 -> s^2 lam4, mq -> mq/s is a symmetry), where the horizon values
    # have the size the d0 scan of chiral_solve_complete was tuned for; d0 scales back by s.
    # sigma_of_T works at mu_g = cs.MU_G; another mu_g is the same problem in rescaled units.
    s = np.sqrt(LAM4_SEARCH / model.lam4)
    scale = cs.MU_G / model.mu_g
    _, _, _, d0s = cs.sigma_of_T(model.mq / s * scale, 0.0, T * scale, mu * scale, model.lam1, 0.0,
                                 10.0 if d0_upper is None else d0_upper / s, 0.0, 1.0,
                                 s * model.lam3, LAM4_SEARCH)
    d0s = [s * d0 for d0 in d0s]
    out = []
    for d0 in d0s:
        r = cs.chiral_shoot(d0, model.lam1, T, mu, model.lam3, model.lam4, mu_g=model.mu_g)
        if r['ok']:
            out.append({'d0': d0, 'sigma': r['sigma'], 'dOmega': 0.0})
    for i in range(1, len(out)):
        d = np.linspace(out[0]['d0'], out[i]['d0'], 2001)
        mq, sig = cs.chiral_scan(d, model.lam1, T, mu, model.lam3, model.lam4, mu_g=model.mu_g)
        if np.all(np.isfinite(mq)) and np.all(np.isfinite(sig)):
            out[i]['dOmega'] = -np.trapz(sig, mq)
        else:
            out[i]['dOmega'] = np.nan
    return out


class Background:
    """Black-hole background at (T, mu) and the chiral solutions on it."""

    def __init__(self, T, mu, model=None, solve_chiral=True, d0=None):
        self.T = float(T)
        self.mu = float(mu)
        self.model = model if model is not None else Model()
        self.b, self.Q2, self.zh_hat = cs.horizon_hat(self.T, self.mu, self.model.mu_g)
        self.branches = []
        if d0 is not None:
            r = cs.chiral_shoot(d0, self.model.lam1, T, mu, self.model.lam3, self.model.lam4,
                                mu_g=self.model.mu_g)
            self.branches = [{'d0': float(d0), 'sigma': r['sigma'], 'dOmega': 0.0}]
        elif solve_chiral:
            self.branches = chiral_branches(T, mu, self.model)

    def pick(self, branch='stable'):
        """Index of a branch: 'stable' (lowest free energy), 'broken' (largest sigma),
        'restored' (smallest sigma), or an integer."""
        if not self.branches:
            raise RuntimeError(f"no chiral solution at T={self.T}, mu={self.mu}, {self.model}")
        if isinstance(branch, (int, np.integer)):
            return int(branch)
        sig = np.array([br['sigma'] for br in self.branches])
        if branch == 'broken':
            return int(np.argmax(sig))
        if branch == 'restored':
            return int(np.argmin(sig))
        if branch == 'stable':
            om = np.array([br['dOmega'] for br in self.branches])
            if np.any(~np.isfinite(om)):
                raise RuntimeError("free-energy difference between branches is not available")
            return int(np.argmin(om))
        raise ValueError(branch)

    def channel(self, kind='axial', branch='stable'):
        if kind == 'vector':
            return Channel(self, None)
        return Channel(self, self.branches[self.pick(branch)])


class Channel:
    """One meson channel on one chiral branch.  All frequencies in MeV."""

    def __init__(self, bg, branch):
        self.bg = bg
        self.branch = branch
        m = bg.model
        s = SETTINGS
        self.kind = 'vector' if branch is None else 'axial'
        n = s['grid_points']
        self.u_grid = np.linspace(s['z_uv'] / bg.zh_hat, 1.0, n)
        self.cu0 = self.u_grid[0]
        self.cdu = self.u_grid[1] - self.u_grid[0]
        if branch is None:
            self.g5sq = 0.0
            self.a_src = 0.0
            self.d_chi = np.zeros(cs.SOLVER_SETTINGS['horizon_order'] + 1)
            self.chi_eps = cs.SOLVER_SETTINGS['horizon_eps_max']
            self.chi = np.zeros(n)
            self.chip = np.zeros(n)
            self.sigma = 0.0
        else:
            self.g5sq = G5SQ
            r = cs.chiral_shoot(branch['d0'], m.lam1, bg.T, bg.mu, m.lam3, m.lam4,
                                u_grid=self.u_grid, mu_g=m.mu_g)
            if not r['ok']:
                raise RuntimeError(f"chiral field failed on the grid at T={bg.T}, d0={branch['d0']}")
            self.a_src = r['a']
            self.chi, self.chip = r['chi'], r['chip']
            self.sigma = r['sigma']
            self.d_chi = cs.horizon_coefficients(branch['d0'], bg.b, bg.Q2, m.lam1, m.lam3, m.lam4,
                                                 cs.SOLVER_SETTINGS['horizon_order'])
            self.chi_eps = r['horizon_eps']
        # matching point: close enough to the horizon that the infalling solution keeps its
        # exponentially small admixture (see module docstring of meson_core)
        self.u_m = min(0.94, max(0.5, np.sqrt(max(0.0, 1.0 - 12.0 / bg.b))))
        self.u_m = min(self.u_m, 1.0 - 1.2 * self.chi_eps)

    # -- the analytic function -------------------------------------------------
    def _args(self, u_m=None):
        s = SETTINGS
        return (self.bg.b, self.bg.Q2, self.g5sq, self.a_src, self.d_chi, self.chi_eps,
                self.cu0, self.cdu, self.chi, self.chip, s['z_uv'],
                self.u_m if u_m is None else u_m, s['order'], s['series_tol'], s['rtol'], s['atol'])

    def _w(self, omega):
        return np.asarray(omega, dtype=np.complex128) * (self.bg.zh_hat / self.bg.model.mu_g)

    def M(self, omega, u_m=None):
        """Mode function at complex omega [MeV] (scalar or array)."""
        w = self._w(omega)
        if w.ndim == 0:
            return mc.mode_function(complex(w), *self._args(u_m))
        return mc.mode_function_array(np.ascontiguousarray(w.ravel()), *self._args(u_m)).reshape(w.shape)

    def rho(self, omega):
        """|Im(B/A)| at real omega [MeV], in the normalization of axial_spectra.BA (psi_1 = u^2)."""
        w = self._w(omega)
        M = self.M(omega)
        pf = np.ones(w.shape)
        r = -1j * w / (2.0 * (2.0 - self.bg.Q2))
        for k in range(1, SETTINGS['order'] + 1):
            pf = pf * np.abs(1.0 + 2.0 * r / k)**2
        return 2.0 * w.real * np.exp(-self.bg.b) * pf / np.abs(M)**2

    def rho_normalized(self, omega):
        """rho/(omega z_h)^2: dimensionless, tends to a constant (pi/2) at large omega for every T,
        so heights can be compared between temperatures and channels."""
        w = self._w(omega).real
        return self.rho(omega) / w**2

    # -- quasinormal modes -----------------------------------------------------
    def find_mode(self, guess, tol=1e-10, max_iter=60):
        """Newton iteration on M from ``guess``; returns complex omega or nan."""
        omega = complex(guess)
        for _ in range(max_iter):
            h = 1e-6 * max(1.0, abs(omega))
            vals = self.M(np.array([omega, omega + h, omega - h]))
            value, slope = vals[0], (vals[1] - vals[2]) / (2.0 * h)
            if not np.isfinite(abs(value)) or not np.isfinite(abs(slope)) or slope == 0.0:
                return complex(np.nan, np.nan)
            step = -value / slope
            limit = 0.25 * max(50.0, abs(omega))
            if abs(step) > limit:
                step *= limit / abs(step)
            omega += step
            if abs(step) < tol * max(1.0, abs(omega)):
                return omega
        return complex(np.nan, np.nan)

    def scan_modes(self, re_max, im_min, re_min=50.0, im_max=30.0, spacing=None, merge=1e-4):
        """
        Every mode in the rectangle [re_min, re_max] x [im_min, im_max] of the complex omega
        plane [MeV], by the argument principle on a grid: the winding number of M around a
        cell is the number of zeros inside (M has no poles there).  Returns a list of complex
        omega sorted by real part.  ``spacing`` (MeV) defaults to a value that resolves the
        phase of M; run with half the spacing to check (tests do).
        """
        if spacing is None:
            spacing = min(25.0, 0.35 * np.pi * self.bg.model.mu_g / self.bg.zh_hat)
        re = np.arange(re_min, re_max + spacing, spacing)
        im = np.arange(im_min, im_max + spacing, spacing)
        grid = re[:, None] + 1j * im[None, :]
        vals = self.M(grid)
        phase = np.angle(vals)

        def wrap(d):
            return (d + np.pi) % (2.0 * np.pi) - np.pi
        bottom = wrap(phase[1:, :-1] - phase[:-1, :-1])
        right = wrap(phase[1:, 1:] - phase[1:, :-1])
        top = wrap(phase[:-1, 1:] - phase[1:, 1:])
        left = wrap(phase[:-1, :-1] - phase[:-1, 1:])
        total = bottom + right + top + left
        winding = np.rint(total / (2.0 * np.pi)).astype(int)
        winding[~np.isfinite(total)] = 0
        modes = []
        for i, j in zip(*np.nonzero(winding)):
            # start Newton from the corner with the smallest |M|
            corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            ci, cj = min(corners, key=lambda c: abs(vals[c]))
            om = self.find_mode(grid[ci, cj])
            if not np.isfinite(om.real):
                om = self.find_mode(0.25 * sum(grid[c] for c in corners))
            if not np.isfinite(om.real):
                continue
            if not (re[i] - spacing <= om.real <= re[i + 1] + spacing and
                    im[j] - spacing <= om.imag <= im[j + 1] + spacing):
                continue
            if all(abs(om - other) > merge * max(1.0, abs(om)) for other in modes):
                modes.append(om)
        return sorted(modes, key=lambda z: z.real)

    # -- spectral peaks --------------------------------------------------------
    def find_peaks(self, w_min, w_max, coarse=5.0, rel_prominence=0.02, tol=1e-3):
        """
        Peaks of the normalized spectral function rho/(omega z_h)^2 between w_min and w_max
        [MeV] (see ``refine_peaks``).  Each peak also carries the quasinormal mode found by
        Newton's method from (omega - i fwhm/2).
        """
        ws = np.arange(w_min, w_max + coarse, coarse)
        y = self.rho_normalized(ws)
        peaks = refine_peaks(ws, y, lambda w: float(self.rho_normalized(w)), rel_prominence, tol)
        for p in peaks:
            guess = complex(p['omega'], -0.5 * p['fwhm'] if np.isfinite(p['fwhm']) else -1e-3 * p['omega'])
            p['mode'] = self.find_mode(guess)
        return peaks


def refine_peaks(ws, y, func, rel_prominence=0.02, tol=1e-3):
    """
    Peaks of a spectrum sampled as y on the ascending grid ws, with ``func(w)`` giving the
    same quantity at any w.

    Each local maximum of the samples is located by a bounded scalar maximization of func
    between its neighbours (to ``tol``), so the position does not depend on the grid and
    no global refinement of the grid is needed.  A maximum counts as a peak if its
    prominence (height above the higher of the two minima separating it from higher
    ground, or from the ends of the window) is at least ``rel_prominence`` times its
    height: a relative criterion, usable for both channels and at every temperature when
    y is the spectrum divided by omega^2.  Returns a list of dicts with omega, height,
    prominence and fwhm (nan if the half maximum is not reached on one side).
    """
    ws = np.asarray(ws, dtype=float)
    y = np.asarray(y, dtype=float)
    peaks = []
    for i in range(1, len(ws) - 1):
        if not (y[i] > y[i - 1] and y[i] >= y[i + 1]):
            continue
        res = minimize_scalar(lambda w: -func(w), bounds=(ws[i - 1], ws[i + 1]),
                              method='bounded', options={'xatol': tol})
        w0, h0 = float(res.x), float(-res.fun)
        if h0 < y[i]:
            w0, h0 = ws[i], y[i]
        base = []
        for step in (-1, 1):
            lowest = h0
            k = i + step
            while 0 <= k < len(ws) and y[k] <= h0:
                lowest = min(lowest, y[k])
                k += step
            base.append(lowest)
        prominence = h0 - max(base)
        if prominence < rel_prominence * h0:
            continue
        half = 0.5 * h0
        crossings = []
        for step in (-1, 1):
            inner, k = w0, i + step
            while 0 <= k < len(ws) and half < y[k] <= h0:
                inner = ws[k]
                k += step
            if not (0 <= k < len(ws)) or y[k] > half:
                crossings.append(np.nan)
                continue
            lo, hi = (ws[k], inner) if step < 0 else (inner, ws[k])
            try:
                crossings.append(brentq(lambda w: func(w) - half, lo, hi, xtol=1e-2 * tol))
            except ValueError:
                crossings.append(np.nan)
        peaks.append({'omega': w0, 'height': h0, 'prominence': prominence,
                      'fwhm': crossings[1] - crossings[0]})
    return peaks


def stable_branch_index(d0s, T, mu, model):
    """
    Index of the solution with the lowest free energy among horizon values d0s (ascending),
    from dOmega = -int sigma dm_q along the curve of solutions between them.
    """
    d0s = np.asarray(d0s, dtype=float)
    if len(d0s) == 1:
        return 0
    order = np.argsort(d0s)
    omega = np.zeros(len(d0s))
    for j in range(1, len(d0s)):
        d = np.linspace(d0s[order[0]], d0s[order[j]], 2001)
        mq, sig = cs.chiral_scan(d, model.lam1, T, mu, model.lam3, model.lam4, mu_g=model.mu_g)
        if not (np.all(np.isfinite(mq)) and np.all(np.isfinite(sig))):
            raise RuntimeError("free-energy difference between branches is not available")
        omega[order[j]] = -np.trapz(sig, mq)
    return int(np.argmin(omega))


def width_resolved(omega):
    """False if the imaginary part of a mode is below what double precision resolves."""
    return abs(omega.imag) > WIDTH_FLOOR * abs(omega)


def track_mode(channels, start, max_jump=0.2):
    """
    Follow one mode through a sequence of channels (e.g. increasing T) from ``start``,
    using each result as the next guess.  Stops (nan from there on) when Newton fails or
    the mode jumps by more than ``max_jump`` of its modulus, which signals that it was
    lost to another mode.
    """
    out = []
    guess = complex(start)
    alive = True
    for ch in channels:
        if not alive:
            out.append(complex(np.nan, np.nan))
            continue
        om = ch.find_mode(guess)
        if not np.isfinite(om.real) or abs(om - guess) > max_jump * abs(guess) or om.real <= 0:
            alive = False
            out.append(complex(np.nan, np.nan))
            continue
        out.append(om)
        guess = om
    return np.array(out)
