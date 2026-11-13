#!/usr/bin/env python3
"""
restoration_analysis.py -- is chiral restoration visible in the meson spectra before the
mesons melt?

For a model (lam1, gamma, lam4, mq) and a chemical potential mu this computes

* the chiral transition: temperature T_chi and order.  First order: T_c where the broken
  and the restored solution have equal free energy (equal-area construction on
  sigma(m_q), meson_spectra.chiral_branches), and both spinodals.  Crossover: the
  temperature of the steepest drop of sigma(T).
* the lowest vector (rho) and axial (a1) quasinormal modes omega = M - i Gamma/2 as
  functions of T, followed continuously from low temperature on the stable branch.
* summary numbers: the a1-rho splitting at low T and just below T_chi, the widths
  Gamma/M of both at T_chi, and the temperature above which the two modes agree.

"Melting" has no sharp definition for these modes: Gamma/M of the rho grows smoothly
through 1 (see the vector table printed by ``--vector``).  The summary therefore quotes
Gamma_V/M_V at T_chi, to be compared with whatever criterion is preferred; for reference
the potential-well criterion of axial_melting_scan.py (T_V = 102.8 MeV at mu = 0,
mu_g = 440 MeV) corresponds to Gamma_V/M_V = 0.77.

Independent parameters: the chiral sector depends only on lam1, kappa = gamma^2/lam4 and
mq sqrt(lam4) (chi -> chi/s, lam3 -> s lam3, lam4 -> s^2 lam4 is a symmetry).  lam4 at
fixed kappa and mq sqrt(lam4) changes nothing but the size of chi, i.e. g5^2 chi^2 in the
axial equation scales like 1/lam4.  ``scan`` therefore takes the reduced parameters.

Usage
-----
    python restoration_analysis.py --model 7.438 -22.6 4.2 9        # one model, table + figure
    python restoration_analysis.py --vector                          # rho mode and peak vs T
    python restoration_analysis.py --scan                            # parameter scan (CSV in results/)
"""
import argparse
import os

import numpy as np

import meson_spectra as ms

T_LOW = 40.0          # MeV: "low temperature" reference point of the summaries
DEGENERATE = 0.05     # |omega_A - omega_V|/|omega_V| below this counts as degenerate


# ---------------------------------------------------------------------------
# Chiral transition
# ---------------------------------------------------------------------------
def _branches(model, T, mu):
    return ms.chiral_branches(T, mu, model)


def _stable_sigma(branches):
    om = [b['dOmega'] for b in branches]
    if not np.all(np.isfinite(om)):
        raise RuntimeError("free-energy difference between branches is not available")
    return branches[int(np.argmin(om))]['sigma']


def chiral_transition(model, mu=0.0, T_min=30.0, T_max=230.0, dT=2.0):
    """
    Returns a dict: order ('first' or 'crossover'), T_chi, for first order also the
    spinodals T_lower < T_chi < T_upper and the jump of sigma^(1/3); for a crossover the
    slope d sigma/dT at T_chi; and the coarse curve (T, sigma^(1/3) of the stable
    branch, number of solutions).
    """
    Ts = np.arange(T_min, T_max + 0.5 * dT, dT)
    curve = [_branches(model, T, mu) for T in Ts]
    n_sol = np.array([len(c) for c in curve])
    keep = n_sol > 0
    Ts, curve, n_sol = Ts[keep], [c for c, k in zip(curve, keep) if k], n_sol[keep]
    s13 = np.array([np.cbrt(_stable_sigma(c)) for c in curve])
    out = {'T': Ts, 'sigma13': s13, 'n_solutions': n_sol}

    # steepest drop on the coarse grid, then look there on a fine grid
    # steepest drop of sigma itself (sigma^(1/3) has an infinite slope where sigma -> 0)
    slope = np.gradient(s13**3, Ts)
    i0 = int(np.argmin(slope))
    lo, hi = Ts[max(i0 - 2, 0)], Ts[min(i0 + 2, len(Ts) - 1)]
    multi = n_sol > 1
    if multi.any():
        lo, hi = min(lo, Ts[multi][0] - dT), max(hi, Ts[multi][-1] + dT)
    fine = np.arange(lo, hi + 1e-9, 0.1)
    fine_br = [_branches(model, T, mu) for T in fine]
    n_fine = np.array([len(c) for c in fine_br])

    if (n_fine > 1).any():
        idx = np.nonzero(n_fine > 1)[0]
        # free energy of the broken (last) branch relative to the restored (first) one
        def gap(T):
            br = _branches(model, T, mu)
            return br[-1]['dOmega'] if len(br) > 1 else np.nan

        def bisect(T_a, T_b, test):
            # test(T) is True at T_a and False at T_b
            for _ in range(40):
                mid = 0.5 * (T_a + T_b)
                if test(mid):
                    T_a = mid
                else:
                    T_b = mid
                if T_b - T_a < 1e-3:
                    break
            return 0.5 * (T_a + T_b)

        g = np.array([fine_br[i][-1]['dOmega'] for i in idx])
        if not np.all(np.isfinite(g)):
            raise RuntimeError("free-energy difference between branches is not available on the fine grid")
        T_multi = fine[idx]
        T_lower = bisect(T_multi[0] - 0.1, T_multi[0], lambda T: len(_branches(model, T, mu)) < 2)
        T_upper = bisect(T_multi[-1], T_multi[-1] + 0.1, lambda T: len(_branches(model, T, mu)) > 1)
        if g[0] >= 0.0:
            T_c = T_lower
        elif g[-1] <= 0.0:
            T_c = T_upper
        else:
            positive = np.nonzero(g > 0.0)[0]
            k = int(positive[0])
            T_c = bisect(T_multi[k - 1], T_multi[k], lambda T: gap(T) < 0.0)   # NaN (branch lost): treated as not below
        br = _branches(model, T_c, mu)
        jump = np.cbrt(br[-1]['sigma']) - np.cbrt(br[0]['sigma']) if len(br) > 1 else np.nan
        out.update(order='first', T_chi=T_c, T_lower=T_lower, T_upper=T_upper, jump13=jump, slope=np.nan)
        return out

    s_fine = np.array([_stable_sigma(c) if c else np.nan for c in fine_br])
    sl = np.gradient(s_fine, fine)
    j = int(np.nanargmin(sl))
    T_chi = fine[j]
    if 0 < j < len(fine) - 1:      # parabola through the three points around the minimum of the slope
        y0, y1, y2 = sl[j - 1], sl[j], sl[j + 1]
        den = y0 - 2.0 * y1 + y2
        if den != 0.0:
            T_chi = fine[j] + 0.5 * 0.1 * (y0 - y2) / den
    out.update(order='crossover', T_chi=float(T_chi), T_lower=np.nan, T_upper=np.nan, jump13=np.nan,
               slope=float(sl[j]))        # d sigma/dT [MeV^2] at T_chi
    return out


# ---------------------------------------------------------------------------
# Modes versus temperature
# ---------------------------------------------------------------------------
def _nearest(modes, target):
    if not modes:
        return complex(np.nan, np.nan)
    return min(modes, key=lambda z: abs(z - target))


def _channels(model, T, mu, branch):
    bg = ms.Background(T, mu, model)
    return bg, bg.channel('vector'), bg.channel('axial', branch)


def ground_modes(model, mu, Ts, T_chi=None, order='crossover', n_axial=1):
    """
    Lowest vector mode and lowest axial mode(s) at the temperatures Ts (ascending).

    The modes are identified at Ts[0] by a full scan of the complex plane and then
    followed with Newton's method; whenever Newton fails or jumps, the plane is scanned
    again and the mode nearest to the previous one is taken.  For a first-order transition
    the broken branch is used below T_chi and the restored one above.
    Returns dict of arrays: T, vector, axial (n_axial columns), sigma13.
    """
    Ts = np.asarray(Ts, dtype=float)
    vec = np.full(len(Ts), np.nan + 0j)
    axi = np.full((len(Ts), n_axial), np.nan + 0j)
    s13 = np.full(len(Ts), np.nan)
    prev_v = None
    prev_a = [None] * n_axial
    for i, T in enumerate(Ts):
        if order == 'first':
            branch = 'broken' if T < T_chi else 'restored'
        else:
            branch = 'stable'
        try:
            bg, v, a = _channels(model, T, mu, branch)
        except RuntimeError:
            continue
        s13[i] = np.cbrt(a.sigma)

        def follow(ch, prev, k):
            if prev is not None and np.isfinite(prev.real):
                om = ch.find_mode(prev)
                if np.isfinite(om.real) and abs(om - prev) < 0.15 * abs(prev) and om.real > 0:
                    return om
            modes = [m for m in ch.scan_modes(3200.0, -1500.0, re_min=20.0) if m.imag < 1.0]
            if prev is not None and np.isfinite(prev.real):
                return _nearest(modes, prev)
            # first temperature: the k-th mode by real part among those that are not overdamped
            under = sorted([m for m in modes if abs(m.imag) < m.real], key=lambda z: z.real)
            return under[k] if len(under) > k else complex(np.nan, np.nan)

        vec[i] = follow(v, prev_v, 0)
        prev_v = vec[i]
        for k in range(n_axial):
            axi[i, k] = follow(a, prev_a[k], k)
            prev_a[k] = axi[i, k]
    return {'T': Ts, 'vector': vec, 'axial': axi, 'sigma13': s13}


def summarize(model, mu=0.0, detail=False):
    """
    One row of the parameter scan.  With detail=True also returns the mode curves on a
    2 MeV grid from T_LOW to T_chi + 30 MeV.
    """
    tr = chiral_transition(model, mu)
    T_chi = tr['T_chi']
    row = {'lam1': model.lam1, 'gamma': model.gamma, 'lam4': model.lam4, 'mq': model.mq,
           'kappa': model.kappa, 'mq_scaled': model.mq_scaled, 'mu': mu,
           'order': tr['order'], 'T_chi': T_chi, 'T_lower': tr['T_lower'], 'T_upper': tr['T_upper'],
           'jump13': tr['jump13'], 'dsigma_dT': tr['slope']}
    below = T_chi - (0.05 if tr['order'] == 'first' else 3.0)
    above = T_chi + (0.05 if tr['order'] == 'first' else 3.0)
    Ts = np.unique(np.concatenate([np.arange(T_LOW, max(T_chi + 30.0, T_LOW + 40.0) + 1e-9, 2.0),
                                   [below, above]]))
    modes = ground_modes(model, mu, Ts, T_chi, tr['order'])
    V, A = modes['vector'], modes['axial'][:, 0]
    dist = np.abs(A - V) / np.abs(V)

    def at(T):
        i = int(np.argmin(np.abs(Ts - T)))
        return V[i], A[i], modes['sigma13'][i]

    v0, a0, s0 = at(T_LOW)
    vb, ab, sb = at(below)
    va, aa, sa = at(above)
    row.update(sigma13_low=s0, sigma13_below=sb, sigma13_above=sa,
               MV_low=v0.real, MA_low=a0.real, split_low=(a0.real - v0.real) / v0.real,
               MV_below=vb.real, GV_below=-2 * vb.imag, MA_below=ab.real, GA_below=-2 * ab.imag,
               split_below=(ab.real - vb.real) / vb.real,
               GV_over_MV=-2 * vb.imag / vb.real, GA_over_MA=-2 * ab.imag / ab.real,
               dist_below=abs(ab - vb) / abs(vb), dist_above=abs(aa - va) / abs(va))
    # lowest temperature above which the two modes stay degenerate
    ok = np.isfinite(dist)
    deg = np.nan
    if ok.any():
        not_deg = np.nonzero(ok & (dist > DEGENERATE))[0]
        if len(not_deg) == 0:
            deg = Ts[ok][0]
        elif not_deg[-1] + 1 < len(Ts):
            deg = Ts[not_deg[-1] + 1]
    row['T_degenerate'] = deg
    if detail:
        return row, tr, modes
    return row


# ---------------------------------------------------------------------------
# Vector channel alone (depends on T/mu_g and mu/mu_g only)
# ---------------------------------------------------------------------------
def vector_table(mu=0.0, Ts=None, model=None):
    model = model or ms.Model()
    Ts = np.arange(30.0, 200.1, 5.0) if Ts is None else Ts
    rows = []
    prev = None
    for T in Ts:
        v = ms.Background(T, mu, model, solve_chiral=False).channel('vector')
        if prev is None:
            modes = [m for m in v.scan_modes(3200.0, -1500.0, re_min=20.0) if abs(m.imag) < m.real]
            om = min(modes, key=lambda z: z.real)
        else:
            om = v.find_mode(prev)
        prev = om
        peaks = v.find_peaks(100.0, 3000.0, rel_prominence=0.0)
        first = peaks[0] if peaks else None
        rows.append({'T': T, 'mu': mu, 'M': om.real, 'Gamma': -2 * om.imag, 'Gamma_over_M': -2 * om.imag / om.real,
                     'peak': first['omega'] if first else np.nan,
                     'peak_prominence': first['prominence'] / first['height'] if first else np.nan})
    return rows


# ---------------------------------------------------------------------------
# Parameter scan
# ---------------------------------------------------------------------------
def _scan_one(args):
    lam1, kappa, mq_scaled, lam4, mu = args
    model = ms.Model.from_reduced(lam1, kappa, mq_scaled, lam4)
    try:
        return summarize(model, mu)
    except Exception as exc:                       # a failed point is reported, never filled in
        return {'lam1': lam1, 'gamma': model.gamma, 'lam4': lam4, 'mq': model.mq, 'kappa': kappa,
                'mq_scaled': mq_scaled, 'mu': mu, 'error': f"{type(exc).__name__}: {exc}"}


def scan(points, n_jobs=8, out=None):
    """points: iterable of (lam1, kappa, mq_scaled, lam4, mu).  Returns a DataFrame."""
    import pandas as pd
    from joblib import Parallel, delayed
    os.environ['NUMBA_NUM_THREADS'] = '1'          # one thread per worker process
    rows = Parallel(n_jobs=n_jobs)(delayed(_scan_one)(p) for p in points)
    df = pd.DataFrame(rows)
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _print_model(model, mu):
    row, tr, modes = summarize(model, mu, detail=True)
    print(model, f"mu = {mu}")
    if tr['order'] == 'first':
        print(f"first-order transition at T_c = {tr['T_chi']:.2f} MeV (spinodals {tr['T_lower']:.2f}, "
              f"{tr['T_upper']:.2f}), jump of sigma^(1/3) = {tr['jump13']:.1f} MeV")
    else:
        print(f"crossover at T_chi = {tr['T_chi']:.2f} MeV, d sigma/dT = {tr['slope']:.3g} MeV^2")
    print("   T    sigma^1/3 |   M_V   Gamma_V |   M_A   Gamma_A | |dw|/|w_V|")
    for T, s, v, a in zip(modes['T'], modes['sigma13'], modes['vector'], modes['axial'][:, 0]):
        print(f"{T:7.2f} {s:8.1f} | {v.real:7.1f} {-2*v.imag:7.1f} | {a.real:7.1f} {-2*a.imag:7.1f} | "
              f"{abs(a - v)/abs(v):.3f}")
    for k in ('split_low', 'split_below', 'GV_over_MV', 'GA_over_MA', 'dist_below', 'dist_above', 'T_degenerate'):
        print(f"{k:>14s} = {row[k]:.3f}")
    return row, tr, modes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', nargs=4, type=float, metavar=('LAM1', 'GAMMA', 'LAM4', 'MQ'))
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--vector', action='store_true')
    args = parser.parse_args()
    if args.vector:
        print("   T      M    Gamma  Gamma/M  peak  prominence/height")
        for r in vector_table(args.mu):
            print(f"{r['T']:6.1f} {r['M']:7.1f} {r['Gamma']:7.1f} {r['Gamma_over_M']:6.3f} {r['peak']:7.1f} "
                  f"{r['peak_prominence']:.3f}")
    if args.model:
        lam1, gamma, lam4, mq = args.model
        _print_model(ms.Model(lam1=lam1, gamma=gamma, lam4=lam4, mq=mq), args.mu)
