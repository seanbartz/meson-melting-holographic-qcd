#!/usr/bin/env python3
"""Does the coordinate/cutoff scheme explain the low-temperature trouble?

Background
----------
The code integrates the chiral EOM in u = z/z_h over [u_i, u_f] with u_i and
u_f held fixed. Since z_h = 1/(pi T) at mu = 0, a fixed u_i means the *physical*
UV cutoff z_i = u_i*z_h drifts like 1/T, and a fixed horizon gap 1-u_f means a
physical gap that also drifts. This script measures what that costs, and
separates it from the other two low-T error sources.

It does not need the equations rewritten in z. Within one solve z_h is a
constant, so u = z/z_h and v = mu_g*z differ by a constant factor and give
identical solutions; the only physical content of the coordinate choice is
where the two cutoffs sit. That is set by `physical_cutoffs` in
chiral_solve_complete.py.

Sections
--------
  (a) integration tolerance, at fixed cutoffs
  (b) UV cutoff z_i, at tight tolerance
  (c) IR cutoff 1-u_f, at tight tolerance
  (d) conditioning of the shooting map d0 -> mq (coordinate independent)
  (e) end-to-end sigma(T), legacy scheme vs physical scheme

Usage
-----
  python diagnostics/cutoff_scheme_study.py            # sections a-d (fast)
  python diagnostics/cutoff_scheme_study.py --full     # all sections
  python diagnostics/cutoff_scheme_study.py --sweep    # sigma(T) both ways -> csv + png
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chiral_solve_complete import (  # noqa: E402
    MU_G, blackness, chiral, calculate_sigma_values, physical_cutoffs, v3, v4,
)

LAMBDA1 = 7.438
MU = 0.0
MQ = 9.0
TIGHT = dict(method='DOP853', rtol=1e-13, atol=1e-18)
TEMPS = [120.0, 60.0, 30.0, 17.0]

ZETA = np.sqrt(3)/(2*np.pi)
cube = lambda x: np.sign(x)*abs(x)**(1/3)  # noqa: E731


def horizon_bc(d0, lambda1, T, mu, uf):
    """Second-order Taylor expansion of chi about the horizon (as in the solver)."""
    zh, q = blackness(T, mu)
    Q = q*zh**3
    l3, l4 = v3, v4
    d1 = (3*d0 - 3*d0**2*l3 - 4*d0**3*l4 + d0*zh**2*lambda1*MU_G**2)/(2*(-2 + Q**2))
    d2 = (1/(16*(-2 + Q**2)**2))*(
        6*d1*(-6 + Q**2 + Q**4)
        + 4*d0**3*(14 - 13*Q**2)*l4
        + d0**2*((42 - 39*Q**2)*l3 - 24*d1*(-2 + Q**2)*l4)
        - 2*d1*(-2 + Q**2)*zh**2*(-8 + 4*Q**2 - lambda1)*MU_G**2
        + 3*d0*(-14 + 13*Q**2 + 8*d1*l3 - 4*d1*Q**2*l3
                + (-2 + 3*Q**2)*zh**2*lambda1*MU_G**2))
    return d1, d2


def integrate(d0, T, uf, u_end, **kw):
    """Integrate from the horizon to u_end; returns a dense solution and z_h."""
    opts = dict(TIGHT)
    opts.update(kw)
    zh, q = blackness(T, MU)
    d1, d2 = horizon_bc(d0, LAMBDA1, T, MU, uf)
    y0 = [d0 + d1*(1 - uf) + d2*(1 - uf)**2, -d1 - 2*d2*(1 - uf)]
    sol = solve_ivp(chiral, [uf, u_end], y0,
                    args=((v3, v4, LAMBDA1, MU_G, 0.0, zh, q),),
                    dense_output=True, **opts)
    return sol, zh


def extract(chi0, chip0, u, zh):
    """mq and sigma from the leading UV behaviour (chiral_solve_complete.py)."""
    s = 1 - 3*v3*(3*chi0 - chip0*u)
    sq = np.sqrt(s)
    mq = (1 - sq)/(3*v3*ZETA*u*zh)
    sigma = ZETA*(1 - 6*chi0*v3 + 3*chip0*u*v3 - sq)/(3*u**3*zh**3*v3)
    return mq, sigma


def probe(d0, T, uf, u_end, **kw):
    sol, zh = integrate(d0, T, uf, u_end, **kw)
    if sol.status != 0:
        return np.nan, np.nan
    m, s = extract(*sol.sol(u_end), u_end, zh)
    return (m, s) if np.isfinite(m) else (np.inf, np.nan)


def physical_d0(T, uf, u_end, target=MQ):
    """Bisect for the d0 that gives mq = target. The map is monotone up to the
    runaway, which is treated as +inf."""
    lo, hi = 1e-4, 12.0
    for _ in range(200):
        mid = 0.5*(lo + hi)
        m, _ = probe(mid, T, uf, u_end)
        if not np.isfinite(m) or abs(m) > 1e8:
            m = np.inf
        if m < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-16*max(1.0, hi):
            break
    return lo


def section_a():
    print("\n(a) INTEGRATION TOLERANCE, at the legacy cutoffs u_i=1e-2, u_f=1-1e-4")
    print("    d0 is the physical one for mq = 9 MeV; a correct solver must return 9.")
    u_legacy = 1e-2 + 10*(1 - 1e-4 - 1e-2)/9999   # the old extract_index = 10
    for T in TEMPS:
        zh, _ = blackness(T, MU)
        d0 = physical_d0(T, 1 - 1e-4, u_legacy)
        print(f"  T = {T:6.1f} MeV   (mu_g z_h)^2 = {(MU_G*zh)**2:7.2f}   d0* = {d0:.10f}")
        for meth, rt, at in [('RK45', 1e-3, 1e-6), ('RK45', 1e-6, 1e-9),
                             ('DOP853', 1e-10, 1e-14), ('DOP853', 1e-13, 1e-18)]:
            m, s = probe(d0, T, 1 - 1e-4, u_legacy, method=meth, rtol=rt, atol=at)
            tag = '   <-- old default' if (meth, rt) == ('RK45', 1e-3) else ''
            print(f"      {meth:7s} rtol={rt:<7.0e} mq = {m:14.6g}   "
                  f"sigma^1/3 = {cube(s):9.3f}{tag}")


def section_b():
    print("\n(b) UV CUTOFF. sigma carries a residual ln(z_i): holding u_i fixed")
    print("    makes the scheme drift with T. Columns: fixed u_i vs fixed z_i.")
    zi_ref = 2.5e-5
    print(f"    {'T':>7} {'z_h':>9} {'u_i=1e-2: z_i':>14} {'sigma^1/3':>10} "
          f"| {'z_i=2.5e-5: u_i':>16} {'sigma^1/3':>10}")
    for T in TEMPS:
        zh, _ = blackness(T, MU)
        d0 = physical_d0(T, 1 - 1e-4, 1e-2)
        sol, _ = integrate(d0, T, 1 - 1e-4, 1e-6)
        _, s_fixed_u = extract(*sol.sol(1e-2), 1e-2, zh)
        u_phys = zi_ref/zh
        _, s_fixed_z = extract(*sol.sol(u_phys), u_phys, zh)
        print(f"    {T:7.1f} {zh:9.5f} {1e-2*zh:14.3e} {cube(s_fixed_u):10.3f} "
              f"| {u_phys:16.3e} {cube(s_fixed_z):10.3f}")
    print("    The first column is the old scheme; its T dependence mixes physics")
    print("    with a moving renormalization point. The second is scheme-fixed.")


def section_c():
    print("\n(c) IR CUTOFF. Error at fixed 1-u_f grows fast with (mu_g z_h)^2.")
    for T in TEMPS:
        zh, _ = blackness(T, MU)
        ui, uf_new = physical_cutoffs(T, MU)
        d0 = physical_d0(T, 1 - 1e-4, ui)
        print(f"  T = {T:6.1f} MeV   (mu_g z_h)^2 = {(MU_G*zh)**2:7.2f}   "
              f"new 1-u_f = {1 - uf_new:.2e}")
        for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            m, s = probe(d0, T, 1 - eps, ui)
            tag = '   <-- old default' if eps == 1e-4 else ''
            print(f"      1-u_f = {eps:7.1e}   mq = {m:14.6g}   "
                  f"sigma^1/3 = {cube(s):9.3f}{tag}")


def section_d():
    print("\n(d) CONDITIONING of the shooting map d0 -> mq. This is a property of")
    print("    the solution in physical z, so NO choice of coordinate changes it.")
    print("    The solver scans d0 on linspace(0, 10, 100), spacing 0.101.")
    print(f"    {'T':>7} {'(mu_g z_h)^2':>13} {'d0 window for mq=1..100':>24} "
          f"{'grid spacing / window':>22}")
    for T in [150.0, 120.0, 100.0, 60.0, 30.0, 17.0]:
        zh, _ = blackness(T, MU)
        ui, uf = physical_cutoffs(T, MU)
        a = physical_d0(T, uf, ui, target=1.0)
        b = physical_d0(T, uf, ui, target=100.0)
        w = max(b - a, 1e-300)
        print(f"    {T:7.1f} {(MU_G*zh)**2:13.2f} {w:24.3e} {0.10101/w:22.3e}")
    print("    Below T ~ 100-120 MeV the whole physical sigma(m_q) curve fits")
    print("    between two adjacent grid points. Fixing that needs a different")
    print("    root-finding strategy (continuation / solve_bvp), not new coordinates.")


def section_e():
    print("\n(e) END-TO-END sigma(T): legacy scheme vs the new physical scheme.")
    print(f"    {'T':>7} {'legacy sigma^1/3':>18} {'physical sigma^1/3':>20} {'shift':>9}")
    for T in TEMPS:
        leg = calculate_sigma_values(MQ, 0.1, T, MU, LAMBDA1, 1e-2, 1 - 1e-4,
                                     0, 10, v3, v4, log_results=False,
                                     solver='legacy')['sigma_values']
        new = calculate_sigma_values(MQ, 0.1, T, MU, LAMBDA1, None, None,
                                     0, 10, v3, v4, log_results=False)['sigma_values']
        lv = cube(leg[0]) if len(leg) else np.nan
        nv = cube(new[0]) if len(new) else np.nan
        print(f"    {T:7.1f} {lv:18.3f} {nv:20.3f} {nv - lv:9.3f}")
        sys.stdout.flush()


def sweep(tmin=17.0, tmax=140.0, n=14, out='sigma_T_scheme_comparison'):
    """sigma(T) under both schemes, written to CSV and plotted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    temps = np.linspace(tmin, tmax, n)
    rows = []
    print(f"\nSWEEP sigma(T), {n} points from {tmin} to {tmax} MeV, both schemes")
    print(f"    {'T':>7} {'legacy sigma^1/3':>18} {'physical sigma^1/3':>20} {'shift':>9}")
    for T in temps:
        def first(res):
            v = res['sigma_values']
            return cube(v[0]) if len(v) else np.nan
        leg = first(calculate_sigma_values(MQ, 0.1, T, MU, LAMBDA1, 1e-2, 1 - 1e-4,
                                           0, 10, v3, v4, log_results=False,
                                           solver='legacy'))
        new = first(calculate_sigma_values(MQ, 0.1, T, MU, LAMBDA1, None, None,
                                           0, 10, v3, v4, log_results=False))
        ui, uf = physical_cutoffs(T, MU)
        zh, _ = blackness(T, MU)
        rows.append((T, leg, new, 1e-2*zh, ui*zh, 1 - uf))
        print(f"    {T:7.1f} {leg:18.3f} {new:20.3f} {new - leg:9.3f}")
        sys.stdout.flush()

    arr = np.array(rows)
    header = ('T_MeV,sigma_cuberoot_legacy_MeV,sigma_cuberoot_physical_MeV,'
              'z_i_legacy_invMeV,z_i_physical_invMeV,horizon_gap_physical')
    np.savetxt(out + '.csv', arr, delimiter=',', header=header, comments='', fmt='%.8g')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(arr[:, 0], arr[:, 1], 'o-', label=r'legacy: $u_i=10^{-2}$, $1-u_f=10^{-4}$')
    ax1.plot(arr[:, 0], arr[:, 2], 's-', label=r'physical: $z_i=1/(40\,$GeV$)$ fixed')
    ax1.set_ylabel(r'$\sigma^{1/3}$ (MeV)')
    ax1.set_title(rf'$m_q={MQ:g}$ MeV, $\lambda_1={LAMBDA1}$, $\mu=0$, $\mu_g={MU_G:g}$ MeV')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(arr[:, 0], arr[:, 3], 'o-', label=r'legacy $z_i=u_i z_h$')
    ax2.plot(arr[:, 0], arr[:, 4], 's-', label=r'physical $z_i$')
    ax2.set_yscale('log')
    ax2.set_xlabel('T (MeV)')
    ax2.set_ylabel(r'$z_i$ (MeV$^{-1}$)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + '.png', dpi=200)
    print(f"\n    wrote {out}.csv and {out}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true',
                    help='also run section (e), the end-to-end pipeline (slow)')
    ap.add_argument('--sweep', action='store_true',
                    help='run only the sigma(T) sweep, writing a CSV and a PNG (slow)')
    args = ap.parse_args()

    if args.sweep:
        sweep()
        return

    print(f"lambda1 = {LAMBDA1}   mu = {MU}   m_q = {MQ} MeV   mu_g = {MU_G}   "
          f"v3 = {v3:.5f}   v4 = {v4}")
    section_a()
    section_b()
    section_c()
    section_d()
    if args.full:
        section_e()
    else:
        print("\n(e) skipped; pass --full to run the end-to-end comparison.")


if __name__ == '__main__':
    main()
