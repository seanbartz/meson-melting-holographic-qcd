#!/usr/bin/env python3
"""
Regression tests for the integration tolerances and the cutoff scheme added to
chiral_solve_complete.py.

Covers:
1. physical_cutoffs holds the UV cutoff fixed in physical units and shrinks the
   horizon gap as (mu_g z_h)^2 grows.
2. solver_settings resolves the specs used across the codebase.
3. The default solver recovers mq = 9 MeV at the physical d0 at T = 17 MeV,
   where the old loose defaults diverge.
4. 'legacy' mode with the old cutoffs still reproduces the pre-change number,
   so published results stay reproducible.
"""

import sys

import numpy as np

from chiral_solve_complete import (
    MU_G, SOLVER_DEFAULT, SOLVER_LEGACY, Z_UV, blackness, calculate_sigma_values,
    chiral_solve_IR, physical_cutoffs, resolve_cutoffs, solver_settings, v3, v4,
)

LAMBDA1 = 7.438
# The d0 that gives mq = 9 MeV at T = 17 MeV, mu = 0, found by bisection with
# rtol = 1e-13 at the legacy cutoffs (see diagnostics/cutoff_scheme_study.py).
D0_T17 = 5.701544031720
LEGACY_UI, LEGACY_UF = 1e-2, 1 - 1e-4


def test_physical_cutoffs():
    print("\n[1] physical_cutoffs")
    prev_gap = None
    for T in [120.0, 60.0, 30.0, 17.0]:
        zh, _ = blackness(T, 0.0)
        ui, uf = physical_cutoffs(T, 0.0)
        zi = ui*zh
        gap = 1 - uf
        print(f"    T={T:6.1f}  u_i={ui:.4e}  z_i={zi:.4e}  1-u_f={gap:.3e}")
        assert abs(zi - Z_UV)/Z_UV < 1e-12, f"z_i drifted with T: {zi} != {Z_UV}"
        if prev_gap is not None:
            assert gap <= prev_gap, "horizon gap must not grow as T falls"
        prev_gap = gap
        assert 0 < ui < 1 and 0 < uf < 1 and ui < uf
    # the gap must actually shrink somewhere in this range, not just stay flat
    assert (1 - physical_cutoffs(17.0, 0.0)[1]) < (1 - physical_cutoffs(120.0, 0.0)[1])
    print("    OK: z_i fixed in physical units, horizon gap shrinks with T")


def test_resolve_cutoffs():
    print("\n[2] resolve_cutoffs / solver_settings")
    assert resolve_cutoffs(17.0, 0.0, 1e-2, 0.9999) == (1e-2, 0.9999)
    ui, uf = resolve_cutoffs(17.0, 0.0, 1e-2, None)
    assert ui == 1e-2 and uf == physical_cutoffs(17.0, 0.0)[1]
    assert solver_settings(None) == SOLVER_DEFAULT
    assert solver_settings('legacy') == SOLVER_LEGACY
    assert solver_settings({'rtol': 1e-8})['rtol'] == 1e-8
    assert solver_settings({'rtol': 1e-8})['method'] == SOLVER_DEFAULT['method']
    assert SOLVER_DEFAULT['extract_index'] == 0, \
        "extraction must sit at u_i, otherwise u_i below ~1e-3 is ignored"
    print("    OK")


def test_tolerance_is_converged():
    """The default tolerances must be converged; the old ones are not.

    Stated as a property of the solution rather than of a particular mq
    inversion, since chiral_solve_IR and new_function invert the UV relation
    differently.
    """
    print("\n[3] integration tolerance at T = 17 MeV")
    tight = {'method': 'DOP853', 'rtol': 1e-13, 'atol': 1e-18}
    _, chi_ref, _, _ = chiral_solve_IR(D0_T17, LAMBDA1, 17.0, 0.0, LEGACY_UI, LEGACY_UF,
                                       v3, v4, solver=tight)
    _, chi_new, _, _ = chiral_solve_IR(D0_T17, LAMBDA1, 17.0, 0.0, LEGACY_UI, LEGACY_UF,
                                       v3, v4)
    _, chi_old, _, _ = chiral_solve_IR(D0_T17, LAMBDA1, 17.0, 0.0, LEGACY_UI, LEGACY_UF,
                                       v3, v4, solver='legacy')
    err_new = abs(chi_new[0] - chi_ref[0])/abs(chi_ref[0])
    err_old = (np.inf if not np.isfinite(chi_old[0])
               else abs(chi_old[0] - chi_ref[0])/abs(chi_ref[0]))
    print(f"    chi(u_i) reference      = {chi_ref[0]:.12e}")
    print(f"    default solver rel err  = {err_new:.2e}")
    print(f"    legacy  solver rel err  = {err_old:.2e}")
    assert err_new < 1e-6, f"default tolerances are not converged: {err_new:.2e}"
    assert err_old > 1e-3, \
        "legacy solver unexpectedly accurate here; the regression guard is stale"
    print("    OK: default tolerances converged, old ones are not")


def test_physical_scheme_sigma():
    print("\n[5] physical scheme end-to-end, T = 17 MeV")
    r = calculate_sigma_values(9.0, 0.1, 17.0, 0.0, LAMBDA1, None, None,
                               0, 10, v3, v4, log_results=False)
    sig = r["sigma_values"]
    assert len(sig) >= 1, "no sigma solution found"
    cube = float(np.sign(sig[0])*abs(sig[0])**(1/3))
    print(f"    sigma^(1/3) = {cube:.3f} MeV   (legacy scheme gives 302.389)")
    assert abs(cube - 305.856) < 0.3, f"physical-scheme sigma moved: {cube}"
    assert abs(r["u_values"][0] - physical_cutoffs(17.0, 0.0)[0]) < 1e-12, \
        "chiral grid does not start at the scheme's u_i"
    print("    OK")


def test_legacy_end_to_end_unchanged():
    print("\n[4] legacy end-to-end reproducibility, T = 120 MeV")
    r = calculate_sigma_values(9.0, 0.1, 120.0, 0.0, LAMBDA1, LEGACY_UI, LEGACY_UF,
                               0, 10, v3, v4, log_results=False, solver='legacy')
    sig = r["sigma_values"]
    assert len(sig) >= 1, "no sigma solution found"
    cube = float(np.sign(sig[0])*abs(sig[0])**(1/3))
    print(f"    sigma^(1/3) = {cube:.3f} MeV   (pre-change value 300.979)")
    assert abs(cube - 300.979) < 0.05, \
        f"legacy mode no longer reproduces the pre-change result: {cube}"
    print("    OK: published numbers remain reproducible via solver='legacy'")


def main():
    print("=" * 70)
    print(f"Cutoff scheme regression tests   (mu_g = {MU_G}, Z_UV = {Z_UV} MeV^-1)")
    print("=" * 70)
    tests = [test_physical_cutoffs, test_resolve_cutoffs,
             test_tolerance_is_converged,
             test_legacy_end_to_end_unchanged,
             test_physical_scheme_sigma]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"    FAIL: {e}")
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if not failed else f"{failed} TEST(S) FAILED")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
