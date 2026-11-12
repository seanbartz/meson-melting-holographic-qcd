"""
Checks of the numerical scheme of PROJECT_ASSESSMENT.md §3.3, §3.3b and §3.4b:

* the UV series with the zhat^n ln(zhat) terms solves the chiral EOM;
* the horizon series recursion reproduces the closed-form d1, d2 used before;
* sigma does not depend on the UV matching point, on the ODE tolerance, or on the
  u grid requested by the spectral codes;
* the exact-Wronskian Im(B/A) equals the direct formula where the latter is well
  conditioned, and stays finite and negative at low T where the direct one is noise.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import chiral_solve_complete as cs
from chiral_uv_series_generated import uv_series

LAM1 = 7.438
MQ = 9.0


@pytest.fixture(scope="module")
def root_T60():
    sig, _, _, d0s = cs.sigma_of_T(MQ, 0.1, 60.0, 0.0, LAM1, 0, 10, 1e-2, 1 - 1e-4)
    assert len(d0s) == 1
    return d0s[0], sig[0]


def test_horizon_hat_matches_blackness():
    for T, mu in [(17.0, 0.0), (60.0, 100.0), (120.0, 300.0)]:
        b, Q2, zh_hat = cs.horizon_hat(T, mu)
        zh, q = cs.blackness(T, mu)
        assert zh_hat == pytest.approx(cs.MU_G * zh, rel=1e-12)
        assert Q2 == pytest.approx((q * zh**3)**2, rel=1e-10, abs=1e-14)


def test_horizon_recursion_reproduces_closed_forms():
    T, mu, d0 = 60.0, 100.0, 0.7
    l3, l4, mug = cs.v3, cs.v4, cs.MU_G
    b, Q2, _ = cs.horizon_hat(T, mu)
    zh, q = cs.blackness(T, mu)
    Q = q * zh**3
    d1 = (3*d0 - 3*d0**2*l3 - 4*d0**3*l4 + d0*zh**2*LAM1*mug**2) / (2*(-2 + Q**2))
    d2 = (1/(16*(-2 + Q**2)**2))*(6*d1*(-6 + Q**2 + Q**4) + 4*d0**3*(14 - 13*Q**2)*l4
          + d0**2*((42 - 39*Q**2)*l3 - 24*d1*(-2 + Q**2)*l4)
          - 2*d1*(-2 + Q**2)*zh**2*(-8 + 4*Q**2 - LAM1)*mug**2
          + 3*d0*(-14 + 13*Q**2 + 8*d1*l3 - 4*d1*Q**2*l3 + (-2 + 3*Q**2)*zh**2*LAM1*mug**2))
    d = cs.horizon_coefficients(d0, b, Q2, LAM1, l3, l4, 10)
    assert d[1] == pytest.approx(d1, rel=1e-12)
    assert d[2] == pytest.approx(d2, rel=1e-12)


def test_uv_series_solves_the_equation():
    """Integrate the EOM from series data at zhat = 0.004 to 0.008; the series must agree to O(zhat^6)."""
    T = 60.0
    b, Q2, zh_hat = cs.horizon_hat(T, 0.0)
    k4 = (1 + Q2) / zh_hat**4
    a, c = 0.006, 0.3
    z0, z1 = 0.004, 0.008
    S0, _ = uv_series(z0, a, c, LAM1, cs.v3, cs.v4, k4)
    u0, u1 = z0 / zh_hat, z1 / zh_hat
    sol = solve_ivp(cs.chiral, [u0, u1], [S0[0], S0[1] * zh_hat], method='DOP853',
                    rtol=1e-13, atol=1e-20, args=(b, Q2, LAM1, cs.v3, cs.v4))
    S1, _ = uv_series(z1, a, c, LAM1, cs.v3, cs.v4, k4)
    # the condensate term c zhat^3 is ~2e-7 here; the residual must be far below it
    assert abs(sol.y[0, -1] - S1[0]) < 1e-6 * c * z1**3


def test_sigma_independent_of_uv_matching_point(root_T60):
    """The old two-term inversion moved sigma^(1/3) by ~3 MeV per decade of z_UV."""
    d0, sigma = root_T60
    ref = cs.chiral_shoot(d0, LAM1, 60.0, 0.0)
    for z in (0.001, 0.002, 0.01):
        other = cs.chiral_shoot(d0, LAM1, 60.0, 0.0, settings={'z_uv': z})
        assert np.cbrt(other['sigma']) == pytest.approx(np.cbrt(ref['sigma']), abs=2e-3)
        assert other['mq'] == pytest.approx(ref['mq'], rel=1e-6)


def test_sigma_converged_in_tolerance(root_T60):
    d0, _ = root_T60
    ref = cs.chiral_shoot(d0, LAM1, 60.0, 0.0, settings={'rtol': 1e-13})
    res = cs.chiral_shoot(d0, LAM1, 60.0, 0.0)
    assert res['sigma'] == pytest.approx(ref['sigma'], rel=1e-8)
    assert res['mq'] == pytest.approx(ref['mq'], rel=1e-8)


def test_root_is_exact(root_T60):
    d0, sigma = root_T60
    res = cs.chiral_shoot(d0, LAM1, 60.0, 0.0)
    assert res['mq'] == pytest.approx(MQ, abs=1e-6)
    assert res['sigma'] == pytest.approx(sigma, rel=1e-12)


def test_sigma_independent_of_output_grid():
    r1 = cs.calculate_sigma_values(MQ, 0.1, 60.0, 0.0, LAM1, 1e-2, 1 - 1e-4, log_results=False)
    r2 = cs.calculate_sigma_values(MQ, 0.1, 60.0, 0.0, LAM1, 1e-4, 1 - 1e-6, log_results=False)
    assert r1['sigma_values'] == pytest.approx(r2['sigma_values'], rel=1e-12)
    # fields agree where the grids overlap
    chi2 = np.interp(r1['u_values'], r2['u_values'], r2['chiral_fields'][0])
    assert np.max(np.abs(chi2 - r1['chiral_fields'][0])) < 1e-6


def test_low_temperature_solution_exists():
    """At T = 17 MeV the old default-tolerance shoot diverged at the physical d0."""
    sig, _, _, d0s = cs.sigma_of_T(MQ, 0.1, 17.0, 0.0, LAM1, 0, 10, 1e-2, 1 - 1e-4)
    assert len(d0s) >= 1
    res = cs.chiral_shoot(d0s[0], LAM1, 17.0, 0.0)
    assert res['ok'] and res['mq'] == pytest.approx(MQ, abs=1e-4)
    assert 290 < np.cbrt(sig[0]) < 310


def test_scheme_shift_is_temperature_independent():
    s1 = cs.condensate_in_scheme(0.0, MQ, LAM1, scale_MeV=1.0)
    s2 = cs.condensate_in_scheme(0.0, MQ, LAM1, scale_MeV=cs.MU_G)
    assert s2 == 0.0
    assert s1 != 0.0


def _chi_interp(T):
    from scipy.interpolate import interp1d
    r = cs.calculate_sigma_values(MQ, 0.1, T, 0.0, LAM1, 1e-2, 1 - 1e-4, log_results=False)
    return r['sigma_values'][0], interp1d(r['u_values'], r['chiral_fields'][0],
                                          bounds_error=False, fill_value="extrapolate")


def _direct_BA(A, ui, uf, w, g5, zh, mug, zeta, sigma, Q, l1, l3, l4):
    """The old formula (numerical Wronskian), at tight tolerance, for comparison."""
    a_1, a_2 = A.a_s(ui, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4)
    def eom2(u, y):
        d1 = A.Axial_eom(y[0:2], u, w, g5, Q, zh, mug, MQ)
        d2 = A.Axial_eom(y[2:4], u, w, g5, Q, zh, mug, MQ)
        return [d1[0], d1[1], d2[0], d2[1]]
    sol = solve_ivp(eom2, [ui, uf], [a_1[0], a_1[1], a_2[0], a_2[1]], method='DOP853', rtol=1e-12, atol=1e-30)
    p1, p1p, p2, p2p = sol.y[:, -1]
    pm, pmp = A.psi_m(uf, w, zh, Q)
    return (pm*p2p - pmp*p2) / (pm*p1p - pmp*p1)


def test_exact_wronskian_matches_direct_where_well_conditioned():
    import axial_spectra as A
    T = 60.0
    sigma, A.chi_interp = _chi_interp(T)
    mug, zh, Q, l1, l3, l4, g5, zeta = A.calculate_variables(LAM1, T, 0.0)
    for w in (1250.0, 1700.0, 2300.0):
        exact = A.BA(1e-2, 1 - 1e-4, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4)
        direct = _direct_BA(A, 1e-2, 1 - 1e-4, w, g5, zh, mug, zeta, sigma, Q, l1, l3, l4)
        assert exact.imag == pytest.approx(direct.imag, rel=1e-5)
        assert exact.real == pytest.approx(direct.real, rel=1e-5)


def test_exact_wronskian_at_low_temperature():
    """§3.4b: at T = 17 MeV the direct formula returns noise of either sign below ~2.3 GeV."""
    import axial_spectra as A
    T = 17.0
    sigma, A.chi_interp = _chi_interp(T)
    mug, zh, Q, l1, l3, l4, g5, zeta = A.calculate_variables(LAM1, T, 0.0)
    im = [A.BA(1e-2, 1 - 1e-4, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4).imag
          for w in (800.0, 1000.0, 1250.0, 2300.0)]
    assert all(v < 0 for v in im)
    # exponentially small away from resonances (assessment table: 1.6e-37 ... 5.8e-30)
    assert abs(im[0]) < 1e-35 and abs(im[2]) < 1e-28
    assert im[3] == pytest.approx(-1.22e-3, rel=0.05)


def test_vector_exact_wronskian_matches_direct():
    import vector_spectra as V
    V.T, V.mu, V.q = 60.0, 0.0, 0.0
    mug, zh, Q, _ = V.calculate_variables()
    w = 1200.0
    exact = V.BA(w, Q, zh, mug)
    v_1, v_2 = V.vs(V.ui, w, zh, mug, Q)
    def eom2(u, y):
        d1 = V.vector_eom(y[0:2], u, w, Q, zh, mug)
        d2 = V.vector_eom(y[2:4], u, w, Q, zh, mug)
        return [d1[0], d1[1], d2[0], d2[1]]
    sol = solve_ivp(eom2, [V.ui, V.uf], [v_1[0], v_1[1], v_2[0], v_2[1]], method='DOP853', rtol=1e-12, atol=1e-30)
    p1, p1p, p2, p2p = sol.y[:, -1]
    pm, pmp = V.psi_m(V.uf, w, zh, Q)
    direct = (pm*p2p - pmp*p2) / (pm*p1p - pmp*p1)
    assert exact.imag == pytest.approx(direct.imag, rel=1e-5)
    assert exact.real == pytest.approx(direct.real, rel=1e-5)


def test_parallel_axial_spectrum_uses_the_chiral_field():
    """
    calculate_data runs BA in joblib workers.  The workers' copy of axial_spectra had
    chi_interp = None, and Axial_eom fell back to chi = 0 (the vector spectrum).
    """
    import axial_spectra as A
    T = 60.0
    sigma, A.chi_interp = _chi_interp(T)
    mug, zh, Q, l1, l3, l4, g5, zeta = A.calculate_variables(LAM1, T, 0.0)
    ws, BAs, _, _ = A.calculate_data(1240, 1260, 3, 100, 1e-2, 1 - 1e-4, g5, zh, MQ, mug, zeta,
                                     sigma, Q, l1, l3, l4, expected_peaks=0)
    for w, ba in zip(ws, BAs):
        serial = A.BA(1e-2, 1 - 1e-4, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4)
        assert ba == pytest.approx(serial, rel=1e-9)
