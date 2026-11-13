"""
Independent checks of meson_core / meson_spectra (numba kernels and high-level API).

Oracles: scipy solve_ivp (DOP853), the legacy vector_spectra / axial_spectra codes, the
analytic T -> 0 vector spectrum, the Sturm oscillation theorem, and a symmetry of the
chiral sector.  Run with:  python -m pytest tests/test_meson_core.py -q -s
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

import chiral_solve_complete as cs
import meson_spectra as ms
from meson_spectra import Model, Background

MU_G = 440.0
LAM1 = 7.438
MQ = 9.0


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def bg60():
    return Background(60.0, 0.0, Model())


@pytest.fixture(scope="module")
def bg17():
    return Background(17.0, 0.0, Model())


@pytest.fixture(scope="module")
def bg10():
    return Background(10.0, 0.0, Model())


@pytest.fixture(scope="module")
def bg12():
    return Background(12.0, 0.0, Model())


# ---------------------------------------------------------------------------- 1
@pytest.mark.parametrize("d0,T,mu", [(0.8, 60.0, 0.0), (1.0, 100.0, 0.0), (0.8, 60.0, 100.0),
                                     (0.3, 60.0, 150.0), (0.6, 60.0, 150.0), (2.0, 30.0, 100.0)])
def test_chiral_shoot_numba_vs_scipy(d0, T, mu):
    u = np.linspace(1e-3, 1.0, 2001)
    kw = dict(u_grid=u)
    a = cs.chiral_shoot(d0, LAM1, T, mu, cs.v3, cs.v4, **kw)
    r = cs.chiral_shoot(d0, LAM1, T, mu, cs.v3, cs.v4, settings={'method': 'DOP853'}, **kw)
    assert a['ok'] and r['ok']
    # mq and sigma against scipy without a grid (with a grid the scipy path's sigma moves
    # by up to ~1e-8 at mu > 0, where sigma is a small difference; the numba path is
    # grid independent)
    r0 = cs.chiral_shoot(d0, LAM1, T, mu, cs.v3, cs.v4, settings={'method': 'DOP853'})
    print(f"\n[1] sigma rel (no grid) {abs(a['sigma'] / r0['sigma'] - 1):.2e}, "
          f"(scipy grid) {abs(a['sigma'] / r['sigma'] - 1):.2e}")
    assert a['mq'] == pytest.approx(r0['mq'], rel=1e-9)
    assert a['sigma'] == pytest.approx(r0['sigma'], rel=1e-9)
    scale = np.max(np.abs(r['chi']))
    pscale = np.max(np.abs(r['chip']))
    err_chi = np.max(np.abs(a['chi'] - r['chi'])) / scale
    err_chip = np.max(np.abs(a['chip'] - r['chip'])) / pscale
    print(f"\n[1] d0={d0} T={T} mu={mu}: chi err {err_chi:.2e}, chip err {err_chip:.2e}")
    assert err_chi < 1e-9
    assert err_chip < 1e-9


# ---------------------------------------------------------------------------- 2
def _pole_factor(w, Q2, order=10):
    r = -1j * w / (2.0 * (2.0 - Q2))
    pf = 1.0 + 0j
    for k in range(1, order + 1):
        pf *= 1.0 + 2.0 * r / k
    return pf


def _scipy_M_vector(w, b, Q2, u_end=0.5, eps=1e-7):
    """Own complex integration of the vector equation (chi = 0), leading-order BCs."""
    def f(u):
        return 1.0 - (1.0 + Q2) * u**4 + Q2 * u**6

    def fp(u):
        return -4.0 * (1.0 + Q2) * u**3 + 6.0 * Q2 * u**5

    def rhs(u, y):
        psi, dpsi = y
        return [dpsi, (1.0 / u - fp(u) / f(u) + 2.0 * b * u) * dpsi - (w * w / f(u)**2) * psi]

    r = -1j * w / (2.0 * (2.0 - Q2))
    u0 = 1e-4
    s1 = solve_ivp(rhs, [u0, u_end], np.array([u0**2, 2.0 * u0], dtype=complex),
                   method='DOP853', rtol=1e-11, atol=1e-30)
    ue = 1.0 - eps
    s2 = solve_ivp(rhs, [ue, u_end], np.array([(1 - ue)**r, -r * (1 - ue)**(r - 1)], dtype=complex),
                   method='DOP853', rtol=1e-11, atol=1e-30)
    p1, p1p = s1.y[:, -1]
    p2, p2p = s2.y[:, -1]
    return (p1 * p2p - p1p * p2) * f(u_end) * np.exp(-b * u_end**2) / u_end


@pytest.mark.parametrize("T,mu", [(60.0, 0.0), (60.0, 100.0)])
def test_mode_function_vs_scipy_vector(T, mu):
    bg = Background(T, mu, Model(), solve_chiral=False)
    ch = bg.channel('vector')
    worst = 0.0
    for om in (1200.0, 800.0 - 50.0j, 1500.0 - 200.0j):
        w = om * bg.zh_hat / MU_G
        mine = complex(ch.M(om)) / _pole_factor(w, bg.Q2)
        ref = _scipy_M_vector(w, bg.b, bg.Q2)
        rel = abs(mine - ref) / abs(ref)
        worst = max(worst, rel)
        print(f"\n[2] T={T} mu={mu} omega={om}: rel discrepancy {rel:.2e}")
        assert rel < 5e-3
    print(f"[2] T={T} mu={mu}: worst {worst:.2e}")


# ---------------------------------------------------------------------------- 3
@pytest.mark.parametrize("kind", ["vector", "axial"])
def test_M_independent_of_matching_point(bg60, kind):
    ch = bg60.channel(kind)
    for om in (1000.0, 1700.0, 1300.0 - 100.0j):
        vals = np.array([complex(ch.M(om, u_m=um)) for um in (0.3, 0.5, 0.7, 0.9)])
        spread = np.max(np.abs(vals - vals[0])) / abs(vals[0])
        print(f"\n[3] {kind} omega={om}: spread {spread:.2e}")
        assert spread < 1e-8


# ---------------------------------------------------------------------------- 4
def test_rho_vs_legacy_vector(bg60):
    import vector_spectra as V
    V.T, V.mu, V.q = 60.0, 0.0, 0.0
    mug, zh, Q, _ = V.calculate_variables()
    ch = bg60.channel('vector')
    for w in (800.0, 1250.0, 1700.0, 2300.0):
        leg = abs(V.BA_scipy(w, Q, zh, mug).imag)
        new = float(ch.rho(w))
        rel = abs(new - leg) / leg
        print(f"\n[4] vector omega={w}: legacy {leg:.6e} new {new:.6e} rel {rel:.2e}")
        assert rel < 1e-3


def test_rho_vs_legacy_axial(bg60):
    import axial_spectra as A
    from scipy.interpolate import interp1d
    r = cs.calculate_sigma_values(MQ, 0.1, 60.0, 0.0, LAM1, 1e-2, 1 - 1e-4, log_results=False)
    sigma = r['sigma_values'][0]
    A.chi_interp = interp1d(r['u_values'], r['chiral_fields'][0], bounds_error=False,
                            fill_value="extrapolate")
    mug, zh, Q, l1, l3, l4, g5, zeta = A.calculate_variables(LAM1, 60.0, 0.0)
    ch = bg60.channel('axial', branch='stable')
    for w in (800.0, 1250.0, 1700.0, 2300.0):
        new = float(ch.rho(w))
        # legacy default horizon cut 1 - 1e-4: leading-order BC error is ~1e-3 at 800 MeV
        leg = abs(A.BA(1e-2, 1 - 1e-4, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4).imag)
        rel = abs(new - leg) / leg
        # legacy with the horizon cut moved closer: its truncation error drops ~10x
        leg5 = abs(A.BA(1e-2, 1 - 1e-5, w, g5, zh, MQ, mug, zeta, sigma, Q, l1, l3, l4).imag)
        rel5 = abs(new - leg5) / leg5
        print(f"\n[4] axial omega={w}: legacy {leg:.6e} new {new:.6e} rel {rel:.2e}; "
              f"legacy(1-1e-5) rel {rel5:.2e}")
        assert rel < 3e-3
        assert rel5 < 1e-4


# ---------------------------------------------------------------------------- 5
def test_vector_normal_modes_low_T(bg10):
    ch = bg10.channel('vector')
    modes = ch.scan_modes(1900.0, -40.0, re_min=20.0, im_max=40.0)
    print("\n[5] modes:", [f"{m:.4f}" for m in modes])
    assert len(modes) == 4
    for n, m in enumerate(modes, start=1):
        assert abs(m.real - 2 * MU_G * np.sqrt(n)) < 5.0
        assert abs(m.imag) < 1e-6


# ---------------------------------------------------------------------------- 6
@pytest.mark.parametrize("T", [17.0, 60.0])
@pytest.mark.parametrize("kind", ["vector", "axial"])
def test_scan_resolution_independent(T, kind, bg17, bg60):
    bg = bg17 if T == 17.0 else bg60
    ch = bg.channel(kind)
    a = ch.scan_modes(2600.0, -400.0, spacing=25.0)
    b = ch.scan_modes(2600.0, -400.0, spacing=10.0)
    print(f"\n[6] T={T} {kind}: {len(a)} vs {len(b)} modes")
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x.real - y.real) <= 1e-7 * abs(y.real)
        assert abs(x - y) <= 1e-6 * abs(y)


# ---------------------------------------------------------------------------- 7
def _count_nodes(ch, omega, u_end):
    """Zeros of the real psi_1 in (u0, u_end) for real omega; own scipy integration."""
    bg = ch.bg
    b, Q2 = bg.b, bg.Q2
    w2 = (omega * bg.zh_hat / MU_G)**2
    spl = CubicHermiteSpline(ch.u_grid, ch.chi, ch.chip)
    g5sq = (2.0 * np.pi)**2

    def f(u):
        return 1.0 - (1.0 + Q2) * u**4 + Q2 * u**6

    def fp(u):
        return -4.0 * (1.0 + Q2) * u**3 + 6.0 * Q2 * u**5

    def rhs(u, y):
        return [y[1], (1.0 / u - fp(u) / f(u) + 2.0 * b * u) * y[1]
                - (w2 / f(u)**2 - g5sq * spl(u)**2 / (u * u * f(u))) * y[0]]

    u0 = ch.u_grid[0]
    sol = solve_ivp(rhs, [u0, u_end], [u0**2, 2.0 * u0], method='DOP853', rtol=1e-11, atol=1e-40,
                    dense_output=True)
    us = np.linspace(u0, u_end, 200001)
    ps = sol.sol(us)[0]
    # nodes of the amplitude: sign changes
    return int(np.sum(np.sign(ps[1:]) != np.sign(ps[:-1])))


def test_node_counting_axial_low_T(bg12):
    ch = bg12.channel('axial', branch='stable')
    modes = ch.scan_modes(2600.0, -40.0, re_min=20.0, im_max=40.0)
    mr = np.array([m.real for m in modes])
    print("\n[7] axial modes at T=12:", np.round(mr, 3))
    n_below = int(np.sum(mr < 2600.0))   # scan_modes may return one just above re_max
    for u_end in (0.85, 0.9, 0.93):
        counts = [_count_nodes(ch, m + 100.0, u_end) for m in mr]
        below = _count_nodes(ch, mr[0] - 100.0, u_end)
        top = _count_nodes(ch, 2600.0, u_end)
        print(f"[7] u_end={u_end}: nodes just above each mode {counts}, below first {below}, "
              f"at 2600: {top}, modes below 2600: {n_below}")
    # u_end = 0.9 is already inside the near-horizon region where psi_1 oscillates
    # (nodes accumulate toward u = 1 and the count is no longer the Sturm count);
    # 0.5 ... 0.85 all give the same count.
    u_end = 0.8
    assert _count_nodes(ch, mr[0] - 100.0, u_end) == 0
    for k, m in enumerate(mr, start=1):
        assert _count_nodes(ch, m + 100.0, u_end) == k, f"mode {k} at {m}"
    assert _count_nodes(ch, 2600.0, u_end) == n_below


# ---------------------------------------------------------------------------- 8
def test_chiral_rescaling_symmetry():
    s = 1.5
    m1 = Model(lam1=6.0, gamma=-22.6, lam4=4.2, mq=9.0)
    m2 = Model(lam1=6.0, gamma=-22.6 * s, lam4=4.2 * s**2, mq=9.0 / s)
    b1 = ms.chiral_branches(100.0, 0.0, m1)
    b2 = ms.chiral_branches(100.0, 0.0, m2)
    print("\n[8] branches:", [(x['d0'], x['sigma']) for x in b1], [(x['d0'], x['sigma']) for x in b2])
    assert len(b1) == len(b2) >= 1
    for x, y in zip(b1, b2):
        assert y['sigma'] == pytest.approx(x['sigma'] / s, rel=1e-6)
        assert y['d0'] == pytest.approx(x['d0'] / s, rel=1e-6)
