"""
meson_core.py -- compiled (numba) numerical kernels shared by the chiral solver, the
axial and vector spectral functions, and the quasinormal-mode search.

Everything here is dimensionless, with mu_g as the unit of energy (same convention as
chiral_solve_complete.py):

    u = z/z_h,  b = (mu_g z_h)^2,  Q2 = (mu z_h)^2,  f = 1 - (1+Q2) u^4 + Q2 u^6,
    w = omega z_h  (complex for quasinormal modes).

Contents
--------
* ``dop853``: scipy's DOP853 step, error estimate and controller, compiled.  The
  coefficient tables are imported from scipy, not retyped.  scipy stays in the code
  base as the oracle (``SOLVER_SETTINGS['method'] = 'scipy'`` in the chiral solver, and
  tests/test_meson_core.py).
* Chiral field: horizon series, one shoot (``chiral_shoot_core``), a scan over horizon
  values, and the solution on a uniform u grid with its derivative (for cubic Hermite
  interpolation inside the meson equation).
* Meson equation (vector: g5sq = 0; axial: g5sq = g5^2 and chi from the grid)

      psi'' = (1/u - f'/f + 2 b u) psi' - (w^2/f^2 - g5^2 chi^2/(u^2 f)) psi .

  - ``meson_uv``: the normalizable solution psi_1 = u^2 + ... from the boundary.
  - ``meson_infalling``: the infalling solution psi_in = (1-u)^r sum p_n (1-u)^n,
    r = -i w/g0, g0 = 2(2 - Q2), from a Frobenius series of any order (numerical
    recursion), then integrated toward the boundary.
  - ``mode_function``: M(w) = W[psi_1, psi_in] f e^{-phi}/u, which does not depend on u
    (Abel's identity) and is analytic in w.  Its zeros are the quasinormal modes, and

          |Im(B/A)| = 2 Re(w) e^{-b} / |M|^2        (real w)

    is the spectral function in the normalization of axial_spectra.BA / vector_spectra.BA.
    (Flux conservation: f e^{-phi} Im(psi* psi')/u is constant; it equals w e^{-b} at
    the horizon and 2 |A|^2 Im(B/A) at the boundary, and M = -2A.)
  - ``legacy_two_solutions``: psi_1 and psi_2 from given initial data to u_f, for the
    B/A formula of axial_spectra.BA (kept for comparison).
"""
import numpy as np
from numba import njit, prange
from scipy.integrate._ivp import dop853_coefficients as _dop

from chiral_uv_series_generated import uv_series as _uv_series_py

uv_series = njit(cache=True)(_uv_series_py)

N_STAGES = _dop.N_STAGES
_A = np.ascontiguousarray(_dop.A[:N_STAGES, :N_STAGES])
_B = np.ascontiguousarray(_dop.B)
_C = np.ascontiguousarray(_dop.C[:N_STAGES])
_E3 = np.ascontiguousarray(_dop.E3)
_E5 = np.ascontiguousarray(_dop.E5)

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
ERROR_EXPONENT = -1.0 / 8.0

KIND_CHIRAL = 0.0     # y = [chi, chi']
KIND_MESON = 1.0      # y = [Re psi, Im psi, Re psi', Im psi'], error norm on the moduli
KIND_MESON_PAIR = 2.0  # same equations, real w: two independent real solutions, error norm per component

_EMPTY = np.zeros(2)


# ---------------------------------------------------------------------------
# Right-hand sides
# ---------------------------------------------------------------------------
@njit(cache=True)
def hermite(u, u0, du, val, der):
    """Cubic Hermite interpolation on a uniform grid (val, der = d val/du at the nodes)."""
    n = val.size
    s = (u - u0) / du
    i = int(np.floor(s))
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    t = s - i
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * val[i] + (t3 - 2.0 * t2 + t) * du * der[i]
            + (-2.0 * t3 + 3.0 * t2) * val[i + 1] + (t3 - t2) * du * der[i + 1])


@njit(cache=True)
def rhs(u, y, dy, par, cu0, cdu, chi, chip):
    kind = par[0]
    b = par[1]
    Q2 = par[2]
    u2 = u * u
    u4 = u2 * u2
    f = 1.0 - (1.0 + Q2) * u4 + Q2 * u4 * u2
    fp = -4.0 * (1.0 + Q2) * u2 * u + 6.0 * Q2 * u4 * u
    if kind == KIND_CHIRAL:
        c = y[0]
        force = (-3.0 - par[3] * b * u2) * c + 3.0 * par[4] * c * c + 4.0 * par[5] * c * c * c
        dy[0] = y[1]
        dy[1] = (3.0 / u - fp / f + 2.0 * b * u) * y[1] + force / (u2 * f)
    else:
        P = 1.0 / u - fp / f + 2.0 * b * u
        A = par[3] / (f * f)
        if par[5] != 0.0:
            x = hermite(u, cu0, cdu, chi, chip)
            A -= par[5] * x * x / (u2 * f)
        B = par[4] / (f * f)
        dy[0] = y[2]
        dy[1] = y[3]
        dy[2] = P * y[2] - (A * y[0] - B * y[1])
        dy[3] = P * y[3] - (A * y[1] + B * y[0])


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
@njit(cache=True)
def dop853(y, u_start, u_end, par, cu0, cdu, chi, chip, rtol, atol, max_steps, ylimit, hbox):
    """
    Advance y (in place) from u_start to u_end.  hbox[0] carries the step size between
    calls (0 = choose).  Returns status: 0 reached u_end; 1 step-size underflow or
    non-finite state; 2 max_steps; 3 |y[0]| exceeded ylimit (runaway).
    """
    n = y.size
    K = np.zeros((N_STAGES + 1, n))
    y_stage = np.empty(n)
    y_new = np.empty(n)
    scale = np.empty(n)
    u = u_start
    direction = 1.0 if u_end > u_start else -1.0
    span = abs(u_end - u_start)
    if span == 0.0:
        return 0
    h_abs = hbox[0] if hbox[0] > 0.0 else 1e-3 * span
    h_min = 1e-14 * max(span, 1e-6)
    paired = par[0] == KIND_MESON
    rhs(u, y, K[0], par, cu0, cdu, chi, chip)
    n_steps = 0
    status = 0
    last_free = h_abs
    while True:
        remaining = abs(u_end - u)
        if remaining <= 1e-15 * max(1.0, abs(u_end)):
            break
        if n_steps >= max_steps:
            status = 2
            break
        clipped = False
        if h_abs >= remaining:
            last_free = h_abs
            h_abs = remaining
            clipped = True
        if h_abs < h_min and not clipped:
            status = 1
            break
        h = direction * h_abs

        for s in range(1, N_STAGES):
            for i in range(n):
                acc = 0.0
                for j in range(s):
                    acc += _A[s, j] * K[j, i]
                y_stage[i] = y[i] + h * acc
            rhs(u + _C[s] * h, y_stage, K[s], par, cu0, cdu, chi, chip)
        for i in range(n):
            acc = 0.0
            for j in range(N_STAGES):
                acc += _B[j] * K[j, i]
            y_new[i] = y[i] + h * acc
        rhs(u + h, y_new, K[N_STAGES], par, cu0, cdu, chi, chip)

        finite = True
        for i in range(n):
            scale[i] = max(abs(y[i]), abs(y_new[i]))
            if not np.isfinite(y_new[i]):
                finite = False
        if paired:
            for i in range(0, n, 2):
                m = max(scale[i], scale[i + 1])
                scale[i] = m
                scale[i + 1] = m
        err5_sq = 0.0
        err3_sq = 0.0
        for i in range(n):
            e5 = 0.0
            e3 = 0.0
            for j in range(N_STAGES + 1):
                e5 += _E5[j] * K[j, i]
                e3 += _E3[j] * K[j, i]
            sc = atol + rtol * scale[i]
            e5 /= sc
            e3 /= sc
            err5_sq += e5 * e5
            err3_sq += e3 * e3
        if not finite or not np.isfinite(err5_sq) or not np.isfinite(err3_sq):
            h_abs *= MIN_FACTOR
            if h_abs < h_min:
                status = 1
                break
            continue
        denominator = err5_sq + 0.01 * err3_sq
        if denominator == 0.0:
            error_norm = 0.0
        else:
            error_norm = h_abs * err5_sq / np.sqrt(denominator * n)

        if error_norm < 1.0:
            if error_norm == 0.0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)
            u = u + h
            for i in range(n):
                y[i] = y_new[i]
                K[0, i] = K[N_STAGES, i]
            n_steps += 1
            if abs(y[0]) > ylimit:
                status = 3
                break
            if clipped:
                h_abs = max(last_free, h_abs * factor)
            else:
                h_abs *= factor
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)
    hbox[0] = h_abs
    return status


# ---------------------------------------------------------------------------
# Chiral field
# ---------------------------------------------------------------------------
POLY_LEN = 12


@njit(cache=True)
def _poly_mul(p, q):
    out = np.zeros(POLY_LEN)
    for i in range(POLY_LEN):
        if p[i] != 0.0:
            for j in range(POLY_LEN - i):
                out[i + j] += p[i] * q[j]
    return out


@njit(cache=True)
def _g_poly(Q2):
    """g(x) = f/x as a polynomial in x = 1 - u, and u = 1 - x."""
    binom4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
    binom6 = np.array([1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0])
    f_x = np.zeros(POLY_LEN)
    f_x[0] = 1.0
    for i in range(5):
        f_x[i] -= (1.0 + Q2) * binom4[i]
    for i in range(7):
        f_x[i] += Q2 * binom6[i]
    g = np.zeros(POLY_LEN)
    for i in range(POLY_LEN - 1):
        g[i] = f_x[i + 1]
    u1 = np.zeros(POLY_LEN)
    u1[0] = 1.0
    u1[1] = -1.0
    return g, u1


@njit(cache=True)
def horizon_coefficients(d0, b, Q2, lam1, lam3, lam4, order):
    """Taylor coefficients d_0..d_order in x = 1 - u of the regular solution at the horizon."""
    g, u1 = _g_poly(Q2)
    u2 = _poly_mul(u1, u1)
    u3 = _poly_mul(u2, u1)
    g_x = np.zeros(POLY_LEN)
    for i in range(POLY_LEN - 1):
        g_x[i] = (i + 1) * g[i + 1]
    P = _poly_mul(u2, g)
    R = 3.0 * _poly_mul(u1, g) + 2.0 * b * _poly_mul(u3, g) + _poly_mul(u2, g_x)
    M = np.zeros(POLY_LEN)
    M[0] = -3.0 - lam1 * b
    M[1] = 2.0 * lam1 * b
    M[2] = -lam1 * b
    d = np.zeros(order + 1)
    d[0] = d0
    sq = np.zeros(order + 1)
    cube = np.zeros(order + 1)
    for n in range(order):
        acc = 0.0
        for k in range(n + 1):
            acc += d[k] * d[n - k]
        sq[n] = acc
        acc = 0.0
        for k in range(n + 1):
            acc += sq[k] * d[n - k]
        cube[n] = acc
        kmax = min(POLY_LEN - 1, n)
        lin = 0.0
        for k in range(kmax + 1):
            lin += M[k] * d[n - k]
        force = lin + 3.0 * lam3 * sq[n] + 4.0 * lam4 * cube[n]
        lower = 0.0
        for k in range(1, kmax + 1):
            m = n + 1 - k
            lower += P[k] * (m * m) * d[m]
        for k in range(0, min(POLY_LEN - 1, n - 1) + 1):
            m = n - k
            lower += R[k] * m * d[m]
        d[n + 1] = (force - lower) / (P[0] * (n + 1)**2)
    return d


@njit(cache=True)
def horizon_evaluate(d, eps):
    """chi and dchi/du at u = 1 - eps."""
    chi = 0.0
    dchi = 0.0
    for n in range(d.size - 1, 0, -1):
        chi = chi * eps + d[n]
        dchi = dchi * eps + n * d[n]
    chi = chi * eps + d[0]
    return chi, -dchi


@njit(cache=True)
def choose_horizon_eps(d, tol, eps_max):
    order = d.size - 1
    if d[0] == 0.0 or d[order] == 0.0:
        return eps_max
    return min(eps_max, (tol * abs(d[0]) / abs(d[order]))**(1.0 / order))


@njit(cache=True)
def extract_uv(z, chi, dchi_dz, lam1, lam3, lam4, k4):
    """Newton inversion of the UV series at zhat = z for (a, c).  Returns (a, c, converged)."""
    a = (3.0 * chi - z * dchi_dz) / (2.0 * z)
    c = (z * dchi_dz - chi) / (2.0 * z**3)
    size = abs(chi) + abs(z * dchi_dz)
    sa = size / z + 1e-300
    sc = size / z**3 + 1e-300
    for _ in range(40):
        S, J = uv_series(z, a, c, lam1, lam3, lam4, k4)
        r0 = chi - S[0]
        r1 = dchi_dz - S[1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if det == 0.0 or not np.isfinite(det):
            return np.nan, np.nan, False
        da = (r0 * J[1, 1] - r1 * J[0, 1]) / det
        dc = (J[0, 0] * r1 - J[1, 0] * r0) / det
        a += da
        c += dc
        rel = max(abs(da) / sa, abs(dc) / sc)
        if not np.isfinite(rel):
            return np.nan, np.nan, False
        if rel < 1e-13:
            return a, c, True
    return a, c, False


@njit(cache=True)
def chiral_shoot_core(d0, b, Q2, lam1, lam3, lam4, z_uv, order, series_tol, eps_max,
                      rtol, atol, blowup):
    """
    One shoot from the horizon value d0 to the UV matching point zhat = z_uv.
    Returns (a, c, status, eps): status 0 ok; 1-3 integrator; 4 extraction failed;
    5 matching point not inside the integration range.
    """
    zh_hat = np.sqrt(b)
    d = horizon_coefficients(d0, b, Q2, lam1, lam3, lam4, order)
    eps = choose_horizon_eps(d, series_tol, eps_max)
    u_x = z_uv / zh_hat
    if not (0.0 < u_x < 1.0 - eps):
        return np.nan, np.nan, 5, eps
    chi0, chip0 = horizon_evaluate(d, eps)
    y = np.array([chi0, chip0])
    par = np.array([KIND_CHIRAL, b, Q2, lam1, lam3, lam4])
    hbox = np.zeros(1)
    status = dop853(y, 1.0 - eps, u_x, par, 0.0, 1.0, _EMPTY, _EMPTY, rtol, atol, 200000, blowup, hbox)
    if status != 0 or not (np.isfinite(y[0]) and np.isfinite(y[1])):
        return np.nan, np.nan, max(status, 1), eps
    k4 = (1.0 + Q2) / (b * b)
    a, c, ok = extract_uv(z_uv, y[0], y[1] / zh_hat, lam1, lam3, lam4, k4)
    if not ok:
        return np.nan, np.nan, 4, eps
    return a, c, 0, eps


@njit(cache=True)
def chiral_scan_core(d0s, b, Q2, lam1, lam3, lam4, z_uv, order, series_tol, eps_max,
                     rtol, atol, blowup):
    """(a, c) along an array of horizon values (NaN where the shoot fails)."""
    n = d0s.size
    a = np.empty(n)
    c = np.empty(n)
    for i in range(n):          # serial: 100 shoots take ~3 ms, and callers may sit in process pools
        ai, ci, status, _ = chiral_shoot_core(d0s[i], b, Q2, lam1, lam3, lam4, z_uv, order,
                                              series_tol, eps_max, rtol, atol, blowup)
        a[i] = ai
        c[i] = ci
    return a, c


@njit(cache=True)
def chiral_on_grid(d0, b, Q2, lam1, lam3, lam4, u_grid, order, series_tol, eps_max,
                   rtol, atol, blowup):
    """
    chi and dchi/du on u_grid (ascending).  Points with u >= 1 - eps come from the
    horizon series; the others from the integration, which lands on every grid point.
    Returns (chi, chip, status).
    """
    n = u_grid.size
    chi = np.full(n, np.nan)
    chip = np.full(n, np.nan)
    d = horizon_coefficients(d0, b, Q2, lam1, lam3, lam4, order)
    eps = choose_horizon_eps(d, series_tol, eps_max)
    u_start = 1.0 - eps
    chi0, chip0 = horizon_evaluate(d, eps)
    y = np.array([chi0, chip0])
    par = np.array([KIND_CHIRAL, b, Q2, lam1, lam3, lam4])
    hbox = np.zeros(1)
    u = u_start
    for i in range(n - 1, -1, -1):
        if u_grid[i] >= u_start:
            chi[i], chip[i] = horizon_evaluate(d, 1.0 - u_grid[i])
            continue
        status = dop853(y, u, u_grid[i], par, 0.0, 1.0, _EMPTY, _EMPTY, rtol, atol, 200000, blowup, hbox)
        if status != 0 or not (np.isfinite(y[0]) and np.isfinite(y[1])):
            return chi, chip, max(status, 1)
        u = u_grid[i]
        chi[i] = y[0]
        chip[i] = y[1]
    return chi, chip, 0


# ---------------------------------------------------------------------------
# Meson equation
# ---------------------------------------------------------------------------
@njit(cache=True)
def meson_uv(w2, b, Q2, g5sq, a_src, u_i, u_m, cu0, cdu, chi, chip, rtol, atol):
    """
    Normalizable solution psi_1 = u^2 (1 + beta u^2 + ...), beta = (4b - w^2 + g5^2 a^2 b)/8,
    started at u_i and integrated to u_m.  Returns (psi, psi', status), complex.
    The neglected terms are O((w u_i)^4/192, (b u_i^2)^2) relative.
    """
    beta = (4.0 * b - w2 + g5sq * a_src * a_src * b) / 8.0
    psi = u_i * u_i * (1.0 + beta * u_i * u_i)
    psip = 2.0 * u_i + 4.0 * beta * u_i**3
    y = np.array([psi.real, psi.imag, psip.real, psip.imag])
    par = np.array([KIND_MESON, b, Q2, w2.real, w2.imag, g5sq])
    hbox = np.zeros(1)
    status = dop853(y, u_i, u_m, par, cu0, cdu, chi, chip, rtol, atol, 2000000, np.inf, hbox)
    return complex(y[0], y[1]), complex(y[2], y[3]), status


@njit(cache=True)
def infalling_series(w, b, Q2, g5sq, d_chi, order):
    """
    Frobenius coefficients p_0 = 1, p_1..p_order of psi_in = x^r sum p_n x^n, x = 1 - u,
    r = -i w/g0.  d_chi are the horizon Taylor coefficients of chi (zeros for the vector).
    The equation times u^2 f^2 is  x^2 A psi_xx + x B psi_x + C psi = 0 with
    A = u^2 g^2,  B = A + x (u^2 g g_x + u g^2 + 2 b u^3 g^2),  C = w^2 u^2 - x g5^2 g chi^2.
    """
    g, u1 = _g_poly(Q2)
    g0 = g[0]
    r = -1j * w / g0
    u2 = _poly_mul(u1, u1)
    u3 = _poly_mul(u2, u1)
    g2 = _poly_mul(g, g)
    g_x = np.zeros(POLY_LEN)
    for i in range(POLY_LEN - 1):
        g_x[i] = (i + 1) * g[i + 1]
    A = _poly_mul(u2, g2)
    inner = _poly_mul(u2, _poly_mul(g, g_x)) + _poly_mul(u1, g2) + 2.0 * b * _poly_mul(u3, g2)
    B = A.copy()
    for i in range(1, POLY_LEN):
        B[i] += inner[i - 1]
    chi_p = np.zeros(POLY_LEN)
    for i in range(min(POLY_LEN, d_chi.size)):
        chi_p[i] = d_chi[i]
    gchi2 = _poly_mul(g, _poly_mul(chi_p, chi_p))
    w2 = w * w
    C = np.zeros(POLY_LEN, dtype=np.complex128)
    for i in range(POLY_LEN):
        C[i] = w2 * u2[i]
        if i >= 1:
            C[i] -= g5sq * gchi2[i - 1]
    p = np.zeros(order + 1, dtype=np.complex128)
    p[0] = 1.0
    for n in range(1, order + 1):
        acc = 0.0j
        for k in range(1, min(n, POLY_LEN - 1) + 1):
            m = n - k
            s = m + r
            acc += (A[k] * s * (s - 1.0) + B[k] * s + C[k]) * p[m]
        p[n] = -acc / (g0 * g0 * n * (n + 2.0 * r))
    return p, r


@njit(cache=True)
def meson_infalling(w, b, Q2, g5sq, d_chi, eps_max, order, series_tol, u_m,
                    cu0, cdu, chi, chip, rtol, atol):
    """Infalling solution integrated from u = 1 - eps to u_m.  Returns (psi, psi', status, eps)."""
    p, r = infalling_series(w, b, Q2, g5sq, d_chi, order)
    eps = eps_max
    pn = abs(p[order])
    if pn > 0.0:
        eps = min(eps_max, (series_tol / pn)**(1.0 / order))
    x = eps
    s0 = 0.0j
    s1 = 0.0j
    for n in range(order, -1, -1):
        s0 = s0 * x + p[n]
        s1 = s1 * x + (n + r) * p[n]
    xr = np.exp(r * np.log(x))
    psi = xr * s0
    psip = -xr * s1 / x
    y = np.array([psi.real, psi.imag, psip.real, psip.imag])
    w2 = w * w
    par = np.array([KIND_MESON, b, Q2, w2.real, w2.imag, g5sq])
    hbox = np.zeros(1)
    status = dop853(y, 1.0 - eps, u_m, par, cu0, cdu, chi, chip, rtol, atol, 2000000, np.inf, hbox)
    return complex(y[0], y[1]), complex(y[2], y[3]), status, eps


@njit(cache=True)
def mode_function(w, b, Q2, g5sq, a_src, d_chi, chi_eps, cu0, cdu, chi, chip,
                  z_uv, u_m, order, series_tol, rtol, atol):
    """
    M(w) = W[psi_1, psi_in](u_m) f(u_m) exp(-b u_m^2)/u_m, times prod_{k<=order}(1 + 2r/k),
    which removes the poles of the Frobenius coefficients at 2r = -k (w = -i k g0/2)
    without adding zeros.  NaN if an integration fails.
    """
    zh_hat = np.sqrt(b)
    u_i = z_uv / zh_hat
    w2 = w * w
    p1, p1p, st1 = meson_uv(w2, b, Q2, g5sq, a_src, u_i, u_m, cu0, cdu, chi, chip, rtol, atol)
    p2, p2p, st2, _ = meson_infalling(w, b, Q2, g5sq, d_chi, chi_eps, order, series_tol, u_m,
                                      cu0, cdu, chi, chip, rtol, atol)
    if st1 != 0 or st2 != 0:
        return complex(np.nan, np.nan)
    um2 = u_m * u_m
    f = 1.0 - (1.0 + Q2) * um2 * um2 + Q2 * um2 * um2 * um2
    M = (p1 * p2p - p1p * p2) * f * np.exp(-b * um2) / u_m
    r = -1j * w / (2.0 * (2.0 - Q2))
    for k in range(1, order + 1):
        M *= (1.0 + 2.0 * r / k)
    return M


@njit(cache=True)
def _pole_factor_sq(w, Q2, order):
    r = -1j * w / (2.0 * (2.0 - Q2))
    out = 1.0
    for k in range(1, order + 1):
        out *= abs(1.0 + 2.0 * r / k)**2
    return out


@njit(cache=True, parallel=True)
def mode_function_array(ws, b, Q2, g5sq, a_src, d_chi, chi_eps, cu0, cdu, chi, chip,
                        z_uv, u_m, order, series_tol, rtol, atol):
    out = np.empty(ws.size, dtype=np.complex128)
    for i in prange(ws.size):
        out[i] = mode_function(ws[i], b, Q2, g5sq, a_src, d_chi, chi_eps, cu0, cdu, chi, chip,
                               z_uv, u_m, order, series_tol, rtol, atol)
    return out


@njit(cache=True)
def spectral_from_mode_function(w, M, b, Q2, order):
    """|Im(B/A)| at real w from M (undoing the pole factor)."""
    return 2.0 * w.real * np.exp(-b) * _pole_factor_sq(w, Q2, order) / (M.real**2 + M.imag**2)


@njit(cache=True)
def legacy_two_solutions(y0, u_i, u_f, w2, b, Q2, g5sq, cu0, cdu, chi, chip, rtol, atol):
    """
    psi_1 and psi_2 (real w) from y0 = [psi1, psi1', psi2, psi2'] at u_i to u_f.
    Returns ([psi1, psi1', psi2, psi2'], status).
    """
    y = np.array([y0[0], y0[2], y0[1], y0[3]])
    par = np.array([KIND_MESON_PAIR, b, Q2, w2, 0.0, g5sq])
    hbox = np.zeros(1)
    status = dop853(y, u_i, u_f, par, cu0, cdu, chi, chip, rtol, atol, 2000000, np.inf, hbox)
    return np.array([y[0], y[2], y[1], y[3]]), status


@njit(cache=True, parallel=True)
def legacy_two_solutions_array(y0s, u_i, u_f, w2s, b, Q2, g5sq, cu0, cdu, chi, chip, rtol, atol):
    """legacy_two_solutions for an array of frequencies (threaded)."""
    n = w2s.size
    out = np.empty((n, 4))
    status = np.empty(n, dtype=np.int64)
    for i in prange(n):
        y, st = legacy_two_solutions(y0s[i], u_i, u_f, w2s[i], b, Q2, g5sq, cu0, cdu, chi, chip, rtol, atol)
        out[i] = y
        status[i] = st
    return out, status
