"""
Derive the near-boundary (UV) series of the chiral field chi and write it out as
numpy code in ``chiral_uv_series_generated.py`` (repository root).

Run by hand, once, whenever the chiral equation of motion changes:

    python derivations/derive_chiral_uv_series.py

It is not imported at run time.  The generated file is imported by
``chiral_solve_complete.py``, which uses it to turn the numerical solution at the
UV matching point into the quark mass and the condensate.

Ported from the 2+1-flavor code (seanbartz/2p1-critical-point,
derivations/derive_uv_series.py) with chi_l = chi_s = chi; on that diagonal the
2+1-flavor equations reduce exactly to the equation used here.

Equation (dimensionless; energy unit mu_g)
------------------------------------------
With zhat = mu_g z (written ``z`` below), the quadratic dilaton Phi = zhat^2 and the
blackness function truncated at the order that matters here, f = 1 - k4 zhat^4 with
k4 = (1 + Q^2) / zhat_h^4, the equation of motion of chiral_solve_complete.chiral
times zhat^2 f is

    z^2 f chi'' - (3 z f + 2 z^3 f - z^2 f') chi' - F = 0,
    F = (-3 - lam1 z^2) chi + 3 lam3 chi^2 + 4 lam4 chi^3,      lam3 = v3, lam4 = v4.

Ansatz (L = ln zhat, so the renormalization scale inside the logarithm is mu_g):

    chi = a z + B z^2 + (c + D L) z^3 + (E + Fl L) z^4 + (G + H L) z^5 + O(z^6 L^2)

with a = m_q zeta / mu_g the source and c = sigma / (zeta mu_g^3) the condensate.
The older extraction kept only a z + B z^2 + c z^3, so its "sigma" absorbed
zeta mu_g^3 D ln(zhat_i): a dependence on the matching point (PROJECT_ASSESSMENT.md
section 3.3, item 2).
"""

import pathlib

import sympy as sp

z, L = sp.symbols("z L")
a, c = sp.symbols("a c")
lam1, lam3, lam4, k4 = sp.symbols("lam1 lam3 lam4 k4")
B, D, E, Fl, G, H = sp.symbols("B D E Fl G H")

chi = a * z + B * z**2 + (c + D * L) * z**3 + (E + Fl * L) * z**4 + (G + H * L) * z**5


def ddz(expr):
    """d/dz treating L = ln z as a function of z."""
    return sp.diff(expr, z) + sp.diff(expr, L) / z


f = 1 - k4 * z**4
fp = sp.diff(f, z)
force = (-3 - lam1 * z**2) * chi + 3 * lam3 * chi**2 + 4 * lam4 * chi**3
res = sp.expand(z**2 * f * ddz(ddz(chi)) - (3 * z * f + 2 * z**3 * f - z**2 * fp) * ddz(chi) - force)


def coeff(expr, n, k):
    return sp.expand(expr).coeff(z, n).coeff(L, k)


solution = {}
for n, k, unknown in [(2, 0, B), (3, 0, D), (4, 1, Fl), (4, 0, E), (5, 1, H), (5, 0, G)]:
    sol = sp.solve(coeff(res, n, k).subs(solution), unknown, dict=True)
    assert len(sol) == 1, (n, k, unknown, sol)
    solution[unknown] = sp.factor(sp.simplify(sol[0][unknown]))

r = sp.expand(res.subs(solution))
for n in range(0, 6):
    for k in range(0, 4):
        assert sp.simplify(coeff(r, n, k)) == 0, (n, k)
print("Residual vanishes through z^5 (all powers of ln z).")
for key, val in solution.items():
    print(f"{key} = {val}")

chi_full = chi.subs(solution)
S = [chi_full, ddz(chi_full)]
params = [a, c]
JS = [[sp.diff(s, p) for p in params] for s in S]
flat = [e.subs(sp.log(z), L) for e in S + [x for row in JS for x in row]]
replacements, reduced = sp.cse(flat, symbols=sp.numbered_symbols("t"), optimizations="basic")

out = []
out.append('"""')
out.append("GENERATED FILE -- do not edit.  Written by derivations/derive_chiral_uv_series.py.")
out.append("")
out.append("UV series of the chiral field through zhat^5, including the zhat^3 ln, zhat^4 ln")
out.append("and zhat^5 ln terms, and its Jacobian with respect to (a, c).  z is zhat = mu_g z;")
out.append("L = ln(zhat), i.e. the scale inside the logarithm is mu_g.")
out.append('"""')
out.append("import numpy as np")
out.append("")
out.append("")
out.append("def uv_series(z, a, c, lam1, lam3, lam4, k4):")
out.append('    """Return S = [chi, dchi/dzhat] and JS = dS/d(a, c) (2x2)."""')
out.append("    L = np.log(z)")
for sym, expr in replacements:
    out.append(f"    {sym} = {sp.pycode(expr)}")
out.append("    S = np.empty(2)")
out.append("    JS = np.empty((2, 2))")
out.append(f"    S[0] = {sp.pycode(reduced[0])}")
out.append(f"    S[1] = {sp.pycode(reduced[1])}")
for i in range(2):
    for j in range(2):
        out.append(f"    JS[{i}, {j}] = {sp.pycode(reduced[2 + 2 * i + j])}")
out.append("    return S, JS")
out.append("")
out.append("")
out.append("def uv_log_coefficient(a, lam1, lam3, lam4):")
out.append('    """Coefficient D of zhat^3 ln(zhat); it sets the scheme dependence of sigma."""')
out.append(f"    return {sp.pycode(solution[D])}")
out.append("")
text = "\n".join(out).replace("math.", "np.")

target = pathlib.Path(__file__).resolve().parent.parent / "chiral_uv_series_generated.py"
target.write_text(text)
print(f"wrote {target}")
