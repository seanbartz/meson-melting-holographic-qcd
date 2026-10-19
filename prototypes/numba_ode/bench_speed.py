#!/usr/bin/env python3
"""Time scipy (current code) vs numba DOPRI5 for the axial B/A and chiral solves."""
import time

import numpy as np
from scipy.integrate import solve_ivp

from common import AxialBackground
import chiral_solve_complete as cs
from nb_ode import dopri5, chiral_rhs

T, mu, lambda1 = 17.0, 0.0, 7.438
bg = AxialBackground(T=T, mu=mu, lambda1=lambda1)
ws = np.linspace(700, 2400, 40)

# Axial B/A per omega
t = time.perf_counter()
for w in ws:
    bg.BA_current(w)
t_scipy = (time.perf_counter() - t) / len(ws)
bg.BA_numba(1000.0)  # compile
t = time.perf_counter()
for w in ws:
    bg.BA_numba(w)
t_numba = (time.perf_counter() - t) / len(ws)
print(f"Axial B/A per omega: scipy {t_scipy*1e3:.1f} ms | numba {t_numba*1e3:.2f} ms "
      f"| speedup {t_scipy/t_numba:.0f}x")

# Chiral field from the horizon, one d0
d0, ui, uf = 0.5, 1e-2, 1 - 1e-4
mu_g, v3, v4 = 440.0, cs.v3, cs.v4
zh, q = cs.blackness(T, mu)
Q = q * zh**3
d1 = (3*d0 - 3*d0**2*v3 - 4*d0**3*v4 + d0*zh**2*lambda1*mu_g**2) / (2*(-2 + Q**2))
d2 = (1/(16*(-2 + Q**2)**2))*(6*d1*(-6 + Q**2 + Q**4) + 4*d0**3*(14 - 13*Q**2)*v4
      + d0**2*((42 - 39*Q**2)*v3 - 24*d1*(-2 + Q**2)*v4)
      - 2*d1*(-2 + Q**2)*zh**2*(-8 + 4*Q**2 - lambda1)*mu_g**2
      + 3*d0*(-14 + 13*Q**2 + 8*d1*v3 - 4*d1*Q**2*v3 + (-2 + 3*Q**2)*zh**2*lambda1*mu_g**2))
y0 = np.array([d0 + d1*(1 - uf) + d2*(1 - uf)**2, -d1 - 2*d2*(1 - uf)])
grid = np.linspace(uf, ui, 10000)
p = np.array([v3, v4, lambda1, mu_g, zh, q])

t = time.perf_counter()
for _ in range(20):
    _, chi_cur, _, _ = cs.chiral_solve_IR(d0, lambda1, T, mu, ui, uf)
t_cur = (time.perf_counter() - t) / 20
args = ((v3, v4, lambda1, mu_g, 0, zh, q),)
t = time.perf_counter()
ref = solve_ivp(cs.chiral, [uf, ui], y0, t_eval=grid, args=args, rtol=1e-10, atol=1e-12)
t_tight = time.perf_counter() - t
chi_ref = ref.y[0][::-1]
for rtol, atol, label in [(1e-3, 1e-6, 'default tol'), (1e-10, 1e-12, 'rtol 1e-10')]:
    dopri5(chiral_rhs, p, uf, ui, y0, rtol, atol, grid)  # compile
    t = time.perf_counter()
    for _ in range(20):
        _, out, _ = dopri5(chiral_rhs, p, uf, ui, y0, rtol, atol, grid)
    t_nb = (time.perf_counter() - t) / 20
    print(f"Chiral solve, numba {label}: {t_nb*1e3:.2f} ms | chi[10] = {out[::-1, 0][10]:.8e}")
print(f"Chiral solve, current code (RK45 default tol): {t_cur*1e3:.1f} ms | chi[10] = {chi_cur[10]:.8e}")
print(f"Chiral solve, scipy RK45 rtol 1e-10: {t_tight*1e3:.0f} ms | chi[10] = {chi_ref[10]:.8e}")
