#!/usr/bin/env python3
"""Compare Im(B/A) from the current code, the direct numba evaluation, and the exact
Wronskian formula. Usage: python check_wronskian.py [T_MeV]  (default 17)."""
import sys

from common import AxialBackground

T = float(sys.argv[1]) if len(sys.argv) > 1 else 17.0
bg = AxialBackground(T=T)
print(f"T = {T} MeV: (mu_g z_h)^2 = {(bg.mug*bg.zh)**2:.1f}")
print(f"{'w':>7} {'current':>12} {'direct 1e-8':>12} {'direct 1e-11':>12} {'exact W':>12} {'W_num/W_exact':>14}")
for w in [800., 1000., 1250., 1512.7, 1700., 2052., 2300.]:
    cur = bg.BA_current(w).imag
    d8 = bg.BA_numba(w, 1e-8, 1e-10).imag
    d11 = bg.BA_numba(w, 1e-11, 1e-13).imag
    ex, ratio = bg.Im_BA_exact(w, 1e-11, 1e-13, return_wronskian_ratio=True)
    print(f"{w:7.1f} {cur:12.4e} {d8:12.4e} {d11:12.4e} {ex:12.4e} {ratio:14.4e}")
