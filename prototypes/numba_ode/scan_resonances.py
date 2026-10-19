#!/usr/bin/env python3
"""Scan the axial spectrum with the exact Wronskian formula on a 1 MeV grid and refine
local maxima of log10 |Im(B/A)|. Usage: python scan_resonances.py [T_MeV]  (default 17)."""
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from common import AxialBackground

T = float(sys.argv[1]) if len(sys.argv) > 1 else 17.0
bg = AxialBackground(T=T)
ws = np.arange(700., 2400., 1.0)

t = time.perf_counter()
L = np.array([np.log10(abs(bg.Im_BA_exact(w))) for w in ws])
print(f"T = {T}: {len(ws)} omegas, {(time.perf_counter() - t)/len(ws)*1e3:.1f} ms each")

neg_log = lambda w: -np.log10(abs(bg.Im_BA_exact(w)))
print("local maxima (omega MeV, log10 height, (omega/mu_g)^2):")
for i in find_peaks(L)[0]:
    r = minimize_scalar(neg_log, bracket=(ws[i] - 1, ws[i], ws[i] + 1), tol=1e-12)
    print(f"  {r.x:9.3f}  {-r.fun:7.2f}  {(r.x/bg.mug)**2:6.2f}")
