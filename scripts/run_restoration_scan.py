#!/usr/bin/env python3
"""
Parameter scan for restoration_analysis.py.

Scan 1 ("phase"): lam1 x kappa x mq sqrt(lam4) at lam4 = 4.2, mu = 0.  kappa = gamma^2/lam4 is
quoted through the gamma it corresponds to at lam4 = 4.2, and mq sqrt(lam4) through mq at 4.2.
Scan 2 ("lam4"): lam4 varied at fixed (lam1, kappa, mq sqrt(lam4)): the phase structure is
identical along it, only the axial coupling g5^2 chi^2 ~ 1/lam4 changes.
Scan 3 ("mu"): selected sets at mu > 0.

    python scripts/run_restoration_scan.py phase|lam4|mu [n_jobs]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import restoration_analysis as ra  # noqa: E402

LAM4_REF = 4.2


def reduced(lam1, gamma_at_ref, mq_at_ref, lam4=LAM4_REF, mu=0.0):
    return (lam1, gamma_at_ref**2 / LAM4_REF, mq_at_ref * np.sqrt(LAM4_REF), lam4, mu)


def main():
    which = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which == 'phase':
        points = [reduced(l1, g, mq)
                  for l1 in (5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.2, 6.5, 7.0, 7.438)
                  for g in (-18.0, -20.0, -22.6, -24.0, -26.0)
                  for mq in (3.0, 9.0, 15.0)]
    elif which == 'lam4':
        points = [reduced(l1, g, mq, lam4)
                  for (l1, g, mq) in ((5.5, -22.6, 9.0), (5.7, -22.6, 9.0), (6.0, -22.6, 9.0), (7.438, -22.6, 9.0),
                                      (5.5, -18.0, 9.0), (5.5, -26.0, 9.0))
                  for lam4 in (0.5, 1.0, 2.0, 4.2, 8.0, 16.0)]
    elif which == 'mu':
        points = [reduced(l1, g, mq, LAM4_REF, mu)
                  for (l1, g, mq) in ((5.5, -22.6, 9.0), (5.7, -22.6, 9.0), (6.0, -22.6, 9.0), (7.438, -22.6, 9.0))
                  for mu in (0.0, 100.0, 200.0, 300.0)]
    else:
        raise SystemExit(__doc__)
    out = os.path.join('results', f'restoration_scan_{which}.csv')
    df = ra.scan(points, n_jobs=n_jobs, out=out)
    print(f"{len(df)} rows -> {out}")
    if 'error' in df:
        print(df[df['error'].notna()][['lam1', 'gamma', 'lam4', 'mq', 'mu', 'error']].to_string())


if __name__ == '__main__':
    main()
