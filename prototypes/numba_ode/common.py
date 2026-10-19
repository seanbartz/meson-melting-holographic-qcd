"""
Shared setup for the numba ODE prototypes.

Builds the chiral background with the existing axial_spectra code and provides
three ways to evaluate the axial B/A at one frequency:
  - BA_current:   the unchanged axial_spectra.BA (scipy LSODA, two solutions)
  - BA_numba:     same direct formula, integrated with the numba DOPRI5 in nb_ode
  - Im_BA_exact:  Im(B/A) from the exact Wronskian, needing only psi_1
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import axial_spectra as ax  # noqa: E402
from nb_ode import dopri5, axial_rhs  # noqa: E402


class AxialBackground:
    """Chiral background and model constants at fixed (T, mu, mq, lambda1)."""

    def __init__(self, T=17.0, mu=0.0, mq=9.0, lambda1=7.438, ui=1e-2, uf=1 - 1e-4):
        self.T, self.mu, self.mq = T, mu, mq
        self.ui, self.uf = ui, uf
        ax.T, ax.mu, ax.mq = T, mu, mq
        (self.mug, self.zh, self.Q, self.lambda1, self.lambda3, self.lambda4,
         self.g5, self.zeta) = ax.calculate_variables(lambda1, T, mu)
        self.sigma = ax.initialize_chiral_field(mq, 0.01, T, mu, self.lambda1, ui, uf)
        self.chi_u = np.asarray(ax.chi_interp.x, dtype=float)
        self.chi = np.asarray(ax.chi_interp.y, dtype=float)

    def _uv_data(self, w):
        return ax.a_s(self.ui, w, self.g5, self.zh, self.mq, self.mug, self.zeta, self.sigma,
                      self.Q, self.lambda1, self.lambda3, self.lambda4)

    def _params(self, w):
        return np.concatenate(([w, self.zh, self.Q, self.mug, self.g5, len(self.chi_u)],
                               self.chi_u, self.chi))

    def BA_current(self, w):
        return ax.BA(self.ui, self.uf, w, self.g5, self.zh, self.mq, self.mug, self.zeta,
                     self.sigma, self.Q, self.lambda1, self.lambda3, self.lambda4)

    def BA_numba(self, w, rtol=1e-8, atol=1e-10):
        a1, a2 = self._uv_data(w)
        y0 = np.array([a1[0], a1[1], a2[0], a2[1]], dtype=float)
        y, _, _ = dopri5(axial_rhs, self._params(w), self.ui, self.uf, y0, rtol, atol, np.empty(0))
        psim = ax.psi_m(self.uf, w, self.zh, self.Q)
        return (psim[0] * y[3] - psim[1] * y[2]) / (psim[0] * y[1] - psim[1] * y[0])

    def Im_BA_exact(self, w, rtol=1e-10, atol=1e-12, return_wronskian_ratio=False):
        """Im(B/A) = k W(uf) / (psi1'^2 + k^2 psi1^2), with W from W' = (1/u - f'/f + phi') W.

        Only psi_1 is needed for the result; psi_2 is integrated here solely so the
        numerical Wronskian can be compared with the exact one.
        """
        a1, a2 = self._uv_data(w)
        y0 = np.array([a1[0], a1[1], a2[0], a2[1]], dtype=float)
        y, _, _ = dopri5(axial_rhs, self._params(w), self.ui, self.uf, y0, rtol, atol, np.empty(0))
        Q, ui, uf = self.Q, self.ui, self.uf
        f = lambda u: 1 - (1 + Q * Q) * u**4 + Q * Q * u**6
        W_ui = a1[0] * a2[1] - a2[0] * a1[1]
        W_uf = W_ui * (uf / ui) * (f(ui) / f(uf)) * np.exp((self.mug * self.zh)**2 * (uf**2 - ui**2))
        k = w * self.zh / (2 * (2 - Q * Q) * (1 - uf))
        im = k * W_uf / (y[1]**2 + k * k * y[0]**2)
        if return_wronskian_ratio:
            return im, (y[0] * y[3] - y[2] * y[1]) / W_uf
        return im
