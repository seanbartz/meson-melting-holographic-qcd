#!/usr/bin/env python3
"""
Figures for the restoration analysis (written to results/):

    python scripts/plot_restoration.py modes LAM1 GAMMA LAM4 MQ [MU]   # sigma, masses, widths vs T
    python scripts/plot_restoration.py spectra LAM1 GAMMA LAM4 MQ [MU] # spectral functions around T_chi
    python scripts/plot_restoration.py scan                             # summary of the phase scan
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import meson_spectra as ms            # noqa: E402
import restoration_analysis as ra     # noqa: E402

OUT = 'results'
C_V, C_A = '#1f77b4', '#d62728'


def _tag(model, mu):
    return f"lam1_{model.lam1:g}_gamma_{model.gamma:g}_lam4_{model.lam4:g}_mq_{model.mq:g}_mu_{mu:g}"


def _title(model, mu):
    return (rf"$\lambda_1={model.lam1:g}$, $\gamma={model.gamma:g}$, $\lambda_4={model.lam4:g}$, "
            rf"$m_q={model.mq:g}$ MeV, $\mu={mu:g}$ MeV")


def plot_modes(model, mu):
    row, tr, modes = ra.summarize(model, mu, detail=True)
    T, V, A = modes['T'], modes['vector'], modes['axial'][:, 0]
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    ax[0].plot(T, modes['sigma13'], 'k-')
    ax[0].set_ylabel(r'$\sigma^{1/3}$ (MeV)')
    ax[1].plot(T, V.real, color=C_V, label=r'vector ($\rho$)')
    ax[1].plot(T, A.real, color=C_A, label=r'axial ($a_1$)')
    ax[1].set_ylabel(r'mass $M=\mathrm{Re}\,\omega$ (MeV)')
    ax[1].legend(frameon=False)
    ax[2].plot(T, -2 * V.imag / V.real, color=C_V)
    ax[2].plot(T, -2 * A.imag / A.real, color=C_A)
    ax[2].axhline(1.0, color='0.6', lw=0.8, ls=':')
    ax[2].set_ylabel(r'$\Gamma/M$')
    ax[2].set_xlabel('T (MeV)')
    kind = 'first order' if tr['order'] == 'first' else 'crossover'
    for a in ax:
        a.axvline(tr['T_chi'], color='0.4', lw=0.8, ls='--')
    ax[0].set_title(_title(model, mu) + f"\n{kind}, $T_\\chi$ = {tr['T_chi']:.1f} MeV", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    name = os.path.join(OUT, f"modes_{_tag(model, mu)}.png")
    fig.savefig(name, dpi=160)
    plt.close(fig)
    return name


def plot_spectra(model, mu):
    tr = ra.chiral_transition(model, mu)
    Tc = tr['T_chi']
    step = 0.05 if tr['order'] == 'first' else 3.0
    temps = [Tc - 30.0, Tc - 15.0, Tc - step, Tc + step, Tc + 15.0]
    ws = np.linspace(200.0, 3000.0, 1401)
    fig, ax = plt.subplots(1, len(temps), figsize=(3.1 * len(temps), 3.4), sharey=True)
    for a, T in zip(ax, temps):
        branch = 'stable' if tr['order'] != 'first' else ('broken' if T < Tc else 'restored')
        bg = ms.Background(T, mu, model)
        a.plot(ws, bg.channel('vector').rho_normalized(ws), color=C_V, label='vector')
        a.plot(ws, bg.channel('axial', branch).rho_normalized(ws), color=C_A, label='axial')
        a.set_title(f"T = {T:.1f} MeV", fontsize=9)
        a.set_xlabel(r'$\omega$ (MeV)')
    ax[0].set_ylabel(r'$|\mathrm{Im}\,B/A|/(\omega z_h)^2$')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_ylim(0, 6)
    fig.suptitle(_title(model, mu) + f",  $T_\\chi$ = {Tc:.1f} MeV ({tr['order']})", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    name = os.path.join(OUT, f"spectra_{_tag(model, mu)}.png")
    fig.savefig(name, dpi=160)
    plt.close(fig)
    return name


def plot_scan():
    import pandas as pd
    d = pd.read_csv(os.path.join(OUT, 'restoration_scan_phase.csv'))
    d = d[np.isfinite(d['T_degenerate']) & (d['dist_above'] < 0.1)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sc = ax[0].scatter(d['T_chi'], d['GV_over_MV'], c=d['lam1'], cmap='viridis', s=18,
                       marker='o', label=r'$\rho$')
    ax[0].scatter(d['T_chi'], d['GA_over_MA'], c=d['lam1'], cmap='viridis', s=18, marker='^', label=r'$a_1$')
    ax[0].axvline(102.8, color='0.5', ls='--', lw=0.8)
    ax[0].text(103.5, 0.05, 'vector potential well\ndisappears', fontsize=7, color='0.4')
    ax[0].set_xlabel(r'$T_\chi$ (MeV)')
    ax[0].set_ylabel(r'$\Gamma/M$ just below $T_\chi$')
    ax[0].legend(frameon=False)
    fig.colorbar(sc, ax=ax[0], label=r'$\lambda_1$')
    sc = ax[1].scatter(d['GV_over_MV'], d['split_below'], c=d['lam1'], cmap='viridis', s=18)
    ax[1].set_xlabel(r'$\Gamma_V/M_V$ at $T_\chi$')
    ax[1].set_ylabel(r'$(M_A-M_V)/M_V$ just below $T_\chi$')
    fig.colorbar(sc, ax=ax[1], label=r'$\lambda_1$')
    fig.suptitle(r'Scan at $\mu=0$, $\lambda_4=4.2$: $\lambda_1$, $\gamma^2/\lambda_4$, $m_q\sqrt{\lambda_4}$', fontsize=10)
    fig.tight_layout()
    name = os.path.join(OUT, 'restoration_scan_phase.png')
    fig.savefig(name, dpi=160)
    plt.close(fig)
    return name


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'scan':
        print(plot_scan())
    else:
        lam1, gamma, lam4, mq = [float(x) for x in sys.argv[2:6]]
        mu = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        model = ms.Model(lam1=lam1, gamma=gamma, lam4=lam4, mq=mq)
        print(plot_modes(model, mu) if what == 'modes' else plot_spectra(model, mu))
