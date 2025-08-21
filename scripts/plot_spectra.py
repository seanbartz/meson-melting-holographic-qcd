#!/usr/bin/env python3
"""
Plot axial and vector spectral functions at T=40 MeV and T=41 MeV.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Determine base directory (MesonMelting folder)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def plot_spectra(axial_file, vector_file, T):
    # Read CSV (skip any comment lines starting with //)
    df_axial = pd.read_csv(axial_file, comment='/')
    df_vector = pd.read_csv(vector_file, comment='/')
    # Choose spectral column depending on file format
    axial_col = 'spectral_function_abs' if 'spectral_function_abs' in df_axial.columns else 'spectral_function'
    vector_col = 'spectral_function' if 'spectral_function' in df_vector.columns else 'spectral_function_abs'

    # Compute dimensionless omega^2 for axial (mu_g = 388 MeV)
    MUG = 388.0
    df_axial['omega_sq_dim'] = (df_axial['omega']**2) / (MUG**2)
    # Vector already has dimensionless column
    if 'omega_squared_dimensionless' in df_vector.columns:
        df_vector['omega_sq_dim'] = df_vector['omega_squared_dimensionless']
    else:
        df_vector['omega_sq_dim'] = (df_vector['omega']**2) / (MUG**2)

    # First plot: spectral function vs omega^2/mu_g^2
    plt.figure(figsize=(8, 5))
    #make all font sizes larger
    plt.rcParams.update({'font.size': 14})
    plt.plot(df_axial['omega_sq_dim'], df_axial[axial_col], label='Axial', lw=2)
    plt.plot(df_vector['omega_sq_dim'], df_vector[vector_col], label='Vector', lw=2)
    plt.xlabel('$\omega^2/\mu_g^2$')
    plt.ylabel('Spectral function')
    # plt.title(f'Meson spectra at T={T} MeV (\mu=0)')
    plt.ylim(0, 300)
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'spectra_T{int(T)}.png')
    plt.close()

    # Second plot: spectral function divided by omega^2
    # compute original omega in MeV
    df_axial['omega2'] = df_axial['omega']**2
    df_vector['omega2'] = df_vector['omega']**2
    y1 = df_axial[axial_col] / df_axial['omega2']
    y2 = df_vector[vector_col] / df_vector['omega2']
    # determine y-axis limit
    y_max = max(y1.max(), y2.max())

    plt.figure(figsize=(8, 5))
    plt.plot(df_axial['omega_sq_dim'], y1, label='Axial / $\omega^2$', lw=2)
    plt.plot(df_vector['omega_sq_dim'], y2, label='Vector / $\omega^2$', lw=2)
    plt.xlabel('$\omega^2/\mu_g^2$')
    plt.ylabel('Spectral function / $\omega^2$')
    # plt.title(f'Meson spectra $/\omega^2$ at T={T} MeV')
    # plt.xlim(0, 300)
    plt.ylim(0, .0003)  # 388 MeV is mu_g
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'spectra_by_omega2_T{int(T)}.png')
    plt.close()


if __name__ == '__main__':
    # Paths relative to the repository structure
    base = BASE_DIR + os.sep
    # T = 40 MeV
    plot_spectra(
        os.path.join(base, 'axial_data', 'axial_spectral_data_T40.0_mu0.0_mq0.1_lambda17.4.csv'),
        os.path.join(base, 'data', 'spectral_data_T40.0_mu0.0_20250408_153052.csv'),
        40
    )
    # T = 41 MeV
    plot_spectra(
        os.path.join(base, 'axial_data', 'axial_spectral_data_T41.0_mu0.0_mq0.1_lambda17.4.csv'),
        os.path.join(base, 'data', 'spectral_data_T41.0_mu0.0_20250408_153109.csv'),
        41
    )
