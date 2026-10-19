# numba ODE prototype

These are standalone prototypes behind §3.4b and §4.5 of `PROJECT_ASSESSMENT.md`. They import the existing `axial_spectra.py` and `chiral_solve_complete.py` without changing them. Nothing in the main code uses these files yet.

| File | Purpose |
|---|---|
| `nb_ode.py` | `@njit` Dormand–Prince 5(4) adaptive integrator (`dopri5`), plus right-hand sides for the axial field (`axial_rhs`) and the chiral field (`chiral_rhs`). |
| `common.py` | Builds the chiral background with `axial_spectra`. Provides `BA_current` (the unchanged code), `BA_numba` (same formula, numba integrator) and `Im_BA_exact` (the exact-Wronskian formula). |
| `bench_speed.py` | Timing comparison: scipy vs numba for the axial B/A and the chiral solve. |
| `check_wronskian.py [T]` | Shows that the current Im(B/A) is round-off noise at low T and matches the exact formula at higher T. |
| `scan_resonances.py [T]` | Scans the spectrum with the exact formula and refines its local maxima. |

## Running

```bash
pip install numpy scipy pandas joblib matplotlib numba
cd prototypes/numba_ode
python bench_speed.py
python check_wronskian.py 17
python check_wronskian.py 60
python scan_resonances.py 17
```

The first call compiles the numba functions. `cache=True` stores the compiled code in `__pycache__/`.

`AxialBackground` calls `axial_spectra.initialize_chiral_field`. That function logs to `sigma_data/sigma_calculations.csv` in the current directory, which is git-ignored.

The model parameters are the current defaults of those modules: μ_g = 440, γ = −22.6, λ₄ = 4.2.

## Key formula

W = ψ₁ψ₂′ − ψ₂ψ₁′ obeys W′ = (1/u − f′/f + φ′) W. With c = −iωz_h/(2(2−Q²)), the infalling factor is ψ_m = (1−u)^c. Then

Im(B/A) = k · W(u_f) / (ψ₁′(u_f)² + k² ψ₁(u_f)²),  where k = ωz_h / [2(2−Q²)(1−u_f)]

and

W(u_f) = W(u_i) (u_f/u_i) (f(u_i)/f(u_f)) e^{φ(u_f)−φ(u_i)}.

This needs only ψ₁, and it avoids the catastrophic cancellation in the current direct formula.

## Results (one core, recorded 2026-10-19)

| Solve | Current code | numba | Speedup |
|---|---|---|---|
| Axial B/A per ω | 206 ms | 3.0 ms | ≈70× |
| Chiral field, current default tolerance | 9.8 ms | 4.4 ms | ≈2× |
| Chiral field, rtol 1e-10 | 80 ms | 4.4 ms | ≈18× |

At T = 17 MeV the current Im(B/A) below about 2.3 GeV is noise. The numerical Wronskian there is off by up to 10²³. See `PROJECT_ASSESSMENT.md` §3.4b.

## Before using this in production

- **χ(u) and its interpolation.** Pass χ(u) on the grid the solver already uses. `np.interp` is linear; switch to cubic-spline coefficients if the interpolation error matters.
- **γ, λ₄, μ_g.** Feed these in from a single parameter module (assessment §3.2). Do not rely on the module defaults.
- **Speed.** Integrate only ψ₁ for the spectrum, which gives another 2×. Batch many ω inside one `@njit` loop, or use `prange`, to avoid per-call Python overhead.
- **Tests.** Add regression tests: the exact-W result against the direct method at T = 60 MeV, and the vector peaks against 4μ_g²(n+1) at low T.
