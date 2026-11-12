# Project Assessment: Status, Corrections, Efficiency, and Path to a PRD Paper

*Assessment date: 2026-10-18 (updated 2026-10-19 with the numba and Wronskian findings; updated 2026-11-05 with the Obsidian cluster data, §8). Covers `main` at commit `2a755a4`, which is also what the cluster checkout is on. No code was changed for this review.*

This document reviews the whole repository against the goals in `ScienceGoals.md`. It covers:

1. what exists and what has actually been computed
2. what must be corrected before any result is publishable
3. where the code can be made much faster
4. how far the project is from a Physical Review D paper
5. a staged plan to get there
6. what is actually on the Obsidian cluster, and how it changes items 1–5 (§8)

File references use `file:line`.

---

## 1. Executive summary

**Bottom line:** the code base has the right ingredients for the paper: a chiral background solver, axial and vector spectral functions, phase-diagram and critical-point search, and cluster batch tooling. However, **none of the headline results needed for the paper are yet in a publishable state.** Three issues are serious enough to invalidate parts of the existing results. They must be fixed before production runs:

1. **Wrong chiral branch and wrong T_c in the first-order region.**
   - When several σ solutions exist, every spectral and melting code path picks the *smallest positive σ* (`axial_spectra.py:504-521`, `axial_melting_scan.py:55-68`, `axial_potential_critical.py:30-45`).
   - The first-order T_c is defined as the *lowest T with three solutions* (`critical_zoom_improved.py:176-199`). That is the spinodal, not the transition.
   - Nothing in the repository computes a free energy. As a result:
     - the first-order T_c values are biased low;
     - the spectra in the coexistence region come from the wrong, metastable branch;
     - the goal "demonstrate a first-order transition in axial spectra" cannot be met as the code stands.

2. **Model parameters are inconsistent between the phase diagram and the spectra.**
   - γ = −22.4 is the default in the phase-diagram tools. γ = −22.6 is hardcoded in the chiral solver default (`chiral_solve_complete.py:15`), in `axial_spectra.py:97` and in `critical_zoom_improved.py:91,217`.
   - **`axial_spectra.py` cannot take γ or λ₄ at all** (`main()` at `axial_spectra.py:920`; `calculate_sigma_values` is called without v3/v4 at line 494). So the goal "assess the effect of γ and λ₄ on meson spectra" is currently impossible.
   - Legacy data files were renamed with `gamma_-22.4` even though they were produced with −22.6 (`rename_cp_data_files.py:43`, `rename_axial_melting_files.py:106`, `fix_phase_plot_names.sh:60,77`). Their provenance is now mislabeled.
   - **The cluster's γ = −22.4 data are not wasted.** The paper's reference point is γ = −22.6 (see item 3), and the plan is to scan the parameters around it. The −22.4 sets are a correctly labeled scan point next to the reference, alongside −18, −20, −24 and −26. What is missing is the reference point itself, γ = −22.6, which has to be added to the scan.
   - *Cluster check (§8.4):* none of the Obsidian data is mislabeled. The SLURM logs show every `gamma_-22.4` cluster file really used γ = −22.4. The mislabel risk is confined to the pre-September 2025 files in the local Dropbox copy.

3. **Two values of μ_g are in use.**
   - The code hardcodes μ_g = 440 MeV everywhere (`chiral_solve_complete.py:272`, `axial_spectra.py:59`, `vector_spectra.py:43`, output paths `mu_g_440/`).
   - The notebooks that compare with experiment use **μ_g = 388 MeV** (`Axial.ipynb`, `Vector_Spectra.ipynb`: `return 388 #440`). Only with 388 does the ρ come out at 775 MeV; with 440 the ρ sits at 880 MeV.
   - Even at 388 the a₁ ground state is not reproduced: the model gives about 1513 MeV against 1230 MeV measured.
   - The zero-temperature calibration, which every finite-T claim rests on, has not been settled.
   - ***Decision (2026-11-05):* μ_g = 440 MeV, γ = −22.6 and λ₄ = 4.2 are fixed, to match the author's earlier published paper with this model.** With μ_g = 440 the ρ sits at about 880 MeV. The paper should say so and point to the earlier paper for the choice; the scale is set by consistency with that paper, not by the ρ mass. The 388 MeV notebooks become a side remark at most.
   - *Cluster check (§8.4):* all cluster data, including `vector_melting.csv`, used μ_g = 440. I recomputed the vector line and it matches 440 to 0.1 MeV.

4. **The low-temperature spectral functions are round-off noise.** *(Added 2026-10-19, §3.4b.)*
   - At T ≈ 17 MeV the true Im(B/A) below about 2 GeV is 10⁻³⁷ to 10⁻¹⁸ away from resonances. The current code returns 10⁻⁷ to 10⁻³ there.
   - The cause is that Im(B/A) is computed from a numerical Wronskian of two solutions that grow by about e⁶⁸. That Wronskian is wrong by up to a factor of 10²³.
   - So the "T ≈ 0" axial spectra in `Axial.ipynb` and the default T = 17 run of `axial_spectra.py` cannot be trusted below about 2 GeV.
   - There is an exact closed-form fix that is also twice as fast (§3.4b), and numba speeds up the axial solve by about 70× (§4).

**Distance to a paper:** the infrastructure is perhaps 70% done. The validated physics is still perhaps 15–20% done. However, the cluster data (§8) already scout most of the parameter space, so the production phase can be targeted rather than exploratory.
- **The cluster holds a 33-set phase-diagram survey** (Sept 2025), at μ = 0–400 MeV, all with μ_g = 440:
  - a λ₁ scan at m_q = 12, γ = −22.4, λ₄ = 4.2;
  - a γ scan with γ ∈ {−18, −20, −24, −26}, m_q ∈ {9, 12, 15} and λ₁ ∈ {5.5, 6.0, 7.0}, at λ₄ = 4.0;
  - axial melting lines for 22 of these sets.
- **Seven clean CEP brackets** (§8.3). The best one: at m_q = 12, λ₁ = 5.9, γ = −22.4, combining `batch_990_5.out` with the job-1002 rerun gives **μ_CEP ∈ (157.9, 160) MeV, T ≈ 108.0 MeV**. The μ = 0 critical coupling is λ₁,c ∈ (5.9, 6.0). These numbers are all at γ = −22.4; at the chosen γ = −22.6, expect μ_CEP to be lower by roughly tens of MeV (§8.5).
- All of these were computed with the inaccurate solver settings of §3.3, so they are scouting data and must be recomputed before publication. The spinodal issue (1 above) turns out to shift T_c by only ≲ 0.5 MeV near the CEPs, and it does not move the CEP μ.
- `vector_melting.csv` is μ_g = 440 with the potential-well criterion, the same as the axial lines. There are also scattered single-point results in `WorkbookJuly2025.ipynb`.
- There are no spectra on the cluster, no λ₄ variation beyond 4.0/4.2, no axial/vector degeneracy analysis, and no branch-resolved spectra.
- **The cluster's bookkeeping is broken.** `sigma_calculations.csv` is corrupted by concurrent writes, which silently disabled σ reuse; 27% of its rows are duplicates. Both task summaries are wrong or empty. The cluster Python is 3.6.8, which is too old for `capture_output` and for current numba.

A realistic first paper is within reach once the corrections in §3 and the validation in §5 (Phases 1–2) are done. The production runs are not the bottleneck; the cluster tooling can handle them.

---

## 2. What the code base contains

### 2.1 Physics core (keep, fix, consolidate)

| File | Role | State |
|---|---|---|
| `chiral_solve_complete.py` | Solves the chiral EOM in the AdS–RN + quadratic dilaton background by shooting from the horizon. Extracts (m_q, σ) and finds all σ for a given m_q. Logs results to `sigma_data/sigma_calculations.csv`. | Works. Has accuracy issues (§3.3) and is slow (§4.2). |
| `axial_spectra.py` | Axial spectral function Im(B/A) from UV series plus horizon infalling BC and Wronskian matching. Includes peak finding, caching, plotting. | Works for γ = −22.6 only. Has peak-location and cache bugs (§3.4, §3.6). |
| `vector_spectra.py` | Vector spectral function, using the same method. Independent of the chiral sector. | Works. Its numerical settings differ from the axial code (§3.5). |
| `vector_eigenvalue_solver.py` | T = 0 Schrödinger shooting for vector masses. | Useful as a validation reference. |
| `axial_melting_scan.py` | Axial melting line from whether the Schrödinger-potential well still exists. | Works. Inherits the wrong branch choice (§3.1). |
| `critical_zoom_improved.py`, `map_phase_diagram_improved.py` | Order of the chiral transition and T_c(μ) for a given (m_q, λ₁, γ, λ₄). | Works. T_c definition is wrong for first-order (§3.1). |

### 2.2 Drivers and pipeline

`batch_phase_diagram_unified.py`, `slurm_*.sh`, `submit_array_job.sh`, `extract_physics_results.py`, `task_summary_logger.py`, `log_task_results.py`, and the axial/vector temperature and melting scans.

Several of these **cannot run as written** (§3.7).

### 2.3 Dead, duplicate, or broken files (delete or move to `legacy/`)

- `log_task_results_fixed.py`: byte-identical to `log_task_results.py`.
- `log_task_results_broken.py`, `extract_physics_results_backup.py`: corrupted merges; SyntaxError.
- `extract_physics_results_clean.py`: an older version of `extract_physics_results.py`.
- `run_axial_temperatures.py`: SyntaxError at line 128. It also passes `-l1`, which `axial_spectra.py` does not accept.
- `test_critical_zoom_v3_v4.py`, `AXIAL_SCAN_ALWAYS_INCLUDE_PARAMS.md`: empty.
- The legacy `map_phase_diagram.py` → `criticalZoom.py` path and its callers `batch_phase_diagrams.py` and `example_phase_mapping.py`. In this path γ/λ₄ are accepted but ignored, and γ = −22.6 is hardcoded at `criticalZoom.py:130`.
- Scripts that read the old pre-`mu_g_440` layout (`data/peaks_data_T…`, `temperature_scan_*`):
  - `clean_peaks_data.py`
  - `clean_temperature_scan_summary.py`
  - `update_summary_with_spectral_values.py`
  - `calculate_peak_widths.py`
  - `plot_numbered_peaks_vs_temperature.py` (this one also uses μ_g = 388)
- One-off debugging scripts: `test_T17.py`, `diagnostic_T17.py`.
- About 10 Markdown summaries, several stale or contradictory (§3.8).

Roughly a third of the 77 tracked files can go. That will make it much easier to keep one source of truth for parameters.

### 2.4 Physics results produced so far

These come from the notebooks, `batch_990_5.out` and `vector_melting.csv`.

- **T ≈ 0 spectra (μ_g = 388, notebooks)**
  - Vector peaks: 775, 1093, 1330, 1519, 1670 MeV. These follow the linear trajectory (ω/μ_g)² ≈ 4(n+1). The ρ matches; the excited states do not follow the experimental trajectory (1282, 1465, 1720, 1909).
  - Axial peaks: 1513, 2052, 2157, 2245 MeV, against experiment 1230, 1647, 1930, 2096, 2270. **The a₁ is about 280 MeV too high.**
  - A log-scale search finds extra structures at 741, 846, 928, 1012 and 1166 MeV. These must be classified as physical narrow peaks or numerical artifacts (Goal 1).
- **Condensate:** σ^(1/3) ≈ 302–309 MeV at T = 17–100 MeV for m_q = 9 and λ₁ = 7.438 (confirmed by the test in §3.3).
- **Critical endpoint:** one bracket, from the cluster run (`batch_990_5.out`, Sep 2025):
  - m_q = 12, λ₁ = 5.9, γ = −22.4;
  - crossover T_c drops from 114.75 MeV at μ = 0 to 108.14 MeV at μ = 157.9;
  - first order from μ = 168.4 (T_c = 107.25 MeV) to μ = 200 (104.41 MeV).
  - **The CEP lies between μ = 158 and 168 MeV, at T ≈ 107–108 MeV.** The axial line could not be generated in this run because of the `capture_output` error.
  - *Cluster update (§8.3):* the job-1002 rerun on a 20 MeV grid has first order at μ = 160. That narrows the bracket to **μ ∈ (157.9, 160), T ≈ 108.0**. Its axial line exists (T_A(0) = 115.1 MeV, within 0.4 MeV of T_c).
- **Cluster phase-diagram survey** (Sept 2025, 33 sets; full table in §8.3):
  - seven clean CEP brackets and two noisy ones;
  - smooth trends: μ_CEP drops as λ₁ grows; more negative γ and smaller m_q favour first order;
  - axial melting tracks the chiral T_c within ±1.5 MeV for γ ≤ −22.4;
  - 9 of the 33 sets must be recomputed outright, because of C/F/C order flips or T_c(μ) jumps in broad weak-γ crossovers.
- **Scattered T_c values and order results** in `WorkbookJuly2025.ipynb`, for several (m_q, λ₁). γ is probably −22.6 and μ_g is unclear. One CEP estimate (m_q = 9, λ₁ = 5: μ ≈ 312–316, T ≈ 68.5–68.9) shows the order flipping 2/1/2, which is a sign of numerical noise near the CEP.
- **Vector melting line** (`vector_melting.csv`; **μ_g = 440, potential-well criterion**, confirmed by recomputation in §8.4): T_melt = 102.8 MeV at μ = 0, 93.5 at μ = 100, 64.1 at μ = 200, and it reaches 0 near μ ≈ 320.

> **Physics warning, now confirmed across the cluster survey (§8.5).** At every CEP parameter set, the ρ has already melted at the CEP: T_V(μ_CEP) = 0–81 MeV, against T_CEP = 68–108 MeV. At μ = 0, the chiral T_c lies above T_V = 102.8 MeV in 30 of 33 sets.
> - The only windows with T_χ < T_V are three weak-γ sets (γ = −18, −20; λ₁ = 5.5), at μ ≲ 60–80 MeV, by ≤ 6 MeV. That is comparable to the solver error.
> - The axial line sits on top of T_c.
>
> The goals "axial/vector degeneracy *below* melting" and "degeneracy for parameter sets with a CEP" need T_χ < T_melt^V. On the existing data the answer is **no in the CEP region**. One QNM or width check of the melting criterion (§8.5) should be done before this is fixed as the paper's story.

---

## 3. Corrections needed (ordered by severity)

### 3.1 CRITICAL: branch selection and the first-order transition temperature

**Problem**
- In the coexistence region the chiral EOM has three solutions: the broken branch (large σ), an unstable branch, and the nearly restored branch (small σ). The stable one is the solution with the **lowest free energy**.
- The code instead picks the smallest positive σ, and defines T_c as the lowest T with three solutions, which is the lower spinodal.

**Consequences**
- First-order T_c values, and so the CEP location in (μ, T), are biased toward low T.
- Axial spectra and the axial melting line in the coexistence region are computed on the wrong branch. This biases the results toward chiral restoration and degeneracy.
- A spectral discontinuity cannot be shown at the correct T_c.

**Fix**
- Compute the free-energy difference between branches. The cheapest route uses quantities the solver already produces. At fixed (T, μ), ∂F/∂m_q ∝ −σ (up to normalization and a scheme-dependent counterterm that cancels between branches at equal m_q). Therefore

  ΔF(branch a − branch b) = −N ∫ σ dm_q along the multivalued σ(m_q) curve.

  `new_function` (`chiral_solve_complete.py:415`) already builds the full σ(m_q) curve by scanning d₀. The equal-area construction on that curve gives T_c with no extra ODE solves.
- Cross-check at one or two points with the renormalized on-shell action.
- Then:
  - (a) select the stable branch in all spectral and melting codes;
  - (b) define first-order T_c by ΔF = 0;
  - (c) also store both spinodals; they are useful for plotting the hysteresis region.

### 3.2 CRITICAL: parameter consistency (γ, λ₄, μ_g, cutoffs)

- **γ and λ₄ never reach the axial spectra:**
  - `axial_spectra.lam_function` hardcodes −22.6/4.2 (`axial_spectra.py:96-97`);
  - the UV coefficients h₃–h₇ use that λ₃;
  - the chiral background is solved with the module default of `chiral_solve_complete`.

  Add `gamma`/`lambda4` arguments to `axial_spectra.main()` and its CLI. Thread them into both `calculate_sigma_values(..., v3, v4)` and `calculate_variables`.
- **γ defaults disagree:**
  - −22.6: `chiral_solve_complete.py:15`, `critical_zoom_improved.py:91,217`, `axial_spectra.py:97`, `vector_spectra.py:67`, `axial_melting_scan.py:477`, `criticalZoom.py:130`.
  - −22.4: `map_phase_diagram_improved.py:32,274,331,528`, `batch_phase_diagram_unified.py:451`, the SLURM scripts, `extract_physics_results.py:607`.
  - `run_axial_melting_scan.py` passes no γ, so it silently uses −22.6 while the phase diagrams use −22.4.
  - Fix: **use γ = −22.6** (decided 2026-11-05, to match the earlier paper). Change the −22.4 defaults in `map_phase_diagram_improved.py`, `batch_phase_diagram_unified.py`, the SLURM scripts and `extract_physics_results.py`. Then create one `model_params.py` (γ, λ₄, μ_g, u_i, u_f, tolerances) that everything imports.
- **μ_g is hardcoded in four places.** The `-mug` flag of `axial_melting_scan.py` changes only the vector part of the potential, not the chiral background (`chiral_solve_complete.py:272`). Make μ_g a real parameter and put it in every cache key and filename.
- **Different cutoffs in different codes:**
  - The phase diagrams solve the chiral field with u_i = 1e-4 (`critical_zoom_improved.py:401`, `map_phase_diagram_improved.py:30`).
  - The axial spectra solve it with u_i = 1e-2 (`axial_spectra.py:25,922`).
  - Vector spectra use u_i = 1e-4, u_f = 1−1e-5. Axial spectra use u_i = 1e-2, u_f = 1−1e-4.
  - So the σ behind a spectrum is not the σ behind the phase diagram; see §3.3 for the size of the difference.
  - *Cluster check (§8.3):* in practice, every phase diagram on Obsidian was run with u_i = 1e-2 and u_f = 0.9999. That matches the axial spectra and melting scan. The inconsistency is between the code defaults, not within the existing cluster data.
- **Provenance:** legacy files renamed `gamma_-22.4` were produced with −22.6. Re-label them, or regenerate them, before using them.

### 3.3 HIGH: accuracy of the chiral solver (σ and m_q extraction)

I ran a test in scratch space, not in the repository. It calls `calculate_sigma_values` for m_q = 9, λ₁ = 7.438, μ = 0. Values are σ^(1/3) in MeV:

| T (MeV) | u_i | default `solve_ivp` (rtol 1e-3) | tight (rtol 1e-10) |
|---|---|---|---|
| 17 | 1e-2 | 302.39 | 302.17 |
| 17 | 1e-4 | 305.88 | 306.12 |
| 100 | 1e-2 | 305.30 | 305.81 |
| 100 | 1e-4 | 306.19 | 308.81 |

What the table shows:

1. **ODE tolerance.** `chiral_solve_IR` calls `solve_ivp` with the default RK45 tolerances (rtol = 1e-3, atol = 1e-6; `chiral_solve_complete.py:296`). This moves σ^(1/3) by up to 0.9%, or about 2.6% in σ.
   - Close to the CEP, m_q(d₀) is nearly flat, so this error is strongly amplified. It can plausibly flip the reported order: the noisy 2/1/2 pattern in the notebook fits that picture.
   - Fix: set rtol ≈ 1e-10 and atol ≈ 1e-12 (or relative to |χ|), or use DOP853/LSODA. Then check convergence.
2. **u_i dependence that tight tolerance does not remove.** σ^(1/3) changes by 1.0–1.3% (σ by 3–4%) between u_i = 1e-2 and 1e-4. The cause is the UV expansion used to extract (m_q, σ) from χ and χ′ at u = u[10] (`chiral_solve_complete.py:391-407, 423-451`). That expansion omits the **u³ log u term**:

   χ = m_q ζ z_h u − 3v₃ (m_q ζ z_h u)² + c (z_h u)³ ln(z_h u) + (σ/ζ)(z_h u)³ + …

   with

   c = m_q ζ μ_g² (1 − λ₁/2) + (m_q ζ)³ (2v₄ − 9v₃²).

   The legacy `criticalZoom.py:167-176` *does* include this term (`s3`). The rewrite dropped it.
   - Without it, the extracted "σ" absorbs ≈ ζ c ln(z_h u_int). That depends on both u_i and T, which is an artifact.
   - Fix: add the log term to the two-equation (χ, χ′) inversion. Then show that σ is independent of u_i.
   - State the scheme (the scale inside the log) explicitly in the paper.
3. **The solver silently fails into garbage data.** If the chiral solve fails, `initialize_chiral_field` falls back to χ = 0 (`axial_spectra.py:525-531`). That fallback itself raises, because `interp1d(u, 0)` is invalid. Separately, `calculate_data` builds a placeholder χ = m_q·u (`axial_spectra.py:600-605`), which is dimensionally wrong (it should be m_q ζ z_h u).
   - In `axial_melting_scan.py:189` a chiral failure counts as "melted".
   - Fix: failures should raise or be flagged. They must never become saved data.
4. **Choice of solutions.** `sigma_of_T` pads with zeros and then filters `sigma != 0` (`chiral_solve_complete.py:514-515, 569-576`). That works but is fragile; return a list instead.

**Status (2026-11-12): items 1 and 2 fixed, item 3 fixed for the chiral solver and the axial spectra.** The solver now follows the scheme of the 2+1-flavor code (`seanbartz/2p1-critical-point`, `physics/model.py`, `physics/series.py`):

- *Scaling by μ_g, not z_h.* The equations are written with ẑ = μ_g z and b = (μ_g z_h)², with a = m_q ζ/μ_g and c = σ/(ζ μ_g³). The UV matching point is a fixed ẑ_UV = 0.005 (`SOLVER_SETTINGS['z_uv']`), not a fixed u, so the renormalization point no longer drifts like 1/T. The horizon start comes from a 10th-order horizon series (numerical recursion; it reproduces the old d₁ and d₂ to 1e-12). 1 − u is chosen so that the truncated term is below 1e-13, which moves the start toward the horizon by itself as b grows. `ui` and `uf` now only set the grid on which χ(u) is handed to the spectral codes.
- *Log terms.* χ = aẑ + Bẑ² + (c + D ln ẑ)ẑ³ + (E + F ln ẑ)ẑ⁴ + (G + H ln ẑ)ẑ⁵, derived by `derivations/derive_chiral_uv_series.py` into `chiral_uv_series_generated.py`, is inverted for (a, c) by Newton iteration. D equals the c of item 2. The scheme is "μ_g inside the logarithm". `condensate_in_scheme` converts to another scale, with M = 1 MeV giving the convention of `criticalZoom.py`. The shift is ∝ m_q and does not depend on T or μ.
- *Tolerances.* DOP853 at rtol = 1e-11. Roots of m_q(d₀) = m_q are solved with Brent's method rather than by linear interpolation across the bracket. A failed shoot returns NaN and is dropped, never saved.
- *Measured* (m_q = 9, λ₁ = 7.438, μ = 0): σ^(1/3) = 300.681, 301.091 and 295.816 MeV at T = 17, 60 and 120 MeV. σ^(1/3) changes by ≤ 2e-5 MeV between ẑ_UV = 0.002 and 0.005 (the old extraction moved ~3 MeV per decade), and by < 3e-6 MeV between rtol = 1e-11 and 1e-13. The returned m_q equals the target to 2e-7 MeV even at T = 17 MeV, where the old solver diverged. These values are in a different scheme from all earlier σ values, so old σ numbers cannot be compared with them directly.
- New results go to `sigma_data/sigma_calculations_v2.csv`, with `z_uv` and `scheme` columns, so they are never mixed with old-scheme rows.
- Checks: `python -m pytest tests -q` (`tests/test_numerical_scheme.py`).
- Still open: the d₀ conditioning at low T (§3.3b of PR #10, item 4) is handled by refining the scan and running Brent on each bracket. The interior-point labelling of the 2+1 code (`physics/interior.py`) has not been ported.

### 3.4 HIGH: axial peak finding (Goal 1: not missing low-lying narrow peaks)

- **The refined points are never used to locate a peak.**
  - In `calculate_data` (`axial_spectra.py:684-707`), points at fine resolution around each peak are *appended* to `ws`/`BAs`.
  - The peak positions are then read as `ws[peaks[i]]`, using indices from the coarse array.
  - So peak positions are only as accurate as the coarse grid. The same bug exists in `vector_spectra.py:316-334`.
  - Fix: re-run the peak search on the sorted, refined array, or fit a Breit–Wigner or Lorentzian to each refined window. The fit also gives the width Γ, which is needed for melting and degeneracy.
- **Global refinement driven by a peak-count guess.** The default expected count is `int((wf/μg)²/4)` = 7 for 700–2400 MeV (`axial_spectra.py:653`). The axial spectrum has only about 4 peaks in that range. So the loop halves the *entire* grid repeatedly, down to `wresolution`. That costs 16× more evaluations at wr = 0.1, and 128× at the CLI default wr = 0.01, without improving any peak location (see the bug above).
- **Thresholds are absolute and differ between channels.** `prominence=0.1, distance=10` for axial (line 664) and `prominence=0.01` for vector (`vector_spectra.py:299`). |Im G| grows like ω² and changes by orders of magnitude with T, so a fixed prominence means different things at different T. Use a relative prominence on the ω²-normalized spectrum, and use the same criterion in both channels.
- **The ω window starts at 700 MeV.** Anything below that is invisible. The log-scale structures found in `Axial.ipynb` (741–1166 MeV) need to be checked.
- **Recommended definitive check: compute quasinormal modes.** Solve the same EOM with infalling BC at the horizon and a normalizable BC at the boundary, for complex ω.
  - A narrow peak is a QNM with small |Im ω|. You cannot miss it in the complex plane the way you can on a real-ω grid.
  - QNMs give mass and width directly. They make the melting and degeneracy statements clean (for example, a₁–ρ degeneracy when Re ω_A → Re ω_V), and they are standard in PRD.
  - The existing UV series and horizon expansion can be reused.
- Smaller issues:
  - `clean_peaks` (`axial_spectra.py:533`) merges peaks within 10 MeV. That can hide nearly degenerate peaks, which is exactly the physics wanted near restoration.
  - The axial UV series a₂ is truncated at u⁴ (`axial_spectra.py:357-364`), but h₅–h₇ are still computed every time.
  - The horizon BC is leading order only (`psi_m`). At low T, z_h is large, and the correction is about (1−u_f)·2μ_g²z_h² ≈ 1% at T = 17 MeV. Check convergence in u_f, or add the next order.

### 3.4b CRITICAL: low-temperature spectral functions are dominated by round-off *(added 2026-10-19)*

**Where it is.** `BA()` in `axial_spectra.py:426-488`, and the identical structure in `vector_spectra.py:209-272`.

**What the code does.** It integrates the two UV solutions ψ₁ (normalizable) and ψ₂ (non-normalizable) to u_f, then forms

B/A = (ψ_m ψ₂′ − ψ_m′ ψ₂) / (ψ_m ψ₁′ − ψ_m′ ψ₁).

**Why it fails at low T.** Some algebra shows the imaginary part is

Im(B/A) = Im(ψ_m′ ψ_m*) · W(ψ₁, ψ₂) / |D|²,

where W = ψ₁ψ₂′ − ψ₂ψ₁′ is the Wronskian and D is the denominator above.
- At low T the dilaton is large at the horizon: (μ_g z_h)² ≈ 68 at T = 17 MeV. Both ψ₁ and ψ₂ therefore grow by about e⁶⁸.
- W is computed numerically as the difference of two products of size about e¹³⁶, so it is mostly cancellation error.

**Test.** I ran this in scratch space, not in the repo: μ_g = 440, m_q = 9, λ₁ = 7.438, μ = 0, γ = −22.6. "Exact formula" means the closed form given below.

| ω (MeV) | current code, T = 17 | exact formula, T = 17 | W_numerical / W_exact, T = 17 | current code, T = 60 | exact formula, T = 60 |
|---|---|---|---|---|---|
| 800 | −1.4e-6 | −1.6e-37 | −8.6e23 | −8.11e-7 | −9.80e-7 |
| 1000 | 1.4e-5 | −2.3e-34 | 1.2e21 | −1.09e-5 | −1.20e-5 |
| 1250 | 6.0e-5 | −5.8e-30 | 4.3e16 | −4.27e-4 | −4.28e-4 |
| 1512.7 | 4.1e-7 | −1.7e-24 | −8.3e10 | −5.32e-2 | −5.32e-2 |
| 1700 | −8.6e-4 | −1.4e-18 | 9.6e6 | −230.2 | −230.2 |
| 2052 | 2.0e-4 | −4.8e-12 | 3.3 | −34.19 | −34.19 |
| 2300 | −1.19e-3 | −1.22e-3 | 1.0 | −125.6 | −125.6 |

What the table shows:
- At T = 17 MeV the current output is noise everywhere below about 2.3 GeV. It even has the wrong sign at some ω. Tightening the tolerance of the direct method does not help either: it gives values of order ±1e-13 with random sign, which is still noise.
- At T = 60 MeV, where (μ_g z_h)² ≈ 5, all methods agree. The only difference is a 10–20% error in the current code at low ω, where the spectrum is small.
- The unexplained low-ω structures in `Axial.ipynb`, and the low-T "peaks" behind the T = 0 comparison, are most likely this noise. That notebook run used μ_g = 388 and T = 18, which has the same conditioning problem.

**Fix (exact, and cheaper).** W satisfies W′ = (1/u − f′/f + φ′) W, so

W(u_f) = W(u_i) · (u_f/u_i) · (f(u_i)/f(u_f)) · exp[φ(u_f) − φ(u_i)],

with W(u_i) taken from the UV series. With ψ_m = (1−u)^c and c = −iωz_h/(2(2−Q²)), this gives |ψ_m| = 1 and Im(ψ_m′ψ_m*) = k ≡ ωz_h/[2(2−Q²)(1−u_f)]. Therefore

**Im(B/A) = k · W(u_f) / (ψ₁′(u_f)² + k² ψ₁(u_f)²).**

- Only **one** integration (ψ₁) is needed, so the cost halves.
- The denominator is a sum of squares, so there is no cancellation.
- Where the direct method is well conditioned, the two agree to all printed digits (T = 60 columns above).

**What remains even after the fix.** At low T the resonances are exponentially narrow, about e^(−2φ(1)) wide. A scan on a 1 MeV grid with the exact formula at T = 17 MeV finds only 3 local maxima between 700 and 2400 MeV (1718, 2194 and 2328 MeV). The lower states are far too narrow for *any* real-ω grid to see.

**Status (2026-11-12): fixed** in `axial_spectra.BA` and `vector_spectra.BA`. Im(B/A) now comes from the formula above, with W(u_i) from the UV series and W(u_f) from Abel's identity. Re(B/A) = (ψ₁′ψ₂′ + k²ψ₁ψ₂)/(ψ₁′² + k²ψ₁²) has no cancellation. Both solutions are integrated in one DOP853 pass at rtol 1e-10. Where the direct formula is well conditioned (T = 60 MeV), the two agree to 1e-5. At T = 17 MeV the result reproduces the "exact formula" column of the table above (tests).

**A second bug found while testing (2026-11-12): parallel axial spectra used χ = 0.** `calculate_data` evaluates `BA` in joblib (loky) workers through a closure that set the module global `chi_interp`. That assignment does not reach the worker's own copy of `axial_spectra`, where `chi_interp` was `None`, so `Axial_eom` silently used `default_chi = 0`. Every spectrum produced through `calculate_data` (that is, `axial_spectra.main` and the temperature scans built on it) was therefore the **χ = 0 spectrum, which is the vector spectrum**. Direct calls to `BA`, such as those behind the table above, were not affected. Fixed: χ is passed explicitly to `BA`/`Axial_eom`, and a missing χ now raises. Existing axial spectra produced by `main()` should be treated as invalid.

This is Goal 1 ("not missing a low-lying narrow peak") in its sharpest form. At low T, masses must come from a quasinormal-mode or eigenvalue search: the condition ψ₁(u → 1) → 0, or the T = 0 Schrödinger problem. Spectral-function scans are only meaningful once (μ_g z_h)² is small enough that the peaks have visible width.

### 3.5 HIGH: melting criteria disagree between channels

- **Axial** (`axial_melting_scan.py:195-206`): melted when the Schrödinger potential has no extremum.
  - "Any sign change" also counts a barrier with no well.
  - It is refined by bisection to 0.1 MeV.
  - It returns T_min = 1 MeV if no zero is found (line 287).
- **Vector** (`run_vector_melting_scan.py:115`): melted when the spectral function has 0 peaks, stepping in T by 5 MeV with no refinement. That script also crashes at startup (§3.7).
- Two different definitions cannot be compared to make a degeneracy or melting-order claim. Choose one criterion for both channels: potential well, peak disappearance, or (best) a QNM width criterion such as Γ/M > 1. Report the sensitivity to that choice.

### 3.6 MEDIUM: caching (Goal 2, partly done)

`--load-if-exists` exists (commit `04e24f9`), so `ScienceGoalsPlan.md` is out of date. However:

- **Cache key collisions.** The key formats T, μ, m_q and λ₁ to **1 decimal** (`axial_spectra.py:798-802`). So 7.438, 7.4 and 7.35 collide, and m_q = 9.0 and 9.04 collide.
- **Missing key fields:** γ, λ₄, μ_g, u_i, u_f, the ω window and the resolution are not in the key. A cached spectrum computed with other settings loads silently.
- **Missing σ.** A cached load returns σ = NaN (`axial_spectra.py:967`), and the params file does not store σ, the branch, γ or λ₄.
- Fix:
  - Key on a hash of the full parameter set.
  - Store the parameters inside the file (a JSON sidecar, or one HDF5/NPZ file per run).
  - Keep human-readable filenames with `%.4g` precision.
- **σ cache.** `sigma_calculations.csv` already stores d₀ for every solution, but `calculate_sigma_values` re-runs the full d₀ scan even when it finds an existing row (`chiral_solve_complete.py:556-567`). Reuse the stored d₀: one ODE solve instead of hundreds (§4).

### 3.7 MEDIUM: pipeline bugs that stop runs or corrupt bookkeeping

**Flag mismatches (verified)**
- `batch_phase_diagram_scan.py:68,72` passes `--gamma`/`--lambda4`, but `map_phase_diagram_improved.py:528` defines `-gamma`/`-lambda4`. Every run fails. `run_batch_scan.py` wraps it and adds `--mu-points` and `--skip-existing True`, which are also wrong.
- `run_vector_melting_scan.py:238,244` reads `args.tstart`, but the argument is `--T_start` (line 196), so it raises AttributeError at startup. It also passes `-wcount`/`-wresolution`, but `vector_spectra.py` accepts only `-wc`/`-wr`. The same mismatch is in `run_temperature_scan.py:323`.

**Python version**
- `subprocess.run(..., capture_output=...)` fails on the cluster's older Python (`batch_phase_diagram_unified.py:120`, `run_axial_melting_scan.py`; the error is visible in `batch_990_5.out`). This is why no axial line was produced in the CEP run.

**`extract_physics_results.py` → `task_summary_logger.py` is wrong in a systematic way**
- The filename regexes do not handle the `_mq_12.0_lambda1_5.9_gamma_-22.4_…` naming.
- It analyzes the *most recently modified* files in the directory, not the current task's files.
- The required field `mq` is missing, so logging fails with an error.
- `axial_vector_merge` is never computed.
- `log_task_results.py` and `TaskSummaryLogger` write to different directories.

**`batch_phase_diagram_unified.py` defaults**
- It defaults to μ_max = 200 with 20 points, and so do `slurm_phase_diagram_array.sh` and `slurm_unified_batch.sh`. That contradicts the 400 MeV / 21-point "alignment" in `MU_ALIGNMENT_SUMMARY.md`.
- It does not forward `--output-dir`/`--plot-dir`.

**SLURM helper scripts**
- `submit_array_job.sh` miscounts lists that contain negative numbers.
- It reads the "Other" CPU field as "Idle" from `sinfo -o "%N %C"`.
- `slurm_batch_array.sh` uses `SHARED_SIGMA_DIR` before defining it, and copies *every* CSV/PNG to shared storage on every task.

**Parallelism**
- `critical_zoom_improved.py:46-61` sets `JOBLIB_N_JOBS=1` to stop nested parallelism. Joblib does not read that variable. The nested loops only run serially because loky refuses to run inside daemonic Pool workers, and that warning is suppressed. The comment is misleading, and it is fragile.

### 3.8 LOW: tests and documentation

- **Tests check no physics.** `test_mu_alignment.py` checks default values with a regex. `test_phase_args.py` tests its own copy of argparse. The others only print text or are empty.
- Add a small `pytest` suite (§6, Phase 1). It is what will make referees, and you, trust the numbers.
- **Documentation contradicts itself:**
  - `README.md` says γ = −22.4 is the default, but the spectra use −22.6.
  - `README_phase_diagram.md:56` says μ_max = 200; `COMMAND_LINE_STANDARDS.md:41` says 400 everywhere.
  - Several READMEs list `--long-flags` that do not exist.
  - `MODERNIZATION_SUMMARY.md` references `program.py`.
- Fix: consolidate into `README.md`, `ScienceGoals.md` and this plan.

### 3.9 Physics caveats the paper must address (referees will ask)

- **Fixed background.** The geometry is AdS–RN with a quadratic dilaton and no backreaction. The chiral transition is the only phase transition in the model. Say this explicitly.
- **A black hole at every T.** Every T > 0 uses the black-hole geometry. In the soft-wall model the Hawking–Page transition is at T_HP ≈ 0.4917 μ_g: about 216 MeV for μ_g = 440, or 191 MeV for 388. All melting and chiral temperatures found so far (about 65–115 MeV) are *below* T_HP, where thermal AdS would dominate and mesons would not melt.
  - This is exactly the setting of the reviewed paper (crossover in the spectra despite a first-order deconfinement transition). The comparison section has to confront it head-on.
- **The vector channel ignores the chiral field.** The vector spectrum depends only on (T, μ, μ_g), not on m_q, λ₁, γ or λ₄. "Axial/vector degeneracy" therefore means the axial channel approaching a chirally blind vector channel. This is fine, but state it.
  - At finite μ the mixing of vector fluctuations with the background gauge field and metric is neglected (probe approximation).
- **Normalization of μ.** κ = 1 in `blackness()`: q = μ/z_h², Q = μ z_h. The paper must say whether μ is the quark or the baryon chemical potential.
- **Units in plot labels.** `critical_zoom_improved.py:343` labels σ in GeV³; the values are in MeV³.

---

## 4. Efficiency improvements

These are ordered by expected payoff. The first four probably make the overall workflow at least an order of magnitude faster.

1. **Stop global ω-grid halving** (`axial_spectra.py:666-682`, `vector_spectra.py:300-314`). Use one moderate grid plus *local* refinement or Lorentzian fits around candidate peaks. Better still, use the QNM solver of §3.4, which needs a few dozen complex root finds per (T, μ) instead of 10⁴–10⁵ ODE pairs.
2. **Reuse cached σ solutions.** Load d₀ from `sigma_calculations.csv`; then a spectrum needs one chiral integration instead of the full d₀ scan.
   - Also do **continuation in T**: use the d₀ from the neighbouring temperature as the starting point for each branch.
   - That makes temperature scans cheap and gives branch tracking, and so hysteresis, for free.
3. **Replace the 100-target `mq_overlay` reconstruction** (`chiral_solve_complete.py:453-477`) with direct root finding. Bracket the sign changes of m_q(d₀) − m_q,target on the coarse d₀ scan, then call `brentq` on each bracket. That is about 10 solves per root instead of 100 re-solves plus interpolation.
   - One d₀ scan per (T, μ) gives σ(m_q) for *all* m_q. Use it both for the free-energy integral and for many m_q at once.
4. **Axial and vector spectra, one integration instead of two.** Use the exact-Wronskian formula from §3.4b: Im(B/A) = k W(u_f)/(ψ₁′² + k²ψ₁²). It needs only ψ₁, which halves the ODE count, and it is also the correctness fix for low T.
5. **Compile the ODE solves with numba** *(measured 2026-10-19)*. In the current code the solves are dominated by Python overhead. Per `axial_spectra.BA` call, about half the time is spent in scipy `interp1d` inside `Axial_eom` (`axial_spectra.py:399`), and most of the rest is per-step overhead in `solve_ivp`. Applying `@njit` to the right-hand side alone helps little, because `solve_ivp` itself is Python. The integrator has to be compiled too.
   - I wrote a ~40-line Dormand–Prince 5(4) adaptive integrator in numba, using `np.interp` for χ(u). `numbalsoda` would also do, but it failed to build in the test environment.
   - Timings on one core, T = 17 MeV:

   | Solve | Current code | numba | Speedup |
   |---|---|---|---|
   | Axial B/A per ω (both solutions, rtol 1e-8) | 206 ms | 3.0 ms | **≈70×** |
   | Chiral field per d₀ (current default tolerance, 10⁴ output points) | 9.8 ms | 4.4 ms | ≈2× |
   | Chiral field per d₀ at rtol 1e-10 (the accuracy §3.3 needs) | 80 ms (scipy RK45) | 4.4 ms | **≈18×** |

   - The chiral solve is limited by the forced output at 10,000 u-points. Saving only χ near u_i plus a coarse grid for the axial interpolation would make it faster still.
   - The same check also shows that at the current default tolerance, scipy's χ(u[10]) is off by 5%. The numba run at rtol 1e-10 matches scipy at rtol 1e-10 to 6 digits.
   - `vector_spectra.py` and `vector_eigenvalue_solver.py` have the same structure and should see similar gains.
   - **Cluster caveat (§8.1):** Obsidian's system Python is 3.6.8. The last numba that supports 3.6 is 0.53, so install a user-space Python ≥ 3.10 first.
6. **Parallelism in one place.** Parallelize over the outermost independent axis: the (T, μ) or ω tasks. Make the inner calls explicitly serial (`n_jobs=1` parameters, not environment variables). Remove the nested `Pool` → `joblib(n_jobs=-1)` pattern.
7. **Bookkeeping I/O.** `check_existing_calculation` reads the whole `sigma_calculations.csv` on every call (`chiral_solve_complete.py:44`). As the file grows this becomes noticeable. Load it once per process, or move to SQLite (also safer under concurrent cluster writes than `flock` on NFS).
8. **Cluster hygiene.** Do not copy every CSV/PNG on every SLURM task. Remove the sinfo-based CPU heuristics, which parse the wrong field; a fixed `--cpus-per-task` is simpler and correct.

---

## 5. Status of each science goal

| # | Goal (from `ScienceGoals.md`) | Status | Blocking items |
|---|---|---|---|
| 1 | Accurate axial spectra; no missed low-lying narrow peak | **Low-T spectra are round-off noise** (§3.4b). Low-T resonances are too narrow for any real-ω grid. | §3.4b: exact-Wronskian formula, QNM or eigenvalue search at low T; §3.4: peak-location bug, ω window, cutoff convergence |
| 2 | Load existing spectra; avoid duplicate work | Partly done (`--load-if-exists`) | §3.6: cache key collisions, missing params, σ = NaN, σ cache unused |
| 3 | First-order transition visible in axial spectra | Not possible yet | §3.1: free energy and branch selection |
| 4 | (m_q, λ₁) of the critical point from spectra | **Scouting map exists** (§8.3): 7 clean CEP brackets over m_q ∈ {9, 12, 15}, λ₁ = 5.5–7.0, γ = −18 to −26; λ₁,c(μ = 0) ∈ (5.9, 6.0) at m_q = 12. Computed with the inaccurate solver. Not linked to spectra. | §3.3 recompute of the 7 sets; §3.1 for spectra only (T_c shift ≲ 0.5 MeV) |
| 5 | Effect of γ, λ₄ | **Phase diagram: γ scan done** (−18 to −26, at λ₄ = 4.0). No λ₄ variation; the γ = −22.4, λ₄ = 4.0 reference is missing. Spectra: impossible. | §3.2; missing λ₄ = 4.4 runs and reference point |
| 6 | Match T = 0 meson spectra | **μ_g = 440 fixed** (consistency with the earlier paper). The ρ comes out at about 880 MeV and the a₁ is also high; report the T = 0 masses and (ω/μ_g)² ratios rather than fit to experiment. | Compute the T = 0 table at μ_g = 440, γ = −22.6 (Phase 3) |
| 7 | Both a CEP and a reasonable melting temperature | Axial lines exist for 22 sets (consistent γ, μ_g = 440, well criterion), and the vector line uses the same criterion. At every CEP, T_melt^V < T_CEP. | §3.3 recompute; Phase 3 μ_g choice |
| 8 | Comparison with the reviewed paper | Not started | Needs 3, 7, and the Hawking–Page discussion in §3.9 |
| 9 | Chiral restoration via axial/vector degeneracy below melting | **Go/no-go answered on the existing data (§8.5): no in the CEP region.** T_χ < T_V only in 3 weak-γ sets at μ ≲ 80, by ≤ 6 MeV. T_A ≈ T_χ. | One QNM/width check of the melting criterion; then write it as a negative result, or map the window |
| 10 | Degeneracy for sets that also have a CEP | Only candidate: (9, 5.5, −20, 4.0), with T_χ < T_V for μ ≤ 60 and its CEP at μ ≈ 390. Marginal. | 4 and 9 |
| 11 | First-order restoration visible directly in spectra | Not started | 3; branch-resolved spectra |

---

## 6. Plan

Each phase ends with a concrete deliverable. Phases 0–2 are corrections and validation; do them before spending cluster time on production.

### Phase 0: consolidation (no physics change)
0. *(Added 2026-11-05, from §8.)* Set up the cluster:
   - install a user-space Python ≥ 3.10 on Obsidian (the system one is 3.6.8);
   - replace the shared `sigma_calculations.csv` with per-task files or SQLite; the current file is corrupted and has disabled σ reuse since Sep 4;
   - archive the 1.1 GB of redundant σ backups;
   - raise memory to ≥ 8 GB and the time limit for strongly first-order points (the λ₁ = 7.0 failures).
1. Create `model_params.py` with one dataclass holding T, μ, m_q, λ₁, γ, λ₄, μ_g, u_i, u_f and the tolerances. Make every module import it, and delete all hardcoded `lam_function`, `mu_g()` and module-level v3/v4 defaults.
2. Thread γ and λ₄ through `axial_spectra.main()` and its CLI, and through `run_axial_temperature_scan.py` and `run_axial_melting_scan.py`.
3. Delete or move the dead files listed in §2.3. Fix the flag mismatches and the `capture_output` problem (§3.7).
4. **Default γ = −22.6** (decided 2026-11-05, to match the earlier paper). Make it the default everywhere.
   - The cluster data are at γ = −22.4 and the scan values −18, −20, −24 and −26 (§8.4). All are correctly labeled, and all stay in the scan. Only the −22.6 reference point itself is missing.
   - The pre-September 2025 Dropbox files that were renamed `gamma_-22.4` but produced with −22.6 should be **re-labelled to −22.6, not regenerated for that reason**. They are actually at the chosen γ. They still carry the solver and spectral issues of §3.3 and §3.4b.
5. **Deliverable:** one command per observable (σ(T), T_c(μ), spectrum, melting line), with every parameter explicit and recorded in the output.

### Phase 1: numerical validation (Goal 1; the referee-proof foundation)
0. **Do this first:** replace the direct B/A evaluation with the exact-Wronskian formula in `axial_spectra.py` and `vector_spectra.py` (§3.4b). At the same time, move the ODE right-hand sides and integrator to numba (§4.5). Then re-check every existing low-T spectrum.
1. Chiral solver:
   - set tight tolerances;
   - add the u³ log u term to the (m_q, σ) extraction;
   - show σ is independent of u_i (1e-2, 1e-3, 1e-4) and converged in tolerance, at low T, near T_c, and in the coexistence region.
2. Spectral solvers:
   - convergence in u_i and u_f;
   - vector check at low T: peaks → 4μ_g²(n+1), and against `vector_eigenvalue_solver.py`;
   - flux conservation check;
   - use the same cutoffs in the axial and vector codes.
3. Fix peak location (use the refined points, or Lorentzian fits) and use one relative prominence criterion in both channels.
4. Implement a **QNM solver** for both channels. Use it to classify the 741–1166 MeV structures and to *prove* no narrow low-lying axial peak is missed (Goal 1).
5. Write a `pytest` suite of the checks above, with tolerances.
6. **Deliverable:** an appendix-ready convergence section with 2–3 figures.

### Phase 2: thermodynamics and branches (Goals 3, 11)
1. Implement ΔF via the equal-area construction on σ(m_q). Cross-check with the on-shell action at one point.
2. Redefine first-order T_c by ΔF = 0. Store both spinodals. Re-run the `batch_990_5` parameter set to see how far the CEP moves. The cluster data predict that it barely moves: the hysteresis width near the CEP is ≲ 0.6 MeV (§8.4). So the main effect of this phase is the branch-resolved spectra, not the phase-diagram numbers.
3. Track branches by continuation in T. Compute **axial spectra on both branches** across the coexistence region, so the peak positions and widths jump at T_c.
4. **Deliverable:**
   - a σ(T) figure with branches, spinodals and T_c;
   - an axial spectral-function figure across the first-order line, compared with the smooth evolution across the crossover.

### Phase 3: calibration and go/no-go (Goals 6, 7, 9)
1. **μ_g = 440, γ = −22.6 and λ₄ = 4.2 are fixed** (decided 2026-11-05, for consistency with the earlier paper). The calibration step becomes:
   - compute the T = 0 vector and axial masses at μ_g = 440, γ = −22.6, using the eigenvalue or QNM method, not the low-T spectra (§3.4b);
   - check how far m_q and λ₁ (and λ₄, through σ) move the a₁–ρ splitting, since it is driven by g₅²χ²;
   - report the masses and the (ω/μ_g)² ratios, and state the comparison with experiment openly, as the earlier paper presumably did.
2. Use the m_q and λ₁ fit to the a₁ (and f_π, if computed) as constraints on the CEP search.
3. **Go/no-go:** for the calibrated set(s), compare the chiral T_χ(μ) with the vector and axial melting lines, using the single criterion from §3.5. *Preliminary answer from the cluster data (§8.5): no-go in the CEP region at μ_g = 440.* What remains is to confirm it with a QNM/width criterion at the γ = −22.6 baseline CEP set and at (9, 5.5, −20).
   - If T_χ > T_melt^V everywhere (as the early data suggests), then "degeneracy below melting" is a *negative result*. That is still publishable and fits the comparison with the reviewed paper.
   - Otherwise, map where T_χ < T_melt.
4. **Deliverable:** a T = 0 mass table and Regge plot (the existing `plot_meson_mass_squared.py` plus model points), and the go/no-go answer.

### Phase 4: production scans (Goals 4, 5, 7, 10)
1. A CEP map in (m_q, λ₁) at fixed (γ, λ₄): μ_CEP and T_CEP versus λ₁, for 2–3 values of m_q. Use the fixed pipeline with stable-branch T_c.
   - **Do not rerun blind. Use the §8.3 scouting map:** recompute the seven clean CEP sets on μ grids refined around each bracket, and bisect λ₁ at m_q = 12 with γ = −22.6, starting from (5.8, 6.0); the −22.4 bracket is (5.9, 6.0).
   - Warm-start each branch from the d₀ values in the cluster `sigma_calculations.csv`, after deduplication and with the corrupt lines dropped.
2. γ/λ₄ sensitivity: a small grid of about 3 × 3 around the default. Record how the CEP, the melting lines, and the T = 0 a₁ mass move.
   - The γ direction has already been scouted (§8.3).
   - Still missing: the γ = −22.4, λ₄ = 4.0 reference point and the never-submitted λ₄ = 4.4 runs.
3. For selected points: axial and vector QNM masses and widths versus T along μ < μ_CEP, μ ≈ μ_CEP and μ > μ_CEP, with the degeneracy metric |M_A − M_V| / M_V versus T.
4. **Deliverable:** the main phase-diagram figure (chiral line with CEP, vector and axial melting lines, degeneracy line), plus a parameter-dependence figure or table.

### Phase 5: paper
1. Comparison with the reviewed paper: transition order seen in the spectra versus the order of the deconfinement or chiral transition, including the Hawking–Page discussion (§3.9).
2. Write: model; methods (spectral functions, QNMs, free energy); T = 0 calibration; phase diagram and CEP; spectra across the transitions; degeneracy; comparison; conclusions; numerical appendix.
3. Archive the code version and data (Zenodo) for reproducibility.

---

## 7. How close is a PRD paper?

*(Rewritten 2026-11-05 to take the cluster data in §8 into account.)*

**What a referee would need, and where each item stands:**

| Need | Status | Change since the cluster data |
|---|---|---|
| 1. A calibrated model | **Fixed: μ_g = 440, γ = −22.6, λ₄ = 4.2**, matching the earlier paper. The T = 0 masses still need computing and reporting. | Decided 2026-11-05. The cluster data become scan points around it; the γ = −22.6 reference point still has to be run. |
| 2. A correct phase structure (stable branch, ΔF-based T_c) | Incorrect in the first-order region | **Less serious.** The T_c error is ≲ 0.5 MeV near the CEPs, and the CEP's μ does not move (§8.4). The fix still matters for branch-resolved spectra. |
| 3. Converged spectra or QNMs | Unverified; the solver errors of §3.3 and §3.4b remain | None. Every cluster number must be recomputed. |
| 4. A clear new result | **Preliminary answer exists** | The go/no-go gives no a₁–ρ degeneracy below melting in the CEP region (§8.5). The paper's story is largely settled. |

**Overall distance.**
- The infrastructure is about the same: roughly 70% done, plus a small new cost to set up the cluster environment (Python ≥ 3.10, and replacing the shared σ log).
- The validated physics is still about 15–20% done. Nothing on the cluster is final.
- **About half of the remaining *risk* is gone.** The open questions that could have reshaped the project are now answered at the scouting level:
  - which parameter sets have a CEP, and where;
  - whether the spinodal T_c is a large error;
  - whether degeneracy happens before melting.
- What remains is mostly well-defined execution: fix the solver, recompute a targeted set of points, confirm with QNMs, and write.

**What a referee would see as new:**
- A CEP map in a model with dilaton–chiral mixing, versus λ₁, γ and m_q. The scouting gives smooth trends and λ₁,c(μ = 0) ∈ (5.9, 6.0) at m_q = 12.
- **Chiral restoration and the CEP occur in the melted phase.** At every CEP, T_melt^V(μ_CEP) < T_CEP. This is a clean negative answer to the degeneracy question, and it sets up the comparison with the reviewed paper.
- **The axial melting line is locked to the chiral transition** (within ±1.5 MeV for γ ≤ −22.4). Under the current criterion, axial melting is a direct probe of chiral restoration.
- Spectral signatures of a first-order transition: jumps in the axial peak positions and widths, and hysteresis.
  - Near the CEPs the coexistence region is only ≲ 1 MeV wide, so these signatures will be hard to see there.
  - **Use the strong-coupling sets (γ = −24, −26)**, whose hysteresis windows are 2–5 MeV.

**Critical path (revised):**
1. **Phase 0.** Python ≥ 3.10 on Obsidian; a σ store that is safe under concurrent writes; `model_params.py`; fix the flag and `capture_output` bugs.
2. **Phase 1, items 0–3.** Exact-Wronskian spectra, the numba integrator, tight chiral tolerances and the log-term σ extraction, then the convergence checks. *This is now the bottleneck.*
3. **Targeted recompute** with the fixed solver, instead of the Phase 4 survey:
   - the 7 clean CEP sets on refined μ grids;
   - a λ₁ bisection in (5.9, 6.0);
   - μ_max ≈ 500 for the two edge cases;
   - **the reference λ₁ scan at γ = −22.6, λ₄ = 4.2**, added next to the −22.4 scan; start the bisection from λ₁ ∈ (5.8, 6.0);
   - a γ = −22.6, λ₄ = 4.0 point, to connect the baseline to the γ scan (λ₄ = 4.0);
   - one λ₄ = 4.4 slice;
   - warm-start each set from the cluster d₀ values.
4. **QNM solver** (Phase 1, item 4). Use it to confirm the go/no-go at the γ = −22.6 baseline CEP set and at (9, 5.5, −20), and to compute masses and widths. If the ρ keeps Γ/M < 1 up to T_CEP, the story changes. This is the one remaining check that could overturn it.
5. **Phase 2, reduced.** ΔF-based T_c and branch-resolved axial spectra, for one strong-|γ| set and the baseline CEP set.
6. **Phase 3, item 1 (T = 0 masses at μ_g = 440, γ = −22.6)** can run alongside steps 3–5. The calibration decision itself is made.
7. **Writing.**

**Revised minimal first paper:**
- the model, with μ_g = 440 and γ = −22.6 as in the earlier paper, and its T = 0 masses (the ρ and a₁ offsets stated openly);
- the corrected CEP map against λ₁ and γ at one or two values of m_q, with ΔF-based T_c;
- vector and axial melting lines with a single criterion, cross-checked with QNMs;
- the main result: restoration and the CEP sit in the melted phase, and axial melting is locked to T_χ;
- first-order spectral signatures in one strong-|γ| set;
- the comparison with the reviewed paper, including the Hawking–Page discussion (§3.9);
- deferred to a follow-up: the full (m_q, λ₁, γ, λ₄) map.

---

## 8. Cluster data inventory and reconciliation *(added 2026-11-05)*

This section reconciles §1–§7, which were written from the code alone, with what has actually been computed on Obsidian. Everything here was read without changing anything on the cluster: no files were modified, moved or deleted, and nothing was pulled, committed or submitted. Analysis ran locally on copies of the small files, using quick `pandas` summaries. (`git status` may refresh the cluster checkout's index file, which is git's own bookkeeping, not repository content.)

### 8.1 State of the cluster checkout

- `~/QGP/meson-melting-holographic-qcd` is on `main` at **`2a755a4`**, which is `origin/main` *before* the two assessment commits (`10041dc`, `bb662fa`). That is the commit §1–§7 reviewed, so the review applies to the cluster code exactly.
- **No uncommitted or unpushed code.** `git diff HEAD` is empty. The only untracked items are `slurm_logs/` and `summary_data/`.
- **One stale stash** (`stash@{0}`, made 2025-09-02 on `5f6b2ff`). It comments out `module load python/3.9` in `slurm_batch_array.sh` and `slurm_unified_batch.sh`. Current `main` already does the same, so the stash is superseded and can be dropped.
- **The cluster Python is 3.6.8** (numpy 1.19.5, scipy 1.5.4, pandas 1.1.5). This is the root cause of the `capture_output` failures (§3.7): that keyword needs Python ≥ 3.7. It also limits numba (§4.5): the last numba release that supports 3.6 is 0.53. Before production, install a conda/venv Python ≥ 3.10 in user space on Obsidian.
- **Which code produced the data.** The checkout's reflog shows which commit was checked out when each dataset was written:

  | Runs | Dates (2025) | Commit at run time |
  |---|---|---|
  | SLURM jobs 970–990 | Sep 2–4 | `5f6b2ff` → `85b8ab2` |
  | Jobs 996, 1002 | Sep 4, 15:15–16:00 | `186aafd` / `70b8614` / `7ecb4f5` |
  | Job 1044 | Sep 11 | `098586b` |
  | Jobs 1065, 1083 | Sep 16–19 | `effd9c3` |
  | Job 1107 (user `awaggoner5`) | Sep 22 | unknown (another user's checkout) |

  Between `85b8ab2` and `effd9c3`, `critical_zoom_improved.py`, `axial_melting_scan.py` and the physics in `chiral_solve_complete.py` did not change. Only CSV logging and locking changed. All Sept 2025 cluster data therefore come from **the same physics code**, and that code matches today's `main` apart from output paths and the axial μ_max default. The cluster was pulled to `2a755a4` on 2026-01-29, and nothing has been run since.

### 8.2 Inventory

There is **no** spectral data on the cluster. There are no axial or vector spectra, no `mu_g_440/` tree, no critical-zoom `.pkl` files, and no peak files. The cluster holds only phase diagrams, axial melting lines (potential criterion), the σ log, and bookkeeping.

| Location | Contents | Files | Size | Dates (2025) | What it is |
|---|---|---|---|---|---|
| checkout `phase_data/` | `phase_diagram_improved_mq_*_lambda1_*_gamma_*_lambda4_*.csv` | 32 | 128 KB | Sep 4 – Sep 19 | T_c(μ) and order, μ = 0–400 in 21 steps |
| `/net/project/QUENCH/phase_data/` | same, plus `mq_12.0_lambda1_6.1_gamma_-22.4` from `awaggoner5` | 33 | 136 KB | copied Sep 19, plus Sep 22 | 32 byte-identical to the checkout; 1 extra |
| checkout `axial_data/` and QUENCH `axial_data/` | `axial_melting_data_*.csv` | 22 + 22 | 92 KB | Sep 4 – Sep 19 | axial melting line, potential-well criterion, μ_g = 440; byte-identical copies |
| `phase_plots/`, `axial_plots/` (both locations) | PNGs of the above | 32–33 / 22 | 9 MB / 4 MB | same | plots only |
| checkout `sigma_data/sigma_calculations.csv` | σ log | 1 | 16 MB | Sep 4 – Sep 19 | 107,876 lines, every σ solve (see §8.4) |
| QUENCH `sigma_data/` | `sigma_backup_task*_*.csv` | 58 | 1.1 GB | Sep 4 – Sep 22 | growing snapshots of the same log. The newest equals the checkout file. **Redundant**: keep the newest, archive the rest. |
| checkout `summary_data/` | `task_summary.csv`, `physics_summary_report.txt` | 2 | 8 KB | Sep 11 | 1 row, wrong (see §8.4) |
| QUENCH `summary_data/task_summary.csv` | task log | 1 | 6 KB | Sep 11 – Sep 22 | 32 rows; parameters correct, physics fields empty |
| checkout `slurm_logs/` | `batch_<job>_<task>.out/.err` | 75 + 75 | 21 MB | Sep 2 – Sep 19 | jobs 970–1083 |
| checkout `batch_990_5.out`, `vector_melting.csv` | tracked in git | 2 | 50 KB | Sep 4 | the CEP log and vector melting line already in the repo |
| `/net/project/QUENCH/DilatonMixingParallel/`, `Dilaton_coupling_exponential/`; `~/QGP/{DilatonMixing, Obsidian_dilaton_coupling_exponential, parallelVersion, BBcollab}` | 2022–2023 projects | — | 295 MB + 251 MB | 2022–2023 | earlier papers and models; not inputs to this paper |

**What the parameter grid covers.**
- **Baseline λ₁ scan:** m_q = 12, γ = −22.4, λ₄ = 4.2, λ₁ ∈ {5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.5}.
- **γ scan:** m_q ∈ {9, 12, 15} × λ₁ ∈ {5.5, 6.0, 7.0} × γ ∈ {−18, −20, −24, −26}, at **λ₄ = 4.0**, not 4.2.
  - So the γ scan does not share λ₄ with the baseline. A γ = −22.4, λ₄ = 4.0 reference point is missing.
  - The submitted grids also listed λ₄ = 4.4, but only array tasks 1–18 of 36 were submitted, so **no λ₄ = 4.4 run exists**.
  - **11 of the 12 λ₁ = 7.0 tasks failed**; only (m_q = 9, γ = −18) finished. Jobs 1065-7/8/9/16/17/18 hit the time limit, and jobs 1083-8/9/16/17/18 were OOM-killed at 3 GB per task. Each failed after only 1–2 μ points, so they left no phase file.

### 8.3 Extracted physics (from the CSV contents)

Common settings in every file:
- u_i = 0.01 and u_f = 0.9999 (so the cluster phase diagrams used u_i = 1e-2, *not* 1e-4 as §3.2 inferred from the code defaults);
- m_q tolerance 0.01; μ_g = 440 (hardcoded);
- RK45 at the default tolerance;
- smallest-positive-σ branch.

"Order" is the code's own classification: 2 = crossover (C), 1 = first order (F). The CEP bracket gives the last crossover point and the first first-order point. "Hysteresis width" is an upper bound on the gap between the code's T_c (the lower spinodal) and the upper spinodal at each first-order μ. It is taken from `sigma_calculations.csv` as (lowest T above T_c with one solution) − (lowest T with three solutions). T_A is the axial melting temperature. T_V is the vector melting temperature from `vector_melting.csv`, which is μ_g = 440 (see §8.4).

| m_q | λ₁ | γ | λ₄ | order, μ = 0→400 (C = crossover, F = first order) | T_c(0) | T_c(400) | CEP bracket: μ; T | max hysteresis width (MeV) | T_A − T_c at μ = 0 | μ with T_c < T_V | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 5.5 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 97.9 | 65.1 | — | — | +7.5 | 0–80 | usable with caveats |
| 9 | 6 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 110.2 | 75.0 | — | — | +1.6 | none | **must recompute** (T_c(μ) jumps up to 5 MeV) |
| 9 | 7 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCFF` | 129.1 | 93.3 | 360–380; 96.3–99.1 | 0.02 (med), 0.02 (max) | +0.1 | none | usable with caveats |
| 12 | 5.5 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 97.1 | 67.0 | — | — | +10.1 | 0–80 | **must recompute** (T_c(μ) jumps up to 8 MeV) |
| 12 | 6 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 111.8 | 77.0 | — | — | +2.4 | none | usable with caveats |
| 15 | 5.5 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 109.6 | 71.9 | — | — | -0.5 | 0–80 | **must recompute** (T_c(μ) jumps up to 13 MeV) |
| 15 | 6 | -18 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 117.1 | 80.4 | — | — | -0.7 | none | **must recompute** (T_c(μ) jumps up to 6 MeV) |
| 9 | 5.5 | -20 | 4 | `CCCCCCCCCCCCCCCCCCCCF` | 100.0 | 67.0 | 380–400; 67.1–69.7 | 0.04 (med), 0.04 (max) | +5.8 | 0–60 | usable with caveats |
| 9 | 6 | -20 | 4 | `CCCCCCCCCCCCCFFFFFFFF` | 112.4 | 77.0 | 240–260; 95.3–97.7 | 0.30 (med), 0.41 (max) | +0.4 | none | usable with caveats |
| 12 | 5.5 | -20 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 103.1 | 69.2 | — | — | +4.6 | none | usable with caveats |
| 12 | 6 | -20 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 114.5 | 79.3 | — | — | +0.9 | none | **must recompute** (T_c(μ) jumps up to 5 MeV) |
| 15 | 5.5 | -20 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 109.8 | 73.7 | — | — | -0.1 | none | **must recompute** (T_c(μ) jumps up to 10 MeV) |
| 15 | 6 | -20 | 4 | `CCCCCCCCCCCCCCCCCCCCC` | 119.1 | 82.4 | — | — | n/a | none | **must recompute** (T_c(μ) jumps up to 5 MeV) |
| 12 | 5.5 | -22.4 | 4.2 | `CCCCCCCCCCCCCCCCFCCFF` | 105.5 | 71.7 | 300–320; 81.9–84.2 (noisy) | 0.04 (med), 0.15 (max) | +2.9 (T_A to μ=220) | none | **must recompute** (order flips C/F/C) |
| 12 | 5.6 | -22.4 | 4.2 | `CCCCCCCCCCCCCCCCCFFFF` | 108.4 | 73.7 | 320–340; 81.6–84.2 | 0.08 (med), 0.28 (max) | +1.3 (T_A to μ=220) | none | usable with caveats |
| 12 | 5.7 | -22.4 | 4.2 | `CCCCCCCCCCCCCFCCFFFFF` | 110.3 | 75.7 | 240–260; 93.5–95.8 (noisy) | 0.06 (med), 0.40 (max) | +1.1 (T_A to μ=220) | none | **must recompute** (order flips C/F/C) |
| 12 | 5.8 | -22.4 | 4.2 | `CCCCCCCCCCCCFFFFFFFFF` | 112.5 | 77.7 | 220–240; 98.1–100.2 | 0.15 (med), 0.54 (max) | +0.7 (T_A to μ=220) | none | usable with caveats |
| 12 | 5.9 | -22.4 | 4.2 | `CCCCCCCCFFFFFFFFFFFFF` | 114.8 | 79.7 | 140–160; 108.0–109.5 | 0.15 (med), 0.64 (max) | +0.4 (T_A to μ=220) | none | usable with caveats |
| 12 | 6 | -22.4 | 4.2 | `FFFFFFFFFFFFFFFFFFFFF` | 116.9 | 81.6 | — | 0.14 (med), 1.00 (max) | +0.2 (T_A to μ=220) | none | usable with caveats |
| 12 | 6.1 | -22.4 | 4.2 | `FFFFFFFFFFFFFFFFFFFFF` | 118.9 | 83.5 | — | — | n/a | none | usable with caveats |
| 12 | 6.5 | -22.4 | 4.2 | `FFFFFFFFFFFFFFFFFFFFF` | 126.5 | 91.1 | — | 0.75 (med), 41 (max, spurious) | -0.1 | none | usable with caveats |
| 9 | 5.5 | -24 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 104.6 | 70.1 | — | 2.56 (med), 3.95 (max) | n/a | none | usable with caveats |
| 9 | 6 | -24 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 115.7 | 80.7 | — | 4.28 (med), 4.78 (max) | n/a | none | usable with caveats |
| 12 | 5.5 | -24 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 107.7 | 73.5 | — | 0.77 (med), 2.24 (max) | +1.3 | none | usable with caveats |
| 12 | 6 | -24 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 118.6 | 83.2 | — | 2.34 (med), 3.80 (max) | n/a | none | usable with caveats |
| 15 | 5.5 | -24 | 4 | `CCCCCCCCFFFFFFFFFFFFF` | 110.2 | 76.2 | 140–160; 103.6–105.1 | 0.38 (med), 1.21 (max) | +1.3 | none | usable with caveats |
| 15 | 6 | -24 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 121.4 | 86.2 | — | 1.09 (med), 2.71 (max) | n/a | none | usable with caveats |
| 9 | 5.5 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 106.7 | 72.2 | — | 3.52 (med), 3.52 (max) | n/a | none | usable with caveats |
| 9 | 6 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 117.8 | 81.1 | — | 3.73 (med), 6.83 (max, spurious) | n/a | none | usable with caveats |
| 12 | 5.5 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 109.7 | 74.9 | — | 3.28 (med), 4.74 (max) | n/a | none | usable with caveats |
| 12 | 6 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 120.7 | 84.8 | — | 4.92 (med), 4.92 (max) | n/a | none | usable with caveats |
| 15 | 5.5 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 112.9 | 78.0 | — | 2.31 (med), 3.96 (max) | n/a | none | usable with caveats |
| 15 | 6 | -26 | 4 | `FFFFFFFFFFFFFFFFFFFFF` | 123.1 | 88.0 | — | 4.37 (med), 4.56 (max) | -0.1 | none | usable with caveats |

**What the flags mean.**
- **No row is usable as final paper numbers.** Every row was computed with the default-tolerance RK45 and with the (m_q, σ) extraction that lacks the log term (§3.3). That error is a few percent in σ, and it can flip the order near a CEP. Every row needs recomputing with the Phase 1 solver before it goes into a figure.
- **Usable with caveats:** good for scouting, i.e. for choosing where to spend the recompute. The order pattern is clean (at most one C→F change) and T_c(μ) is smooth.
- **Must recompute:** do not quote at all. Either the order flips back and forth (C/F/C), or T_c(μ) jumps by 4.5 MeV or more between neighbouring μ, or it rises with μ. All the jump cases are broad crossovers at weak coupling (γ = −18, −20), where the code's T_c is poorly defined.
- Hysteresis widths: for (12, 6.5, −22.4) the 41 MeV maximum, and for (9, 6, −26) the 6.8 MeV maximum, come from isolated spurious three-solution rows. They are not real widths; the medians are reliable.

**What the data say**

1. **There are seven CEP brackets, not one.**
   - Clean CEPs: (m_q, λ₁, γ, λ₄) = (12, 5.6, −22.4, 4.2), (12, 5.8, −22.4, 4.2), (12, 5.9, −22.4, 4.2), (15, 5.5, −24, 4.0), (9, 6.0, −20, 4.0), (9, 5.5, −20, 4.0) and (9, 7.0, −18, 4.0).
   - Noisy CEPs: (12, 5.5, −22.4) and (12, 5.7, −22.4), where the order flips C/F/C.
   - Edge cases: the last two clean ones sit at μ = 360–400, at the edge of the grid. They need μ_max > 400 to confirm.
2. **The batch_990_5 CEP is narrower than §1 says.**
   - `batch_990_5.out` ran on a 0–200 grid with 20 points. It has crossover at μ = 157.9 (T_c = 108.14).
   - The rerun in `phase_data` (job 1002, same physics code) has first order at μ = 160 (T_c = 107.96).
   - Together: **μ_CEP ∈ (157.9, 160) MeV, T_CEP ≈ 108.0–108.1 MeV**, at m_q = 12, λ₁ = 5.9, γ = −22.4, λ₄ = 4.2. The bracket is still subject to the §3.3 solver error.
3. **The trends are smooth and physical.**
   - At m_q = 12, γ = −22.4: μ_CEP falls from ≈ 330 MeV (λ₁ = 5.6) to ≈ 230 (5.8) to ≈ 159 (5.9). At λ₁ = 6.0 the transition is already first order at μ = 0, with a hysteresis of only 0.14 MeV. So the μ = 0 critical coupling is **λ₁,c ∈ (5.9, 6.0)**.
   - More negative γ drives the transition toward first order. At m_q = 12, λ₁ = 5.5: γ = −20 is a crossover everywhere, −22.4 has a CEP near μ ≈ 310, and −24 is first order everywhere.
   - Larger m_q drives it toward crossover. At λ₁ = 5.5, γ = −24: first order for m_q = 9 and 12, a CEP at μ ≈ 150 for m_q = 15.
4. **T_A tracks T_c.** For γ ≤ −22.4 the axial melting temperature is within ±1.5 MeV of the chiral T_c at μ = 0. At γ = −18 to −20 with λ₁ = 5.5 and m_q ≤ 12, the axial line is 5–10 MeV above T_c at μ = 0 and crosses below it at larger μ. In this model, with this criterion, axial melting is essentially *locked to* the chiral transition.

### 8.4 Checks against the known problems

**γ provenance (§3.2): no mislabeled files on the cluster.**
- Every phase-diagram task log prints the γ it actually used, as `Using gamma = …` and `Using v3=…`.
  - All jobs labeled γ = −22.4 (970–1044) print **v3 = −2.640** = −22.4/(6√2). For −22.6 the value would be −2.663.
  - The γ-scan jobs print v3 = −2.121, −2.357, −2.828 and −3.064, which are γ = −18, −20, −24 and −26.
- `map_phase_diagram_improved.py` passed `-gamma`/`-lambda4` explicitly to `axial_melting_scan.py` at every run-time commit, so the axial files share the phase diagram's γ.
- `sigma_calculations.csv` contains no γ = −22.6 rows.
- The collaborator's λ₁ = 6.1 run: its σ backup shows γ = −22.4, λ₄ = 4.2 throughout.
- Every cluster file is dated Sept 4 or later, all produced by `map_phase_diagram_improved.py`. None was renamed by `rename_cp_data_files.py` or `rename_axial_melting_files.py`.
- **The mislabel risk is on the laptop/Dropbox copy instead.** Its `phase_data/` has July 2025 `chiral_transition_*_gamma_-22.4_*.pkl` files. Its `axial_data/` has 336 files, many from July 2025. There is also `mu_g_440/`. These predate the git history (which starts 2025-08-21), so their γ cannot be checked from code. Treat every Dropbox data file dated before 2025-09-04 as **γ = −22.6 unless shown otherwise**.

**First-order T_c is the lower spinodal (§3.1): the effect is small for T_c and for the CEP.**
- Every first-order (F) T_c in §8.3 is the *lower spinodal, not ΔF = 0*.
- The ΔF = 0 temperature lies between the two spinodals, so the hysteresis widths in the table bound the error.
- Near every clean CEP the median width is ≤ 0.4 MeV and the maximum ≤ 1.2 MeV. Deep in the first-order region (γ = −24, −26) the width grows to 2–5 MeV.
- The μ location of a CEP does not depend on how T_c is defined. It is where three solutions first appear. **So CEP μ brackets do not move under the ΔF fix, and T_CEP moves by ≲ 0.5 MeV.**
- §3.1 still matters for *branch selection*: spectra and axial melting in the coexistence region use the restored branch. But that region is a sliver ≲ 1 MeV wide near the CEPs.

**Low-T axial spectra (§3.4b): nothing on the cluster is affected.**
- No spectra were computed on the cluster.
- The axial melting lines use the Schrödinger-potential criterion, which does not involve the Wronskian.
- They also sit at T ≥ 67 MeV, where (μ_g z_h)² = (440/(πT))² ≤ 4.4 even at μ = 0, which is in the well-conditioned regime.
- There are no T = 0 mass comparisons on the cluster. The suspect low-T spectra are the local ones (`Axial.ipynb`, `mu_g_440/axial_data/`, `data/axial_params_T17.0_*`), as §3.4b says.

**μ_g: every cluster dataset used μ_g = 440.**
- 440 is hardcoded in the chiral solver, and `axial_melting_scan.py` defaults to `-mug 440`.
- `vector_melting.csv` is **μ_g = 440 with the potential-well criterion**. I recomputed the vector well-disappearance temperature with `axial_melting_scan.dVT`: 102.72, 93.42 and 64.02 MeV at μ = 0, 100 and 200. The file has 102.80, 93.50 and 64.10, which agree within its 0.1 MeV resolution.
  - With μ_g = 388 the same criterion gives 90.6, 80.0 and 46.3 MeV.
- So the vector and axial lines on the cluster use **the same melting criterion and the same μ_g**, and can be compared with each other. This is better than §3.5 feared for this dataset, although `run_vector_melting_scan.py` itself uses a different criterion and is still broken.

**Logging bugs (§3.7): confirmed, and nothing in the summaries is usable.**
- **The checkout's `summary_data/task_summary.csv` has one row (`batch_1044_1`) with the wrong parameters.**
  - It records λ₁ = 6.5 and order 2, but task 1044_1 ran λ₁ = 5.5. The λ₁ = 6.5 file is first order everywhere anyway, so the order is wrong too.
  - The cause is the "most recent file" bug: 1044_3 (λ₁ = 6.5) finished just before it.
- The QUENCH `task_summary.csv` (32 rows) takes its parameters from the SLURM arguments, so these are correct. But **every physics field is empty, and `has_critical_point` is `False` in all rows**, including the five rows whose files contain a CEP.
- Use the phase-data CSVs and §8.3, not the summaries.

**`sigma_calculations.csv`: corrupt, so the σ cache never worked.**
- 123 of 107,876 lines are malformed: rows from concurrent tasks interleaved without a newline (for example line 2050). About 70 more rows are NaN or garbage, with values such as γ = 1.0, u_i = 6.0 or num_solutions = 0.303.
- Because of these lines, every later `check_existing_calculation` call failed with `Error tokenizing data … line 2050` (≈ 2,000–5,000 warnings per task log). **Deduplication and reuse were silently disabled for every run after Sep 4.**
- 29,121 rows (27%) duplicate an earlier key exactly. The duplicated σ values agree to ≤ 2 × 10⁻¹¹ relative, so the solver is deterministic and this is pure wasted compute.
- The QUENCH backups repeat the same corruption. The collaborator's backup also has 8 repeated header lines.
- Fix: move to per-task output files merged afterwards, or SQLite (§4.7). Do not use `flock` on NFS.

**SLURM failures.**
- Several early jobs died before computing anything:
  - job 970: `python: command not found`;
  - job 972: `batch_phase_diagram_scan.py` was called without `--parameter`;
  - jobs 976-2/3/4, 980-2/3 and 984: argument parsing failed. In 984, a division by zero in `N_GAMMA` left `-gamma` without a value.
- Jobs 976-1, 980-1, 987 and 990 computed the phase diagram, but lost the axial line to `capture_output` on Python 3.6.
- The 11 λ₁ = 7.0 tasks above failed on time limit or OOM. Strong first-order points need more T iterations, and at 20 T points × joblib the memory exceeded 3 GB.
- Every other task finished.

### 8.5 Go/no-go (§6, Phase 3), with the data that exist

The question: does chiral restoration (T_c) happen *before* melting, i.e. T_c < T_melt? This is with μ_g = 440 and the potential-well criterion in both channels.

- **Vector channel: no for every parameter set with a CEP.**
  - At each CEP, T_V(μ_CEP) is far below T_CEP. The ρ has already melted:

    | CEP set | T_V(μ_CEP) | T_CEP |
    |---|---|---|
    | (12, 5.9, −22.4) | 81 MeV | 108 MeV |
    | (15, 5.5, −24) | 81 MeV | 104 MeV |
    | (12, 5.8, −22.4) | 51 MeV | 99 MeV |
    | (9, 6.0, −20) | 41 MeV | 96 MeV |
    | (12, 5.6), (9, 5.5, −20), (9, 7.0, −18) | 0 (fully melted for μ ≳ 320) | 68–98 MeV |

  - At μ = 0 the chiral T_c lies above T_V = 102.8 MeV in 30 of the 33 sets, by up to 26 MeV.
- **The only windows with T_c < T_V** are the three weakest-coupling sets, and only at small μ:

  | Set | Window | Margin at μ = 0 | Status |
  |---|---|---|---|
  | (12, 5.5, −18, 4.0) | μ ≤ 80 | 5.7 MeV | must recompute |
  | (9, 5.5, −18, 4.0) | μ ≤ 80 | 4.9 MeV | usable with caveats |
  | (9, 5.5, −20, 4.0) | μ ≤ 60 | 2.9 MeV | usable with caveats; **also has a CEP**, at μ ≈ 390 |

  - These margins are comparable to the §3.3 solver error, and all three are broad crossovers where T_c is poorly defined.
  - (9, 5.5, −20) is the only candidate for Goal 10, "degeneracy for sets with a CEP". Even there, the degeneracy window (μ ≤ 60) is far from the CEP (μ ≈ 390).
- **Axial channel: T_A ≈ T_c (±1.5 MeV) for γ ≤ −22.4.** "Restoration before axial melting" therefore cannot be resolved with the current criterion and solver accuracy.
- **μ_g is fixed at 440**, so the comparison above is the relevant one. (For reference only: at μ_g = 388 the vector line would drop by 12%.)
- **The effect of γ = −22.6 versus the −22.4 data.** From the γ scan, T_c(μ = 0) rises by about 1.1 MeV per unit of −γ (for example, at m_q = 12, λ₁ = 6.0: 111.8, 114.5, 116.9, 118.6 and 120.7 MeV at γ = −18, −20, −22.4, −24 and −26). So δγ = −0.2 raises T_c by about 0.2 MeV.
  - The CEP moves more. More negative γ favours first order, so μ_CEP at fixed λ₁ should drop by roughly tens of MeV, and λ₁,c(μ = 0) should fall slightly below the (5.9, 6.0) bracket.
  - This estimate is crude: the scan mixes λ₄ = 4.0 and 4.2.
  - Neither shift changes the no-go: T_V is independent of γ, and T_c moves up, not down.

**Answer:** with the existing data, **no-go for degeneracy below melting in the CEP region**. There is at most a marginal window, at weak γ and small μ, that is not yet resolved. The paper's story then becomes the negative result in §6 Phase 3: "chiral restoration and the CEP occur in the melted phase; axial melting is locked to the chiral transition." That still answers Goals 9–10 and fits the comparison with the reviewed paper.

**What could overturn it.** The potential-well criterion is the earliest possible melting signal. Spectral peaks (or QNMs with Γ/M < 1) can survive somewhat above the point where the well disappears. Before writing this up, run **one** QNM or width check for both channels at (12, 5.9, −22.4) and at (9, 5.5, −20), using the corrected solver. If the ρ QNM keeps Γ/M < 1 up to T ≈ 108 MeV at μ ≈ 160, the answer changes.

### 8.6 What can be reused

- **Reuse directly:**
  - the parameter-space map itself, i.e. which (m_q, λ₁, γ) give a CEP and roughly where;
  - `vector_melting.csv` (μ_g = 440, well criterion) as a cross-check for the new vector code;
  - the d₀ values in `sigma_calculations.csv`, as starting guesses for continuation in T (§4.2), after deduplication and with corrupt lines dropped. These are good warm starts even though the σ values themselves need the corrected extraction.
- **Recompute with the Phase 1 solver (targeted, not a full rerun):**
  - the seven clean CEP sets, on a μ grid refined around each bracket, with tight tolerance and the log-term extraction;
  - the λ₁ ∈ (5.9, 6.0) interval at m_q = 12, γ = −22.4, to pin λ₁,c at μ = 0;
  - extend μ_max to about 500 for (9, 5.5, −20) and (9, 7.0, −18);
  - axial lines for the γ = −22.4 sets beyond μ = 220.
- **Run for the first time:**
  - γ = −22.4 at λ₄ = 4.0 (the missing reference point for the γ scan);
  - the never-submitted λ₄ = 4.4 half of the grid;
  - the λ₁ = 7.0 tasks, with more memory (≥ 8 GB) and a longer time limit;
  - the reference λ₁ scan at γ = −22.6, λ₄ = 4.2 (the earlier paper's values), as a new scan point next to −22.4.
- **Keep every physics dataset.** Every phase diagram and axial line is a valid scan point at the parameters in its name.
  - Every row needs recomputing with the corrected solver before it goes into a figure (§8.3). The old files are still worth keeping, to compare against.
  - Don't *rely on* the two `task_summary.csv` files or `physics_summary_report.txt`. Their physics fields are wrong or empty, so use §8.3 instead.
  - The 1.1 GB of σ backups on QUENCH are redundant snapshots of the same log. Whether to archive them is only a disk-space question.

A copy of the small cluster files (phase and axial CSVs, summaries, `vector_melting.csv`) and the derived tables from this section is on branch `data/obsidian-snapshot-20261105`. That branch is not merged into `main`.

---

## Appendix B: numba and Wronskian tests (2026-10-19)

All tests ran in scratch space, outside the repository. They imported `axial_spectra` and `chiral_solve_complete` unchanged.

**Background.** `axial_spectra.initialize_chiral_field` with T = 17 (or 60) MeV, μ = 0, m_q = 9, λ₁ = 7.438, u_i = 1e-2, u_f = 1 − 1e-4. The model defaults were γ = −22.6, λ₄ = 4.2, μ_g = 440.

**Integrator.**
- Fixed-coefficient Dormand–Prince 5(4) under `@njit`, with error control on an RMS norm with atol + rtol·max(|y|).
- An optional output grid; the integrator lands exactly on each grid point.
- Integration can run in either direction.
- The axial state vector [ψ₁, ψ₁′, ψ₂, ψ₂′] is integrated in one pass. χ(u) comes in as arrays and is evaluated with `np.interp`.
- The UV initial data come from the existing `axial_spectra.a_s`.

**Comparisons.**
- Axial: the current `axial_spectra.BA` (LSODA, rtol 1e-8), scipy DOP853 and LSODA at rtol 1e-12, numba at rtol 1e-8 and 1e-11, and the exact-Wronskian formula.
- Chiral: the current `chiral_solve_IR` (RK45, default tolerance), scipy RK45 at rtol 1e-10, and numba at both tolerances, all from the same horizon data at d₀ = 0.5.

## Appendix A: how the tolerance and u_i test was run

I ran a scratch script outside the repository, with `log_results=False` so no files were written. It imported `chiral_solve_complete` and wrapped `scipy.integrate.solve_ivp` to pass `rtol`/`atol`. It ran under `joblib.parallel_backend("threading")` so that the patch reached the parallel workers: with the default loky backend the workers re-import the module and ignore the patch. Parameters: m_q = 9 MeV, m_q tolerance 0.1, λ₁ = 7.438, μ = 0, u_f = 1 − 1e-4, and the default γ = −22.6, λ₄ = 4.2. Results are in the table in §3.3.
