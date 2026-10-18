# Project Assessment: Status, Corrections, Efficiency, and Path to a PRD Paper

*Assessment date: 2026-10-18. Covers `main` at commit `2a755a4`. No code was changed for this review.*

This document reviews the whole repository against the goals in `ScienceGoals.md`. It covers:

1. what exists and what has actually been computed
2. what must be corrected before any result is publishable
3. where the code can be made much faster
4. how far the project is from a Physical Review D paper
5. a staged plan to get there

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

3. **Two values of μ_g are in use.**
   - The code hardcodes μ_g = 440 MeV everywhere (`chiral_solve_complete.py:272`, `axial_spectra.py:59`, `vector_spectra.py:43`, output paths `mu_g_440/`).
   - The notebooks that compare with experiment use **μ_g = 388 MeV** (`Axial.ipynb`, `Vector_Spectra.ipynb`: `return 388 #440`). Only with 388 does the ρ come out at 775 MeV; with 440 the ρ sits at 880 MeV.
   - Even at 388 the a₁ ground state is not reproduced: the model gives about 1513 MeV against 1230 MeV measured.
   - The zero-temperature calibration, which every finite-T claim rests on, has not been settled.

**Distance to a paper:** the infrastructure is perhaps 70% done. The validated physics is perhaps 15–20% done.
- One clean critical-endpoint bracket exists, from the cluster run in `batch_990_5.out`:
  - parameters: m_q = 12, λ₁ = 5.9, γ = −22.4, λ₄ = 4.2;
  - crossover up to μ ≈ 158 MeV, first order from μ ≈ 168 MeV;
  - T ≈ 107–108 MeV.
  - However, its first-order T_c values are spinodal values (see 1 above).
- There is one vector melting curve (`vector_melting.csv`, with unknown μ_g) and scattered single-point results in `WorkbookJuly2025.ipynb`.
- There are no axial melting curves, no γ/λ₄ study, no axial/vector degeneracy analysis, and no branch-resolved spectra.

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
- **Scattered T_c values and order results** in `WorkbookJuly2025.ipynb`, for several (m_q, λ₁). γ is probably −22.6 and μ_g is unclear. One CEP estimate (m_q = 9, λ₁ = 5: μ ≈ 312–316, T ≈ 68.5–68.9) shows the order flipping 2/1/2, which is a sign of numerical noise near the CEP.
- **Vector melting line** (`vector_melting.csv`, μ_g not recorded): T_melt = 102.8 MeV at μ = 0, 93.5 at μ = 100, 64.1 at μ = 200, and it reaches 0 near μ ≈ 320.

> **Early physics warning.** At the one clean CEP parameter set, the chiral crossover at μ = 0 (114.8 MeV) lies *above* the vector melting temperature (102.8 MeV, if that file used μ_g = 440). Near the CEP (T ≈ 107 MeV, μ ≈ 160 MeV) the vector line is around 78 MeV. **The ρ is therefore already melted where the chiral transition happens.**
>
> The goals "axial/vector degeneracy *below* melting" and "degeneracy for parameter sets with a CEP" need T_χ < T_melt^V. This should be the **first go/no-go check** of the calibration study (§6, Phase 3). It is cheap to run and decides the story of the paper.

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
  - Fix: decide the value. Then create one `model_params.py` (γ, λ₄, μ_g, u_i, u_f, tolerances) that everything imports.
- **μ_g is hardcoded in four places.** The `-mug` flag of `axial_melting_scan.py` changes only the vector part of the potential, not the chiral background (`chiral_solve_complete.py:272`). Make μ_g a real parameter and put it in every cache key and filename.
- **Different cutoffs in different codes:**
  - The phase diagrams solve the chiral field with u_i = 1e-4 (`critical_zoom_improved.py:401`, `map_phase_diagram_improved.py:30`).
  - The axial spectra solve it with u_i = 1e-2 (`axial_spectra.py:25,922`).
  - Vector spectra use u_i = 1e-4, u_f = 1−1e-5. Axial spectra use u_i = 1e-2, u_f = 1−1e-4.
  - So the σ behind a spectrum is not the σ behind the phase diagram; see §3.3 for the size of the difference.
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
4. **Axial spectrum, one integration instead of two.** Integrate the infalling solution from the horizon to the boundary and read off the retarded Green's function, or compute Im G from the conserved flux Im(ψ* ψ′). This halves the ODE count, and flux conservation along u gives a built-in accuracy check.
5. **Faster RHS evaluation.** `Axial_eom` calls a scipy `interp1d` on every evaluation (`axial_spectra.py:399`). Precompute a `CubicSpline` of χ² once, or integrate χ together with the axial field. Optionally JIT the RHS with `numba`, or vectorize over many ω at once with a fixed-step high-order integrator.
6. **Parallelism in one place.** Parallelize over the outermost independent axis: the (T, μ) or ω tasks. Make the inner calls explicitly serial (`n_jobs=1` parameters, not environment variables). Remove the nested `Pool` → `joblib(n_jobs=-1)` pattern.
7. **Bookkeeping I/O.** `check_existing_calculation` reads the whole `sigma_calculations.csv` on every call (`chiral_solve_complete.py:44`). As the file grows this becomes noticeable. Load it once per process, or move to SQLite (also safer under concurrent cluster writes than `flock` on NFS).
8. **Cluster hygiene.** Do not copy every CSV/PNG on every SLURM task. Remove the sinfo-based CPU heuristics, which parse the wrong field; a fixed `--cpus-per-task` is simpler and correct.

---

## 5. Status of each science goal

| # | Goal (from `ScienceGoals.md`) | Status | Blocking items |
|---|---|---|---|
| 1 | Accurate axial spectra; no missed low-lying narrow peak | Not verified. Unexplained structures at 741–1166 MeV in the notebook. | §3.4: peak-location bug, ω window, QNM check, cutoff convergence |
| 2 | Load existing spectra; avoid duplicate work | Partly done (`--load-if-exists`) | §3.6: cache key collisions, missing params, σ = NaN, σ cache unused |
| 3 | First-order transition visible in axial spectra | Not possible yet | §3.1: free energy and branch selection |
| 4 | (m_q, λ₁) of the critical point from spectra | One CEP bracket (m_q = 12, λ₁ = 5.9). No map. Not linked to spectra. | §3.1, §3.3, §3.7; then a production scan |
| 5 | Effect of γ, λ₄ | Impossible for spectra; possible for the phase diagram only | §3.2 |
| 6 | Match T = 0 meson spectra | ρ OK only at μ_g = 388; a₁ 280 MeV too high | Calibration decision (§6, Phase 3) |
| 7 | Both a CEP and a reasonable melting temperature | Ingredients exist; not run consistently | §3.1, §3.2, §3.5 |
| 8 | Comparison with the reviewed paper | Not started | Needs 3, 7, and the Hawking–Page discussion in §3.9 |
| 9 | Chiral restoration via axial/vector degeneracy below melting | Not started. Early indication: T_χ > T_melt^V at the known CEP set. | §3.5; QNM masses; go/no-go check |
| 10 | Degeneracy for sets that also have a CEP | Not started | 4 and 9 |
| 11 | First-order restoration visible directly in spectra | Not started | 3; branch-resolved spectra |

---

## 6. Plan

Each phase ends with a concrete deliverable. Phases 0–2 are corrections and validation; do them before spending cluster time on production.

### Phase 0: consolidation (no physics change)
1. Create `model_params.py` with one dataclass holding T, μ, m_q, λ₁, γ, λ₄, μ_g, u_i, u_f and the tolerances. Make every module import it, and delete all hardcoded `lam_function`, `mu_g()` and module-level v3/v4 defaults.
2. Thread γ and λ₄ through `axial_spectra.main()` and its CLI, and through `run_axial_temperature_scan.py` and `run_axial_melting_scan.py`.
3. Delete or move the dead files listed in §2.3. Fix the flag mismatches and the `capture_output` problem (§3.7).
4. Decide the default γ (−22.4 or −22.6). Re-label or regenerate the legacy files whose names now say −22.4.
5. **Deliverable:** one command per observable (σ(T), T_c(μ), spectrum, melting line), with every parameter explicit and recorded in the output.

### Phase 1: numerical validation (Goal 1; the referee-proof foundation)
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
2. Redefine first-order T_c by ΔF = 0. Store both spinodals. Re-run the `batch_990_5` parameter set to see how far the CEP moves.
3. Track branches by continuation in T. Compute **axial spectra on both branches** across the coexistence region, so the peak positions and widths jump at T_c.
4. **Deliverable:**
   - a σ(T) figure with branches, spinodals and T_c;
   - an axial spectral-function figure across the first-order line, compared with the smooth evolution across the crossover.

### Phase 3: calibration and go/no-go (Goals 6, 7, 9)
1. Decide μ_g:
   - μ_g = 388 fits the ρ;
   - check whether the axial ground state can be fixed by m_q and λ₁ (and by γ, λ₄ through σ), since the a₁–ρ splitting is driven by g₅²χ²;
   - if the a₁ cannot be fitted, say so and fit to the a₁–ρ splitting or to (ω/μ_g)² ratios instead.
2. Use the m_q and λ₁ fit to the a₁ (and f_π, if computed) as constraints on the CEP search.
3. **Go/no-go:** for the calibrated set(s), compare the chiral T_χ(μ) with the vector and axial melting lines, using the single criterion from §3.5.
   - If T_χ > T_melt^V everywhere (as the early data suggests), then "degeneracy below melting" is a *negative result*. That is still publishable and fits the comparison with the reviewed paper.
   - Otherwise, map where T_χ < T_melt.
4. **Deliverable:** a T = 0 mass table and Regge plot (the existing `plot_meson_mass_squared.py` plus model points), and the go/no-go answer.

### Phase 4: production scans (Goals 4, 5, 7, 10)
1. A CEP map in (m_q, λ₁) at fixed (γ, λ₄): μ_CEP and T_CEP versus λ₁, for 2–3 values of m_q. Use the fixed pipeline with stable-branch T_c.
2. γ/λ₄ sensitivity: a small grid of about 3 × 3 around the default. Record how the CEP, the melting lines, and the T = 0 a₁ mass move.
3. For selected points: axial and vector QNM masses and widths versus T along μ < μ_CEP, μ ≈ μ_CEP and μ > μ_CEP, with the degeneracy metric |M_A − M_V| / M_V versus T.
4. **Deliverable:** the main phase-diagram figure (chiral line with CEP, vector and axial melting lines, degeneracy line), plus a parameter-dependence figure or table.

### Phase 5: paper
1. Comparison with the reviewed paper: transition order seen in the spectra versus the order of the deconfinement or chiral transition, including the Hawking–Page discussion (§3.9).
2. Write: model; methods (spectral functions, QNMs, free energy); T = 0 calibration; phase diagram and CEP; spectra across the transitions; degeneracy; comparison; conclusions; numerical appendix.
3. Archive the code version and data (Zenodo) for reproducibility.

---

## 7. How close is a PRD paper?

**What a referee would need:**
1. a calibrated model;
2. a correct phase structure (stable branch, ΔF-based T_c);
3. converged spectra or QNMs;
4. a clear new result.

Today, item 1 is undecided, item 2 is incorrect in the first-order region, item 3 is unverified, and item 4 is not yet computed.

**What a referee would see as new:**
- Direct spectral signatures of a first-order chiral transition and a CEP: discontinuous axial peak positions and widths, and hysteresis.
- Where the CEP sits relative to the meson melting lines, in a model with dilaton–chiral mixing (λ₁).
- The answer to whether a₁–ρ degeneracy happens before melting. A negative answer would also be interesting, set against the reviewed paper.

**Critical path:** Phase 0 → Phase 1 (items 1–3) → Phase 2 → Phase 3, item 3 (go/no-go). Once the go/no-go answer is in, the paper's story is fixed, and Phases 4–5 are mostly production and writing with tooling that already exists. The QNM solver (Phase 1, item 4) is the largest single piece of new code. It pays for itself in speed (§4.1) and in how credible the narrow-peak and degeneracy claims are.

**Suggested minimal first paper**, if you want to publish sooner:
- one calibrated parameter set;
- the corrected phase diagram with CEP and melting lines;
- axial spectra or QNMs across the first-order and crossover regions;
- the degeneracy go/no-go result;
- γ/λ₄ sensitivity as a short section;
- the full (m_q, λ₁) CEP map left for a follow-up.

---

## Appendix A: how the tolerance and u_i test was run

I ran a scratch script outside the repository, with `log_results=False` so no files were written. It imported `chiral_solve_complete` and wrapped `scipy.integrate.solve_ivp` to pass `rtol`/`atol`. It ran under `joblib.parallel_backend("threading")` so that the patch reached the parallel workers: with the default loky backend the workers re-import the module and ignore the patch. Parameters: m_q = 9 MeV, m_q tolerance 0.1, λ₁ = 7.438, μ = 0, u_f = 1 − 1e-4, and the default γ = −22.6, λ₄ = 4.2. Results are in the table in §3.3.
