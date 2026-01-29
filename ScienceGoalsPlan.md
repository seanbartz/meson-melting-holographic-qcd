# Science Goals Plan

This plan summarizes the current status of each goal in `ScienceGoals.md` and the concrete next steps to address them.

## Goal Status (as of 2026-01-29)

- **Accurate axial spectra (avoid missing low-lying narrow peak)**
  - **Status:** Partially supported, not verified.
  - **Evidence:** `axial_spectra.py` computes spectra + peaks; `run_axial_temperature_scan.py` cleans duplicate peaks and saves plots/summary; `calculate_peak_widths.py` can compute FWHM. No dedicated low-ω validation workflow.

- **Load existing spectra to avoid duplicate calculations**
  - **Status:** Not implemented.
  - **Evidence:** `axial_spectra.py` always writes new files. `run_axial_temperature_scan.py` calls `axial_spectra.main()` without any cache check.

- **Demonstrate a first-order transition for axial spectra**
  - **Status:** Not implemented (for axial spectra).
  - **Evidence:** First-order vs crossover logic exists in phase-diagram tooling (`critical_zoom_improved.py`, `map_phase_diagram_improved.py`, `batch_phase_diagram_unified.py`), but there is no axial-spectra-specific demonstration.

- **Find (mq, lambda1) giving a critical point evidenced by spectra**
  - **Status:** Partially supported (critical point detection), not tied to spectra.
  - **Evidence:** `extract_physics_results.py` can detect critical points and log results, but does not connect to spectral features.

- **Assess effect of gamma and lambda4 parameters**
  - **Status:** Not implemented.
  - **Evidence:** Parameter support exists across scripts, but no dedicated gamma/lambda4 sweep tied to spectra/phase conclusions.

- **Match T=0 meson spectra**
  - **Status:** Not implemented.
  - **Evidence:** No systematic T≈0 fitting workflow or target-matching comparison.

- **Check if setup yields both critical point and reasonable melting temperature**
  - **Status:** Supported in tooling, not executed.
  - **Evidence:** `extract_physics_results.py` logs both `axial_melting_T_mu0` and critical point presence into `summary_data/task_summary.csv` if runs are performed.

- **Compare with recent paper (crossover spectra despite first-order deconfinement)**
  - **Status:** Not implemented.
  - **Evidence:** No comparison workflow or notes in repo.

- **Chiral symmetry restoration via axial/vector degeneracy below melting**
  - **Status:** Not implemented.
  - **Evidence:** Axial and vector spectra exist separately; no automated degeneracy/overlap analysis or temperature-threshold metric.

- **Degeneracy co-occurring with critical point**
  - **Status:** Not implemented.
  - **Evidence:** No joined analysis between degeneracy metric and critical-point detection.

- **First-order chiral restoration evidenced in spectra**
  - **Status:** Not implemented.
  - **Evidence:** No spectral-order diagnostic beyond phase-diagram order.

## Prioritized Plan

1) **Add spectrum caching to avoid duplicate runs**
   - Implement “load if exists” in `axial_spectra.py` and expose in `run_axial_temperature_scan.py`.
   - Use the existing filename convention:
     `mu_g_440/axial_data/axial_spectral_data_T_{T}_mu_{mu}_mq_{mq}_lambda1_{lambda1}.csv`.

2) **Validate low-lying narrow peaks**
   - Add a high-resolution low-ω scan mode (smaller `wr` / larger `wcount`) with a diagnostic plot.
   - Use `calculate_peak_widths.py` to compute FWHM for low-T, μ=0 spectra.

3) **Map critical-point region in (mq, lambda1)**
   - Run `batch_phase_diagram_unified.py` over a grid of mq/λ1 (with fixed gamma/lambda4).
   - Use auto-logging to populate `summary_data/task_summary.csv`.
   - Filter for `has_critical_point=True` and collect `axial_melting_T_mu0` ranges.

4) **Tie axial spectra to transition order**
   - For candidate (mq, λ1) values, run `run_axial_temperature_scan.py` near the critical region.
   - Compare peak behavior vs transition order from phase-diagram results.

5) **Assess gamma/lambda4 sensitivity**
   - Sweep gamma and lambda4 for a fixed (mq, λ1) and repeat Steps 3–4.
   - Track changes in critical point location, melting temperatures, and spectral features.

6) **Chiral restoration via axial/vector degeneracy**
   - Define a degeneracy metric between axial and vector spectra (e.g., peak alignment or integrated difference).
   - Evaluate it vs temperature to find a degeneracy temperature below melting.

7) **Degeneracy + critical point overlap**
   - Cross-reference degeneracy temperature findings with parameter sets that show critical points.

8) **Match T≈0 meson spectra**
   - Run axial and vector spectra at T≈0, μ=0 for candidates.
   - Compare against target masses and pick best-fit parameters.

9) **Paper comparison figure/table**
   - Create a comparison figure/table showing axial/vector spectral behavior vs transition order in your model and in the reviewed paper.

10) **First-order chiral restoration in spectra**
   - Look for discontinuities or hysteresis in spectral peak behavior vs T across the transition region.

## Suggested Next Step

Start with Step 1 (caching). It reduces redundant work for all later scans and is straightforward to implement.
