# Obsidian data snapshot, 2026-11-05

Read-only copies of the small output files from the Obsidian cluster, taken for the reconciliation in `PROJECT_ASSESSMENT.md` §8 (branch `claude/cluster-reconciliation`).

- `checkout/`: from `~/QGP/meson-melting-holographic-qcd` (code at `2a755a4`; data written Sept 2025, see §8.1 for the commit at run time).
  - `phase_data/`, `axial_data/`: phase diagrams and axial melting lines.
  - `summary_data/`: the task summary. Its single row is wrong; see §8.4.
  - `slurm_logs/`: the evidence for the γ actually used (`Using v3=…` lines).
  - `vector_melting.csv`: μ_g = 440, potential-well criterion.
- `quench/`: from `/net/project/QUENCH/`. Identical to `checkout/` except for the extra `phase_diagram_improved_mq_12.0_lambda1_6.1_gamma_-22.4_lambda4_4.2.csv` (from user awaggoner5) and the 32-row `task_summary.csv`.
- `derived/`: tables made from the above (`phase_summary.csv`, the per-set summary; `spinodal_windows.csv`, the hysteresis bounds).

Not included:
- `sigma_data/sigma_calculations.csv` (16 MB, corrupted; §8.4);
- the 1.1 GB of σ backups;
- PNG plots.

These remain on the cluster.

All values were computed with the pre-fix solver. They are for scouting only (§8.3).
