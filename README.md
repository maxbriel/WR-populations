# WR Population

Population-synthesis study of Wolf-Rayet (WR) stars formed through binary evolution, using POSYDON. All runs cover 5 metallicities: 0.01, 0.1, 0.2, 0.45, 1.0 Zsun.

## Directory map

- `population_synthesis/runs/2025_11_04/` — POSYDON popsynth runs + HPC submission
  - `WR_run.ini` — POSYDON population config
  - `run_metallicity.py` / `merge_metallicity.py` — evolve / merge one metallicity
  - `*_Zsun_slurm_array.slurm`, `*_Zsun_merge_popsyn.slurm`, `slurm_submit.sh` — HPC array runs + merge
  - Output: `*_Zsun_population.h5` (1.6–1.8 GB each; git-ignored)
- `population_synthesis/scripts/` — post-processing of the runs
  - `optical_depth.py` — optical-depth machinery (Langer 1989): τ, v_esc, v_inf, Γ_Edd, τ_calc
  - `extract_WR_optical_depth.py` — **current** extraction: strips WR phases per binary, computes optical depth, saves pickles
  - `calculate_model_weights.py` — computes `extended_IMF` model weights
  - `extract_WR_phases.py` — **superseded**, see below
- `data/` — extracted phases + observed catalogues
  - `optical_depth_test/` — **canonical pickles** read by all figure notebooks
  - `optical_depth/`, `Z-dependent_L_limit/` — **superseded** pickle variants, not used
  - `observations/` — Galactic (`Galactic_WRs.json`), LMC (`LMC_WRs.csv`), SMC (`SMC_WRs.data`) WR catalogues
- `figure_scripts/` — final analysis notebooks + `helper_functions.py` (optical depth, WR classification, Tout ZAMS)
- `HPC_high_detail_grids/` — Case A/B "partial stripping" figures from high-res POSYDON grids (needs POSYDON grid access)

## Pipeline to reproduce

1. Run popsynth on HPC:
   `run_metallicity.py <Z>` (via `*_Zsun_slurm_array.slurm`, array 0–49), then `merge_metallicity.py <Z>` → `*_Zsun_population.h5`.
2. `calculate_model_weights.py` → stores `extended_IMF` model weights (population params hardcoded in script).
3. `extract_WR_optical_depth.py` → `data/optical_depth_test/WR_phases_Z*_Zsun.pkl` (per-bin DataFrames with `weight`, `model_weight`, `optical_depth`).
4. Open the `figure_scripts/*.ipynb` notebooks to reproduce figures.

## WR classification rulebook

- Optical depth τ (Langer 1989): v0 = 20 km/s, κ = 0.2(1+X_H), v_inf = type-dependent factor × v_esc.
- WR phase: τ > 1.5.
- WNL: 0.05 < X_H ≤ 0.5, not H-burning.
- WNE: X_H ≤ 0.05, X_He ≥ 0.7.
- WC: X_H ≤ 0.05, X_He < 0.7.
- Binary mask: state in {detached, RLO1, RLO2, contact}.

## Population parameters (reweighting in `calculate_model_weights.py`)

- binary fraction: const = 1.0
- primary mass: Kroupa2001, 0.01–150 Msun
- secondary: flat mass ratio, 0.01–150 Msun
- period: power law (slope 0), 0.75–6000 d
- q range: 0.0–1.0

*Original sampling (`WR_run.ini`): Kroupa2001 primary 7–150 Msun, flat mass ratio secondary, Sana+12 extended period 0.75–6000 d.*

## Superseded / not used

- `extract_WR_phases.py` — older extraction (Z-dependent luminosity limit `4.9 − log10(Z)`, X_H < 0.4); replaced by `extract_WR_optical_depth.py`.
- `data/Z-dependent_L_limit/` and `data/optical_depth/` — pickle outputs of that older approach; do not use.

## Caveats

- `Case_A_SMT_example_models.ipynb` and `Case_B_SMT_example_models.ipynb` save to the **same** output filename (`WR_evolution_Zsun_045Zsun_01Zsun.*`); running both overwrites one.
- `observations_summary.ipynb` writes figures via relative paths (run from `figure_scripts/`).
- `relative_contributions.ipynb` uses hardcoded percentages; no data-file inputs.
- `binary_fraction.ipynb` and `stripped_stars.ipynb` print/explore only; save no figures.