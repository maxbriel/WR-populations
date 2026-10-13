# WR Population

Population-synthesis study of Wolf-Rayet (WR) stars formed through binary evolution, using POSYDON. All runs cover 5 metallicities: 0.01, 0.1, 0.2, 0.45, 1.0 Zsun.

## Setup

```bash
git clone --branch v2.2.8 https://github.com/POSYDON-code/POSYDON.git <POSYDON checkout>   # version used for the population run
conda env create -p ./env -f environment.yml          # git-ignored env in the repo
conda env config vars set -p ./env PATH_TO_POSYDON=<POSYDON checkout> PATH_TO_POSYDON_DATA=<POSYDON data>
./env/bin/python -m pip install -e <POSYDON checkout>
./env/bin/python -m pip install --no-deps -e .          # the wr_population package
./env/bin/python -m ipykernel install --user --name wr_population --display-name "Python (wr_population)"
```

All notebooks use the `wr_population` kernel and import from the `wr_population` package, so they run from any folder.

## Directory map

- `wr_population/` — shared code (installed with `pip install -e .`)
  - `paths.py` — data/run/figure locations relative to the repo, `METALLICITIES`, `population_file()`, `WR_phases_file()`
  - `io.py` — `load_metallicity` (WR-phase pickles → arrays), `load_interaction_classes`, observed catalogues (`load_galactic_observations`, `load_LMC_observations`, `load_SMC_observations`, `*_periods`)
  - `classification.py` — `classify_WR` (WNL/WNE/WC), `make_binary_mask`, `formation_channels` / `formation_channel_masks`, `CHANNEL_COLOURS`
  - `optical_depth.py` — optical-depth machinery (Langer 1989): τ, v_esc, v_inf, Γ_Edd
  - `stellar.py` — `calc_Teff`, Tout et al. (1996) ZAMS
  - `plotting.py` — `use_posydon_style` (paper figure style), `plot_period_histogram`, `METALLICITY_COLOURS`

- `population_synthesis/runs/2026_08_14/` — POSYDON popsynth run, read by all figure notebooks
  - `WR_run.ini` — POSYDON population config (300,000 binaries per metallicity, log-normal kicks)
  - Output: `*_Zsun_population.h5` (~5.2–5.6 GB each; git-ignored), with `extended_IMF` model weights stored inside
- `population_synthesis/scripts/` — post-processing of the runs
  - `calculate_model_weights.py` — computes `extended_IMF` model weights and stores them in the `.h5` files
  - `extract_WR_optical_depth.py` — extraction: keeps H-poor phases (X_H ≤ 0.7) per binary, computes optical depth, saves pickles
- `data/`
  - `optical_depth/` — (from run `2026_08_14`), read by all paper figure notebooks
  - `observations/` — Galactic (`Galactic_WRs.json`), LMC (`LMC_WRs.csv`), SMC (`SMC_WRs.data`) WR catalogues
- `figure_scripts/` — paper figure notebooks
  - `extra_figures/` — proposal/presentation figures; not part of the paper
- `HPC_high_detail_grids/` — Case A/B "partial stripping" figures from high-res POSYDON grids (needs access to the internal grids on `/srv/astro`)

## Pipeline to reproduce

1. Run popsynth on HPC with `runs/2026_08_14/WR_run.ini`
2. `calculate_model_weights.py` → stores `extended_IMF` model weights (population params hardcoded in script).
3. `extract_WR_optical_depth.py` → `data/optical_depth/WR_phases_Z*_Zsun.pkl` (per-binary DataFrames with `weight` = Δt, `model_weight`, `optical_depth`).
4. Run the `figure_scripts/*.ipynb` notebooks. They also read the `interp_class_*` columns from the `oneline` table of the `.h5` files to split WR stars by formation channel.

## Figures and the notebooks that make them (`figure_scripts/figures/`)

| Figure                                                                               | Notebook                                                                                                                           |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `MW_WR_period_distributions_MT`, `MW_subtype_period_distributions_MT`                | `MW_formation_channels.ipynb`                                                                                                      |
| `SMC_LMC_WR_period_distributions_MT`                                                 | `SMC_LMC_formation_channels.ipynb`                                                                                                 |
| `metallicity_dependence`, `_Sana12`, `_lower_optical_depth`                          | `metallicity_dependence.ipynb`                                                                                                     |
| `relative_contributions`                                                             | `relative_contributions.ipynb`                                                                                                     |
| `MW_other_selection_criteria`                                                        | `WR_selection.ipynb`                                                                                                               |
| `HR_diagrams`                                                                        | `HR_diagrams.ipynb`                                                                                                                |
| `observed_period_distributions`                                                      | `observations_summary.ipynb`                                                                                                       |
| `Case_A_WR_evolution_Zsun_045Zsun_01Zsun`, `Case_B_WR_evolution_Zsun_045Zsun_01Zsun` | `Case_A_SMT_example_models.ipynb`, `Case_B_SMT_example_models.ipynb` (need the POSYDON v2 HMS-HMS grids in `PATH_TO_POSYDON_DATA`) |

`binary_fraction.ipynb` and `stripped_stars.ipynb` only print/explore and save no figures.

## WR classification rulebook

- Optical depth τ (Langer 1989): v0 = 20 km/s, κ = 0.2(1+X_H), v_inf = type-dependent factor × v_esc.
- WR phase: τ > 1.5.
- WNL: 0.05 < X_H ≤ 0.5, not H-burning.
- WNE: X_H ≤ 0.05, X_He ≥ 0.7.
- WC: X_H ≤ 0.05, X_He < 0.7.
- Binary mask: state in {detached, RLO1, RLO2, contact}.
- Formation channel: S1 from `interp_class_HMS_HMS` (`stable_MT`, `unstable_MT`, `stable_reverse_MT`, `no_MT`/NaN → No MT); S2 from `interp_class_CO_HMS_RLO` (`stable_MT`, `unstable_MT`, `no_MT`/NaN → No MT).
- Number of WR stars ∝ Σ Δt × `model_weight`.

## Sampling vs. population parameters

Sampled in `runs/2026_08_14/WR_run.ini`:
- primary: Kroupa2001, 7–150 Msun
- secondary: flat mass ratio, 0.5–150 Msun
- period: Sana+12 extended, 0.75–50,000 d

Reweighted in `calculate_model_weights.py` (`extended_IMF` model weights):
- binary fraction: const = 1.0
- primary mass: Kroupa2001, 0.01–150 Msun
- secondary: flat mass ratio, 0.01–150 Msun, q = 0–1
- period: power law (slope 0, i.e. flat in log P), 0.75–6000 d

Binaries sampled with P > 6000 d get `model_weight = 0`, so all analysis is limited to 0.75–6000 d. The "Sana+12" figures use the sampled distribution directly (Δt weights only), restricted to the same `model_weight > 0` binaries.