# WR Population

These are the notebooks used to generate the figures in this paper: https://ui.adsabs.harvard.edu/abs/2026arXiv260905398B/abstract

## Setup

You need to install POSYDON v2.2.8, the version used for the population synthesis. `environment.yml` installs it from the `posydon` conda channel together with the other dependencies.

```bash
conda env create -p ./env -f environment.yml   # POSYDON v2.2.8 + dependencies in ./env (git-ignored)
conda activate ./env
conda env config vars set \
    PATH_TO_POSYDON=$(python -c "import site; print(site.getsitepackages()[0])") \
    PATH_TO_POSYDON_DATA=<POSYDON data>
conda activate ./env                           # re-activate so the variables take effect
pip install --no-deps -e .                     # the wr_population package
python -m ipykernel install --user --name wr_population --display-name "Python (wr_population)"
```

`PATH_TO_POSYDON` points to the `site-packages` folder of `./env`, where conda installed POSYDON.

`PATH_TO_POSYDON_DATA` must be an existing directory, but it can be empty. Only the Case A/B notebooks (`Case_A_SMT_example_models.ipynb`, `Case_B_SMT_example_models.ipynb`) need POSYDON data: the v2 HMS-HMS grids in `<POSYDON data>/POSYDON_data/HMS-HMS/`.

All notebooks use the `wr_population` kernel and import from the `wr_population` package, so they run from any folder.

## Directory structure

- `wr_population/` — shared Python code used by the notebooks and scripts
- `population_synthesis/` — POSYDON population run config (`WR_run.ini`), its output, and the post-processing scripts
- `data/` — extracted WR phases from the population run and the observed WR catalogues
- `figure_scripts/` — notebooks that make the paper figures
- `HPC_high_detail_grids/` — Case A/B example figures from the high-resolution POSYDON grids

The large data files in `data/` and `population_synthesis/` are available on Zenodo.

## Pipeline to reproduce

1. Run popsynth on HPC with `runs/2026_08_14/WR_run.ini`
2. `calculate_model_weights.py` → stores `extended_IMF` model weights (population params hardcoded in script).
3. `extract_WR_optical_depth.py` → `data/optical_depth/WR_phases_Z*_Zsun.pkl` (per-binary DataFrames with `weight` = Δt, `model_weight`, `optical_depth`).
4. Run the `figure_scripts/*.ipynb` notebooks. They also read the `interp_class_*` columns from the `oneline` table of the `.h5` files to split WR stars by formation channel.

## WR selection

- Optical depth $\tau$ (Langer 1989), with $v_0 = 20\,\mathrm{km\,s^{-1}}$, $\kappa = 0.2\,(1 + X_\mathrm{H})$, and $v_\infty$ a type-dependent factor $\times\, v_\mathrm{esc}$.
- WR phase: $\tau > 1.5$.
- WNL: $0.05 < X_\mathrm{H} \leq 0.5$, not H-burning.
- WNE: $X_\mathrm{H} \leq 0.05$, $X_\mathrm{He} \geq 0.7$.
- WC: $X_\mathrm{H} \leq 0.05$, $X_\mathrm{He} < 0.7$.
- Binary mask: `state` in {`detached`, `RLO1`, `RLO2`, `contact`}.
- Formation channel: S1 from `interp_class_HMS_HMS` (`stable_MT`, `unstable_MT`, `stable_reverse_MT`, `no_MT`/NaN → No MT); S2 from `interp_class_CO_HMS_RLO` (`stable_MT`, `unstable_MT`, `no_MT`/NaN → No MT).

### Weighting

Each binary $i$ gets a model weight $w_i$ (the `model_weight`, POSYDON's `extended_IMF` weights from `calculate_model_weights.py`). It converts the sampled initial distributions to the target ones (see [Initial distributions](#initial-distributions)) and normalises to the stellar mass formed:

$$
w_i = \frac{p_\mathrm{target}(M_{1,i}, q_i, P_i)}{p_\mathrm{sampled}(M_{1,i}, q_i, P_i)} \times \frac{1}{M_\mathrm{sim}} \frac{\langle M \rangle_\mathrm{sampled}}{\langle M \rangle_\mathrm{target}},
$$

where $p$ are the joint PDFs of primary mass, mass ratio and orbital period, $M_\mathrm{sim}$ is the total stellar mass of the simulated population, and $\langle M \rangle$ is the mean mass per system of each distribution. $w_i$ is thus the number of such binaries per $\mathrm{M_\odot}$ of star formation.

Each history step $j$ of binary $i$ that passes the WR selection contributes its duration $\Delta t_{ij}$ (the `weight` column), so the number of WR stars is

$$
N_\mathrm{WR} \propto \sum_{i} \sum_{j \in \mathrm{WR}} \Delta t_{ij}\, w_i .
$$

## Initial distributions

The population is sampled from one set of initial distributions and then reweighted to another with the `extended_IMF` model weights.

| Parameter       | Sampled (`WR_run.ini`)            | Reweighted (`calculate_model_weights.py`)      |
| --------------- | --------------------------------- | ---------------------------------------------- |
| Binaries        | 300,000 per metallicity           | —                                              |
| Binary fraction | 1.0                               | 1.0                                            |
| Primary mass    | Kroupa (2001), 7–150 M☉           | Kroupa (2001), 0.01–150 M☉                     |
| Secondary mass  | flat in q, 0.5–150 M☉             | flat in q (0–1), 0.01–150 M☉                   |
| Orbital period  | Sana+12 (extended), 0.75–50,000 d | flat in log P, 0.75–6,000 d                    |
| Metallicity     | 0.01, 0.1, 0.2, 0.45, 1.0 Z☉      | —                                              |

Binaries sampled with P > 6,000 d get `model_weight = 0`, so all analysis covers 0.75–6,000 d. The "Sana+12" figures use the sampled distribution directly (Δt weights only), restricted to the same `model_weight > 0` binaries.