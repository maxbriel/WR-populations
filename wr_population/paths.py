"""Locations of the data, population runs and figures, relative to the repository."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / 'data'
OPTICAL_DEPTH_DIR = DATA_DIR / 'optical_depth'
OBSERVATIONS_DIR = DATA_DIR / 'observations'

RUNS_DIR = REPO_ROOT / 'population_synthesis' / 'runs'
POPULATION_RUN_DIR = RUNS_DIR / '2026_08_14'

FIGURE_DIR = REPO_ROOT / 'figure_scripts' / 'figures'
EXTRA_FIGURE_DIR = REPO_ROOT / 'figure_scripts' / 'extra_figures'

# metallicities of the population runs, in units of Zsun
METALLICITIES = [0.01, 0.1, 0.2, 0.45, 1.0]


def population_file(metallicity, run_dir=POPULATION_RUN_DIR):
    """Path to the POSYDON population file of one metallicity."""
    from posydon.utils.common_functions import convert_metallicity_to_string
    return Path(run_dir) / f'{convert_metallicity_to_string(metallicity)}_Zsun_population.h5'


def WR_phases_file(metallicity, data_dir=OPTICAL_DEPTH_DIR):
    """Path to the extracted WR-phase pickle of one metallicity."""
    from posydon.utils.common_functions import convert_metallicity_to_string
    return Path(data_dir) / f'WR_phases_Z{convert_metallicity_to_string(metallicity)}_Zsun.pkl'
