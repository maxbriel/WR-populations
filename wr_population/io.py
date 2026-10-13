"""Loading of the extracted WR phases, population files and observed catalogues."""
import json
import pickle

import numpy as np
import pandas as pd

from .paths import OBSERVATIONS_DIR, OPTICAL_DEPTH_DIR, population_file, WR_phases_file
from .stellar import calc_Teff


def load_metallicity(metallicity, data_dir=OPTICAL_DEPTH_DIR):
    """Load the extracted WR phases of one metallicity.

    Rows of S1 and S2 are stacked; quantities refer to the star in its
    (possible) WR phase, `companion_*` to the other star.
    Returns a dict of arrays ready for analysis.
    """
    from posydon.utils.common_functions import orbital_separation_from_period

    with open(WR_phases_file(metallicity, data_dir), 'rb') as f:
        data = pickle.load(f)

    out = {}
    for star, other in [('S1', 'S2'), ('S2', 'S1')]:
        WR_data = data[f'{star}_WR_data']
        s = pd.concat(WR_data.values(), keys=WR_data.keys())
        out[star] = {
            'time':            s['time'].values,
            'weight':          s['weight'].values,
            'model_weight':    s['model_weight'].values,
            'state':           s['state'].values,
            'key':             s.index.get_level_values(0).values,
            'star':            np.full(len(s), star),
            'period':          s['orbital_period'].values,
            'separation':      orbital_separation_from_period(s['orbital_period'].values,
                                                              s[f'{star}_mass'].values,
                                                              s[f'{other}_mass'].values),
            'Teff':            10**calc_Teff(s[f'{star}_log_L'].values, s[f'{star}_log_R'].values),
            'L':               10**s[f'{star}_log_L'].values,
            'radius':          10**s[f'{star}_log_R'].values,
            'Mdot':            10**s[f'{star}_lg_mdot'].values,
            'mass':            s[f'{star}_mass'].values,
            'optical_depth':   np.asarray(s['optical_depth'].values, dtype=float),
            'surface_H':       s[f'{star}_surface_h1'].values,
            'surface_He':      s[f'{star}_surface_he4'].values,
            'star_state':      s[f'{star}_state'].values,
            'mass_companion':  s[f'{other}_mass'].values,
            'companion_state': s[f'{other}_state'].values,
        }
        del s, WR_data
    del data
    return {k: np.concatenate([out['S1'][k], out['S2'][k]]) for k in out['S1']}


def load_interaction_classes(metallicity):
    """Mass-transfer interaction classes of every binary, indexed by binary index."""
    return pd.read_hdf(population_file(metallicity), 'oneline',
                       columns=['interp_class_HMS_HMS', 'interp_class_CO_HMS_RLO'])


def load_galactic_observations():
    """Galactic WR binaries; [lower, value, upper] columns are split into
    `<col>_err_lo`, `<col>`, `<col>_err_hi`."""
    with open(OBSERVATIONS_DIR / 'Galactic_WRs.json', 'r') as f:
        observational_data = json.load(f)
    obs_df = pd.DataFrame.from_dict(observational_data, orient='index')
    obs_df.index.name = 'WR_name'
    list_cols = ['Period', 'Eccentricity', 'M1', 'M2', 'Mass Function', 'M1_sin3i', 'M2_sin3i', 'RA', 'Dec']
    for col in list_cols:
        if col in obs_df.columns:
            unpacked = pd.DataFrame(obs_df[col].tolist(), index=obs_df.index,
                                    columns=[f'{col}_err_lo', col, f'{col}_err_hi'])
            obs_df = obs_df.drop(columns=col).join(unpacked)
    return obs_df


# BAT99 entries dropped from the LMC catalogue
LMC_BAT99_TO_DROP = [6, 12, 27, 33, 42, 59, 68, 77, 80, 92, 93, 97, 98, 99, 102, 104, 105, 106,
                     107, 108, 109, 110, 112, 113, 114, 116, 117, 118, 119, 126]


def load_LMC_observations():
    """LMC WR catalogue without the dropped BAT99 entries; adds a boolean `is_binary` column."""
    LMC = pd.read_csv(OBSERVATIONS_DIR / 'LMC_WRs.csv', sep=',')
    LMC = LMC[~LMC['BAT99'].isin(LMC_BAT99_TO_DROP)].copy()
    LMC['is_binary'] = LMC['Binary?'] != '-'
    return LMC


def load_SMC_observations():
    """SMC WR catalogue; adds a boolean `is_binary` column."""
    SMC = pd.read_csv(OBSERVATIONS_DIR / 'SMC_WRs.data', sep=',')
    SMC['is_binary'] = SMC['binary_type'].str.contains('binary', na=False)
    return SMC


def galactic_periods():
    """Orbital periods (days) of the Galactic WR binaries."""
    return load_galactic_observations()['Period'].dropna().astype(float).values


def LMC_binary_periods():
    """Orbital periods (days) of the LMC WR binaries with a measured period."""
    LMC = load_LMC_observations()
    return LMC['P [d]'][LMC['is_binary'] & (LMC['P [d]'] != '-')].astype(float).values


def SMC_binary_periods():
    """Orbital periods (days) of the SMC WR binaries with a measured period."""
    SMC = load_SMC_observations()
    return SMC['binary_period[days]'].dropna().astype(float).values
