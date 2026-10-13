'''
This script extracts the WR phases from the posydon population runs and
stores it in a pickle file for the analysis notebooks.
'''

from posydon.popsyn.synthetic_population import TransientPopulation
import numpy as np
import pickle
from tqdm import tqdm

from wr_population.paths import METALLICITIES, OPTICAL_DEPTH_DIR, population_file, WR_phases_file
from wr_population.optical_depth import optical_depth_calculation
from wr_population.stellar import calc_Teff

# metallicities to process
metallicities = METALLICITIES

def get_WR_phases(history, model_weights, metallicity):
    '''Get the WR phases for both stars in the binary history.'''
    S1_WR_data = {}
    S2_WR_data = {}

    # unique binaries
    nr_systems = np.unique(history.index).size
    
    for i in tqdm(range(nr_systems), total=nr_systems):
        binary = history.loc[i]
        weight = model_weights.loc[i]
        out = filter_WR_phases(binary, weight, which_star='S1')
        if out is not None:
            S1_WR_data[i] = out
            
        out = filter_WR_phases(binary, weight, which_star='S2')
        if out is not None:
            S2_WR_data[i] = out
    return S1_WR_data, S2_WR_data

def set_star_parameters( which_star):
    """Set the column names and final event based on requested star."""
    if which_star not in ['S1', 'S2']:
        raise ValueError("which_star must be 'S1' or 'S2'")
    
    names = (f'{which_star}_surface_h1',
            f'{which_star}_surface_he4',
            f'{which_star}_mass',
            f'{which_star}_log_R',
            f'{which_star}_log_L',
            f'{which_star}_lg_mdot',
            'CC1' if which_star == 'S1' else 'CC2')
    
    return names
        

def filter_WR_phases(binary, weight, which_star):
    
    surface_h_column, surface_he_column, mass_column, radius_column, log_L_column, mdot_column, final_event = set_star_parameters(which_star)
    
    Teff_column = calc_Teff(binary[log_L_column].values, binary[radius_column].values)
    
    delta_t = binary['time'].diff().fillna(0)
    
    optical_depth_array = optical_depth_calculation(binary[mass_column],
                                                    10**binary[radius_column],
                                                    10**binary[log_L_column],
                                                    10**binary[mdot_column],
                                                    binary[surface_h_column],
                                                    binary[surface_he_column],
                                                    10**Teff_column)
    
    # Basically no hydrogen limit!
    hydrogen_poor_mask = (binary[surface_h_column] <= 0.7)
    mask_final_event = binary['event'] != final_event
    
    stripped_phase = binary[mask_final_event & hydrogen_poor_mask].copy()
    if len(stripped_phase) == 0:
        return None

    stripped_phase['weight'] = delta_t[mask_final_event & hydrogen_poor_mask].values
    stripped_phase['model_weight'] = np.full(len(stripped_phase), weight.to_numpy())
    stripped_phase['optical_depth'] = optical_depth_array[mask_final_event & hydrogen_poor_mask]

    return stripped_phase



for metallicity in metallicities:
    print(f"Processing metallicity Z={metallicity} Zsun")

    pop = TransientPopulation(str(population_file(metallicity)), 'all')
    model_weights = pop.model_weights('extended_IMF')
    
    # extract relevant columns
    tmp_hist = pop.history[['time', 'state', 'event', 'step_names',
                            'S1_log_L', 'S1_mass', 'S2_mass', 'orbital_period',
                            'S1_state', 'S2_state', 'S2_log_L', 'S1_log_R',
                            'S2_log_R', 'S1_surface_h1', 'S1_surface_he4',
                            'S2_surface_h1', 'S2_surface_he4', 'S1_lg_mdot', 'S2_lg_mdot',]]
    
    S1_WR_data, S2_WR_data = get_WR_phases(tmp_hist,
                                           model_weights,
                                           metallicity)
    
    OPTICAL_DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(WR_phases_file(metallicity), 'wb') as f:
        pickle.dump({'S1_WR_data': S1_WR_data, 'S2_WR_data': S2_WR_data}, f)
    print(f"Saved WR phases for Z={metallicity} Zsun")
    