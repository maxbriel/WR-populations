'''
This notebooks extracts the WR phases from the posydon population runs and
stores it in a pickle file for the analysis notebooks.
'''

from posydon.utils.common_functions import convert_metallicity_to_string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from posydon.config import PATH_TO_POSYDON
from tqdm import tqdm
from posydon.popsyn.synthetic_population import Population
import pickle

data_folder = Path('/Users/max/Documents/projects/WR_population/population_synthesis/runs/2025_11_04/')

output_folder = Path('/Users/max/Documents/projects/WR_population/data/')

# metallicities to process
metallicities = [0.01, 0.1, 0.2, 0.45, 1]

def get_WR_phases(history, metallicity):
    '''Get the WR phases for both stars in the binary history.'''
    luminosity_threshold = 4.9 - np.log10(metallicity)
    surface_threshold = 0.4

    S1_WR_data = {}
    S2_WR_data = {}

    # unique binaries
    nr_systems = np.unique(history.index).size
    

    for i in tqdm(range(nr_systems), total=nr_systems):
        binary = history.loc[i]
        out = filter_WR_phases(binary,
                               surface_threshold=surface_threshold,
                               luminosity_threshold=luminosity_threshold,
                               which_star='S1')
        if out:
            S1_WR_data[i] = out
            
        out = filter_WR_phases(binary,
                               surface_threshold=surface_threshold,
                               luminosity_threshold=luminosity_threshold,
                               which_star='S2')
        if out:
            S2_WR_data[i] = out
        
    return S1_WR_data, S2_WR_data
    
    
def set_star_parameters(which_star):
    """Set the column names and final event based on requested star."""
    if which_star == 'S1':
        surface_h_column = 'S1_surface_h1'
        log_L_column = 'S1_log_L'
        final_event = 'CC1'
    elif which_star == 'S2':
        surface_h_column = 'S2_surface_h1'
        log_L_column = 'S2_log_L'
        final_event = 'CC2'
    else:
        raise ValueError("which_star must be 'S1' or 'S2'")
    return surface_h_column, log_L_column, final_event
        

def filter_WR_phases(binary,
                     surface_threshold,
                     luminosity_threshold,
                     which_star):
    
    surface_h_column, log_L_column, final_event = set_star_parameters(which_star)
    
    delta_t = binary['time'].diff().fillna(0)
    mask_h1 = binary[surface_h_column] < surface_threshold
    mask_L = binary[log_L_column] > luminosity_threshold
    mask_final_event = binary['event'] != final_event
    
    stripped_phase = binary[mask_h1 & mask_final_event & mask_L]
    if len(stripped_phase) == 0:
        return {}
    weights = delta_t[mask_h1 & mask_final_event & mask_L]
    duration = stripped_phase['time'].iloc[-1] - stripped_phase['time'].iloc[0]
    
    out_dict = {
        'duration': duration,
        'data': stripped_phase,
        'weights': weights
    }
    
    return out_dict



for metallicity in metallicities:
    print(f"Processing metallicity Z={metallicity} Zsun")

    met_str = convert_metallicity_to_string(metallicity)
    metallicity_file = data_folder / f"{met_str}_Zsun_population.h5"

    pop = Population(str(metallicity_file))
    
    # extract relevant columns
    tmp_hist = pop.history[['time', 'state', 'event', 'step_names', 'S1_log_L', 'S1_mass', 'S2_mass', 'orbital_period', 'S1_state', 'S2_state', 'S2_log_L', 'S1_log_R', 'S2_log_R', 'S1_surface_h1', 'S1_surface_he4', 'S2_surface_h1', 'S2_surface_he4']]
    
    S1_WR_data, S2_WR_data = get_WR_phases(tmp_hist, metallicity)
    
    with open(output_folder / f'WR_phases_Z{met_str}_Zsun.pkl', 'wb') as f:
        pickle.dump({'S1_WR_data': S1_WR_data, 'S2_WR_data': S2_WR_data}, f)
    print(f"Saved WR phases for Z={metallicity} Zsun")
    