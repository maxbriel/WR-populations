from posydon.popsyn.synthetic_population import Population, TransientPopulation
from posydon.utils.common_functions import convert_metallicity_to_string
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

def trans_func(hist_chunk, oneline_chunk, form_chunk):
    
    df = pd.DataFrame()
    df['time'] = oneline_chunk['time_i']
    df['metallicity'] = oneline_chunk['metallicity']
    return df
    


metallicities = [0.01, 0.1, 0.2, 0.45, 1]
data_folder = Path('/Users/max/Documents/projects/WR_population/population_synthesis/runs/2026_08_14/')


for metallicity in metallicities:
    print(f"Processing metallicity Z={metallicity} Zsun")

    # open file
    met_str = convert_metallicity_to_string(metallicity)
    metallicity_file = data_folder / f"{met_str}_Zsun_population.h5"
    pop = Population(str(metallicity_file))
    # I want to calculate the model_weight for each binary in the population.
    # check if transient population already exists, if not create it
    try:
        print('Loading file')
        trans = TransientPopulation(str(metallicity_file), 'all')
    except ValueError:
        print('Failed to load transient population, creating it now...')
        trans = pop.create_transient_population(trans_func, 'all')

    population_properties = {
        'binary_fraction_scheme': 'const',
        'binary_fraction_const': 1.0,
        'primary_mass_scheme': 'Kroupa2001',
        'primary_mass_min': 0.01,
        'primary_mass_max': 150.0,
        'secondary_mass_scheme': 'flat_mass_ratio',
        'secondary_mass_min': 0.01,
        'secondary_mass_max': 150.0,
        'orbital_scheme': 'period',
        'orbital_period_scheme': 'power_law',
        'orbital_period_min': 0.75,
        'orbital_period_max': 6000.0,
        'power_law_slope': 0.0,
        'q_min': 0.0,
        'q_max': 1.0,
    }
    print('calculating model weights for each binary in the population...')

    trans.calculate_model_weights(population_parameters = population_properties,
                                  model_weights_identifier = 'extended_IMF')
    
    