from posydon.utils.common_functions import convert_metallicity_to_string, orbital_separation_from_period
import matplotlib.pyplot as plt
import numpy as np


import pickle
from pathlib import Path

data_folder = Path('/Users/max/Documents/projects/WR_population/data/')
for metallicity in [0.1, 0.2, 0.45, 1.0]:
    print(metallicity)
    met_str = convert_metallicity_to_string(metallicity)
    file_path = data_folder / f'WR_phases_Z{met_str}_Zsun.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    S1_WR_data = data['S1_WR_data']
    S2_WR_data = data['S2_WR_data']

    time_array = []
    weight_array = []
    state_array = []
    key_array = []
    period_array = []
    separation_array = []

    for key in S1_WR_data:
        data = S1_WR_data[key]['data']
        weights = S1_WR_data[key]['weights']
        
        time_array.extend(data['time'].values)
        weight_array.extend(weights.values)
        state_array.extend(data['state'].values)
        key_array.extend([key] * len(data))
        period_array.extend(data['orbital_period'].values)
        sep = orbital_separation_from_period(data['orbital_period'].values,
                                            data['S1_mass'].values,
                                            data['S2_mass'].values)
        
        separation_array.extend(sep)
        
    for key in S2_WR_data:
        data = S2_WR_data[key]['data']
        weights = S2_WR_data[key]['weights']
        
        time_array.extend(data['time'].values)
        weight_array.extend(weights.values)
        state_array.extend(data['state'].values)
        key_array.extend([key] * len(data))
        period_array.extend(data['orbital_period'].values)
        sep = orbital_separation_from_period(data['orbital_period'].values,
                                            data['S1_mass'].values,
                                            data['S2_mass'].values)
        separation_array.extend(sep)
        
    time_array = np.array(time_array)
    weight_array = np.array(weight_array)
    state_array = np.array(state_array)
    key_array = np.array(key_array)
    period_array = np.array(period_array)
    separation_array = np.array(separation_array)

    # only 3 keys: 'detached', 'merged', 'disrupted'

    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | (state_array == 'contact') | (state_array == 'RLO2')
    single_star_mask = (state_array == 'disrupted') | (state_array == 'merged')

    binary_fraction = np.sum(weight_array[binary_mask]) / np.sum(weight_array[single_star_mask | binary_mask])

    # Analyze binary fraction among WR phases
    # as a function of orbital period
    # plot period against, cumulative weight fraction

    sorted_indices = np.argsort(separation_array[binary_mask])
    sorted_periods = separation_array[binary_mask][sorted_indices]
    sorted_weights = weight_array[binary_mask][sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= np.sum(weight_array)  # normalize
    rsun2au = 0.004650476

    plt.plot(sorted_periods * rsun2au, cumulative_weights, label=f"{metallicity:.2f} Z$_\odot$")


plt.axvline(35, color='black', linestyle='-', label='35 au')

plt.xlabel('Separation [au]')
plt.ylabel('Cumulative Companion Fraction')
plt.xscale('log')
plt.legend()
plt.savefig('/Users/max/Documents/projects/WR_population/analysis/figures/png/separation_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/max/Documents/projects/WR_population/analysis/figures/pdf/separation_figure.pdf')
