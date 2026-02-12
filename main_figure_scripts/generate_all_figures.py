"""
Script to generate all WR population figures for all metallicities.

This script processes the optical depth data and creates comprehensive
figures analyzing WR populations across different metallicities.
"""

from posydon.utils.common_functions import convert_metallicity_to_string, orbital_separation_from_period
from posydon.popsyn.synthetic_population import Population
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmap import Colormap
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.stats import gaussian_kde

# Custom helper scripts
from helper_functions import calc_Teff, tout_L_ZAMS, tout_R_ZAMS, classify_WR_type


def load_WR_data(metallicity):
    """Load WR phase data for a given metallicity."""
    data_folder = Path('/Users/max/Documents/projects/WR_population/data/optical_depth/')
    met_str = convert_metallicity_to_string(metallicity)
    file_path = data_folder / f'WR_phases_Z{met_str}_Zsun.pkl'
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['S1_WR_data'], data['S2_WR_data'], met_str


def process_WR_data(S1_WR_data, S2_WR_data):
    """Process and combine S1 and S2 WR data into arrays."""
    WR_types = []
    time_array = []
    weight_array = []
    state_array = []
    key_array = []
    star_array = []
    period_array = []
    separation_array = []
    Teff_array = []
    L_array = []
    optical_depth_array = []
    surface_H_array = []
    surface_He_array = []
    mass_array = []
    radius_array = []
    Mdot_array = []
    companion_array = []

    # Process S1 data
    for key in tqdm(S1_WR_data, desc="Processing S1 data"):
        data = S1_WR_data[key]['data']
        weights = S1_WR_data[key]['weights']
        optical_depth = S1_WR_data[key]['optical_depth']
        
        surface_H = data['S1_surface_h1']
        surface_He = data['S1_surface_he4']
        Teff = 10**calc_Teff(data['S1_log_L'], data['S1_log_R'])
        
        # WR classification
        WN_mask, WC_mask, wr_types = classify_WR_type(surface_H, surface_He, optical_depth)
        
        WR_types.extend(wr_types)
        time_array.extend(data['time'].values)
        weight_array.extend(weights)
        state_array.extend(data['state'].values)
        key_array.extend([key] * len(data))
        star_array.extend(['S1'] * len(data))
        Teff_array.extend(Teff)
        L_array.extend(10**data['S1_log_L'].values)
        optical_depth_array.extend(optical_depth)
        surface_H_array.extend(surface_H.values)
        surface_He_array.extend(surface_He.values)
        mass_array.extend(data['S1_mass'].values)
        radius_array.extend(10**data['S1_log_R'].values)
        Mdot_array.extend(10**data['S1_lg_mdot'].values)
        companion_array.extend(data['S2_state'].values)
        period_array.extend(data['orbital_period'].values)
        
        sep = orbital_separation_from_period(data['orbital_period'].values,
                                            data['S1_mass'].values,
                                            data['S2_mass'].values)
        separation_array.extend(sep)
    
    # Process S2 data
    for key in tqdm(S2_WR_data, desc="Processing S2 data"):
        data = S2_WR_data[key]['data']
        weights = S2_WR_data[key]['weights']
        optical_depth = S2_WR_data[key]['optical_depth']
        
        surface_H = data['S2_surface_h1']
        surface_He = data['S2_surface_he4']
        Teff = 10**calc_Teff(data['S2_log_L'], data['S2_log_R'])
        
        # WR classification
        WN_mask, WC_mask, wr_types = classify_WR_type(surface_H, surface_He, optical_depth)
        
        WR_types.extend(wr_types)
        time_array.extend(data['time'].values)
        weight_array.extend(weights)
        state_array.extend(data['state'].values)
        key_array.extend([key] * len(data))
        star_array.extend(['S2'] * len(data))
        Teff_array.extend(Teff)
        L_array.extend(10**data['S2_log_L'].values)
        optical_depth_array.extend(optical_depth)
        surface_H_array.extend(surface_H.values)
        surface_He_array.extend(surface_He.values)
        mass_array.extend(data['S2_mass'].values)
        radius_array.extend(10**data['S2_log_R'].values)
        Mdot_array.extend(10**data['S2_lg_mdot'].values)
        companion_array.extend(data['S1_state'])
        period_array.extend(data['orbital_period'].values)
        
        sep = orbital_separation_from_period(data['orbital_period'].values,
                                            data['S2_mass'].values,
                                            data['S1_mass'].values)
        separation_array.extend(sep)
    
    # Convert to arrays
    return {
        'WR_types': np.array(WR_types),
        'time_array': np.array(time_array),
        'weight_array': np.array(weight_array),
        'state_array': np.array(state_array),
        'key_array': np.array(key_array),
        'star_array': np.array(star_array),
        'period_array': np.array(period_array),
        'separation_array': np.array(separation_array),
        'Teff_array': np.array(Teff_array),
        'L_array': np.array(L_array),
        'optical_depth_array': np.array(optical_depth_array),
        'surface_H_array': np.array(surface_H_array),
        'surface_He_array': np.array(surface_He_array),
        'mass_array': np.array(mass_array),
        'radius_array': np.array(radius_array),
        'Mdot_array': np.array(Mdot_array),
        'companion_array': np.array(companion_array)
    }


def plot_HR_diagram(arrays, met_str, output_dir):
    """Create HR diagram with WN and WC contours."""
    WR_types = arrays['WR_types']
    Teff_array = arrays['Teff_array']
    L_array = arrays['L_array']
    weight_array = arrays['weight_array']
    
    WN_mask = WR_types == 'WN'
    WC_mask = WR_types == 'WC'
    
    cm_grays = Colormap('colorbrewer:Greys')
    
    T_bins = np.linspace(4.0, 5.5, 61)
    L_bins = np.linspace(4.0, 7.0, 61)
    
    plt.figure(figsize=(8, 6))
    
    # Total WR histogram
    (H0, xedges, yedges) = np.histogram2d(
        np.log10(Teff_array)[WN_mask | WC_mask],
        np.log10(L_array)[WN_mask | WC_mask],
        weights=weight_array[WN_mask | WC_mask],
        bins=[T_bins, L_bins]
    )
    
    H0 /= (np.diff(T_bins)[:, None] * np.diff(L_bins))
    H0_sorted = np.sort(H0.flatten())[::-1]
    H0_cumsum = H0_sorted.cumsum()
    H0_total = H0_cumsum[-1]
    
    level0_1 = H0_sorted[np.searchsorted(H0_cumsum, 0.683 * H0_total)]
    level0_2 = H0_sorted[np.searchsorted(H0_cumsum, 0.954 * H0_total)]
    level0_3 = H0_sorted[np.searchsorted(H0_cumsum, 0.997 * H0_total)]
    levels0 = [level0_3, level0_2, level0_1]
    
    plt.imshow(H0.T, origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap=cm_grays.to_mpl(), aspect='auto', interpolation='bilinear')
    
    plt.contour(H0.T, levels=levels0, colors=cm_grays([0.2, 0.6, 1.0]),
                linewidths=[1.0, 1.0, 1.0],
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # WN contours
    H_WN, _, _ = np.histogram2d(
        np.log10(Teff_array)[WN_mask],
        np.log10(L_array)[WN_mask],
        weights=weight_array[WN_mask],
        bins=[T_bins, L_bins]
    )
    H_WN /= (np.diff(T_bins)[:, None] * np.diff(L_bins))
    H_WN_sorted = np.sort(H_WN.flatten())[::-1]
    H_WN_cumsum = H_WN_sorted.cumsum()
    H_WN_total = H_WN_cumsum[-1]
    level_WN_3 = H_WN_sorted[np.searchsorted(H_WN_cumsum, 0.683 * H_WN_total)]
    
    plt.contour(H_WN.T, levels=[level_WN_3], colors='blue',
                linewidths=[1.0], linestyles='dashed',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # WC contours
    H_WC, _, _ = np.histogram2d(
        np.log10(Teff_array)[WC_mask],
        np.log10(L_array)[WC_mask],
        weights=weight_array[WC_mask],
        bins=[T_bins, L_bins]
    )
    H_WC /= (np.diff(T_bins)[:, None] * np.diff(L_bins))
    H_WC_sorted = np.sort(H_WC.flatten())[::-1]
    H_WC_cumsum = H_WC_sorted.cumsum()
    H_WC_total = H_WC_cumsum[-1]
    level_WC_3 = H_WC_sorted[np.searchsorted(H_WC_cumsum, 0.683 * H_WC_total)]
    
    plt.contour(H_WC.T, levels=[level_WC_3], colors='red',
                linewidths=[1.0], linestyles='dashed',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # ZAMS
    M_zams = np.logspace(0.7, 2.5, 100)
    L_zams = tout_L_ZAMS(M_zams)
    R_zams = tout_R_ZAMS(M_zams)
    log_Teff_zams = calc_Teff(np.log10(L_zams), np.log10(R_zams))
    log_L_zams = np.log10(L_zams)
    
    plt.plot(log_Teff_zams, log_L_zams, 'g--', linewidth=1.5,
             label='ZAMS (Tout+96)', alpha=0.7)
    plt.plot([], [], 'b--', linewidth=1.5, label='$1\sigma$ WN contour', alpha=0.7)
    plt.plot([], [], 'r--', linewidth=1.5, label='$1\sigma$ WC contour', alpha=0.7)
    
    plt.xlim(5.25, 4.0)
    plt.ylim(4.2, 7.0)
    plt.xlabel('log Teff [K]')
    plt.ylabel('log L [L$_\odot$]')
    plt.legend(loc='upper right')
    plt.title(f'HR Diagram - Z = {met_str} Z$_\odot$')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'png' / f'HR_diagram_Z{met_str}.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / f'HR_diagram_Z{met_str}.pdf')
    plt.close()


def plot_cumulative_separation(arrays, met_str, output_dir):
    """Plot cumulative binary fraction over separation."""
    WR_types = arrays['WR_types']
    separation_array = arrays['separation_array']
    weight_array = arrays['weight_array']
    state_array = arrays['state_array']
    
    WN_mask = WR_types == 'WN'
    WC_mask = WR_types == 'WC'
    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | \
                  (state_array == 'contact') | (state_array == 'RLO2')
    
    rsun2au = 0.004650476
    
    plt.figure(figsize=(8, 6))
    
    # Total WR
    sorted_indices = np.argsort(separation_array[binary_mask & (WN_mask | WC_mask)])
    sorted_periods = separation_array[binary_mask & (WN_mask | WC_mask)][sorted_indices]
    sorted_weights = weight_array[binary_mask & (WN_mask | WC_mask)][sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= np.sum(weight_array[(WN_mask | WC_mask)])
    plt.plot(sorted_periods * rsun2au, cumulative_weights, color='black', label='Total WR')
    
    # WN
    sorted_indices = np.argsort(separation_array[binary_mask & WN_mask])
    sorted_periods = separation_array[binary_mask & WN_mask][sorted_indices]
    sorted_weights = weight_array[binary_mask & WN_mask][sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= np.sum(weight_array[WN_mask])
    plt.plot(sorted_periods * rsun2au, cumulative_weights, color='blue', label='WN')
    
    # WC
    sorted_indices = np.argsort(separation_array[binary_mask & WC_mask])
    sorted_periods = separation_array[binary_mask & WC_mask][sorted_indices]
    sorted_weights = weight_array[binary_mask & WC_mask][sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= np.sum(weight_array[WC_mask])
    plt.plot(sorted_periods * rsun2au, cumulative_weights, color='red', label='WC')
    
    plt.xscale('log')
    plt.xlabel('Separation [au]')
    plt.ylabel('Cumulative Companion Fraction')
    plt.legend()
    plt.title(f'Cumulative Companion Fraction - Z = {met_str} Z$_\odot$')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'png' / f'cumulative_separation_Z{met_str}.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / f'cumulative_separation_Z{met_str}.pdf')
    plt.close()


def plot_period_distributions(arrays, met_str, output_dir):
    """Plot period distributions for WN, WC, and total WR."""
    WR_types = arrays['WR_types']
    period_array = arrays['period_array']
    weight_array = arrays['weight_array']
    state_array = arrays['state_array']
    
    WN_mask = WR_types == 'WN'
    WC_mask = WR_types == 'WC'
    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | \
                  (state_array == 'contact') | (state_array == 'RLO2')
    
    plt.figure(figsize=(8, 6))
    
    # Total WR
    period_selection = period_array[(binary_mask) & (WN_mask | WC_mask)]
    kde = gaussian_kde(np.log10(period_selection),
                      weights=weight_array[(binary_mask) & (WN_mask | WC_mask)])
    x_eval = np.linspace(np.log10(np.min(period_selection)),
                        np.log10(np.max(period_selection)), 1000)
    kde_eval = kde(x_eval)
    plt.plot(10**x_eval, kde_eval, color='black', label='Total WR')
    
    # WN
    period_selection = period_array[(binary_mask) & (WN_mask)]
    if len(period_selection) > 0:
        kde = gaussian_kde(np.log10(period_selection),
                          weights=weight_array[(binary_mask) & (WN_mask)])
        x_eval = np.linspace(np.log10(np.min(period_selection)),
                            np.log10(np.max(period_selection)), 1000)
        kde_eval = kde(x_eval)
        plt.plot(10**x_eval, kde_eval, color='blue', label='WN')
    
    # WC
    period_selection = period_array[(binary_mask) & (WC_mask)]
    if len(period_selection) > 0:
        kde = gaussian_kde(np.log10(period_selection),
                          weights=weight_array[(binary_mask) & (WC_mask)])
        x_eval = np.linspace(np.log10(np.min(period_selection)),
                            np.log10(np.max(period_selection)), 1000)
        kde_eval = kde(x_eval)
        plt.plot(10**x_eval, kde_eval, color='red', label='WC')
    
    plt.xscale('log')
    plt.xlabel('Orbital Period [days]')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Period Distributions - Z = {met_str} Z$_\odot$')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'png' / f'period_distributions_Z{met_str}.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / f'period_distributions_Z{met_str}.pdf')
    plt.close()


def plot_binary_interactions(arrays, met_str, pop, output_dir):
    """Plot binary interactions period distributions."""
    WR_types = arrays['WR_types']
    period_array = arrays['period_array']
    weight_array = arrays['weight_array']
    state_array = arrays['state_array']
    key_array = arrays['key_array']
    star_array = arrays['star_array']
    
    WR_mask = (WR_types == 'WN') | (WR_types == 'WC')
    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | \
                  (state_array == 'contact') | (state_array == 'RLO2')
    
    # Get interaction classes
    HMS_HMS_interaction = pop.oneline['interp_class_HMS_HMS']
    CO_HMS_RLO_interaction = pop.oneline['interp_class_CO_HMS_RLO']
    
    # S1 indices
    S1_stable_indices = set(np.where(HMS_HMS_interaction == 'stable_MT')[0])
    S1_unstable_indices = set(np.where(HMS_HMS_interaction == 'unstable_MT')[0])
    S1_no_MT_indices = set(np.where((HMS_HMS_interaction == 'no_MT') | 
                                    pd.isna(HMS_HMS_interaction))[0])
    
    # S2 indices
    S2_stable_indices = set(np.where(CO_HMS_RLO_interaction == 'stable_MT')[0])
    S2_unstable_indices = set(np.where(CO_HMS_RLO_interaction == 'unstable_MT')[0])
    S2_no_MT_indices = set(np.where((CO_HMS_RLO_interaction == 'no_MT') | 
                                    pd.isna(CO_HMS_RLO_interaction))[0])
    
    # Create masks
    S1_entries = (star_array == 'S1')
    S2_entries = (star_array == 'S2')
    
    mask_stable = S1_entries & np.isin(key_array, np.array(list(S1_stable_indices)))
    mask_unstable = S1_entries & np.isin(key_array, np.array(list(S1_unstable_indices)))
    mask_no_MT = S1_entries & np.isin(key_array, np.array(list(S1_no_MT_indices)))
    
    S2_mask_stable = S2_entries & np.isin(key_array, np.array(list(S2_stable_indices)))
    S2_mask_unstable = S2_entries & np.isin(key_array, np.array(list(S2_unstable_indices)))
    S2_mask_no_MT = S2_entries & np.isin(key_array, np.array(list(S2_no_MT_indices)))
    
    # Common x_eval range
    period_selection_all = period_array[(binary_mask) & (WR_mask)]
    x_eval = np.linspace(np.log10(np.min(period_selection_all)),
                        np.log10(np.max(period_selection_all)), 1000)
    
    plt.figure(figsize=(10, 6))
    
    # Total WR distribution
    period_selection = period_array[(binary_mask) & (WR_mask)]
    kde = gaussian_kde(np.log10(period_selection),
                      weights=weight_array[(binary_mask) & (WR_mask)])
    kde_eval = kde(x_eval)
    plt.plot(10**x_eval, kde_eval, color='black', label='Total WR', linewidth=2)
    
    # Plot each interaction type
    interaction_configs = [
        (mask_stable, 'green', 'Stable MT (S1)'),
        (S2_mask_stable, 'blue', 'Stable MT (S2)'),
        (mask_unstable, 'orange', 'Unstable MT (S1)'),
        (S2_mask_unstable, 'red', 'Unstable MT (S2)'),
        (mask_no_MT, 'purple', 'No MT (S1)'),
        (S2_mask_no_MT, 'brown', 'No MT (S2)')
    ]
    
    for mask, color, label in interaction_configs:
        period_selection = period_array[(binary_mask) & (WR_mask) & (mask)]
        if len(period_selection) > 0:
            kde = gaussian_kde(np.log10(period_selection),
                              weights=weight_array[(binary_mask) & (WR_mask) & (mask)])
            kde_eval = kde(x_eval)
            fraction = np.sum(weight_array[(binary_mask) & (WR_mask) & (mask)]) / \
                      np.sum(weight_array[(binary_mask) & (WR_mask)])
            plt.plot(10**x_eval, kde_eval*fraction, color=color, label=label)
    
    plt.xscale('log')
    plt.xlabel('Orbital Period [days]')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Binary Interactions - Z = {met_str} Z$_\odot$')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'png' / f'binary_interactions_Z{met_str}.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / f'binary_interactions_Z{met_str}.pdf')
    plt.close()


def save_statistics(arrays, met_str, output_file):
    """Save key statistics to a text file."""
    WR_types = arrays['WR_types']
    weight_array = arrays['weight_array']
    state_array = arrays['state_array']
    companion_array = arrays['companion_array']
    
    WN_mask = WR_types == 'WN'
    WC_mask = WR_types == 'WC'
    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | \
                  (state_array == 'contact') | (state_array == 'RLO2')
    single_star_mask = (state_array == 'disrupted') | (state_array == 'merged')
    
    with open(output_file, 'w') as f:
        f.write(f"WR Population Statistics - Z = {met_str} Zsun\n")
        f.write("=" * 60 + "\n\n")
        
        # Binary fractions
        WR_binary_fraction = np.sum(weight_array[binary_mask & (WN_mask | WC_mask)]) / \
                            np.sum(weight_array[(binary_mask | single_star_mask) & (WN_mask | WC_mask)])
        f.write("Binary Fractions:\n")
        f.write(f"  WR binary fraction: {WR_binary_fraction:.4f}\n")
        f.write(f"  WN binary fraction: {np.sum(weight_array[binary_mask & WN_mask]) / np.sum(weight_array[(binary_mask | single_star_mask) & WN_mask]):.4f}\n")
        f.write(f"  WC binary fraction: {np.sum(weight_array[binary_mask & WC_mask]) / np.sum(weight_array[(binary_mask | single_star_mask) & WC_mask]):.4f}\n")
        f.write(f"  WC/WN ratio: {np.sum(weight_array[binary_mask & WC_mask]) / np.sum(weight_array[binary_mask & WN_mask]):.4f}\n\n")
        
        # Companion types
        f.write("Companion Fractions (for binaries only):\n")
        bh_mask = companion_array == 'BH'
        w = np.sum(weight_array[binary_mask & (WN_mask | WC_mask)])
        sel = np.sum(weight_array[((binary_mask & bh_mask) & (WN_mask | WC_mask))])
        f.write(f"  BH: {np.sum(weight_array[(binary_mask & bh_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        ns_mask = companion_array == 'NS'
        sel += np.sum(weight_array[((binary_mask & ns_mask) & (WN_mask | WC_mask))])
        f.write(f"  NS: {np.sum(weight_array[(binary_mask & ns_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        wd_mask = companion_array == 'WD'
        sel += np.sum(weight_array[((binary_mask & wd_mask) & (WN_mask | WC_mask))])
        f.write(f"  WD: {np.sum(weight_array[(binary_mask & wd_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        ss_mask = np.core.defchararray.find(companion_array, 'stripped') != -1
        sel += np.sum(weight_array[(binary_mask & ss_mask) & (WN_mask | WC_mask)])
        f.write(f"  Stripped: {np.sum(weight_array[(binary_mask & ss_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        ms_mask = companion_array == 'H-rich_Core_H_burning'
        sel += np.sum(weight_array[(binary_mask & ms_mask) & (WN_mask | WC_mask)])
        f.write(f"  MS: {np.sum(weight_array[(binary_mask & ms_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        hg_mask = companion_array == 'H-rich_Shell_H_burning'
        sel += np.sum(weight_array[(binary_mask & hg_mask) & (WN_mask | WC_mask)])
        f.write(f"  HG: {np.sum(weight_array[(binary_mask & hg_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")
        
        g_mask = (companion_array == 'H-rich_Core_He_burning') | \
                 (companion_array == 'H-rich_Shell_He_burning') | \
                 (companion_array == 'H-rich_He_depleted')
        sel += np.sum(weight_array[(binary_mask & g_mask) & (WN_mask | WC_mask)])
        f.write(f"  Giants: {np.sum(weight_array[(binary_mask & g_mask) & (WN_mask | WC_mask)]) / w:.4f}\n")

        # sum weights combined
        f.write(f"  Total (check): {sel / w:.4f}\n")
        f.write("unique companion types considered above.\n")
        f.write(str(np.unique(companion_array[binary_mask & (WN_mask | WC_mask)]).tolist()))
        f.write('\n')
        
        

def process_metallicity(metallicity):
    """Process a single metallicity and generate all figures."""
    print(f"\n{'='*60}")
    print(f"Processing Z = {metallicity} Zsun")
    print(f"{'='*60}\n")
    
    # Load data
    S1_WR_data, S2_WR_data, met_str = load_WR_data(metallicity)
    
    # Process arrays
    arrays = process_WR_data(S1_WR_data, S2_WR_data)
    
    # Load population
    population_run_folder = Path('/Users/max/Documents/projects/WR_population/population_synthesis/runs/2025_11_04/')
    population_file = population_run_folder / f"{met_str}_Zsun_population.h5"
    pop = Population(str(population_file))
    
    # Create output directories
    output_dir = Path('/Users/max/Documents/projects/WR_population/main_figure_scripts/figures/')
    output_dir_png = output_dir / 'png'
    output_dir_pdf = output_dir / 'pdf'    
    output_dir_png.mkdir(parents=True, exist_ok=True)
    output_dir_pdf.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("Generating HR diagram...")
    plot_HR_diagram(arrays, met_str, output_dir)
    
    print("Generating cumulative separation plot...")
    plot_cumulative_separation(arrays, met_str, output_dir)
    
    print("Generating period distributions...")
    plot_period_distributions(arrays, met_str, output_dir)
    
    print("Generating binary interactions plot...")
    plot_binary_interactions(arrays, met_str, pop, output_dir)
    
    # Save statistics
    print("Saving statistics...")
    stats_file = output_dir / f'statistics_Z{met_str}.txt'
    save_statistics(arrays, met_str, stats_file)
    
    print(f"Completed Z = {met_str} Zsun\n")


def main():
    """Main function to process all metallicities."""
    metallicities = [0.01, 0.1, 0.2, 0.45, 1.0]
    
    print("\nGenerating WR population figures for all metallicities")
    print("="*60)
    
    for metallicity in metallicities:
        try:
            process_metallicity(metallicity)
        except Exception as e:
            print(f"Error processing Z = {metallicity}: {e}")
            continue
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Figures saved to: /Users/max/Documents/projects/WR_population/main_figure_scripts/figures/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
