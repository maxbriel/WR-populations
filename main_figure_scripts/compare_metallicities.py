"""
Script to create comparison plots across different metallicities.

This script loads pre-processed data for all metallicities and creates
comparison figures showing trends with metallicity.
"""

from posydon.utils.common_functions import convert_metallicity_to_string
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Custom helper scripts
from helper_functions import calc_Teff


def load_all_metallicities():
    """Load WR data for all available metallicities."""
    data_folder = Path('/Users/max/Documents/projects/WR_population/data/optical_depth/')
    metallicities = [0.01, 0.1, 0.2, 0.45, 1.0]
    
    all_data = {}
    
    for met in metallicities:
        met_str = convert_metallicity_to_string(met)
        file_path = data_folder / f'WR_phases_Z{met_str}_Zsun.pkl'
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            all_data[met] = data
            print(f"Loaded Z = {met_str} Zsun")
        else:
            print(f"Warning: File not found for Z = {met_str}")
    
    return all_data


def compute_binary_fractions(S1_WR_data, S2_WR_data):
    """Compute binary fractions for a given metallicity."""
    weight_array = []
    state_array = []
    WR_types = []
    
    # Process S1 data
    for key in S1_WR_data:
        data = S1_WR_data[key]['data']
        weights = S1_WR_data[key]['weights']
        optical_depth = S1_WR_data[key]['optical_depth']
        
        surface_H = data['S1_surface_h1']
        surface_He = data['S1_surface_he4']
        
        # WR classification
        WN_mask = (surface_H <= 0.4) & (optical_depth >= 1.5)
        WC_mask = (surface_H <= 0.4) & (surface_He < 0.7) & (optical_depth >= 1.5)
        
        WR_types.extend(np.where(WC_mask, 'WC', np.where(WN_mask, 'WN', 'stripped-He')))
        weight_array.extend(weights)
        state_array.extend(data['state'].values)
    
    # Process S2 data
    for key in S2_WR_data:
        data = S2_WR_data[key]['data']
        weights = S2_WR_data[key]['weights']
        optical_depth = S2_WR_data[key]['optical_depth']
        
        surface_H = data['S2_surface_h1']
        surface_He = data['S2_surface_he4']
        
        # WR classification
        WN_mask = (surface_H <= 0.4) & (optical_depth >= 1.5)
        WC_mask = (surface_H <= 0.4) & (surface_He < 0.7) & (optical_depth >= 1.5)
        
        WR_types.extend(np.where(WC_mask, 'WC', np.where(WN_mask, 'WN', 'stripped-He')))
        weight_array.extend(weights)
        state_array.extend(data['state'].values)
    
    WR_types = np.array(WR_types)
    weight_array = np.array(weight_array)
    state_array = np.array(state_array)
    
    WN_mask = WR_types == 'WN'
    WC_mask = WR_types == 'WC'
    binary_mask = (state_array == 'detached') | (state_array == 'RLO1') | \
                  (state_array == 'contact') | (state_array == 'RLO2')
    single_star_mask = (state_array == 'disrupted') | (state_array == 'merged')
    
    # Compute fractions
    WR_binary_frac = np.sum(weight_array[binary_mask & (WN_mask | WC_mask)]) / \
                     np.sum(weight_array[(binary_mask | single_star_mask) & (WN_mask | WC_mask)])
    
    WN_binary_frac = np.sum(weight_array[binary_mask & WN_mask]) / \
                     np.sum(weight_array[(binary_mask | single_star_mask) & WN_mask])
    
    WC_binary_frac = np.sum(weight_array[binary_mask & WC_mask]) / \
                     np.sum(weight_array[(binary_mask | single_star_mask) & WC_mask])
    
    WC_WN_ratio = np.sum(weight_array[WC_mask]) / np.sum(weight_array[WN_mask])
    
    return {
        'WR': WR_binary_frac,
        'WN': WN_binary_frac,
        'WC': WC_binary_frac,
        'WC_WN_ratio': WC_WN_ratio
    }


def plot_binary_fraction_vs_metallicity(all_data, output_dir):
    """Plot binary fractions as a function of metallicity."""
    metallicities = sorted(all_data.keys())
    
    WR_fractions = []
    WN_fractions = []
    WC_fractions = []
    WC_WN_ratios = []
    
    print("\nComputing binary fractions for each metallicity...")
    for met in tqdm(metallicities):
        data = all_data[met]
        fracs = compute_binary_fractions(data['S1_WR_data'], data['S2_WR_data'])
        WR_fractions.append(fracs['WR'])
        WN_fractions.append(fracs['WN'])
        WC_fractions.append(fracs['WC'])
        WC_WN_ratios.append(fracs['WC_WN_ratio'])
    
    # Plot binary fractions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(metallicities, WR_fractions, 'o-', color='black', linewidth=2,
             markersize=8, label='Total WR')
    ax1.plot(metallicities, WN_fractions, 's-', color='blue', linewidth=2,
             markersize=8, label='WN')
    ax1.plot(metallicities, WC_fractions, '^-', color='red', linewidth=2,
             markersize=8, label='WC')
    
    ax1.set_xlabel('Metallicity [Z$_\odot$]', fontsize=12)
    ax1.set_ylabel('Binary Fraction', fontsize=12)
    ax1.set_title('Binary Fraction vs Metallicity', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot WC/WN ratio
    ax2.plot(metallicities, WC_WN_ratios, 'o-', color='purple',
             linewidth=2, markersize=8)
    ax2.set_xlabel('Metallicity [Z$_\odot$]', fontsize=12)
    ax2.set_ylabel('WC/WN Ratio', fontsize=12)
    ax2.set_title('WC/WN Ratio vs Metallicity', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'png' / 'binary_fraction_vs_metallicity.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / 'binary_fraction_vs_metallicity.pdf')
    plt.close()


def plot_HR_diagrams_comparison(all_data, output_dir):
    """Create a multi-panel HR diagram comparing all metallicities."""
    from cmap import Colormap
    from helper_functions import tout_L_ZAMS, tout_R_ZAMS
    
    metallicities = sorted(all_data.keys())
    n_met = len(metallicities)
    
    fig, axes = plt.subplots(1, n_met, figsize=(5*n_met, 4.5))
    if n_met == 1:
        axes = [axes]
    
    cm_grays = Colormap('colorbrewer:Greys')
    T_bins = np.linspace(4.0, 5.3, 51)
    L_bins = np.linspace(4.0, 6.5, 51)
    
    # ZAMS line (same for all)
    M_zams = np.logspace(0.7, 2.5, 100)
    L_zams = tout_L_ZAMS(M_zams)
    R_zams = tout_R_ZAMS(M_zams)
    log_Teff_zams = calc_Teff(np.log10(L_zams), np.log10(R_zams))
    log_L_zams = np.log10(L_zams)
    
    for idx, met in enumerate(metallicities):
        ax = axes[idx]
        data = all_data[met]
        met_str = convert_metallicity_to_string(met)
        
        # Collect data
        Teff_array = []
        L_array = []
        weight_array = []
        WR_types = []
        
        for key in data['S1_WR_data']:
            data_s1 = data['S1_WR_data'][key]['data']
            weights = data['S1_WR_data'][key]['weights']
            optical_depth = data['S1_WR_data'][key]['optical_depth']
            
            surface_H = data_s1['S1_surface_h1']
            surface_He = data_s1['S1_surface_he4']
            Teff = 10**calc_Teff(data_s1['S1_log_L'], data_s1['S1_log_R'])
            
            WN_mask = (surface_H <= 0.4) & (optical_depth >= 1.5)
            WC_mask = (surface_H <= 0.4) & (surface_He < 0.7) & (optical_depth >= 1.5)
            
            WR_types.extend(np.where(WC_mask, 'WC', np.where(WN_mask, 'WN', 'stripped-He')))
            Teff_array.extend(Teff)
            L_array.extend(10**data_s1['S1_log_L'].values)
            weight_array.extend(weights)
        
        for key in data['S2_WR_data']:
            data_s2 = data['S2_WR_data'][key]['data']
            weights = data['S2_WR_data'][key]['weights']
            optical_depth = data['S2_WR_data'][key]['optical_depth']
            
            surface_H = data_s2['S2_surface_h1']
            surface_He = data_s2['S2_surface_he4']
            Teff = 10**calc_Teff(data_s2['S2_log_L'], data_s2['S2_log_R'])
            
            WN_mask = (surface_H <= 0.4) & (optical_depth >= 1.5)
            WC_mask = (surface_H <= 0.4) & (surface_He < 0.7) & (optical_depth >= 1.5)
            
            WR_types.extend(np.where(WC_mask, 'WC', np.where(WN_mask, 'WN', 'stripped-He')))
            Teff_array.extend(Teff)
            L_array.extend(10**data_s2['S2_log_L'].values)
            weight_array.extend(weights)
        
        WR_types = np.array(WR_types)
        Teff_array = np.array(Teff_array)
        L_array = np.array(L_array)
        weight_array = np.array(weight_array)
        
        WN_mask = WR_types == 'WN'
        WC_mask = WR_types == 'WC'
        
        # Create histogram
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
        
        ax.imshow(H0.T, origin='lower',
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                 cmap=cm_grays.to_mpl(), aspect='auto', interpolation='bilinear')
        
        ax.contour(H0.T, levels=[level0_1], colors='black',
                  linewidths=[1.5],
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        # ZAMS
        ax.plot(log_Teff_zams, log_L_zams, 'g--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlim(5.25, 4.0)
        ax.set_ylim(4.2, 6.5)
        ax.set_xlabel('log T$_{\mathrm{eff}}$ [K]', fontsize=11)
        if idx == 0:
            ax.set_ylabel('log L [L$_\odot$]', fontsize=11)
        ax.set_title(f'Z = {met_str} Z$_\odot$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'png' / 'HR_diagrams_comparison.png', dpi=300)
    plt.savefig(output_dir / 'pdf' / 'HR_diagrams_comparison.pdf')
    plt.close()


def main():
    """Main function to create comparison plots."""
    print("\nLoading data for all metallicities...")
    all_data = load_all_metallicities()
    
    output_dir = Path('/Users/max/Documents/projects/WR_population/main_figure_scripts/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_png = output_dir / 'png'
    output_dir_pdf = output_dir / 'pdf'
    output_dir_png.mkdir(parents=True, exist_ok=True)
    output_dir_pdf.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating binary fraction comparison plot...")
    plot_binary_fraction_vs_metallicity(all_data, output_dir)
    
    print("Generating HR diagram comparison...")
    plot_HR_diagrams_comparison(all_data, output_dir)
    
    print("\n" + "="*60)
    print("All comparison figures generated successfully!")
    print(f"Figures saved to: {output_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
