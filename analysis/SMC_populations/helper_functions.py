import numpy as np

def tout_L_ZAMS(M):
    """Calculate ZAMS luminosity and Teff from Tout et al. (1996) for solar metallicity.
    
    M : np.array
        Mass in Msun
    Returns
    -------
    log_L : np.array
        Logarithmic luminosity in L_sun
    log_Teff : np.array
        Logarithmic effective temperature in K
    """
    alpha = 0.39704170
    beta = 8.52762600
    gamma = 0.00025546
    delta = 5.43288900
    epsilon = 5.56357900
    zeta = 0.78866060
    eta = 0.00586685
    return (alpha * M**5.5 + beta * M**11.0) / (gamma + M**3 + delta*M**5 + epsilon*M**7 + zeta*M**8 + eta*M**9.5)

def tout_R_ZAMS(M):
    """Calculate ZAMS radius from Tout et al. (1996) for solar metallicity.
    
    M : np.array
        Mass in Msun
    Returns
    -------
    log_R : np.array
        Logarithmic radius in R_sun
    """
    theta = 1.71535900
    iota = 6.59778800
    kappa = 10.08855000
    lambda_ = 1.01249500
    mu = 0.07490166
    nu = 0.01077422
    zeta = 3.08223400
    o = 17.8477800
    pi = 0.00022582
    
    return (theta*M**2.5 + iota*M**6.5 + kappa*M**11.0 + lambda_*M**19.0 + mu*M**19.5) / (nu + zeta*M**2 + o*M**8.5 + M**18.5 + pi*M**19.5)



def calc_Teff(L, R):
    """Calculate the effective temperature.
    
    L : np.array
        Logarithmic luminosity in L_sun
    R : np.array
        Logarithmic radius in R_sun
    Returns
    -------
    Teff : np.array
        Logarithmic effective temperature in K
    """
    L_vals = 10**L * 3.828e26  # L_sun to Watts
    R_vals = 10**R* 6.957e8   # R_sun to meters
    sigma = 5.67037442e-8
    Teff_vals = (L_vals / (4 * np.pi * R_vals**2 * sigma))**0.25
    
    return np.log10(Teff_vals)



from posydon.utils.common_functions import convert_metallicity_to_string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from posydon.config import PATH_TO_POSYDON
from tqdm import tqdm
from posydon.popsyn.synthetic_population import Population
import pickle




def optical_depth(Mdot, R, v_inf, kappa,  v0=20):
    '''
    Optical depth calculation following Langer 1989
    https://ui.adsabs.harvard.edu/abs/1989A%26A...210...93L/abstract
    
    tau = - (kappa * Mdot)/(4 * pi * R * (v_inf-v_0) ln (v_inf/v_0))
    
    where kappa is the opacity, Mdot the mass loss rate, v_inf the terminal wind velocity, and v0 the expansion velocity at the base of the wind.
    
    Parameters
    ----------
    Mdot : float or np.array
        Mass loss rate in Msun/yr
    R : float or np.array
        Stellar radius in Rsun
    v_inf : float or np.array
        Terminal wind velocity in km/s
    kappa : float or np.array
        Opacity in cm^2/g
    v0 : float, optional
        Expansion velocity at the base of the wind in km/s, by default 20 km/s
        
    Returns
    -------
    tau : float or np.array
        Optical depth
    '''
    
    Mdot_cgs = Mdot * 1.989e33 / (365.25*24*3600)  # convert Msun/yr to g/s
    R_cgs = R * 6.955e10  # convert Rsun to cm
    v_inf_cgs = v_inf * 1e5  # convert km/s to cm/s
    v0_cgs = v0 * 1e5  # convert km/s to cm/s
    
    tau = (kappa * Mdot_cgs) / (4 * np.pi * R_cgs * (v_inf_cgs - v0_cgs)) * np.log(v_inf_cgs / v0_cgs)
    
    return  tau
    
    
def escape_velocity(mass, radius, eddington_factor):
    '''
    The esscape velocity from the stellar surface.
    
    Parameters
    ----------
    mass : float or np.array
        Stellar mass in Msun
    radius : float or np.array
        Stellar radius in Rsun
    eddington_factor : float or np.array
        Eddington factor (dimensionless)
    
    Returns
    -------
    v_esc : float or np.array
        Escape velocity in km/s
    '''
    
    G_cgs = 6.67430e-8  # gravitational constant in cm^3 g^-1 s^-2
    mass_cgs = mass * 1.989e33  # convert Msun to g
    radius_cgs = radius * 6.955e10  # convert Rsun to cm
    
    v_esc = np.sqrt(2 * G_cgs * mass_cgs * (1 - eddington_factor) / radius_cgs)  # in cm/s
    v_esc_kms = v_esc / 1e5  # convert to km/s
    
    return v_esc_kms
    
    

def terminal_wind_velocity(v_esc, Teff, surface_H, surface_He):
    '''
       Terminal wind velocity as a function of the escape velocity.
    Sort of following Pauli+22, we set different factors based on the effective temperature and surface hydrogen content.
    
    WC stars (no helium envelope):              factor = 1.6
    (H-rich, H-poor, H-free) WN stars:          factor = 1.3
    Hot OB stars (27 500 K ≤ Teff ≤ 50 000 K):  factor = 2.6
    Cool OB stars (12 500 K ≤ Teff ≤ 22 500 K): factor = 1.3
    Supergiants (Teff < 10 000 K):              factor = 0.7

    
    Parameters
    ----------
    v_esc : float or np.array
        Escape velocity in km/s
    Teff : float or np.array
        Effective temperature in K
    surface_H : float or np.array
        Surface hydrogen mass fraction
    surface_He : float or np.array
        Surface helium mass fraction
        
    Returns
    -------
    v_inf : float or np.array
        Terminal wind velocity in km/s
    '''
    # Convert inputs to arrays for vectorization
    v_esc = np.atleast_1d(v_esc)
    Teff = np.atleast_1d(Teff)
    surface_H = np.atleast_1d(surface_H)
    surface_He = np.atleast_1d(surface_He)
    
    # Initialize factor array with default value
    factor = np.full_like(v_esc, 1.3, dtype=float)
    
    # H-rich stars (surface_H > 0.4)
    h_rich_mask = surface_H > 0.4
    
    # Hot OB stars
    hot_ob_mask = h_rich_mask & (Teff >= 27500) & (Teff <= 50000)
    factor[hot_ob_mask] = 2.6
    
    # Cool OB stars
    cool_ob_mask = h_rich_mask & (Teff >= 12500) & (Teff <= 22500)
    factor[cool_ob_mask] = 1.3
    
    # Supergiants
    supergiant_mask = h_rich_mask & (Teff < 10000)
    factor[supergiant_mask] = 0.7
    
    # H-poor or H-free stars (surface_H <= 0.4)
    h_poor_mask = ~h_rich_mask
    
    # WC stars
    wc_mask = h_poor_mask & (surface_He <= 0.7) & (surface_H < 0.1)
    factor[wc_mask] = 1.6
    
    v_inf = factor * v_esc
    
    # Return scalar if input was scalar
    if v_inf.size == 1:
        return float(v_inf[0])
    
    return v_inf
    
def eddington_factor(kappa, luminosity, mass):
    '''
    
    Eddington factor calculation.
    
    Parameters
    ----------
    kappa : float or np.array
        Opacity in cm^2/g
    luminosity : float or np.array
        Stellar luminosity in Lsun
    mass : float or np.array
        Stellar mass in Msun
    
    
    Returns
    -------
    gamma : float or np.array
        Eddington factor (dimensionless)
    '''
    
    L_cgs = luminosity * 3.828e33  # convert Lsun to erg/s
    M_cgs = mass * 1.989e33  # convert Msun to g
    c_cgs = 2.99792458e10  # speed of light in cm/s
    G_cgs = 6.67430e-8  # gravitational constant in cm^3 g^-1 s^-2
    gamma = (kappa * L_cgs) / (4 * np.pi * G_cgs * M_cgs * c_cgs)
    
    return gamma
    
    
def optical_depth_calculation(M, R, L, Mdot, surface_H, surface_He, Teff):
    '''
    Optical depth calculation given stellar parameters.
    
    Assumptions:
    - expansion velocity at the base of the wind v_0 = 20 km/s
    - terminal wind velocity v_inf = 1.3 * v_esc (following Gräfener et al. 2011)
    
    Parameters
    ----------
    M : float or np.array
        Stellar mass in Msun
    R : float or np.array
        Stellar radius in Rsun
    L : float or np.array
        Stellar luminosity in Lsun
    Mdot : float or np.array
        Mass loss rate in Msun/yr
        
    Returns
    -------
    tau : float or np.array
        Optical depth
    '''
    # Assumptions:
    v_0 = 20  # km/s, expansion velocity at the base of the wind
    
    # Opacity
    kappa = 0.2 * (1 + surface_H)  # cm^2/g
    
    # Eddington factor
    gamma = eddington_factor(kappa, L, M)

    # Escape velocity
    v_esc = escape_velocity(M, R, gamma)
    
    # Terminal wind velocity
    # adaptive factor based on types
    v_inf = terminal_wind_velocity(v_esc, Teff, surface_H, surface_He)

    # Optical depth
    tau = optical_depth(Mdot, R, v_inf, kappa,  v_0)
    
    return tau


def classify_WR_type(surface_H, surface_He, optical_depth, H_threshold=0.0001):
    """Classify stellar objects as WN, WC, or stripped-He stars.
    
    Parameters
    ----------
    surface_H : np.array or pd.Series
        Surface hydrogen mass fraction
    surface_He : np.array or pd.Series
        Surface helium mass fraction
    optical_depth : np.array
        Optical depth of the stellar wind
    H_threshold : float, optional
        Hydrogen mass fraction threshold for WR classification (default: 0.0001)
    
    Returns
    -------
    WN_mask : np.array
        Boolean mask for WN stars
    WC_mask : np.array
        Boolean mask for WC stars
    WR_types : np.array
        String array with classifications: 'WN', 'WC', or 'stripped-He'
    """    
    WN_mask = (surface_H <= 0.4) & (optical_depth >= 1.5)
    WC_mask = (surface_H <= H_threshold) & (surface_He < 0.7) & (optical_depth >= 1.5)
    
    WR_types = np.where(WC_mask, 'WC', np.where(WN_mask, 'WN', 'stripped-He'))
    
    return WN_mask, WC_mask, WR_types