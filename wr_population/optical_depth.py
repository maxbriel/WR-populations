"""
Optical depth of the stellar wind for the WR population.
Similar to Pauli+22 (https://ui.adsabs.harvard.edu/abs/2022A&A...667A..58P)
"""
import numpy as np


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
    - terminal wind velocity v_inf = type-dependent factor * v_esc (see terminal_wind_velocity)
    
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
