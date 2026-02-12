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