import numpy as np
import scipy.interpolate as itpl
import scipy.integrate as itgr
from seawater import pden, pres, fp


def melt_rate(d, T_o, S_o, u, Γ_T=0.022, Γ_S=0.00062, C_d=0.0025, T_i=-10.0, λ_1=-0.0573, λ_2=0.0832, λ_3=0.000761, L=335000.0, c_i=2009.0, c_w=3974.0):
    """The 3-equation melt rate parameterization (e.g. Holland and Jenkins 1999).
    
    Parameters
    ----------
        d : float
            Depth (m).
        T_o : float 
            Outer boundary layer temperature (C).
        S_o : float
            Outer boundary layer salinity (PSU).
        u : float
            Outer boundary layer speed (m s-1).
        Γ_T : float, optional
            Turbulent transfer coefficient for temperature, default 0.022.
        Γ_S : float, optional
            Turbulent transfer coefficient for salinity, default 0.00062.
        C_d : float, optional
            Drag coefficient, default 0.0025.
        T_i : float, optional
            Ice temperature, default -10.0 (C).
        λ_1 : float, optional
            Variation of freezing point with salinity in linear freezing point of seawater, default -0.0573 (C PSU-1).
        λ_2 : float, optional
            Freezing point offset in linear freezing point of seawater, default 0.0832 (C).
        λ_3 : float, optional
            Variation of freezing point with depth in linear freezing point of seawater, default 0.000761 (C m-1).
        L : float, optional
            Latent heat of fusion, default 335000.0 (J kg-1).
        c_i : float, optional
            Heat capacity of ice, default 2009.0 (J kg-1 C-1).
        c_w : float, optional
            heat capacity of seawater, default 3974.0 (J kg-1 C-1).
            
    Returns
    -------
        m : float
            Melt rate (m s-1).
        T_b : float
            Ice-ocean interface temperature (C).
        S_b : float
            Ice-ocean interface salinity (PSU).
    """
    
    # Linearized freezing point temperature
    T_b = λ_1 * S_o + λ_2 + λ_3*d  # Temperature of the ice-ocean interface (C)

    # Coefficients of the quadratic equation for interface salinity
    a = λ_1 * (Γ_T * c_w - Γ_S * c_i)
    b = Γ_S * c_i * (λ_1 * S_o - λ_2 - λ_3*d + T_i - (L/c_i)) - Γ_T * c_w * (T_o - λ_2 - λ_3*d)
    c = Γ_S * S_o * (c_i * (λ_2 + λ_3*d - T_i) + L)

    S_b = (-b - (b**2 - 4 * a * c)**0.5) / (2 * a)  # Salinity of the ice-ocean interface (PSU)
    m = Γ_S * C_d**0.5 * u * (S_o - S_b) / S_b  # Melt rate (m s-1)
    return m, T_b, S_b


def bpt(z, fluxes, T, S, α=0.1, ρ_0=1025.0, g=-9.81, lat=60.0, Γ_T=0.022, Γ_S=0.00062, C_d=0.0025, **melt_kwargs):
    """Differential equations of buoyant plume theory for a line plume with ice melt.
    
    Parameters
    ----------
        z : float
            Height or negative depth (m).
        fluxes : array_like of float
            Plume fluxes of [mass, momentum, temperature, salt].
        T : callable
            Ocean (far-field) in-situ temperature profile as a function of height, T(z) (C).
        S : callable
            Ocean (far-field) practical salinity profile as a function of height, S(z) (PSU).
        α : float, optional 
            Entrainment coefficient (m s-1).
        ρ_0 : float, optional
            Density constant, default 1025.0 (kg m-3).
        g : float, optional
            Gravitational constant, default -9.81 (m s-2).
        lat : float, optional
            Latitude used in the pressure calculation, default 60 (degree_north).
        Γ_T : float, optional
            Turbulent transfer coefficient for temperature, default 0.022 .
        Γ_S : float, optional
            Turbulent transfer coefficient for salinity, default 0.00062.
        C_d : float, optional
            Drag coefficient, default 0.0025.
        melt_kwargs : dict, optional
            Other optional parameters for the melt rate calculation.
            
    Returns
    -------
        dfluxes : list
            Vertical derivative of plume fluxes of [mass, momentum, temperature, salt]
    """
    
    mass_flux = fluxes[0]  # Mass flux (m2 s-1)
    mom_flux = fluxes[1]  # Momentum flux (m3 s-2)
    T_flux = fluxes[2]  # Temperature flux (C m2 s-1)
    S_flux = fluxes[3]  # Salt flux (PSU m2 s-1)
    
    w = mom_flux/mass_flux  # Plume vertical velocity (m s-1)
    R = mass_flux/w  # Plume thickness or radius (m)
    T_o = T_flux/mass_flux  # Plume temperature (C)
    S_o = S_flux/mass_flux  # Plume salinity (PSU)
    
    m, T_b, S_b = melt_rate(-z, T_o, S_o, w, Γ_T=Γ_T, Γ_S=Γ_S, C_d=C_d, **melt_kwargs)
    
    p = pres(-z, lat)  # Pressure (dbar)
    ρ_o = pden(S_o, T_o, p, pr=0)  # Plume potential density (kg m-3)
    ρ = pden(S(z), T(z), p, pr=0)  # Ocean (far-field) potential density (kg m-3)
    gp = g * (ρ_o - ρ) / ρ_0  # Reduced gravity (m s-2)
    
    dfluxes = [0, 0, 0, 0]
    # Mass
    dfluxes[0] = α * w + m
    # Momentum
    dfluxes[1] = R * gp - C_d * w**2
    # Temperature
    dfluxes[2] = α * w * T(z) + m * T_b - C_d**0.5 * w * Γ_T * (T_o - T_b)
    # Salinity
    dfluxes[3] = α * w * S(z) + m * S_b - C_d**0.5 * w * Γ_S * (S_o - S_b)
    return dfluxes


def Δρ(z, fluxes, T, S, α=0.1, ρ_0=1025.0, g=-9.81, lat=60.0, Γ_T=0.022, Γ_S=0.00062, C_d=0.0025, **melt_kwargs):
    T_o = fluxes[2]/fluxes[0]  # Plume temperature (C)
    S_o = fluxes[3]/fluxes[0]  # Plume salinity (PSU)
    p = pres(-z, lat)  # Pressure (dbar)
    ρ_o = pden(S_o, T_o, p, pr=0)  # Plume potential density (kg m-3)
    ρ = pden(S(z), T(z), p, pr=0)  # Ocean (far-field) potential density (kg m-3)
    return ρ - ρ_o


# Δρ.terminal = True
# Δρ.direction = -1