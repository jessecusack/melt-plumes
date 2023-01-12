"""Test cases for the bpt module."""

import numpy as np
import scipy.integrate as itgr
import scipy.interpolate as itpl
import seawater as sw

from melt_plumes import bpt
from melt_plumes import helpers


def test_linearized_freezing_point_equation() -> None:
    pass


def test_plume_homogeneous_ocean_without_melt() -> None:
    """"""

    d = np.arange(1.0, 201.0, 1.0)
    T_o = 15.0
    S_o = 25.0
    # Initial conditions
    W = 100  # Channel width (m)
    Q = 0.001  # Discharge rate (m3 s-1)
    α = 0.1  # Entrainment coefficient
    lat = 60.0  # Latitude
    zi = -165  # Height (m)
    ρ_0 = 1025  # Density constant (kg m-3)
    g = -9.81  # Gravity (m s-2)
    Si = 0.0000001  # Need a small initial salinity apparently!

    # Derived initial conditions
    Qm2s = Q / W  # Discharge rate per m (m2 s-1)
    pi = sw.pres(-zi, lat)  # Pressure (dbar)
    Ti = sw.fp(Si, pi)  # Temperature is the freezing point (C)
    ρi = sw.pden(S_o, T_o, pi, pr=0)  # Far-field ocean density (kg m-3)
    ρ_oi = sw.pden(Si, Ti, pi, pr=0)  # Plume density (kg m-3)
    gpi = g * (ρ_oi - ρi) / ρ_0  # Reduced gravity (m s-2)
    # Vertical velocity (m s-1), from balance of momentum and buoyancy for a line plume
    wi = (gpi * Qm2s / α) ** (1 / 3)
    Ri = Qm2s / wi  # Radius or thickness (m)

    mass_flux = Qm2s  # Mass flux (m2 s-1)
    mom_flux = Qm2s * wi  # Momentum flux (m3 s-2)
    T_flux = Qm2s * Ti  # Temperature flux (C m2 s-1)
    S_flux = Qm2s * Si  # Salt flux (PSU m2 s-1)

    fluxes0 = [mass_flux, mom_flux, T_flux, S_flux]
    args = (T_o, S_o, α, ρ_0, g, lat)

    z_eval = np.arange(zi, -d[0], 1)

    result = itgr.solve_ivp(
        bpt.bpt,
        [zi, -d[0]],
        fluxes0,
        method="RK45",
        t_eval=z_eval,
        args=args,
        events=bpt.Δρ,
    )

    mass_flux = result.y[0, :]
    mom_flux = result.y[1, :]
    T_flux = result.y[2, :]
    S_flux = result.y[3, :]

    w = mom_flux / mass_flux  # Plume vertical velocity (m s-1)
    R = mass_flux / w  # Plume thickness or radius (m)
    T_o = T_flux / mass_flux  # Plume temperature (C)
    S_o = S_flux / mass_flux  # Plume salinity (PSU)


def test_plume_stratified_ocean_without_melt() -> None:
    """"""

    d, S, T = helpers.load_LeConte_CTD()

    Sz = itpl.interp1d(-d, S)
    Tz = itpl.interp1d(-d, T)

    # Initial conditions
    W = 100  # Channel width (m)
    Q = 10.0  # Discharge rate (m3 s-1)
    α = 0.1  # Entrainment coefficient
    lat = 60.0  # Latitude
    zi = -165  # Height (m)
    ρ_0 = 1025  # Density constant (kg m-3)
    g = -9.81  # Gravity (m s-2)
    Si = 0.0000001  # Need a small initial salinity apparently!

    # Derived
    Qm2s = Q/W  # Discharge rate per m (m2 s-1)
    pi = sw.pres(-zi, lat)  # Pressure (dbar)
    Ti = sw.fp(Si, pi)  # Temperature is the freezing point (C)
    ρi = sw.pden(Sz(zi), Tz(zi), pi, pr=0)  # Far-field ocean density (kg m-3)
    ρ_oi = sw.pden(Si, Ti, pi, pr=0)  # Plume density (kg m-3)
    gpi = g * (ρ_oi - ρi) / ρ_0  # Reduced gravity (m s-2)
    wi = (gpi * Qm2s / α)**(1/3)  # Vertical velocity (m s-1), from balance of momentum and buoyancy for a line plume
    Ri = Qm2s / wi  # Radius or thickness (m)

    mass_flux = Qm2s  # Mass flux (m2 s-1)
    mom_flux = Qm2s * wi  # Momentum flux (m3 s-2)
    T_flux = Qm2s * Ti  # Temperature flux (C m2 s-1)
    S_flux = Qm2s * Si  # Salt flux (PSU m2 s-1)

    fluxes0 = [mass_flux, mom_flux, T_flux, S_flux]
    args = (Tz, Sz, α, ρ_0, g, lat)

    z_eval = np.arange(zi, -d[0], 1)

    result = itgr.solve_ivp(
        bpt.bpt,
        [zi, -d[0]],
        fluxes0,
        method="RK45",
        t_eval=z_eval,
        args=args,
        events=bpt.Δρ,
    )

    mass_flux = result.y[0, :]
    mom_flux = result.y[1, :]
    T_flux = result.y[2, :]
    S_flux = result.y[3, :]

    w = mom_flux / mass_flux  # Plume vertical velocity (m s-1)
    R = mass_flux / w  # Plume thickness or radius (m)
    T_o = T_flux / mass_flux  # Plume temperature (C)
    S_o = S_flux / mass_flux  # Plume salinity (PSU)
    pass


def test_melt_rate() -> None:
    pass
