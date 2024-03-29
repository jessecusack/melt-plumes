from textwrap import dedent
from typing import Callable
from typing import Tuple
from typing import Union

import numpy as np
import seawater as sw
from scipy.integrate import solve_ivp

from . import bpt


def initialize_fluxes(
    z: float,
    Q: float,
    T: Union[float, Callable[[float], float]] = 5.0,
    S: Union[float, Callable[[float], float]] = 25.0,
    W: float = 100.0,
    α: float = 0.1,
    Sp: float = 1e-8,
    ρ_0: float = 1025.0,
    g: float = -9.81,
    lat: float = 60.0,
) -> Tuple[float]:
    """Calculate initial conditions for plume mass, momentum, temperature and salinity fluxes
    for given ocean parameters.

    Parameters
    ----------
        z : float
            Plume starting height (m), MUST BE A NEGATIVE NUMBER.
        Q : float
            Plume discharge rate (m3 s-1).
        T : callable or float, optional
            Ocean (far-field) in-situ temperature profile as a function of height, T(z) (C),
            or a constant value. Default 5.
        S : callable or float, optional
            Ocean (far-field) practical salinity profile as a function of height, S(z) (PSU),
            or a constant value. Default 25.
        W : float, optional
            Plume width (m), default 100.
        α : float, optional
            Entrainment coefficient (m s-1), default.
        Sp : float, optional
            Plume initial salinity (PSU), default 1e-8.
        ρ_0 : float, optional
            Density constant (kg m-3), default 1025.0.
        g : float, optional
            Gravitational constant (m s-2), default -9.81.
        lat : float, optional
            Latitude used in the pressure calculation, default 60 (degree_north).

    Returns
    -------
        mass_flux : float
        mom_flux : float
        T_flux : float
        S_flux : float
    """

    if callable(T) and callable(S):
        T_o = T(z)
        S_o = S(z)
    else:
        T_o = T
        S_o = S

    Qm2s = Q / W  # Discharge rate per m (m2 s-1)
    pi = sw.pres(-z, lat)  # Pressure (dbar)
    Tp = sw.fp(Sp, pi)  # Plume temperature is the freezing point (C)
    ρi = sw.pden(S_o, T_o, pi, pr=0)  # Far-field ocean density (kg m-3)
    ρ_oi = sw.pden(Sp, Tp, pi, pr=0)  # Plume density (kg m-3)
    gpi = g * (ρ_oi - ρi) / ρ_0  # Reduced gravity (m s-2)
    # Vertical velocity (m s-1), from a balance of momentum and buoyancy
    # for a line plume
    wi = (gpi * Qm2s / α) ** (1 / 3)
    Ri = Qm2s / wi  # Radius or thickness (m)

    mass_flux = Qm2s  # Mass flux (m2 s-1)
    mom_flux = Qm2s * wi  # Momentum flux (m3 s-2)
    T_flux = Qm2s * Tp  # Temperature flux (C m2 s-1)
    S_flux = Qm2s * Sp  # Salt flux (PSU m2 s-1)

    return mass_flux, mom_flux, T_flux, S_flux


def convert_fluxes(mass_flux, mom_flux, T_flux, S_flux) -> Tuple[float]:
    """Convert flux variables to non-flux variable.

    Parameters
    ----------
        mass_flux : float or array_like
            Plume mass flux per unit width (m2 s-1).
        mom_flux : float or array_like
            Plume momentum flux (m3 s-2).
        T_flux : float or array_like
            Plume temperature flux (C m2 s-1).
        S_flux : float or array_like
            Plume salt flux (PSU m2 s-1).

    Returns
    -------
        w : float or array_like
            Plume vertical velocity (m s-1).
        R : float or array_like
            Plume thickness (m).
        T : float or array_like
            Plume temperature (C).
        S : float or array_like
            Plume salinity (PSU).
    """
    w = mom_flux / mass_flux  # Plume vertical velocity (m s-1)
    R = mass_flux / w  # Plume thickness or radius (m)
    T = T_flux / mass_flux  # Plume temperature (C)
    S = S_flux / mass_flux  # Plume salinity (PSU)
    return w, R, T, S


def plume(
    z: float,
    Q: float,
    dz: float,
    z_max: float = -1.0,
    T: Union[float, Callable[[float], float]] = 5.0,
    S: Union[float, Callable[[float], float]] = 25.0,
    u: Union[float, Callable[[float], float]] = 0.0,
    W: float = 100.0,
    α: float = 0.1,
    Sp: float = 1e-8,
    ρ_0: float = 1025.0,
    g: float = -9.81,
    lat: float = 60.0,
) -> Tuple[np.array]:
    """Solve the buoyant plume theory equations.

    Parameters
    ----------
        z : float
            Plume starting height (m), MUST BE A NEGATIVE NUMBER.
        Q : float
            Plume discharge rate (m3 s-1).
        dz : float
            Output spacing (m).
        z_max : float
            Maximum plume height (m).
        T : callable or float, optional
            Ocean (far-field) in-situ temperature profile as a function
            of height, T(z) (C), or a constant value. Default 5.
        S : callable or float, optional
            Ocean (far-field) practical salinity profile as a function
            of height, S(z) (PSU), or a constant value. Default 25.
        u : callable or float
            Additional horizontal ocean velocity as a function of height,
            u(z) (m s-1), or a constant value. Default 0.
        W : float, optional
            Plume width (m), default 100.
        α : float, optional
            Entrainment coefficient (m s-1), default.
        Sp : float, optional
            Plume initial salinity (PSU), default 1e-8.
        ρ_0 : float, optional
            Density constant (kg m-3), default 1025.0.
        g : float, optional
            Gravitational constant (m s-2), default -9.81.
        lat : float, optional
            Latitude used in the pressure calculation, default 60 (degree_north).

    Returns
    -------
        w : numpy.array
            Plume vertical velocity (m s-1).
        R : numpy.array
            Plume thickness (m).
        T_o : numpy.array
            Plume temperature (C).
        S_o : numpy.array
            Plume salinity (PSU).
        z_out : numpy.array
            Evaluated heights (m).
    """
    fluxes0 = initialize_fluxes(z, Q, T, S, W, α, Sp, ρ_0, g, lat)

    args = (T, S, u, α, ρ_0, g, lat)

    z_eval = np.arange(np.ceil(z / dz) * dz, z_max - dz / 2, dz)

    result = solve_ivp(
        bpt.bpt,
        [z, z_max],
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
    z_out = result.t

    w, R, T_o, S_o = convert_fluxes(mass_flux, mom_flux, T_flux, S_flux)

    return w, R, T_o, S_o, z_out


def plume_chain(
    z: float,
    Q: float,
    dz: float,
    n_plumes_max: int = 1000,
    z_max: float = -1.0,
    T: Union[float, Callable[[float], float]] = 5.0,
    S: Union[float, Callable[[float], float]] = 25.0,
    u: Union[float, Callable[[float], float]] = 0.0,
    W: float = 100.0,
    α: float = 0.1,
    Sp: float = 1e-8,
    ρ_0: float = 1025.0,
    g: float = -9.81,
    lat: float = 60.0,
) -> Tuple[np.array]:
    """Solve the buoyant plume theory equations.

    Parameters
    ----------
        z : float
            Plume starting height (m), MUST BE A NEGATIVE NUMBER.
        Q : float
            Plume discharge rate (m3 s-1).
        dz : float
            Output spacing (m).
        n_plumes_max : int, optional
            Maximum number of plumes. Default 1000.
        z_max : float
            Maximum plume height (m).
        T : callable or float, optional
            Ocean (far-field) in-situ temperature profile as a function
            of height, T(z) (C), or a constant value. Default 5.
        S : callable or float, optional
            Ocean (far-field) practical salinity profile as a function
            of height, S(z) (PSU), or a constant value. Default 25.
        u : callable or float
            Additional horizontal ocean velocity as a function of height,
            u(z) (m s-1), or a constant value. Default 0.
        W : float, optional
            Plume width (m), default 100.
        α : float, optional
            Entrainment coefficient (m s-1), default.
        Sp : float, optional
            Plume initial salinity (PSU), default 1e-8.
        ρ_0 : float, optional
            Density constant (kg m-3), default 1025.0.
        g : float, optional
            Gravitational constant (m s-2), default -9.81.
        lat : float, optional
            Latitude used in the pressure calculation, default 60 (degree_north).

    Returns
    -------
        w : numpy.array
            Plume vertical velocity (m s-1).
        R : numpy.array
            Plume thickness (m).
        T_o : numpy.array
            Plume temperature (C).
        S_o : numpy.array
            Plume salinity (PSU).
        z_out : numpy.array
            Evaluated heights (m).
        idx : numpy.array
            Plume counter. idx == 1 is first plume, idx == 2 is second, etc.
    """
    fluxes0 = initialize_fluxes(z, Q, T, S, W, α, Sp, ρ_0, g, lat)
    args = (T, S, u, α, ρ_0, g, lat)
    z_eval = np.arange(np.ceil(z / dz) * dz, z_max - dz / 2, dz)

    mass_flux = np.array([])
    mom_flux = np.array([])
    T_flux = np.array([])
    S_flux = np.array([])
    z_out = np.array([])
    idx = np.array([])

    for n in range(1, n_plumes_max + 1):
        result = solve_ivp(
            bpt.bpt,
            [z, z_max],
            fluxes0,
            method="RK45",
            t_eval=z_eval,
            args=args,
            events=bpt.Δρ,
        )

        mass_flux = np.hstack((mass_flux, result.y[0, :]))
        mom_flux = np.hstack((mom_flux, result.y[1, :]))
        T_flux = np.hstack((T_flux, result.y[2, :]))
        S_flux = np.hstack((S_flux, result.y[3, :]))
        z_out = np.hstack((z_out, result.t))
        idx = np.hstack((idx, np.full_like(result.t, n)))

        if np.isclose(result.t[-1], z_max - dz):
            text = f"""
            Number of plumes = {n}
            Starting height of last plume = {z:1.2f}  m
            """
            print(dedent(text))

            break

        z = result.t_events[0][0]  # Final plume height
        fluxes0 = initialize_fluxes(z, Q, T, S, W, α, Sp, ρ_0, g, lat)
        z_eval = np.arange(np.ceil(z / dz) * dz, z_max - dz / 2, dz)

    w, R, T_o, S_o = convert_fluxes(mass_flux, mom_flux, T_flux, S_flux)

    return w, R, T_o, S_o, z_out, idx
