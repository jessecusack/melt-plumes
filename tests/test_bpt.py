"""Test cases for the bpt module."""

import numpy as np
import scipy.integrate as itgr
import scipy.interpolate as itpl
import seawater as sw

from melt_plumes import bpt, helpers, solve


def test_linearized_freezing_point_equation() -> None:
    pass


def test_plume_homogeneous_ocean_without_melt() -> None:
    """"""
    w, R, T_o, S_o, z_out = solve.plume(-165, 100, 0.5)


def test_plume_stratified_ocean_without_melt() -> None:
    """"""
    d, S, T = helpers.load_LeConte_CTD()

    S = itpl.interp1d(-d, S)
    T = itpl.interp1d(-d, T)

    w, R, T_o, S_o, z_out = solve.plume(-165, 100, 0.5, -d[0], T, S)


def test_melt_rate() -> None:
    pass
