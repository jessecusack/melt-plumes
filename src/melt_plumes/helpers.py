from importlib import resources
import numpy as np


def load_LeConte_CTD():
    file = resources.open_text("melt_plumes.data", "LeConte_profile.csv")
    data = np.loadtxt(file, delimiter=",", skiprows=1)
    depth = data[:, 0]
    salinity = data[:, 1]
    temperature = data[:, 2]
    return depth, salinity, temperature