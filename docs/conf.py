"""Sphinx configuration."""
project = "Melt Plumes"
author = "Jesse Cusack"
copyright = "2022, Jesse Cusack"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
