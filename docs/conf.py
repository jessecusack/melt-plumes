"""Sphinx configuration."""
project = "Melt Plumes"
author = "Jesse Cusack"
copyright = "2022, Jesse Cusack"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",
]
autodoc_typehints = "description"
html_theme = "furo"
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
exclude_patterns = ["*.ipynb"]
nb_execution_excludepatterns = [
    "*.ipynb",
    "usage.md",
    "codeofconduct.md",
    "contributing.md",
    "index.md",
    "license.md",
    "reference.md",
]
