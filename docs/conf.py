"""Sphinx configuration."""

project = "Melt Plumes"
author = "Jesse Cusack"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",
]
autodoc_typehints = "description"
html_theme = "furo"
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nbsphinx_execute = "always"
