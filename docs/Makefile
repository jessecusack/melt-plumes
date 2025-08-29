# Minimal Makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: html
html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: html
	python -m http.server --directory  "$(BUILDDIR)"/html 8000