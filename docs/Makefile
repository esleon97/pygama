<<<<<<< HEAD
SOURCEDIR = source
BUILDDIR = build

# docs for all project versions will be placed here
BUILDDIRALL = $(BUILDDIR)/allver/html

# list in source/.py_api_exclude files or directories that should be excluded
# from the API auto-generated documentation. No regex allowed
PY_SOURCE_EXCLUDE = $(shell cat source/.py_api_exclude)

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

allver:
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIRALL)"
	cp source/_assets/redirect.html "$(BUILDDIRALL)/index.html"

allver-info:
	sphinx-multiversion --dump-metadata "$(SOURCEDIR)" "$(BUILDDIRALL)"

apidoc: clean-apidoc
	sphinx-apidoc --private --module-first --force --output-dir "$(SOURCEDIR)/generated" ../src/pygama $(PY_SOURCE_EXCLUDE)

clean-apidoc:
	rm -rf "$(SOURCEDIR)/generated"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
=======
# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	
# Since you want to keep the source and output separate, but still be able to 
# both publish on GitHub Pages and preview builds locally, you can add an option
# to your Makefile to do both.
github:
	@make html
	@cp -a _build/html/. ../docs
>>>>>>> Modified processors.py
