<<<<<<< HEAD
# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

sys.path.insert(0, Path('../../src').resolve().as_posix())

project = 'pygama'
copyright = '2022, the LEGEND Collaboration'

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx_multiversion',
    'sphinx_copybutton',
    'myst_parser'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
language = 'python'
# in _templates/ we have a custom layout.html to include the version menu
# (adapted from sphinx-multiversion docs)
templates_path = ['_templates']
pygments_style = 'sphinx'

# readthedocs.io Sphinx theme
html_theme = 'sphinx_rtd_theme'

# list here pygama dependencies that are not required for building docs and
# could be unmet at build time
autodoc_mock_imports = [
    'pygama._version',
    'pandas',
    # 'numpy',
    'matplotlib',
    'mplhep',
    'scipy',
    'numba',
    'pytest',
    'pyhf',
    'awkward',
    'iminuit',
    'boost-histogram',
    'hepunits',
    'hepstats',
    'uproot',
    'h5py',
    'pint',
    'pyfftw',
    'tqdm',
    'tinydb',
    'parse'
]

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'numba': ('https://numba.readthedocs.io/en/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'iminuit': ('https://iminuit.readthedocs.io/en/stable', None),
    'h5py': ('https://docs.h5py.org/en/stable', None),
    'pint': ('https://pint.readthedocs.io/en/stable', None)
}

# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = 'both'

# sphinx-multiversion

# For now, we include only (certain) branches when building docs.
# To add a specific release to the list of versions for which docs should be build,
# one must create a new branch named `releases/...`
smv_branch_whitelist = r'^(main|refactor|releases/.*)$'
smv_tag_whitelist = '^$'
smv_released_pattern = '^$'
smv_outputdir_format = '{ref.name}'
smv_prefer_remote_refs = False

# HACK: we need to regenerate the API documentation before the actual build,
# but it's not possible with the current sphinx-multiversion. Changes have been
# proposed in this PR: https://github.com/Holzhaus/sphinx-multiversion/pull/62
# but there's no timeline for merging yet. For the following option to be considered,
# one needs to install the sphinx-multiversion-pre-post-build fork from PyPI
smv_prebuild_command = 'make -ik apidoc'

# The right way to find all docs versions is to look for matching branches on
# the default remote
smv_remote_whitelist = r'^origin$'
=======
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# NOTE: I mainly did this following
# https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
#
# but there are also some cool things you can do with sphinx-gallery
# https://sphinx-gallery.readthedocs.io/en/latest/

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../../pygama'))


# -- Project information -----------------------------------------------------

project = 'pygama'
copyright = '2020, LEGEND'
author = 'C. Wiseman'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = 'v1'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['modules']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'bizstyle'
html_theme = "sphinx_rtd_theme"


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pygamadoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pygama.tex', 'pygama Documentation',
     'C. Wiseman', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'pygama', 'pygama Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'pygama', 'pygama Documentation',
     author, 'pygama', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
>>>>>>> Modified processors.py
