# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

autodoc_mock_imports = [
    'pandas',
    'pytest',
    'numpy',
    'scipy',
    'lmfit',
    'matplotlib',
    'numba',
    'matplotlib.rcParams',
    'sklearn',
    'numerics',
    'blank',
    'dot',
]

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../pysprint'))
sys.path.insert(0, os.path.abspath('../pysprint/core/bases'))
sys.path.insert(0, os.path.abspath('../pysprint/core/methods'))
sys.path.insert(0, os.path.abspath('../pysprint/utils'))

# -- Project information -----------------------------------------------------

project = 'Pysprint'
copyright = '2020, Peter Leeh'
author = 'Peter Leeh'

release = '0.17.0'

autodoc_default_options = {
    'special-members': '__init__'
}

# -- General configuration ---------------------------------------------------
master_doc = 'index'

extensions = [
    'recommonmark',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']


exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'pysprint.tests.rst',
    'pysprint.mpl_tools.rst',
    'pysprint.bases.io.rst',
    '**.ipynb_checkpoints'
]


intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None)
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

autodoc_member_order = 'bysource'
add_module_names = False


def maybe_skip_member(app, what, name, obj, skip, options):
    EXCLUDE_CLASSES = (
        'DatasetBase', 'DatasetApply', 'pysprint.core.bases.apply',
        'pysprint.core.bases.dataset_base', '**/*dataset_base*'
    )
    exclude = name in EXCLUDE_CLASSES
    return skip or exclude


def setup(app):
    app.connect('autodoc-skip-member', maybe_skip_member)
