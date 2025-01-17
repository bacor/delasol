# Configuration file for the Sphinx documentation builder.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(CUR_DIR, os.path.pardir, os.path.pardir)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT_DIR)))
# sys.path.insert(0, os.path.abspath(os.path.join(ROOT_DIR, "delasol")))

# -- Project information

project = "Delasol"
copyright = "2024-%Y, Bas Cornelissen"
author = "Bas Cornelissen"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # "jupyter_sphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "music21": ("https://www.music21.org/music21docs/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

autodoc_member_order = "bysource"

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# Make sure delasol is loadable by jupyter-sphinx:
# https://jupyter-sphinx.readthedocs.io/en/latest/#configuration-options
# package_path = os.path.abspath("../")
# os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))
