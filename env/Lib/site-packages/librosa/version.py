#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Version info"""

import sys
import importlib

short_version = "0.9"
version = "0.9.1"


def __get_mod_version(modname):

    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return "installed, no version number available"

    except ImportError:
        return None


def show_versions():
    """Return the version information for all librosa dependencies."""

    core_deps = [
        "audioread",
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "decorator",
        "soundfile",
        "resampy",
        "numba",
    ]

    extra_deps = [
        "numpydoc",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinxcontrib.versioning",
        "sphinx-gallery",
        "pytest",
        "pytest-mpl",
        "pytest-cov",
        "matplotlib",
        "presets",
    ]

    print("INSTALLED VERSIONS")
    print("------------------")
    print("python: {}\n".format(sys.version))
    print("librosa: {}\n".format(version))
    for dep in core_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
    print("")
    for dep in extra_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
