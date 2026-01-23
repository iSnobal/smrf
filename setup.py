#!/usr/bin/env python

import os

import numpy
from setuptools import Extension, setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Give user option to specify local compiler name
if "CC" not in os.environ:
    os.environ["CC"] = "gcc"

print("Compiler set to: " + os.environ["CC"])

extension_params = dict(
    extra_compile_args=[
        '-fopenmp',
        '-O3',
    ],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

directives = {
    'language_level': "3str",
    'embedsignature': True,
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'binding': True,
}

extensions = []

# detrended kriging
extensions += [
    Extension(
        'smrf.spatial.dk.detrended_kriging',
        sources=[
            os.path.join('smrf/spatial/dk', source_file) for source_file in [
                "detrended_kriging.pyx",
                "krige.c",
                "lusolv.c",
                "array.c"
            ]
        ],
        **extension_params
    ),
]

# envphys core c functions
extensions += [
    Extension(
        'smrf.envphys.core.envphys_c',
        sources=[
            os.path.join('smrf/envphys/core', val) for val in [
                "envphys_c.pyx",
                "topotherm.c",
                "dewpt.c",
                "iwbt.c"
            ]
        ],
        **extension_params
    ),
    Extension(
        'smrf.envphys.solar.toposplit',
        sources=['smrf/envphys/solar/toposplit.pyx'],
        **extension_params
    ),
    Extension(
        'smrf.utils.wind.wind_c',
        sources=[
            os.path.join('smrf/utils/wind', val) for val in [
                "wind_c.pyx",
                "breshen.c",
                "calc_wind.c"
            ]
        ],
        **extension_params
    ),
]

setup(
    cmdclass={
        'build_ext': build_ext
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives=directives,
        annotate=False,
    ),
)
