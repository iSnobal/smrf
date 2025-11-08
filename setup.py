#!/usr/bin/env python

import os

import numpy
from setuptools import Extension, setup

try:
    # Use the Cython build_ext module if the library is installed.
    from Cython.Distutils import build_ext
    print('Using Cython')
    ext = '.pyx'
except ImportError:
    # Falling back to setuptools as the default if the Cython import is
    # unsuccessful
    from setuptools.command.build_ext import build_ext
    print('Using GCC')
    ext = '.c'

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

extensions = []

# detrended kriging
extensions += [
    Extension(
        'smrf.spatial.dk.detrended_kriging',
        sources=[
            os.path.join('smrf/spatial/dk', source_file) for source_file in [
                "detrended_kriging" + ext,
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
                "envphys_c" + ext,
                "topotherm.c",
                "dewpt.c",
                "iwbt.c"
            ]
        ],
        **extension_params
    ),
]
extensions += [
    Extension(
        'smrf.envphys.solar.toposplit',
        sources=['smrf/envphys/solar/toposplit.pyx'],
        **extension_params
    )
]

# wind model c functions
extensions += [
    Extension(
        'smrf.utils.wind.wind_c',
        sources=[
            os.path.join('smrf/utils/wind', val) for val in [
                "wind_c" + ext,
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
    ext_modules=extensions,
)
