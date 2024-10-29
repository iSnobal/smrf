#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy
from setuptools import Extension, setup

# Test if compiling with cython or using the C source
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext
    ext = '.c'
    print('Using GCC')
else:
    ext = '.pyx'
    print('Using Cython')


def c_name_from_path(location, name):
    return os.path.join(location, name).replace('/', '.')


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)


# Give user option to specify local compiler name
if "CC" not in os.environ:
    os.environ["CC"] = "gcc"
print(os.environ["CC"])

# extension parameters
extension_params = dict(
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp', '-O3'],
    include_dirs=[numpy.get_include()],
)

ext_modules = []

# detrended kriging
source_folder = 'smrf/spatial/dk'
ext_modules += [
    Extension(
        c_name_from_path(source_folder, 'detrended_kriging'),
        sources=[os.path.join(source_folder, val) for val in [
            "detrended_kriging" + ext,
            "krige.c",
            "lusolv.c",
            "array.c"
        ]],
        **extension_params
    ),
]

# envphys core c functions
source_folder = 'smrf/envphys/core'
ext_modules += [
    Extension(
        c_name_from_path(source_folder, 'envphys_c'),
        sources=[os.path.join(source_folder, val) for val in [
            "envphys_c" + ext,
            "topotherm.c",
            "dewpt.c",
            "iwbt.c"
        ]],
        **extension_params
    ),
]

# wind model c functions
source_folder = 'smrf/utils/wind'
ext_modules += [
    Extension(
        c_name_from_path(source_folder, 'wind_c'),
        sources=[os.path.join(source_folder, val) for val in [
            "wind_c" + ext,
            "breshen.c",
            "calc_wind.c"
        ]],
        **extension_params
    ),
]

setup(
    cmdclass={
        'build_ext': build_ext
    },
    ext_modules=ext_modules,
)
