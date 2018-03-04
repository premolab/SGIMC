"""Setup script for the Sparse Group IMC."""
import os
from os.path import join

import numpy

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

from numpy.distutils.system_info import get_info


def get_blas_info():
    """A utility function from sklearn."""
    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])

    return cblas_libs, blas_info


# liblinear module
cblas_libs, blas_info = get_blas_info()
if os.name == 'posix':
    cblas_libs.append('m')

sources = [join("sgimc", "ops.pyx")]

depends = [join("sgimc", "src", "include", "*.h")]

include_dirs = [join("sgimc", "src", "include"), numpy.get_include()]
include_dirs.extend(blas_info.pop("include_dirs", []))

extra_compile_args = blas_info.pop("extra_compile_args", [])
extra_compile_args.extend(["-std=c99", "-O3"])

extensions = [
    Extension("sgimc.ops", sources=sources, libraries=cblas_libs,
              include_dirs=include_dirs, extra_compile_args=extra_compile_args,
              depends=[], **blas_info)
]

# refer to http://python-packaging.readthedocs.io/en/latest/metadata.html
# python setup.py build_ext [--inplace]
setup(
    name="sgimc",
    version="0.2.dev0",
    description="""Fast sparse operrations for the Sparse Group IMC.""",
    # url="",
    author="Ivan Nazarov",
    # author_email="ivan.nazaor@skolkovotech.ru",
    # license='GNU',
    packages=find_packages(),
    ext_modules=cythonize(extensions, quiet=True, nthreads=4),
    # cmdclass={"build_ext": build_ext},
    install_requires=[],
)
