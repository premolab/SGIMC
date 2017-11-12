"""Setup script for the L1_tf wrapper."""
import os
from os.path import join

import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize


# python setup.py build_ext [--inplace]

extensions = [
    Extension(
        name="imc_ops.ops",
        sources=[join("imc_ops", "ops.pyx")],
        include_dirs=[numpy.get_include(), join("imc_ops", "src", "include")],
        libraries=["blas", "lapack", "m"],
        depends=[],
        extra_link_args=['-lstdc++'])
]

# refer to http://python-packaging.readthedocs.io/en/latest/metadata.html
setup(
    name="imc_ops",
    version="0.4.dev0",
    description="""Fast sparse operrations for IMC.""",
    # url="",
    author="Ivan Nazarov",
    # author_email="ivan.nazaor@skolkovotech.ru",
    # license='GNU',
    packages=["imc_ops"],
    ext_modules=cythonize(extensions, quiet=True, nthreads=4),
    # cmdclass={"build_ext": build_ext},
    install_requires=[],
)
