"""Setup script for the Sparse Group IMC."""
import os

import numpy

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


# python setup.py build_ext [--inplace]

extensions = [
    Extension(
        name="sgimc.ops",
        sources=[os.path.join("sgimc", "ops.pyx")],
        include_dirs=[numpy.get_include(),
                      os.path.join("sgimc", "src", "include")],
        libraries=["blas", "lapack", "m"],
        extra_compile_args=["-std=c99"],
        extra_link_args=["-lstdc++"],
        depends=[])
]

# refer to http://python-packaging.readthedocs.io/en/latest/metadata.html
setup(
    name="sgimc",
    version="0.1.dev0",
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
