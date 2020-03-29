"""
python3 compile.py build_ext --inplace
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("./Spelling_Correction_c.pyx"),include_dirs=[numpy.get_include()])
