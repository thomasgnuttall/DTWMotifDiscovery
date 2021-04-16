#!/usr/bin/env python

from setuptools import find_packages, setup, Extension
from distutils.sysconfig import *
import numpy

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

try:
    from Cython.Distutils import build_ext
except ImportError:
    from distutils.command import build_ext
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

py_inc = [get_python_inc()]

if use_cython:
    ext_modules += [
        Extension("dtw.dtw", ["dtw/cdtw.pyx", "dtw/dtw.c"],
                  include_dirs=['dtw', numpy.get_include()] + py_inc),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("dtw.dtw", ["dtw/dtw.c"],
                  include_dirs=['dtw', numpy.get_include()] + py_inc),
    ]

#setup(
#    name='dtw',
#    ext_modules=cythonize("cdtw.pyx"),
#    include_dirs=[np.get_include()],
#    zip_safe=False,
#)


setup(name='dtw',
      version='0.4-beta',
      description='DTW',
      packages=find_packages(),
      ext_modules=ext_modules
      )

#