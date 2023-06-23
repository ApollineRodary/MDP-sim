#!/usr/bin/env python

from distutils.core import setup, Extension

pymdp_module = Extension("_pymdp",
                         sources=["swig/pymdp_wrap.cxx", "src/mdp.cpp"]
                         )

setup(name="pymdp",
      version="0.1",
      author="Apolline Rodary",
      description="""Markov decision processes""",
      ext_modules=[pymdp_module],
      py_modules=["pymdp"]
      )

