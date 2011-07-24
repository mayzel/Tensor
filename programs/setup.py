from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
setup(
    name="CPWOPTcritical",
    cmdclass={"build_ext":build_ext},
    ext_modules=[Extension("CPWOPTcritical",["CPWOPTcritical.pyx"]
                 ,include_dirs=[numpy.get_numpy_include()]
                 )]
)
