from distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy,os

#Options = "-unroll4 -mtune=native -march=native"
#os.environ["CC"] = "gcc " + Options
#os.environ["LDSHARED"] = "gcc -shared " + Options

setup(
      cmdclass={'build_ext':build_ext},
      ext_modules = [Extension("fft_cython",
                     sources=["fft_cython_py.pyx", "fft_cython_c.c"],
                     libraries = ["fftw3"],
                     extra_compile_args = ["-Ofast","-mtune=native","-unroll4","-march=native","-fopenmp","-pthread","-lpthread"],
                     extra_link_args = ["-lfftw3","-lfftw3_omp","-lm","-fopenmp","-pthread","-lpthread"],
                     library_dirs=['/usr/lib64'],
                     include_dirs=['/usr/lib64',numpy.get_include()])])
