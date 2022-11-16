from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "optcython_n_body_sim",
        ["optcython_n_body_sim.pyx"],
# Uncomment the following line if you are on MacOS Catalina
        extra_compile_args=['-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include'],
    )
]

setup(name="optcython_n_body_sim",
      include_dirs = [np.get_include()],
      ext_modules=cythonize(ext_modules))