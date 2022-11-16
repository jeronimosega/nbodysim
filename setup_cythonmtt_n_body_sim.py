from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cythonmtt_n_body_sim",
        ["cythonmtt_n_body_sim.pyx"],
# Uncomment the following line if you are on MacOS Catalina
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"]
    )
]

setup(name="cythonmtt_n_body_sim",
      ext_modules=cythonize(ext_modules))

