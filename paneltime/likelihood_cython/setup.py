from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# List of Cython files to compile
cython_modules = ["cfunctions.pyx", "calculus.pyx" ,"calculus_functions.pyx", "function.pyx", "main.pyx"]

# Define the list of Extension objects for each Cython file
extensions = [Extension(name=module[:-4], sources=[module]) for module in cython_modules]

setup(
    ext_modules = cythonize(extensions), 
    include_dirs=[np.get_include()]
)

