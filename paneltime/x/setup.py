from setuptools import setup, Extension
from Cython.Build import cythonize

# List of Cython files to compile
cython_modules = ["module1.pyx"]

# Define the list of Extension objects for each Cython file
extensions = [Extension(name=module[:-4], sources=[module]) for module in cython_modules]

setup(
    ext_modules = cythonize(extensions)
)