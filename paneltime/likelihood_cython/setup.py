from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

def find_cython_files():
    """Recursively finds all .pyx files in the given directory."""
    cython_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(".pyx"):
                cython_files.append(file)
    return cython_files

if len(sys.argv) == 1:
    sys.argv.append("build_ext")
    sys.argv.append("--inplace")
    
cython_modules = find_cython_files()

cython_modules = ['arma.pyx']
extensions = [Extension(name=module[:-4], sources=[module]) for module in cython_modules]
setup(
    ext_modules=cythonize(extensions)
)
