from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys
import glob

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
    
wd = os.getcwd()
os.chdir(os.path.dirname(__file__))
cython_modules = find_cython_files()

#delete_pyds()
#cython_modules = ['arma.pyx', 'main.pyx']
extensions = [Extension(name=module[:-4], 
                        sources=[module], 
                        extra_compile_args=['/Ox', '/GL', '/fp:fast'],
                        extra_link_args=['/LTCG']
    
                        ) for module in cython_modules]
setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={'boundscheck': False, 'wraparound': False}, 
                          annotate=True,
                          
                          )
)

os.chdir(wd)