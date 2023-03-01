import sys
from pydoc import importfile
import os
path = os.path.dirname(__file__)
parallel =  importfile(os.path.join(path,'parallel.py'))
parallel_slave =  importfile(os.path.join(path,'parallel_slave.py'))

parallel_slave.run(parallel.Transact(sys.stdin,sys.stdout), True)

