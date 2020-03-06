import sys
sys.path.append('..//build//lib.win-amd64-3.5')
sys.path.append('..//build//lib.linux-x86_64-3.5')
import numpy
import ctypes as ct


gamma=numpy.array([0.5,0.2,-0.3])
lambda_=numpy.array([0.5,0.2,-0.3])



import cfunctions as c
bbb=c.bandinverse(gamma,lambda_,1500)
