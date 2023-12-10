
import ctypes as ct
import os
import numpy.ctypeslib as npct
from pathlib import Path

p = Path(__file__).parent.absolute()

if os.name=='nt':
  cfunct = npct.load_library('ctypes.dll',p)
elif os.name == 'posix':
  cfunct = npct.load_library('ctypes.dylib',p)
else:
  cfunct = npct.ßload_library('ctypes.so',p)


CDPT = ct.POINTER(ct.c_double) 
CIPT = ct.POINTER(ct.c_uint) 

def armas(parameters,lmbda, rho,  gmma, psi,  
        AMA_1, AMA_1AR,  GAR_1, GAR_1MA,  
        u,e,var,  h,  G,T_arr ):

  cfunct.armas(parameters.ctypes.data_as(CIPT), 
                     lmbda.ctypes.data_as(CDPT), rho.ctypes.data_as(CDPT),
                                                  gmma.ctypes.data_as(CDPT), psi.ctypes.data_as(CDPT),
                                                  AMA_1.ctypes.data_as(CDPT), AMA_1AR.ctypes.data_as(CDPT),
                                                  GAR_1.ctypes.data_as(CDPT), GAR_1MA.ctypes.data_as(CDPT),
                                                  u.ctypes.data_as(CDPT), 
                                                  e.ctypes.data_as(CDPT), 
                                                  var.ctypes.data_as(CDPT),
                                                  h.ctypes.data_as(CDPT),
                                                  G.ctypes.data_as(CDPT), 
                                                  T_arr.ctypes.data_as(CIPT)
                                                  )	


def fast_dot(r,a,b, cols):

     cfunct.fast_dot(r.ctypes.data_as(CIPT), 
              a.ctypes.data_as(CDPT),
              b.ctypes.data_as(CDPT), len(a), cols)