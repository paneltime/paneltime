
import ctypes as ct
import os
import numpy.ctypeslib as npct
from pathlib import Path
import numpy as np


p = Path(__file__).parent.absolute()

if os.name=='nt':
	cfunct = npct.load_library('ctypes.dll',p)
elif os.name == 'posix':
	cfunct = npct.load_library('ctypes.dylib',p)
else:
	cfunct = npct.load_library('ctypes.so',p)


CDPT = ct.POINTER(ct.c_double) 
CIPT = ct.POINTER(ct.c_uint) 



def armas(parameters,lmbda, rho,  gmma, psi,  
          AMA_1, AMA_1AR,  GAR_1, GAR_1MA,  
          u,e,var,  h,  G,T_arr, h_expr):

	cfunct.armas(np.array(parameters, dtype=np.float64).ctypes.data_as(CDPT), 
	            lmbda.ctypes.data_as(CDPT), rho.ctypes.data_as(CDPT),
	           	gmma.ctypes.data_as(CDPT), psi.ctypes.data_as(CDPT),
	            AMA_1.ctypes.data_as(CDPT), AMA_1AR.ctypes.data_as(CDPT),
				GAR_1.ctypes.data_as(CDPT), GAR_1MA.ctypes.data_as(CDPT),
				u.ctypes.data_as(CDPT), 
				e.ctypes.data_as(CDPT), 
				var.ctypes.data_as(CDPT),
				h.ctypes.data_as(CDPT),
				G.ctypes.data_as(CDPT), 
				T_arr.ctypes.data_as(CIPT),
	            ct.c_char_p(h_expr)
				)	


def fast_dot(r,a,b, cols):
	r = r.astype(float)
	a = a.astype(float)
	b = b.astype(float)
	cfunct.fast_dot(r.ctypes.data_as(CDPT), 
	                a.ctypes.data_as(CDPT),
	                b.ctypes.data_as(CDPT), len(a), cols)
	return r, a, b



