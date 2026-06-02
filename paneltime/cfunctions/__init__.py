
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
CI64PT = ct.POINTER(ct.c_int64) 

cfunct.armas.restype    = ct.c_int
cfunct.fast_dot.restype = ct.c_int



def armas(parameters, lmbda, rho, gmma, psi,
		  AMA_1, AMA_1AR, GAR_1, GAR_1MA,
		  u, e, var, h, G, T_arr, h_expr):

	params  = np.asarray(parameters, dtype=np.float64)
	lmbda   = np.asarray(lmbda,      dtype=np.float64)
	rho     = np.asarray(rho,        dtype=np.float64)
	gmma    = np.asarray(gmma,       dtype=np.float64)
	psi     = np.asarray(psi,        dtype=np.float64)
	u       = np.asarray(u,          dtype=np.float64)
	G       = np.asarray(G,          dtype=np.float64)
	T_arr   = np.asarray(T_arr,      dtype=np.int64)
	h_bytes = h_expr.encode('utf-8') if isinstance(h_expr, str) else h_expr
	


	cfunct.armas(params.ctypes.data_as(CDPT),
				 lmbda.ctypes.data_as(CDPT),   rho.ctypes.data_as(CDPT),
				 gmma.ctypes.data_as(CDPT),    psi.ctypes.data_as(CDPT),
				 AMA_1.ctypes.data_as(CDPT),   AMA_1AR.ctypes.data_as(CDPT),
				 GAR_1.ctypes.data_as(CDPT),   GAR_1MA.ctypes.data_as(CDPT),
				 u.ctypes.data_as(CDPT),
				 e.ctypes.data_as(CDPT),
				 var.ctypes.data_as(CDPT),
				 h.ctypes.data_as(CDPT),
				 G.ctypes.data_as(CDPT),
				 T_arr.ctypes.data_as(CI64PT),
				 ct.c_char_p(h_bytes))

	return AMA_1, AMA_1AR, GAR_1, GAR_1MA, e, var, h

def fast_dot(r, a, b, cols):
	#rn = np.array(r, dtype=np.float64)
	r = np.asarray(r, dtype=np.float64)
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)

	assert np.isfortran(r) == np.isfortran(b), "fast_dot: r and b must have the same memory layout"
	
	n = len(a)

	cfunct.fast_dot(r.ctypes.data_as(CDPT),
					a.ctypes.data_as(CDPT),
					b.ctypes.data_as(CDPT), n, cols)
	
	if False:
		rn = fast_dot_numpy(rn, a, b, cols)
		if not np.all(np.isclose(r, rn, atol=1e-12, equal_nan=True)):
			print("Discrepancy between C and NumPy results in fast_dot:")
			print("C result:", r)
			print("NumPy result:", rn)
			raise ValueError("fast_dot results do not match")

	return r



def fast_dot_numpy(r, a, b, m):
	"""
	NumPy-versjon av fast_dot.
	r, a, b er 1D-arrays, alle i column-major layout (Fortran style).
	Oppdaterer r in-place.
	"""
	r, a, b = np.array(r), np.array(a), np.array(b)
	n = len(a)
	# Reshape til matriseformat, men behold column-major memory mapping
	R = np.reshape(r, (n, m), order='F')
	B = np.reshape(b, (n, m), order='F')

	for i in range(1, n):
		# a[i] * B[0:n-i, :] er en rad-blokk
		R[i:n, :] += a[i] * B[0:n-i, :]

	return r

