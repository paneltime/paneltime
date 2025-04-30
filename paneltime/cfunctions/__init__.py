
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




class hFunction:
	"""Class for handling the h functions"""
	def __init__(self, h_dict, included):
		
		h= ''
		self.h_dict = h_dict

		if self.valid():
			h = h_dict['h']
		
		self.z_active=False
		if not np.all(np.array([h_dict[k] for k in ['h_z', 'h_z2', 'h_e_z']]) == ''):
			self.z_active=True

		h = h.replace(':=', '=')
		h = h.replace('==', '||$$~&|').replace('=', ':=').replace('||$$~&|', '==')
		
		self.h_func_str = h
		self.h_func_bstr = h.encode('utf-8')

		for k in h_dict:
			setattr(self, k, self.eval(h_dict[k], included))


	def valid(self):
		if type(self.h_dict) != dict:
			print("Warning: Your custom h-function must be a dictionary. Default is used")
			return False

		if set(self.h_dict.keys()) != {'h', 'h_e', 'h_e2', 'h_z', 'h_z2', 'h_e_z'}:
			print("Warning: Your custom h-function must be a dictionary with these exact keys 'h', 'h_e', 'h_e2', 'h_z','h_z2', and 'h_e_z'. Default is used")	
			return False

		test_values = []
		for e in [-1000, 0, 1000]:
			for k in self.h_dict:
				exp = self.h_dict[k]
				if exp != '':
					test_values.append(self.test_expr(e, 0, self.h_dict[k]))
		if np.any(np.isnan(test_values)):
			print("Warning: Your custom h-function must return non-nans when e=-1000, e=0 or e=1000. Default is used")
			return False

		return True

	
	def eval(self, expr, included):
		# Return a function that takes variables dynamically
		def numpy_func(e,z):
			if expr == '':
				return None
			expr_clean = " ".join(expr.split()).replace('; ', '\n').replace(';', '\n').replace('^', '**')
			expr_clean = expr_clean.replace(':=', '=')
			# Local namespace: numpy functions + variables
			safe_namespace = np.__dict__.copy()
			safe_namespace.update({'e': e, 'z': z})
			for s in expr_clean.split('\n'):
				try:
					r = eval(s, {"__builtins__": {}}, safe_namespace)
				except SyntaxError as ex:
					exec(s, {"__builtins__": {}}, safe_namespace)
			return r*included
		
		return numpy_func

	def test_expr(self, e, z, h_expr):
		h_expr = h_expr.encode('utf-8')
		cfunct.expression_test.argtypes = [ct.c_double, ct.c_double, ct.c_char_p]
		cfunct.expression_test.restype = ct.c_double
		x = cfunct.expression_test(e, z, h_expr)
		return x