import os
import numpy as np
import numpy.ctypeslib as npct
from pathlib import Path

p = Path(__file__).parent.absolute()

if os.name=='nt':
	cfunct = npct.load_library('../cfunctions/ctypes.dll',p)
elif os.name == 'posix':
	cfunct = npct.load_library('../cfunctions/ctypes.dylib',p)
else:
	cfunct = npct.load_library('../cfunctions/ctypes.so',p)




class hClass:
	"""Class for handling the h functions"""
	def __init__(self, h_dict, included):


		self.h_func_bstr = b''
		
		h= ''
		self.h_dict = h_dict

		self.test()
			
		h = h_dict['h']

		if h == '':
			return
		
		self.in_use = True

		self.h_func_bstr = self.syntax_adapt(h).encode('utf-8')

		self.h_dict = {}
		for k in h_dict:
			self.h_dict[k] = self.eval(h_dict[k], included)

	def syntax_adapt(self, expr):
		# Temporarily encode multi-char operators to avoid overlap during substitution
		placeholders = {
			'<=': '__LE__',
			'>=': '__GE__',
			'!=': '__NE__',
			'==': '__EQ__',
			'1e-': '__SCI__'
		}

		# Encode complex operators
		for op, ph in placeholders.items():
			expr = expr.replace(op, ph)

		expr = expr.replace('=', ':=')

		# Decode placeholders
		for op, ph in placeholders.items():
			expr = expr.replace(ph, op)

		# Convert Pythonic power operator
		expr = expr.replace('**', '^')

		# Finally, rewrite == to = for relaxed equality
		expr = expr.replace('==', '=')

		return expr
	
	def syntax_reverse(self, expr):
		placeholders = {
			'1e-': '__SCI__'
		}
		
		for op, ph in placeholders.items():
			expr = expr.replace(op, ph)

		# Reverses the syntax adaptation done in syntax_adapt to be compatible with python eval
		expr = " ".join(expr.split()).replace('; ', '\n').replace(';', '\n').replace('^', '**')
		expr = expr.replace(':=', '=').replace('-', '-1*')

		for op, ph in placeholders.items():
			expr = expr.replace(ph, op)

		return expr
	
	def test(self):
		if type(self.h_dict) != dict:
			raise ValueError("Your custom h-function must be a dictionary.")

		if set(self.h_dict.keys()) < {'h', 'h_e', 'h_e2'}:
			raise ValueError("Your custom h-function must be a dictionary with atleast these keys 'h', 'h_e', 'h_e2'. \n"
					"In addition 'h_z','h_z2', and 'h_e_z' can be specified if you need a shape parameters.")	
		for k in {'h_z', 'h_z2', 'h_e_z'}:
			if k not in self.h_dict:
				self.h_dict[k] = ''

		for e in [-4, 0, 4]:
			for k in self.h_dict:
				exp = self.h_dict[k]
				if '?' in exp:
					raise ValueError("Logical operations are not allowed currently.")
				if exp != '':
					self.test_expr(e, 0, self.h_dict[k], k)


	def test_expr(self, e, z, h_expr, k):
		h_expr = self.syntax_adapt(h_expr).encode('utf-8')
		cfunct.expression_test.argtypes = [ct.c_double, ct.c_double, ct.c_char_p]
		cfunct.expression_test.restype = ct.c_char_p
		x = cfunct.expression_test(e, z, h_expr).decode('utf-8')
		if "Error:" in x:
			raise ValueError(f"Error in h function: {x}")
		elif x == 'nan':
			raise ValueError(f"h function returns NaN for e={e} and z={z}.")
		elif x == 'inf' or x == '-inf':	
			raise ValueError(f"{h_expr} function returns inf for e={e} and z={z}.")

		x = float(x)
	

	def eval(self, expr, included):
			# Return a function that takes variables dynamically
		if expr.strip() == '':
			return lambda e, z: None
		funcs = {
			'abs': np.abs,
			'sqrt': np.sqrt,
			'log': np.log,
			'exp': np.exp,
			'sin': np.sin,
			'cos': np.cos,
			'tan': np.tan,
			'floor': np.floor,
			'ceil': np.ceil,
			'min': np.minimum,
			'max': np.maximum,
			'sign': np.sign, 
			'included': included
		}
		expr_clean = self.syntax_reverse(expr)
		expr_list = expr_clean.split('\n')
		func_code = (
		f"def h_func(e, z):\n"
		f"	{'\n\t'.join(expr_list[:-1])}\n"
		f"	return  ({expr_list[-1]})*included\n"
		)
		d = {}
		try:
			exec(func_code, funcs, d)
			f = d['h_func']
			test = f(0, 0)
		except SyntaxError as e:
			if "instead of '='?" in str(e):
				raise ValueError("Last item in a h function expression cannot be an assignment.")
			raise e
		
		f._source = func_code
		return  f
