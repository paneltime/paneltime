import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
from pathlib import Path

p = Path(__file__).parent.absolute()

if os.name=='nt':
	cfunct = npct.load_library('../cfunctions/ctypes.dll',p)
elif os.name == 'posix':
	cfunct = npct.load_library('../cfunctions/ctypes.dylib',p)
else:
	cfunct = npct.load_library('../cfunctions/ctypes.so',p)


def get_model_string(model, e, e2, z):
	"""Returns a string with the model name and parameters"""
	if model is None:
		s = b''
	else:
		s = syntax_adapt(model.h_val_cpp).encode('utf-8')
	if not s == b'':
		test_expr(s, e, e2, z)
	return s




def syntax_adapt(expr):
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



def test_expr(h_expr, e, e2, z):
	cfunct.expression_test.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_char_p]
	cfunct.expression_test.restype = ct.c_char_p
	e = -100
	z = -100
	x = cfunct.expression_test(e, e2, z, h_expr).decode('utf-8')
	if "Error:" in x:
		raise ValueError(f"Error in h function: {x}")
	elif x == 'nan':
		raise ValueError(f"h function returns NaN for e={e} and z={z}.")
	elif x == 'inf' or x == '-inf':	
		raise ValueError(f"{h_expr} function returns inf for e={e} and z={z}.")

	x = float(x)




