"""Symbolic heteroskedasticity-function support."""

import numpy as np


def build_h_function(expression, e_symbol=None, z_symbol=None):
	"""Compile a SymPy expression and its derivatives for paneltime.

	Parameters
	----------
	expression : sympy.Expr
		Expression in the symbols ``e`` and optionally ``z``.

	Returns
	-------
	 dict
		Numeric functions and an ExprTk-compatible expression string.
	"""
	try:
		import sympy as sp
	except ImportError as exc:
		raise ImportError('SymPy is required for symbolic h_function expressions') from exc
	if not isinstance(expression, sp.Expr):
		raise TypeError('h_function must be a SymPy expression')
	expr = expression
	free_symbols = {symbol.name: symbol for symbol in expr.free_symbols}
	e = e_symbol if e_symbol is not None else free_symbols.get('e', sp.Symbol('e'))
	z = z_symbol if z_symbol is not None else free_symbols.get('z', sp.Symbol('z'))
	unsupported = set(expr.free_symbols) - {e, z}
	if unsupported:
		names = ', '.join(sorted(symbol.name for symbol in unsupported))
		raise ValueError(f'h_function may only use e and z; unsupported symbols: {names}')
	derivatives = {
		'h_val': expr, 'h_e_val': sp.diff(expr, e),
		'h_2e_val': sp.diff(expr, e, 2), 'h_z_val': sp.diff(expr, z),
		'h_2z_val': sp.diff(expr, z, 2), 'h_ez_val': sp.diff(expr, e, z)}
	functions = {name: sp.lambdify((e, z), value, 'numpy') for name, value in derivatives.items()}
	cpp = sp.sstr(expr).replace('**', '^')
	return {'expression': expr, 'cpp': cpp, **functions}


class SymbolicHFunction:
	"""Callable adapter for the likelihood model's ``h(e, e2, v)`` interface."""

	def __init__(self, expression, z=None):
		self.expression = expression
		self.z = z
		self.functions = build_h_function(expression)
		self.has_z = any(symbol.name == 'z' for symbol in expression.free_symbols)

	def __call__(self, e, e2, v):
		return self.functions['h_val'](e, self.z)

	def __getstate__(self):
		return self.expression, self.z

	def __setstate__(self, state):
		self.expression, self.z = state
		self.functions = build_h_function(self.expression)


def apply_h_function(model, expression):
	h_function = SymbolicHFunction(expression, model.z)
	model.h = h_function
	model.h_val = h_function(model.e, model.e2, model.v)
	model.h_val_cpp = h_function.functions['cpp']
	for name in ('h_e_val', 'h_2e_val', 'h_z_val', 'h_2z_val', 'h_ez_val'):
		if name in ('h_z_val', 'h_2z_val', 'h_ez_val') and not h_function.has_z:
			value = None
		else:
			value = h_function.functions[name](model.e, model.z)
		setattr(model, name, value)