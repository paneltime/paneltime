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
	e = e_symbol or sp.Symbol('e')
	z = z_symbol or sp.Symbol('z')
	if not isinstance(expression, sp.Expr):
		raise TypeError('h_function must be a SymPy expression')
	expr = expression
	derivatives = {
		'h_val': expr, 'h_e_val': sp.diff(expr, e),
		'h_2e_val': sp.diff(expr, e, 2), 'h_z_val': sp.diff(expr, z),
		'h_2z_val': sp.diff(expr, z, 2), 'h_ez_val': sp.diff(expr, e, z)}
	functions = {name: sp.lambdify((e, z), value, 'numpy') for name, value in derivatives.items()}
	cpp = sp.sstr(expr).replace('**', '^')
	return {'expression': expr, 'cpp': cpp, **functions}