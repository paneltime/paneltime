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
	e = e_symbol if e_symbol is not None else free_symbols.get('e', sp.Symbol('e', real=True))
	z = z_symbol if z_symbol is not None else free_symbols.get('z', sp.Symbol('z', real=True))
	unsupported = set(expr.free_symbols) - {e, z}
	if unsupported:
		names = ', '.join(sorted(symbol.name for symbol in unsupported))
		raise ValueError(f'h_function may only use e and z; unsupported symbols: {names}')
	# Residuals and shape parameters are real even when symbols have no assumptions.
	real_e, real_z = sp.symbols('e z', real=True)
	expr = expr.xreplace({e: real_e, z: real_z})
	e, z = real_e, real_z
	derivatives = {
		'h_val': expr, 'h_e_val': sp.diff(expr, e),
		'h_2e_val': sp.diff(expr, e, 2), 'h_z_val': sp.diff(expr, z),
		'h_2z_val': sp.diff(expr, z, 2), 'h_ez_val': sp.diff(expr, e, z)}
	for name, value in derivatives.items():
		if value.has(sp.DiracDelta, sp.Derivative):
			raise ValueError(
				f'h_function has an unsupported derivative ({name}); '
				'use a smooth expression, for example sqrt(e**2 + epsilon) instead of Abs(e)'
			)
	functions = {name: sp.lambdify((e, z), value, 'numpy') for name, value in derivatives.items()}
	cpp = sp.sstr(expr).replace('**', '^')
	return {'expression': expr, 'cpp': cpp, **functions}


class SymbolicHFunction:
	"""Callable adapter for the likelihood model's ``h(e, e2, v)`` interface."""

	def __init__(self, expression, z=None):
		self.expression = expression
		self.z = None if z is None else float(np.asarray(z).item())
		self.functions = build_h_function(expression)
		self.has_z = any(symbol.name == 'z' for symbol in expression.free_symbols)

	def __call__(self, e, e2, v):
		return self.functions['h_val'](e, self.z)

	def __getstate__(self):
		return self.expression, self.z

	def __setstate__(self, state):
		self.expression, self.z = state
		self.functions = build_h_function(self.expression)
		self.has_z = any(symbol.name == 'z' for symbol in self.expression.free_symbols)


def apply_h_function(model, expression):
	h_function = SymbolicHFunction(expression, model.z)
	model.h = h_function
	model.h_val = _broadcast_value(h_function(model.e, model.e2, model.v), model.e)
	model.h_val_cpp = h_function.functions['cpp']
	for name in ('h_e_val', 'h_2e_val', 'h_z_val', 'h_2z_val', 'h_ez_val'):
		if name in ('h_z_val', 'h_2z_val', 'h_ez_val') and not h_function.has_z:
			value = None
		else:
			value = _broadcast_value(h_function.functions[name](model.e, h_function.z), model.e)
		setattr(model, name, value)


def _broadcast_value(value, e):
	"""Constant symbolic derivatives must have the residual array's shape."""
	result = np.array(np.broadcast_to(value, np.shape(e)), dtype=float, copy=True)
	return result.item() if result.ndim == 0 else result