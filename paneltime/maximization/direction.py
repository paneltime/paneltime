#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Search-direction computation for the maximum-likelihood optimizer.

Conventions
-----------
The optimizer MAXIMIZES the log-likelihood (LL). g is the gradient and H the
Hessian of LL, so at a proper maximum H is negative definite and the Newton
direction is dx = -H^{-1} g. A direction is an ascent direction iff g @ dx > 0.

Handling of non-concavity
-------------------------
Where LL is not concave, H has positive eigenvalues and the plain Newton step
can point downhill or toward a saddle. With concave=True, H (or hessin) is
replaced by a negative definite matrix with the same eigenvectors and
eigenvalues -max(|lambda|, floor). Where H is already negative definite (in
particular near the optimum) nothing changes, so fast local convergence and
the Hessian used for standard errors are unaffected. In addition, in
unconstrained problems a step along the most convex direction is added
(signed uphill), which lets the optimizer escape saddle points where the
gradient is ~0.

Feasibility guarantee
---------------------
The step returned by new() always satisfies the interval constraints:
x + dx_alam lies within [min, max] for every interval constraint, equals the
value of every value constraint (to VALUE_TOL), and fixed variables do not
move. This holds whatever happens inside solve(), provided the returned step
is applied as x + dx_alam. It is enforced in three layers:

  1. solve(): components of active constraints are set exactly to their
     targets after the linear solve, so numerical error in the KKT solve
     cannot push them across a bound.
  2. new(): the constrained step is only accepted if it is both uphill and
     feasible; otherwise the largest feasible fraction of the uphill
     direction is used (fraction to boundary).
  3. new(): if even that is infeasible (e.g. x itself started outside a
     bound), the step is projected onto the feasible box.

get() returns an UNCONSTRAINED direction (used for convergence measures and
as input to new()); it must not be applied to x directly.
"""

import numpy as np
from .. import functions as fn

REL_EIG_FLOOR = 1e-8	# eigenvalue floor relative to the largest |eigenvalue|
NEG_CURV_TOL = 1e-6		# relative size of a positive eigenvalue that counts as convexity
VALUE_TOL = 1e-10		# relative tolerance for value (equality) constraints


# =============================================================== public API

def get(g, x, H, constr, f, hessin, simple=True, concave=True, neg_curv=True):
	"""Compute the (unconstrained w.r.t. intervals) search direction.

	Fixed variables are never moved. Returns (dx, dx_norm, H). H is returned
	UNMODIFIED so it can still be used for standard errors; the concavity fix
	is only used for the direction. Pass dx through new() before applying it."""
	g = _vec(g)
	if simple or H is None:
		Hi = hessin
		if concave:
			Hi, _ = make_concave(Hi)
		dx = _zero_fixed(constr, -np.dot(Hi, g))
	else:
		dx, _, _ = solve(constr, H, g, x, f, concave)
		if neg_curv and not has_constraints(constr):
			dx = add_negative_curvature(dx, g, H, x)
	return dx, normalize(dx, x), H


def new(g, x, H, constr, f, dx, alam, concave=True):
	"""Scale the direction by the line-search factor alam and enforce the
	constraints. Returns (dx_alam, slope, rev, applied_constraints).

	x + dx_alam is guaranteed to satisfy the interval and fixed constraints."""
	g = _vec(g)
	x = _vec(x)
	gz = _zero_fixed(constr, g)
	dx, slope, rev = slope_check(gz, _zero_fixed(constr, dx), H)

	# 1. The plain scaled step is feasible: use it.
	step = alam*dx
	if (not has_intervals(constr)) or within(constr, x + step):
		return step, slope, rev, []

	# 2. Constrained Newton step, accepted only if uphill AND feasible.
	if H is not None:
		dxalam, _, applied = solve(constr, H, g*alam, x, f, concave)
		if np.dot(g, dxalam) > 0.0 and within(constr, x + dxalam):
			return dxalam, slope, rev, applied
		rev = True

	# 3. Largest feasible fraction of the (uphill) unconstrained direction.
	dxalam = alam*fraction_to_boundary(constr, x, dx)*dx

	# 4. Last resort (e.g. x itself infeasible, or value constraints not yet
	#    met): project the step onto the feasible box.
	if not within(constr, x + dxalam):
		dxalam = project(constr, x, dxalam)
		rev = True
	return dxalam, slope, rev, []


def slope_check(g, dx, H=None):
	"""Ensure dx is an ascent direction. With concave=True this should
	practically never trigger; if it does, fall back to a diagonally scaled
	gradient step (always uphill) instead of just flipping the sign of dx."""
	slope = np.dot(g, dx)
	if slope > 0.0:
		return dx, slope, False
	if H is not None and np.all(np.isfinite(H)):
		d = g/np.maximum(np.abs(np.diag(H)), 1e-12)
	else:
		d = np.array(g, dtype=float)
	return d, np.dot(g, d), True


def solve(constr, H, g, x, f=None, concave=True):
	"""Solve the second-order Taylor expansion for the step dx with dLL/dx = 0,
	subject to fixed constraints (variables removed) and interval constraints
	(activated one by one through Kuhn-Tucker rows).

	Returns (dx, H, applied_constraints). H is returned unmodified. The
	components of active constraints are set exactly to their targets, but
	constraints that could not be activated (singular system) may remain
	violated; new() takes care of that."""
	if H is None:
		raise RuntimeError('Cannot solve with no coefficient matrix')
	g = _vec(g)
	x = _vec(x)
	H_orig = H
	H = np.asarray(H, dtype=float)

	if not has_constraints(constr):
		Hc = make_concave(H)[0] if concave else H
		return -fn.solve(Hc, g), H_orig, []

	m = len(g)
	free = np.ones(m, dtype=bool)
	free[list(getattr(constr, 'fixed', {}).keys())] = False
	pos = np.cumsum(free) - 1				# index of each free variable in the reduced system

	Hr = H[free][:, free]
	if concave:
		Hr, _ = make_concave(Hr)			# concavity on the free subspace
	n = len(Hr)
	intervals = getattr(constr, 'intervals', {})
	keys = [k for k in intervals if free[k]]
	K, b = _kkt_system(Hr, g[free], len(keys))
	d = -fn.solve(K, b)

	# Active-set loop: activate ONE constraint at a time, always the violated
	# constraint that is hit FIRST along the current step, then re-solve.
	# (Activating in key order can pin a variable at a bound the corrected
	# step would never reach, e.g. one of two collinear parameters at its max.)
	applied = []
	targets = {}							# key -> exact step for active constraints
	skipped = set()
	for _ in range(len(keys) + 1):
		if within(constr, x + _full_step(d, n, free, targets)):
			break
		first = None
		for j, key in enumerate(keys):
			if key in targets or key in skipped:
				continue
			c = intervals[key]
			i = pos[key]
			q = _violation(c, x[key], d[i])
			if q is None:
				continue
			t_hit = _hit_fraction(c, x[key], d[i])
			if first is None or t_hit < first[0]:
				first = (t_hit, j, key, i, q)
		if first is None:
			break
		_, j, key, i, q = first
		K_old, b_old = K.copy(), b.copy()
		K[i, n+j] = K[n+j, i] = 1.0
		K[n+j, n+j] = 0.0
		b[n+j] = q
		try:
			d_new = -fn.solve(K, b)
			if not np.all(np.isfinite(d_new)):
				raise np.linalg.LinAlgError('non-finite solution')
		except np.linalg.LinAlgError:
			K, b = K_old, b_old			# could not activate; skip this constraint
			skipped.add(key)
			continue
		d = d_new
		applied.append(key)
		targets[key] = -q

	return _full_step(d, n, free, targets), H_orig, applied


def within(constr, xn):
	"""True if xn satisfies all interval constraints (bounds inclusive,
	value constraints to VALUE_TOL)."""
	xn = _vec(xn)
	for key, c in getattr(constr, 'intervals', {}).items():
		v = xn[key]
		if not np.isfinite(v):
			return False
		if c.value is not None:
			if abs(v - c.value) > VALUE_TOL*max(1.0, abs(c.value)):
				return False
			continue
		if c.min is not None and v < c.min:
			return False
		if c.max is not None and v > c.max:
			return False
	return True


def project(constr, x, dx):
	"""Modify dx so that x + dx satisfies all constraints: fixed variables do
	not move, value constraints are met, and components crossing a bound are
	stopped a hair inside it. Other components are left unchanged."""
	x = _vec(x)
	dx = _zero_fixed(constr, dx)
	for key, c in getattr(constr, 'intervals', {}).items():
		xn = x[key] + dx[key]
		if c.value is not None:
			dx[key] = c.value - x[key]
		elif c.min is not None and not xn >= c.min:		# also catches NaN
			dx[key] = _inside(c.min, +1) - x[key]
		elif c.max is not None and not xn <= c.max:
			dx[key] = _inside(c.max, -1) - x[key]
	return dx


def projected_gradient(g, x, constr, tol=1e-10):
	"""Gradient with components removed where a variable sits at a bound and
	the gradient points out of the feasible region. Its norm is the correct
	(KKT) convergence measure with interval constraints; the norm of g itself
	never goes to zero when a constraint is active at the optimum."""
	pg = np.array(_vec(g))
	if constr is None:
		return pg
	for k in getattr(constr, 'fixed', {}):
		pg[k] = 0.0
	for k, c in getattr(constr, 'intervals', {}).items():
		if c.value is not None:
			pg[k] = 0.0
			continue
		if c.min is not None and x[k] <= c.min + tol and pg[k] < 0:
			pg[k] = 0.0
		if c.max is not None and x[k] >= c.max - tol and pg[k] > 0:
			pg[k] = 0.0
	return pg


def normalize(dx, x):
	"""Relative step size (absolute for |x| < 1e-2)."""
	ax = np.abs(x)
	rel = np.where(ax >= 1e-2, dx/np.where(ax > 0, ax, 1.0), dx)
	return rel


# ======================================================= concavity handling

def make_concave(H, rel_floor=REL_EIG_FLOOR):
	"""Return (H_mod, modified): a negative definite version of symmetric H.

	Eigenvalues lambda are replaced by -max(|lambda|, floor), keeping the
	eigenvectors. Directions of positive curvature are thereby treated as
	concave with the same curvature magnitude, which gives an ascent step
	that is well scaled. Works equally for the Hessian and its inverse."""
	if H is None:
		return H, False
	H = np.asarray(H, dtype=float)
	if H.size == 0 or not np.all(np.isfinite(H)):
		return H, False
	Hs = 0.5*(H + H.T)
	w, V = np.linalg.eigh(Hs)
	scale = np.max(np.abs(w))
	if scale == 0.0:
		return -np.eye(len(H)), True
	floor = rel_floor*scale
	if w[-1] <= -floor:						# already negative definite
		return Hs, False
	w_mod = -np.maximum(np.abs(w), floor)
	return (V*w_mod) @ V.T, True


def add_negative_curvature(dx, g, H, x, rel_tol=NEG_CURV_TOL):
	"""If LL is convex along some direction (H has a positive eigenvalue),
	make sure the step moves a minimum distance along the most convex
	direction v, signed uphill. Along v both slope and curvature increase LL,
	so this escapes saddles where g @ v ~ 0 and the modified Newton step
	would be tiny. The final length is left to the line search."""
	H = np.asarray(H, dtype=float)
	if not np.all(np.isfinite(H)):
		return dx
	w, V = np.linalg.eigh(0.5*(H + H.T))
	scale = np.max(np.abs(w))
	if scale == 0.0 or w[-1] <= rel_tol*scale:
		return dx
	v = V[:, -1]
	s = np.sign(np.dot(g, v)) or 1.0
	target = 0.5*max(np.linalg.norm(dx), 1e-2*max(1.0, np.linalg.norm(x)))
	c = s*np.dot(v, dx)						# current uphill movement along v
	if c < target:
		dx = dx + (target - c)*s*v
	return dx


# =================================================================== helpers

def has_constraints(constr):
	try:
		return len(list(constr.keys())) > 0
	except (AttributeError, TypeError):
		return False


def has_intervals(constr):
	return constr is not None and len(getattr(constr, 'intervals', {})) > 0


def fraction_to_boundary(constr, x, dx, tau=0.99):
	"""Largest t in [0, 1] (times tau) such that x + t*dx respects the bounds,
	assuming x itself is feasible. Value constraints are not handled here
	(a scalar t cannot enforce them); new() projects afterwards if needed."""
	x = _vec(x)
	t = 1.0
	for key, c in getattr(constr, 'intervals', {}).items():
		if c.value is not None or dx[key] == 0:
			continue
		xn = x[key] + dx[key]
		if c.min is not None and xn < c.min:
			t = min(t, tau*(c.min - x[key])/dx[key])
		if c.max is not None and xn > c.max:
			t = min(t, tau*(c.max - x[key])/dx[key])
	return max(t, 0.0)


def _violation(c, xk, dxk):
	"""Kuhn-Tucker target q (the step becomes dx_k = -q), or None if inactive."""
	if c.value is not None:
		return xk - c.value
	if c.min is not None and xk + dxk < c.min:
		return xk - _inside(c.min, +1)
	if c.max is not None and xk + dxk > c.max:
		return xk - _inside(c.max, -1)
	return None


def _hit_fraction(c, xk, dxk):
	"""Fraction t of the step dxk at which variable k reaches its bound."""
	if c.value is not None or dxk == 0:
		return 0.0
	if dxk < 0 and c.min is not None:
		return max((c.min - xk)/dxk, 0.0)
	if dxk > 0 and c.max is not None:
		return max((c.max - xk)/dxk, 0.0)
	return np.inf


def _inside(bound, sign, rel=1e-12):
	"""Target a hair inside the bound, so rounding in x + dx cannot violate it."""
	return bound + sign*rel*max(1.0, abs(bound))


def _kkt_system(Hr, gr, k):
	"""Hessian and gradient enlarged with k (initially inactive) slack variables."""
	n = len(Hr)
	K = np.zeros((n + k, n + k))
	K[:n, :n] = Hr
	K[n:, n:] = np.eye(k)
	b = np.concatenate([gr, np.zeros(k)])
	return K, b


def _full_step(d, n, free, targets):
	"""Full-length step from the reduced solution, with active constraints
	set exactly to their targets (immune to error in the linear solve)."""
	full = _expand(d[:n], free)
	for key, t in targets.items():
		full[key] = t
	return full


def _expand(d, free):
	full = np.zeros(len(free))
	full[free] = d
	return full


def _zero_fixed(constr, v):
	v = np.array(_vec(v))
	if constr is not None:
		for k in getattr(constr, 'fixed', {}):
			v[k] = 0.0
	return v


def _vec(g):
	return np.asarray(g, dtype=float).ravel()