#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtracking line search for MAXIMIZING the log-likelihood.

Armijo (sufficient increase) condition with quadratic/cubic interpolation,
after Numerical Recipes' lnsrch.

Consistency with direction.py
-----------------------------
The search direction p is computed ONCE by direction.new (with alam = step),
which applies the concavity-safe direction, the ascent check and the interval
constraints. Trial points are then x + t*p for t in (0, 1]. Since the feasible
set (box constraints) is convex and x is feasible, every trial point is
feasible, the step is exactly linear in t, and the directional derivative used
in the Armijo test and in the interpolation is the true slope g @ p of the step
actually taken.

Return codes (self.conv)
  1  sufficient increase
  2  no sufficient increase within max_iter (best point found is returned)
  3  step became negligible (best point found, or x, is returned)
  4  no usable step: zero/non-ascent direction or undefined likelihood
"""

import numpy as np
from .. import likelihood as logl
from . import direction

STPMX = 100.0


class LineSearch:
	def __init__(self, x, comput, panel, ll_old, step=1.0):
		self.alf = 1.0e-3			# Armijo constant (sufficient increase)
		self.tolx = 1.0e-14			# convergence criterion on relative step
		self.max_iter = 100			# backtracking iterations
		self.max_invalid = 40		# halvings allowed to leave an undefined region
		self.adaptive_step = False	# step_adj was disabled in the original; kept optional
		self.step = step
		self.stpmax = STPMX*max(np.sqrt(np.sum(np.asarray(x, dtype=float)**2)), len(x))
		self.comput = comput
		self.panel = panel
		self.ll_old = ll_old
		self.applied_constraints = []
		self.rev = False
		self.conv = 0
		self.msg = ""
		self.k = 0

	def lnsrch(self, x, f, g, H, dx):
		if f is None:
			raise RuntimeError('f cannot be None')
		x = np.asarray(x, dtype=float)
		g = np.asarray(g, dtype=float).ravel()
		dx = np.asarray(dx, dtype=float).ravel()

		self.conv = 0
		self.rev = False
		self.g = g
		self.msg = ""
		self.k = 0
		self.applied_constraints = []

		if not np.all(np.isfinite(dx)):
			return self.default(f, x, 0, "dx is not finite", 4)
		norm = np.linalg.norm(dx)
		if norm == 0:
			return self.default(f, x, 0, "dx is zero", 4)
		if norm > self.stpmax:
			dx = dx*self.stpmax/norm

		# The full, constrained, ascent-checked step. All trials lie on x + t*p.
		p, _, self.rev, self.applied_constraints = direction.new(
			g, x, H, self.comput.constr, f, dx, self.step)
		p = np.asarray(p, dtype=float).ravel()
		slope = np.dot(g, p)					# d LL(x + t*p)/dt at t = 0
		if not np.any(p):
			return self.default(f, x, 0, "Constrained step is zero", 4)
		if not slope > 0:
			return self.default(f, x, 0, "Step is not an ascent direction", 4)

		tmin = self.tolx/np.max(np.abs(p)/np.maximum(np.abs(x), 1.0))

		# Largest t (from the full step) where the likelihood is defined
		t = 1.0
		for _ in range(self.max_invalid):
			ft, llt = self.func(x + t*p)
			if ft is not None:
				break
			t *= 0.5
		else:
			return self.default(f, x, 0, "Likelihood undefined along the search direction", 4)

		best = [f, x, self.ll_old, 0.0]
		t2 = f2 = None
		for self.k in range(self.max_iter):
			if ft is not None:
				if ft > best[0]:
					best = [ft, x + t*p, llt, t]
				if ft >= f + self.alf*t*slope:
					self._set(ft, x + t*p, llt, t, "Sufficient function increase", 1)
					self.step_adj()
					return
			if t < tmin:
				return self._finish(best, "Convergence on delta dx", 3)

			tmp = self._backtrack(t, ft, t2, f2, f, slope)
			if ft is not None:
				t2, f2 = t, ft
			t = max(tmp, 0.1*t)					# lambda >= 0.1*lambda1
			ft, llt = self.func(x + t*p)

		self._finish(best, f"No sufficient function increase after {self.max_iter} iterations", 2)

	@staticmethod
	def _backtrack(t, ft, t2, f2, f, slope):
		"""Next trial t from a quadratic (first backtrack) or cubic model of
		phi(t) = LL(x + t*p), with phi(0) = f and phi'(0) = slope > 0."""
		if ft is None:							# undefined point: just halve
			return 0.5*t
		if t2 is None:							# quadratic through phi(0), phi'(0), phi(t)
			tmp = -slope*t*t/(2.0*(ft - f - slope*t))
		else:									# cubic through the two last points
			rhs1 = ft - f - t*slope
			rhs2 = f2 - f - t2*slope
			a = (rhs1/t**2 - rhs2/t2**2)/(t - t2)
			b = (-t2*rhs1/t**2 + t*rhs2/t2**2)/(t - t2)
			if a == 0.0:
				tmp = -slope/(2.0*b)
			else:
				disc = b*b - 3.0*a*slope
				if disc < 0.0:
					tmp = 0.5*t
				elif b >= 0.0:
					tmp = -(b + np.sqrt(disc))/(3.0*a)
				else:
					tmp = slope/(-b + np.sqrt(disc))
		if not (np.isfinite(tmp) and tmp > 0):
			tmp = 0.5*t
		return min(tmp, 0.5*t)					# lambda <= 0.5*lambda1

	def step_adj(self):
		if not self.adaptive_step:
			return
		if self.alam == self.step:
			self.step += self.step
		elif self.step > 1:
			self.step = 0.5*self.step if self.step > 2.0 else 1.0

	def func(self, x):
		ll = logl.LL(x, self.panel, constraints=self.comput.constr)
		if ll is None or ll.LL is None or not np.isfinite(ll.LL):
			return None, None
		return ll.LL, ll

	def _set(self, f, x, ll, t, msg, conv):
		self.f, self.x, self.ll = f, x, ll
		self.alam = t*self.step					# step length relative to dx
		self.msg, self.conv = msg, conv

	def _finish(self, best, msg, conv):
		"""Terminate without Armijo success: keep the best point seen, if any
		improved on the start, else stay at x."""
		f, x, ll, t = best
		self._set(f, x, ll, t, msg, conv)

	def default(self, f, x, alam, msg, conv):
		self.msg = msg
		self.conv = conv
		self.f = f
		self.x = x
		self.alam = alam
		self.ll = self.ll_old
