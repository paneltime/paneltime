#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constraints for the LL maximization.

Two kinds of constraints are supported:
  fixed     : the parameter is held at a value (removed from the Newton system)
  intervals : the parameter must stay within [min, max]; either bound may be
              None (one-sided), stored as -inf/+inf

Only static constraints are used: the ARMA/GARCH extreme bounds and user
constraints. The dynamic (multicollinearity based) constraining has been
removed; weakly identified directions are handled by the direction module
(concavity fix and active-set solve) instead of by fixing parameters.
Multicollinearity is still diagnosed and reported (multicoll_report), but
nothing is constrained because of it. The exception is a common AR/MA factor
(arma_cancellation), where the highest AR and MA lags are fixed at zero.
"""

import numbers
import numpy as np

ARMA_CANCELLATION = 'AR/MA cancellation'
# Wald statistic of rho + lambda below which the AR and MA factors are treated
# as cancelling.
ARMA_MIN_WALD = 3.0


def arma_resultant(rho, lmbda):
	"""Resultant of z^p - sum(rho_j z^(p-j)) and z^q + sum(lambda_j z^(q-j)),
	the Sylvester determinant; zero iff the AR and MA polynomials share a root."""
	a = np.concatenate(([1.0], -np.asarray(rho, dtype=float)))
	b = np.concatenate(([1.0], np.asarray(lmbda, dtype=float)))
	p, q = len(a) - 1, len(b) - 1
	S = np.zeros((p + q, p + q))
	for i in range(q):
		S[i, i:i + p + 1] = a
	for i in range(p):
		S[q + i, i:i + q + 1] = b
	return np.linalg.det(S)


class Constraint:
	def __init__(self, index, assco, cause, value, interval, names, category, ci=0):
		self.name = names[index]
		self.index = index
		self.cause = cause
		self.category = category
		self.ci = ci
		self.intervalbound = None
		self.value = None
		self.value_str = None
		self.min = None
		self.max = None
		if interval is None:
			self.value = value
			self.value_str = str(round(value, 8))
		else:
			lo, hi = interval
			self.min = -np.inf if lo is None else lo
			self.max = np.inf if hi is None else hi
			if self.min > self.max:
				raise ValueError(f"Lower constraint exceeds upper for {self.name}: {interval}")
		self.assco_ix = assco
		self.assco_name = None if assco is None else names[assco]


class Constraints(dict):
	"""Stores the constraints of the LL maximization, keyed by parameter index."""

	def __init__(self, panel, args, its, armaconstr):
		dict.__init__(self)
		self.categories = {}
		self.fixed = {}
		self.intervals = {}
		self.associates = {}
		self.args = args
		self.panel_args = panel.args
		self.its = its
		self.pqdkm = panel.pqdkm
		self.m_zero = panel.m_zero
		self.ARMA_constraint = armaconstr
		self.GARCH_min = panel.options.GARCH_min

		# Multicollinearity diagnostics, set by multicoll_report (report only)
		self.ci = None
		self.ci_n = 0
		self.mc_report = {}
		self.mc_details = []


	# ------------------------------------------------------------ add/delete

	def add(self, name, assco, cause, interval=None, replace=True, value=None, ci=0):
		"""Adds constraints for all parameters matching `name` (a name, group
		or index). Fixed if interval is None, otherwise an interval constraint."""
		name, index = self.panel_args.get_name_ix(name)
		name_assco, assco = self.panel_args.get_name_ix(assco, True)
		for i in index:
			self.add_item(i, assco, cause, interval, replace, value, ci)

	def add_item(self, index, assco, cause, interval, replace, value, ci=0):
		"""Adds a constraint at parameter position `index`.

		interval None    -> fixed constraint at `value` (default: current args)
		interval (lo,hi) -> interval constraint; lo or hi may be None
		replace=False    -> an existing constraint at `index` is kept

		A fixed value outside an existing interval is rejected, and not all
		parameters can be fixed. Returns True if the constraint was added."""
		existing = self.get(index)
		if existing is not None and not replace:
			return False

		if interval is None:
			if value is None:
				value = self.args[index]
			if existing is not None and existing.value is None:
				if not (existing.min <= value <= existing.max):
					return False
			n_fixed_other = len(self.fixed) - (index in self.fixed)
			if n_fixed_other >= len(self.panel_args.caption_v) - 1:
				return False							# can't fix all variables

		if existing is not None:
			self.delete(index)

		_, category, _ = self.panel_args.positions_map[index]
		c = Constraint(index, assco, cause, value, interval,
					   self.panel_args.caption_v, category, ci)
		self[index] = c
		self.categories.setdefault(category, []).append(index)
		if interval is None:
			self.fixed[index] = c
		else:
			self.intervals[index] = c
		if assco is not None:
			lst = self.associates.setdefault(assco, [])
			if index not in lst:
				lst.append(index)
		return True

	def delete(self, index):
		if index not in self:
			return False
		self.pop(index)
		self.intervals.pop(index, None)
		self.fixed.pop(index, None)

		_, category, _ = self.panel_args.positions_map[index]
		cat = self.categories.get(category, [])
		if index in cat:
			cat.remove(index)
		if not cat:
			self.categories.pop(category, None)

		for a in list(self.associates):
			if index in self.associates[a]:
				self.associates[a].remove(index)
			if not self.associates[a]:
				self.associates.pop(a)
		return True

	def clear(self, cause=None):
		for i in list(self.keys()):
			if cause is None or self[i].cause == cause:
				self.delete(i)

	# --------------------------------------------------------------- queries

	def set_fixed(self, x):
		"""Sets all elements of x that have fixed constraints to their values."""
		for i, c in self.fixed.items():
			x[i] = c.value

	def within(self, x, fix=False):
		"""Returns True if x satisfies all interval constraints.

		fix=False: returns False at the first violation (x is not changed).
		fix=True : violating elements of x are moved to the nearest bound (in
		           place) and True is returned."""
		for i, c in self.intervals.items():
			if c.min <= x[i] <= c.max:
				c.intervalbound = None
			elif fix:
				x[i] = min(max(x[i], c.min), c.max)
				c.intervalbound = str(round(x[i], 8))
			else:
				return False
		return True

	# ------------------------------------------------ multicollinearity report

	def multicoll_report(self, H, limit):
		"""Belsley collinearity diagnostics on the information matrix -H of the
		non-fixed parameters. Reports only; nothing is constrained.

		-H is scaled to unit diagonal and eigendecomposed, -H_s = V L V'.
		Condition index of dimension k: sqrt(l_max/l_k). Variance-decomposition
		proportion of parameter j in dimension k: (v_jk^2/l_k)/sum_k(v_jk^2/l_k),
		i.e. the share of the parameter's variance due to that dimension.

		Sets
		  ci         : largest condition index
		  ci_n       : number of parameters with proportion > 0.5 in that dimension
		  mc_report  : {index: associate} for each dimension with condition index
		               >= limit and at least two parameters with proportion > 0.5
		               (largest proportion -> second largest)
		  mc_details : [(condition index, [(index, name, proportion), ...]), ...]
		               for the same dimensions, largest condition index first"""
		self.ci, self.ci_n, self.mc_report, self.mc_details = 0.0, 0, {}, []
		if H is None:
			return
		H = np.asarray(H, dtype=float)
		incl = np.ones(len(H), dtype=bool)
		incl[list(self.fixed)] = False
		idx = np.flatnonzero(incl)
		if len(idx) < 2:
			return
		C = -H[np.ix_(idx, idx)]
		if not np.all(np.isfinite(C)):
			return
		d = np.sqrt(np.maximum(np.abs(np.diag(C)), 1e-300))
		C = 0.5*(C + C.T)/np.outer(d, d)
		lam, V = np.linalg.eigh(C)
		lam = np.abs(lam)						# sign problems are handled elsewhere
		lam_max = lam.max()
		if lam_max == 0:
			return
		lam = np.maximum(lam, lam_max*1e-30)	# singular -> condition index ~1e15
		cond = np.sqrt(lam_max/lam)
		phi = V**2/lam							# phi[j, k] = v_jk^2/l_k
		prop = phi/phi.sum(axis=1, keepdims=True)

		order = np.argsort(cond)[::-1]
		self.ci = float(cond[order[0]])
		self.ci_n = int(np.sum(prop[:, order[0]] > 0.5))
		names = self.panel_args.caption_v
		for k in order:
			if cond[k] < limit:
				break
			p = prop[:, k]
			if np.sum(p > 0.5) < 2:
				continue
			top = np.argsort(p)[::-1]
			self.mc_report[int(idx[top[0]])] = int(idx[top[1]])
			self.mc_details.append((float(cond[k]),
				[(int(idx[j]), names[idx[j]], float(p[j])) for j in top if p[j] > 0.5]))

	# ------------------------------------------------- AR/MA cancellation

	def arma_cancellation(self, x, H, min_wald=ARMA_MIN_WALD):
		"""Positions of the highest AR and MA lags if a common AR/MA factor is
		detected, otherwise [].

		The AR and MA polynomials share a factor, which then cancels and leaves
		the likelihood flat along a ridge, exactly when their resultant is zero.
		The resultant of the inverse-root polynomials
		  z^p - rho_1 z^(p-1) - ... - rho_p  and  z^q + lambda_1 z^(q-1) + ... + lambda_q
		is used (for ARMA(1,1) it is rho + lambda). The factor is treated as
		cancelling if the Wald statistic of the resultant (delta method, with
		the inverse of -H over the free mean-equation parameters) is below min_wald. Under the
		null the factor's root is not identified, so the statistic is not
		standard normal and min_wald is set above the usual 1.96.
		Removing a common factor turns ARMA(p,q) into ARMA(p-1,q-1), so the
		highest AR and MA lags are returned, to be fixed at zero. Lags already
		fixed at zero are trimmed first; if the highest remaining lag of either
		polynomial is fixed, nothing is reduced."""
		positions = self.panel_args.positions
		rho = self._trim_zero_lags(list(positions.get('rho', [])), x)
		lam = self._trim_zero_lags(list(positions.get('lambda', [])), x)
		if not rho or not lam or rho[-1] in self.fixed or lam[-1] in self.fixed:
			return []
		# Mean-equation block only: the information matrix is asymptotically
		# block diagonal between mean and variance parameters, and a weakly
		# identified GARCH block would otherwise inflate the variance.
		mean = [k for g in ('beta', 'rho', 'lambda') for k in positions.get(g, [])]
		free = [k for k in mean if k not in self.fixed]
		try:
			cov = np.linalg.inv(-np.asarray(H)[np.ix_(free, free)])
		except np.linalg.LinAlgError:
			return []
		idx = rho + lam
		c = np.asarray(x, dtype=float)[idx]
		p = len(rho)
		res = lambda c: arma_resultant(c[:p], c[p:])
		r = res(c)
		grad = np.zeros(len(free))
		h = 1e-6
		for n, k in enumerate(idx):
			if k not in self.fixed:
				e = np.zeros(len(c))
				e[n] = h
				grad[free.index(k)] = (res(c + e) - res(c - e))/(2*h)
		var = grad @ cov @ grad
		if not np.isfinite(var) or var <= 0:
			return []
		if abs(r)/np.sqrt(var) < min_wald:
			return [rho[-1], lam[-1]]
		return []

	def _trim_zero_lags(self, lags, x):
		while lags and lags[-1] in self.fixed and x[lags[-1]] == 0:
			lags.pop()
		return lags

	# ---------------------------------------------------- static constraints

	def add_static_constraints(self, comput, its=0, ll=None):
		"""ARMA/GARCH extreme bounds and user constraints."""
		panel = comput.panel
		c = self.ARMA_constraint
		bounds = [('rho', -c, c), ('lambda', -c, c)]
		if comput.grestricted:	
			bounds = bounds + [('gamma', -1e-12, c), ('psi', -1e-12, c)]
		else:
			bounds = bounds + [('gamma', -c, c), ('psi', -c, c)]

		for name, lo, hi in bounds:
			self.add(name, None, 'ARMA/GARCH extreme bounds', [lo, hi])
		self.add_custom_constraints(panel, self.panel_args.user_constraints, True, 'user constraints')
		self.set_init_constr(its, panel)

	def add_custom_constraints(self, panel, constraints, replace, cause):
		"""Adds user constraints. For each group:
		  tuple (min, max) -> interval on the whole group (a bound may be None)
		  number           -> the whole group fixed at that value
		  list             -> one element per parameter in the group, each a
		                      tuple, a number, a one-element list or None"""
		for grp, c in constraints.items():
			if c is None:
				continue
			elif isinstance(c, tuple):
				self.add(grp, None, cause, c, replace)
			elif _is_number(c):
				self.add(grp, None, cause, replace=replace, value=float(c))
			else:
				for i, name in enumerate(panel.args.caption_d[grp]):
					self.add_custom_constraint_subgroup(c, i, name, replace, cause, grp)

	def add_custom_constraint_subgroup(self, constraints, i, name, replace, cause, grp):
		if len(constraints) <= i or constraints[i] is None:
			return
		ci = constraints[i]
		if isinstance(ci, tuple):
			self.add(name, None, cause, ci, replace)
		elif _is_number(ci):
			self.add(name, None, cause, value=float(ci), replace=replace)
		elif isinstance(ci, list) and len(ci) == 1 and _is_number(ci[0]):
			self.add(name, None, cause, value=float(ci[0]), replace=replace)
		else:
			raise RuntimeError(f"When using the constraints option, the elements of {grp} "
							   "must be a tuple (min, max), a number, or a one-element "
							   "list with a number")

	def set_init_constr(self, its, panel):
		return
		p, q, d, k, m = panel.pqdkm

		if its*(k>1)>7 or its==0:
			return
		
		
		constr = [f'gamma{i}' for i in range(1, k)] + ['omega', 'psi']
		
		for name in constr:
			self.add(name, None,'user constraint')

		a =0

	def __str__(self):
		s = ''
		for desc, obj in [('All', self), ('Fixed', self.fixed), ('Intervals', self.intervals)]:
			s += f"{desc} constraints:\n"
			for i, c in obj.items():
				s += (f"constraint: {i}, associate:{c.assco_ix}, max:{c.max}, "
					  f"min:{c.min}, value:{c.value}, cause:{c.cause}\n")
		return s


def _is_number(v):
	return isinstance(v, numbers.Real) and not isinstance(v, bool)