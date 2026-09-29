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
nothing is constrained because of it.
"""

import numbers
import numpy as np
from ..processing import arguments


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

	# ---------------------------------------------------- static constraints

	def add_static_constraints(self, comput, its=0, ll=None):
		"""ARMA/GARCH extreme bounds and user constraints."""
		panel = comput.panel
		c = self.ARMA_constraint
		bounds = [('rho', -c, c), ('lambda', -c, c),  ('psi', -c, c)]
		if comput.grestricted:	
			bounds.append(('gamma', -1e-12, c))
		else:
			bounds.append(('gamma', -c, c))

		if panel.options.include_initvar:
			bounds.append((arguments.INITVAR, 1e-50, 1e+10))
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