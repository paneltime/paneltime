#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import dfpmax
from . import computation
#from ..likelihood_simple import dfpmax as dfpmax_smpl
from .. import likelihood as logl


import numpy as np


def maximize(args, panel, gtol, tolx, slave_id, slave_server):
	args = np.array(args)
	comput = computation.Computation(args, panel, gtol, tolx)

	initval = InitialValues(panel, comput)
	
	x, ll, f, g, hessin, H = initval.calc_init_dir(args, panel)

	armaconstr = panel.options.ARMA_constraint

	res = dfpmax.dfpmax(x, f, g, hessin, H, comput, panel, slave_id, ll, armaconstr, slave_server)
	if res['conv']==6:
		armaconstr = 0.9
		print(f"Overflow in dfpmax. Maximum absolute value for ARMA/GARCH coefficients set to {armaconstr}")
		res = dfpmax.dfpmax(x, f, g, hessin, H, comput, panel, slave_id, ll, armaconstr, slave_server)
	
	res['node'] = slave_id
	return res


class InitialValues:
	def __init__(self, panel, comput):
		self.comput = comput
		self.panel = panel
		

	def init_ll(self,args=None, ll=None):
		if args is None:
			args = ll.args.args_v
		
		try:
			args=args.args_v
		except:
			pass#args must be a vector
		for i in self.comput.constr.fixed:
			args[i]=self.comput.constr.fixed[i].value
		if ll is None:
			ll=logl.LL(args, self.panel, constraints=self.comput.constr,print_err=True)
		if ll.LL is None:
			print("WARNING: Initial arguments failed, attempting default OLS-arguments ...")
			self.panel.args.set_init_args(self.panel,default=True)
			ll=logl.LL(self.panel.args.args_OLS,self.panel,constraints=self.constr,print_err=True)
			if ll.LL is None:
				raise RuntimeError("OLS-arguments failed too, you should check the data")
			else:
				print("default OLS-arguments worked")
		return ll
	
	def calc_init_dir(self, p0, panel):
		"""Calculates the initial computation""" 
		ll = self.init_ll(p0)
		g, G = self.comput.calc_gradient(ll)
		if sum(np.isnan(g)):
			a=0
		if self.panel.options.use_analytical==0:
			H = -np.identity(len(g))
			hessin = H
			return p0, ll, ll.LL , g, hessin, H
		H = self.comput.calc_hessian(ll)
		try:
			hessin = np.linalg.inv(H)
		except np.linalg.LinAlgError:
			hessin = -np.identity(len(g))*panel.args.init_var
		return p0, ll, ll.LL , g, hessin, H