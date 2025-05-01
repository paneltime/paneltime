#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import init

import numpy as np
import time
import itertools
import os


EPS=3.0e-16 
TOLX=(4*EPS) 



def maximize(panel, args, mp, t0):


		gtol = panel.options.tolerance

		if mp is None or panel.args.initial_user_defined:
				d = maximize_node(panel, args.args_v, gtol, 0)    
				return d

		tasks = []
		a = get_directions(panel, args, mp.n_slaves)
		for i in range(len(a)):
				tasks.append(
				f'max.maximize_node(panel, {list(a[i])}, {gtol}, {i}, slave_server)\n'
																)
				
		mp.eval(tasks)
 
		r_base = maximize_node(panel, args.args_v, gtol, len(a))  
		res = mp.collect(True)
		res[len(a)] = r_base
		f = [res[k]['f'] for k in res]
		r = res[list(res.keys())[f.index(max(f))]]
		return r



def get_directions(panel, args, n):
		if n == 1:
				return [args.args_v]
		d = args.positions
		size = panel.options.initial_arima_garch_params
		pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
		perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
		z = np.nonzero(np.sum(perm**2,1)==0)[0][0]
		perm = perm[np.arange(len(perm))!=z]
		perm[:,:] =perm[:,:]*0.1
		perm = perm[:-1]
		a = np.array([args.args_v for i in range(len(perm))])
		a[:,pos] = perm
		return a


def maximize_node_new(panel, args, gtol = 1e-5, slave_id =0 , slave_server = None):
		constr = []

		res = init.maximize(args, panel, gtol, TOLX, slave_id, slave_server, False, constr)

		return res


#Need to implement this again
def maximize_node(panel, args, gtol = 1e-5, slave_id =0 , slave_server = None):
	res = init.maximize(args, panel, gtol, TOLX, slave_id, slave_server)
	return res
	# Possibly unneccessary code:
	constr = []
	while True:
		if not res['constr'].is_collinear:
			break
		f = res['constr'].fixed
		coll = []
		coll_preferred = []
		for k in f:
			if not k in constr:
				coll.append(k)
				if not k in panel.args.positions['beta']:#prefers not to constraint independents
					coll_preferred.append(k)
		if len(coll)==0:
			break
		if len(coll_preferred):
			coll=coll_preferred
		for c in ['rho', 'lambda','gamma','psi']:
			#avoding constraining first ARMA coefficient
			avoid_first_arma(coll, c, f, panel, constr)

		k = coll[np.argsort([f[k].ci for k in coll])[-1]]
		if not panel.args.names_v[k]=='initvar':
			args[k] = 0
		print(f'Added multicollinearity constraint for {panel.args.names_v[k]}')
		constr.append(k)

		#trying another time
		res = init.maximize(args, panel, gtol, TOLX, slave_id, slave_server, False, constr)
	return res
		

def avoid_first_arma(coll, c, f, panel, constr):
	"Ensures the first ARMA coefficient is not constrained, if possible"
	pos = panel.args.positions[c]
	if len(pos)>1:
		if pos[0] in coll:
			found = False
			for k in pos[1:]:
				if k in coll:
					found = True
					coll.pop(coll.index(pos[0]))
					break
			if not found:
				for j in pos[1:]: 
					if not j in constr:
						coll.pop(coll.index(pos[0]))
						coll.append(pos[1])
						f[pos[1]] = f[pos[0]]