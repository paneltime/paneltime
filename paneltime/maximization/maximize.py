#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import init
from ..processing import arguments

import numpy as np
import time
import itertools
import os


EPS=3.0e-16 
TOLX=(4*EPS) 



def maximize(panel, args, mp, t0):	

	if mp is None or panel.args.initial_user_defined or mp.n_slaves== 1:
		res, a = maximize_single(panel, args)
	else:
		res, a = maximize_multiproc(panel, args, mp)
	f = [res[k]['f'] for k in res]
	r = res[list(res.keys())[f.index(max(f))]]

	if len(a)>0:
		res2 = np.array([[res[i]['f'], res[i]['x'][7], res[i]['x'][8], a[i][7], res[i]['its'], i] for i in res])
		print(np.round(res2[res2[:,0].argsort()], 4))

	return r

def maximize_single(panel, args, a = None):
	res = {}
	gtol = panel.options.tolerance
	if a is None:
		a = get_directions(args)
	for i,x in enumerate(a):
		d = maximize_node(panel, x, gtol, 0)    
		res[i] = d
	return res, a

def maximize_multiproc(panel, args, mp):
	tasks = []
	gtol = panel.options.tolerance
	a = get_directions(args, [0.3, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99])
	for i in range(len(a)):
			tasks.append(
			f'maximize.maximize_node(panel, {list(a[i])}, {gtol}, {i}, slave_server)\n')
	mp.eval(tasks)
	res = mp.collect(force_quit=True)
	return res, a

def get_directions(args, gammas = None):
		if len(args.positions['gamma'])==0 or gammas is None:
			return [np.array(args.args_v)]
		p = args.positions['gamma'][0]
		a = []
		
		for x in gammas:
			ai = np.array(args.args_v)
			ai[p] = x
			a.append(ai)
		
		return a


def maximize_node_new(panel, args, gtol = 1e-5, slave_id =0 , slave_server = None):
		constr = []

		res = init.maximize(args, panel, gtol, TOLX, slave_id, slave_server, False, constr)

		return res


#Need to implement this again
def maximize_node(panel, args, gtol = 1e-5, slave_id =0 , slave_server = None):
	res0 = init.maximize(args, panel, gtol, TOLX, slave_id, slave_server)
	res = init.maximize(res0['x'], panel, gtol, TOLX, slave_id, slave_server, grestricted=True)
	res['its'] += res0['its']
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