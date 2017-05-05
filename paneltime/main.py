#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import regstats
import regobj
import warnings
import multi_core
import loaddata
import paneltime_functions as ptf
import maximize
import os
warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)





def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, groups_name=None, sort_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=True,add_intercept=True,
            h=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""

	r=[]#restrictions


	print ("Analyzing variables ...")
	X,x_names,Y,y_name,groups,groups_name,W,w_names,has_intercept=ptf.get_variables(dataframe,model_string,groups_name,w_names,add_intercept,sort_name)
	N,cols=X.shape
	if N*(cols**0.5)>200000 and os.cpu_count()>1:#paralell computing will not increase computation time for 'small' data sets
		master=multi_core.master('regprocs','rp')#for paralell computing
	else:
		master=None

	print ("Creating panel")
	panel=regobj.panel(p, d, q, m, k, X, Y, groups,x_names,y_name,groups_name,fixed_random_eff,W,w_names,master,descr,dataframe,h,has_intercept,loadargs)
	print ("Maximizing")

	ll,g,G,H, conv = maximize.maximize(panel,_print=True)	
	panel.args_bank.save(ll.args_d, conv)

	return panel,g,G,H,ll


