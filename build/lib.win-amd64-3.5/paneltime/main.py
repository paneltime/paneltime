#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import regstats
import panel
import warnings
import multi_core as mc
import loaddata
import paneltime_functions as ptf
import maximize
import os
import loglikelihood as logl

warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)





def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, groups_name=None, sort_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=True,direction_testing=True,add_intercept=True,
            h=None,user_constraints=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""

	print ("Analyzing variables ...")
	X,x_names,Y,y_name,groups,groups_name,W,w_names,has_intercept=ptf.get_variables(dataframe,model_string,groups_name,w_names,add_intercept,sort_name)
	


	print ("Creating panel")
	pnl=panel.panel(p, d, q, m, k, X, Y, groups,x_names,y_name,groups_name,fixed_random_eff,W,w_names,descr,dataframe,h,has_intercept,loadargs,model_string,user_constraints)
	direction=logl.direction(pnl)
	 
	N,k=X.shape
	if (N*(k**0.5)>200000 and os.cpu_count()>=2) or os.cpu_count()>=24:#numpy all ready have multiprocessing, so there is no purpose unless you have a lot of processors or the dataset is very big
		mp=mc.multiprocess()
		mp.send_dict({'panel':pnl,'direction':direction},'static dictionary')		
	else:
		mp=None
	print ("Maximizing")

	ll,g,G,H, conv = maximize.maximize(pnl,direction,mp,direction_testing,_print=True,user_constraints=user_constraints)	

	return pnl,g,G,H,ll


