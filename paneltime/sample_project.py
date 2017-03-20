#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import paneltime
import numpy as np


filters="""
    nTrades>50 and MeanRelSpread/RelTickSize>1 and MeanRelSpread/RelTickSize<50
    """ 
transforms="""
         def Ln(x,subtractmean=False):
             "Calculates the natural log of an numpy array, but cencors zero" 
             try:
                 res=np.log(x+1e-15)
             except RuntimeWarning:
                 errrow=np.nonzero(x<0)[0][0]
                 print (x[max(0,errrow-20):min(len(x),errrow+20)])
                 raise RuntimeError('Log of negative number attempted in row %s' %(errrow,))
             if subtractmean:
                 res=res-np.mean(res)
             return res

         tz_type1=(tz_type==1)
         tz_type2=(tz_type==2)
         Turnover=SumTrades*avgPrice
         LNRange=Ln(HauVol,False)
         LNTurnover=Ln(Turnover,False)
         LNnTrades=Ln(nTrades,False)     
         LNMeanRelSpread=Ln(MeanRelSpread,False)               
         LNRelTick=Ln(RelTickSize,False)
         LNMeanNticks=np.maximum(LNMeanRelSpread-LNRelTick,0)
         DayRet=DayChange


         LNBestOBVol=Ln(bestOBVol*avgPrice,False)
         LNTotOBVol=Ln(orderBookVol*avgPrice,False)

         #fu.SaveVar(fu.Concat((MeanRelSpread,RelTickSize)))

         LNBindFreq=Ln(BindFreq,False)
         LNmaxLevels=Ln(maxLevels,False)
         LNMktShare=Ln(MktShare,False)
         MktShare0=MktShare<0.01
         BindFreq0=BindFreq<0.01
         BindFreq1=BindFreq>0.99	
         LNAvgQDiff=Ln(AvgQDiff,False)

    """

model_string="""LNRange ~ 
            Constant
            LNRelTick
            LNRelTick*LNTurnover
            LNTurnover
            LNMeanNticks
            LNBindFreq
            BindFreq0
            BindFreq1		    
            LNMktShare
            MktShare0
            LNmaxLevels
            DayRet	
                """
h="""def h(e,z):
        e2=e**2+z
        i=np.abs(e2)<1e-100
        h_val		=	 np.log(e2+i)	
        h_e_val		=	 2*e/(e2+i)
        h_2e_val	=	 2*(z-e**2)/((e2+i)**2)
        h_z_val		=	 1/(e2+i)
        h_2z_val	=	-1/(e2+i)**2
        h_ez_val	=	-2*e/(e2+i)**2
        return h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val"""

dataframe=paneltime.load(fname='../Input/DataFTT_small.csv',filters=filters,transforms=transforms)

if True:
	panel,g,G,H,ll=paneltime.execute(dataframe,model_string,
		                             1,0,1,1,1,'ISINCode','date',
		                             descr='test_proj',
		                             fixed_random_eff=0,
		                             h=h,loadargs=False)

if False:
	dataset=[]
	headings=[]
	for i in dataframe:
		if type(dataframe[i])==np.ndarray:
			dataset.append(dataframe[i])
			headings.append(i)
	dataset=np.concatenate(dataset,1)
	dataframe2=paneltime.from_matrix(dataset,headings,filters, transforms)
	
	panel,g,G,H,ll=paneltime.execute(dataframe2,model_string,
		                             1,1,1,1,1,'ISINCode',
		                             descr='test_proj',
		                             fixed_random_eff=2,
		                             h=h)

	X=[]
	y_name='LNRange'
	x_names=['ones',
		            'LNRelTick',
		            'LNRelTick*LNTurnover',
		            'LNTurnover',
		            'LNMeanNticks',
		            'LNBindFreq',
		            'BindFreq0',
		            'BindFreq1',	    
		            'LNMktShare',
		            'MktShare0',
		            'LNmaxLevels',
		            'DayRet']
	dataframe['LNRelTick*LNTurnover']=dataframe['LNRelTick']*dataframe['LNTurnover']
	for i in x_names:
		X.append(dataframe[i])
	Y=dataframe[y_name]
	X=np.concatenate(X,1)
	model=paneltime.model(X,Y,x_names,y_name,dataframe['ISINCode'],groups_name='ISINCode')
	
	
	panel,g,G,H,ll=paneltime.execute_model(model,
		                                 1,1,1,1,1,
		                                 fixed_random_eff=2,
		                                 h=h)
	
	
	
	

paneltime.diagnostics(panel,g,G,H,ll)
