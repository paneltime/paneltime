#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
import numpy as np


#Creates factor portfolios

tbl_name='factors_tmp'
db= 'OSEData'

def calc_factors():
	
	conn,crsr=DB.Connect(db)
	DB.DropTable(tbl_name, conn, crsr,db)
	DB.createTable(tbl_name, conn, crsr)
	(isins, month_counter, 
	 months_list, isins_list, size, BookMkt, returns ,isins_dict)= get_factor_criterias(conn,crsr)
	p_isins,p_month_counter,p_date, dp, p_day,p_month,p_year = get_prices(conn, crsr, isins_dict)
	
	n=len(isins)
	momentum=np.zeros(n)
	mom_include=np.zeros(n,dtype=bool)
	for i in isins_list:
		calc_momentum_return(returns, i==isins, momentum, mom_include)	
	for m in months_list:
		sel=m==month_counter
		
		#HML and SMB
		portfolios,month_size,month_isins=get_portfolios(sel, isins,size, BookMkt)
		monthly_return(m,portfolios,p_isins,p_month_counter,p_date,p_day,p_month,p_year,dp,month_size,month_isins,conn,crsr)
		
		#Momentum
		portfolios,month_size,month_isins=get_portfolios(sel, isins,size, momentum, attr_lbl='_mom' ,include=mom_include) 
		monthly_return(m,portfolios,p_isins,p_month_counter,p_date,p_day,p_month,p_year,dp,month_size,month_isins,conn,crsr)
	
	maketable(conn,crsr)
	
def monthly_return(m, portfolios,p_isins,p_month_counter,p_date,p_day,p_month,p_year,dp,month_size,month_isins,conn,crsr):
	if portfolios is None:
		return
	month_sel=np.nonzero(p_month_counter==m)

	for p in portfolios:
		add_portfolio_return(month_sel,p,portfolios[p],p_isins,p_date,p_day,p_month,p_year,dp,month_size,month_isins,conn,crsr)
	
def add_portfolio_return(month_sel,p_name,members,p_isins,p_date,p_day,p_month,p_year,dp,month_size,month_isins,conn,crsr):
	if len(members)==0:
		return
	dp=dp[month_sel]
	p_date=p_date[month_sel]
	p_isins=p_isins[month_sel]
	p_day=p_day[month_sel]
	p_month=p_month[month_sel]
	p_year=p_year[month_sel]
	port_isins=month_isins[members]
	port_isins=port_isins.reshape((1,len(port_isins)))
	share=month_size[members]/np.sum(month_size[members])
	r=np.zeros(32)
	ret=np.zeros(32)
	n=len(members)
	for i in range(n):
		r=r*0
		isin_sel=np.nonzero(p_isins==month_isins[members[i]])
		r[p_day[isin_sel]] = dp[isin_sel]
		ret+=r*share[i]
	days=np.sort(np.unique(p_day))
	table=[]
	month=p_month[0]
	year=p_year[0]
	if not np.all(month==p_month) or not np.all(year==p_year):
		raise RuntimeError('something wrong here')
	for i in days:
		dt="%s-%s-%s" %(year,str(month).rjust(2,'0'),str(i).rjust(2,'0'))
		table.append((dt,ret[i],p_name))
		
	DB.InsertTableIntoDB(conn, crsr, tbl_name, ['Date','return','factor_name'], table, db)

def calc_momentum_return(returns,sel,momentum,mom_include):
	returns=returns[sel]
	n=len(returns)
	include=np.ones(n,dtype=bool)
	m=np.mean(np.concatenate([np.roll(returns.reshape((n,1)),i,0) for i in range(2,12)],1),1)
	m[0:11]=0
	include[0:11]=False
	momentum[sel]=m
	mom_include[sel]=include
	pass

def get_portfolios(sel, isins,size, attribute,attr_lbl='',include=None):
	month_size           =size[sel]
	month_attribute  =attribute[sel]
	month_isins          =isins[sel]
	if not include is None:
		incld=include[sel]
		month_size=month_size[incld]
		month_attribute=month_attribute[incld]
		month_isins=month_isins[incld]
	nantest=np.isnan(month_size*month_attribute)==False
	month_size=month_size[nantest]
	month_attribute=month_attribute[nantest]
	month_isins=month_isins[nantest]	
	
	size_srt           =  np.argsort(month_size)
	attribute_srt  =  np.argsort(month_attribute)
	
	n=len(month_attribute)
	if n==0:
		return None, None, None
	
	small=size_srt[:int(n/2)]
	big=  size_srt[:int(n/2)+1]
	low=  attribute_srt[:int(n*0.3)]
	high= attribute_srt[int(n*0.7)+1:]
	med=  attribute_srt[int(n*0.3)+1:int(n*0.7)]
	portfolios=dict()
	for s in ['small','big']:
		for i in locals()[s]:
			for a in ['low','med','high']:
				port_str=a+'_'+s+attr_lbl
				if not port_str in portfolios.keys():
					portfolios[port_str]=[]			
				if i in locals()[a]:
					portfolios[port_str].append(i)
	for p in portfolios:
		n=len(portfolios[p])
		portfolios[p]=np.array(portfolios[p])
	return portfolios,month_size,month_isins
	
def get_prices(conn,crsr,isins_dict):
	crsr.execute("""
				SELECT
					[ISIN]
	                ,[Date]
	                ,year([Date])
	                ,month([Date])
	                ,day([Date])
	                ,[lnDeltaP]
	            FROM [OSEData].[dbo].[equity_extended1]
	            where [Date]>='1993-01-01'
	            order by [Date],[ISIN]""")
	f=np.array(crsr.fetchall())
	n=len(f)
	p_isins=-np.ones(n,dtype=int)
	p_isins_str=f[:,0]
	for i in range(n):
		if p_isins_str[i] in isins_dict.keys():
			p_isins[i]=isins_dict[p_isins_str[i]]

	p_date=f[:,1]
	f=f[:,2:].astype(float)
	p_month_counter=((f[:,0]-1980)*12+f[:,1]).astype(int)
	p_day=f[:,2].astype(int)
	p_month=f[:,1].astype(int)
	p_year=f[:,0].astype(int)
	dp=f[:,3]
	
	return p_isins,p_month_counter,p_date, dp,p_day,p_month,p_year
	
def get_factor_criterias(conn,crsr):
	crsr.execute("""SELECT distinct 
		[ISIN]
		,[Year]
	    ,[Month]
	    ,[AvgMktCap]
	    ,[BM]
	    ,[lnReturn]
  FROM [OSEData].[dbo].[factor_criteria]
  where [Year]>=1993
  order by [ISIN],[Year],[Month]
  """)	
	f=crsr.fetchall()	
	f=np.array(f)
	n=len(f)
	isins_unq=np.unique(f[:,0])
	isins_list=np.arange(len(isins_unq))
	isins_dict=dict(zip(isins_unq,isins_list))
	isins=np.array([isins_dict[k] for k in f[:,0]]).flatten()
	f=f[:,1:].astype(float)
	size=f[:,2]
	BookMkt=f[:,3]
	returns=f[:,4]
	month_counter=((f[:,0]-1980)*12+f[:,1]).astype(int)
	months_list=np.unique(month_counter)

	return isins, month_counter, months_list, isins_list,size,BookMkt,returns,isins_dict
	

