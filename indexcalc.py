#!/usr/bin/python
# -*- coding: UTF-8 -*-
import datetime
import numpy as np
import DB


def make_index(conn,crsr):
	mnth_strt=get_month_start(crsr)
	sqlstr1="""
SELECT [Date]
      ,[ISIN]
      ,[CorpAdj]
	  ,[DividendAdj]
      ,[Price]
      ,[NoShares]
	  ,[bills_DayLnrate]
  FROM [OSEData].[dbo].[OSEBX_mcaps]
  where ((year([Date])=%s  and [Date]<='%s')  or (year([Date])=%s and month([Date])=12)) and not [NoShares] is NULL
  order by [ISIN],[Date]"""

	sqlstr2="""SELECT [Date]
      ,[ISIN]
      ,[CorpAdj]
	  ,[DividendAdj]
      ,[Price]
      ,[NoShares]
	  ,[bills_DayLnrate]
  FROM [OSEData].[dbo].[OSEBX_mcaps]
  where (year([Date])=%s and (month([Date])>%s and [Date]<='%s')) and not [NoShares] is NULL
  order by [Date]
"""

	indx=[[] for i in range(7)]
	mktshares=[]
	for i in range(1996,datetime.datetime.now().year):
		s="%s-%s" %(i,6)
		if s in mnth_strt:
			seminannual_index(sqlstr1 %(i,mnth_strt["%s-%s" %(i,6)],i-1),indx,mktshares,conn,crsr)
		s="%s-%s" %(i,12)
		if s in mnth_strt:
			seminannual_index(sqlstr2 %(i,5,mnth_strt["%s-%s" %(i,12)]),indx,mktshares,conn,crsr)
		if i==datetime.datetime.now().year-1:
			seminannual_index(sqlstr2 %(i,11,"%s-%s-%s" %(i,12,31)),indx,mktshares,conn,crsr)
		print(i)

	n=len(indx[0])
	for j in range(7):
		if j<4:
			indx[j][0]=indx[j][0]/indx[j][0][0]
		for i in range(1,n):
			if j<4:
				indx[j][i]=indx[j][i]*indx[j][i-1][-1]/indx[j][i][0]
			indx[j][i]=indx[j][i][:-1]
	for j in range(7):
		indx[j]=np.concatenate(indx[j],0)
	n=len(indx[0])
	for j in range(4):
		ret=np.log(indx[j])-np.log(np.roll(indx[j],1))
		ret[0]=0
		indx.append(ret)
	tbl=[]
	for i in range(n):
		tbl_elem=[]
		for j in indx:
			tbl_elem.append(j[i])
		tbl.append(tuple(tbl_elem))
	tblname='OSEBX_recalc'
	DB.DropTable(tblname, conn, crsr,'OSEData')
	DB.createTable(tblname, conn, crsr)
	DB.InsertTableIntoDB(conn, crsr, tblname, 
	                     "[index],[index_d],[index_c],[index_p],[adj_d],[adj_p],[Date],[return],[return_d],[return_c],[return_p]", tbl, 'OSEData')

	tblname='OSEBX_mktshares'
	DB.DropTable(tblname, conn, crsr,'OSEData')
	DB.createTable(tblname, conn, crsr)
	DB.InsertTableIntoDB(conn, crsr,tblname, 
	                     "[Date],[ISIN],[mktshare],[alpha],[beta],[marketcap]", mktshares, 'OSEData')


def make_index_caps(conn,crsr):
	sqlstr="""


SELECT distinct S.[Date],S.[ISIN],S.[CorpAdj],S.[DividendAdj],S.[Price],T2.[NoShares],S.[bills_DayLnrate]
	into [OSEData].[dbo].[OSEBX_mcaps]
		from
			 (select distinct [Date],[ISIN],[CorpAdj],[DividendAdj],[Price],[bills_DayLnrate] 
			 from [OSEData].[dbo].[equity_extended1]) S

		left join 
			(SELECT distinct  [Date]
			  ,[ISIN]
			  ,[NoShares]
			FROM [OSEData].[dbo].[OSEBX_number_shares]) T2
		on T2.[ISIN]=S.[ISIN] and S.[Date]=T2.[Date]

"""
	DB.DropTable('OSEBX_mcaps',conn,crsr,'OSEData')
	crsr.execute(sqlstr)
	conn.commit()
	DB.CreateIndex(conn,crsr,'OSEBX_mcaps','OSEData',
	               IndexFields="""
                    [Date]
                    ,[ISIN]""")		

def get_ret_matrix(sqlstr,conn,crsr,isin_dict=None):
	crsr.execute(sqlstr)
	r=np.array(crsr.fetchall())
	if len(r)<100:
		return None,None,None,None
	n,k=r.shape
	isin, isinrng,isin_dict,isin_list=makeint(r[:,1],isin_dict)
	m=len (isinrng)
	dates, dates_rng, dates_dict,dates_list= makeint(r[:,0])
	pivoted=[]
	long_format=[]
	for i in range(k-2):
		pivoted.append(np.zeros((len(dates_rng),m)))
		long_format.append(r[:,i+2])
	dates_str=r[:,0]
	isin_str=r[:,1]
	for i in isinrng:
		sel=isin==i
		for j in range(k-2):
			pivoted[j][dates[sel],i]=long_format[j][sel]
	return pivoted,long_format,isin_list,dates_list

def get_month_start(crsr):
	sqlstr="""select distinct year([Date]),month([Date]),min([Date]) 
from [OSEData].[dbo].[OSEBX_mcaps] 
where  month([Date])=6 or month([Date])=12
group by year([Date]),month([Date])
order by year([Date]),month([Date])"""
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	d=dict()
	for i in r:
		d["%s-%s" %(i[0],i[1])]=i[2]
	return d




def seminannual_index(sqlstr,indx_lst,mktshares,conn,crsr):	
	pivoted,long_format,isins,dates=get_ret_matrix(sqlstr,conn,crsr)
	if pivoted is None:
		return
	adj_c=pivoted[0]
	adj_d=pivoted[1]
	p=pivoted[2]
	nshares=pivoted[3]
	bills=pivoted[4]
	nshares[:,:]=np.max(nshares,0)
	fill_matrix(p)
	fill_matrix(adj_c)
	fill_matrix(adj_d)
	adj_c=adj_c/adj_c[0]
	adj_d=adj_d/adj_d[0]

	mcap=adj_c*adj_d*p*nshares
	n,k=mcap.shape
	mkt_shares=mcap/np.sum(mcap,1).reshape((n,1))
	indx=np.sum(mcap,1)
	ret=(np.log(p)-np.log(np.roll(p,1,0))-bills)[1:]
	ix_ret=np.sum(mkt_shares[1:]*ret,1).reshape((n-1,1))
	ix_ret=np.concatenate((np.ones((n-1,1)),ix_ret),1)
	beta=OLS(ix_ret,ret)
	for i in range(n-1):
		for j in range(k):
			mktshares.append((dates[i],isins[j],mkt_shares[i,j],beta[0,j],beta[1,j],mcap[i,j]))
	indx_d=np.sum(adj_d*p*nshares,1)
	indx_c=np.sum(adj_c*p*nshares,1)
	indx_p=np.sum(p*nshares,1)
	adj_d=np.mean(adj_d,1)
	adj_c=np.mean(adj_c,1)

	indx_lst[0].append(indx)
	indx_lst[1].append(indx_d)
	indx_lst[2].append(indx_c)
	indx_lst[3].append(indx_p)
	indx_lst[4].append(adj_d)
	indx_lst[5].append(adj_c)
	indx_lst[6].append(dates)

def OLS(X,Y):
	XX=np.dot(X.T,X)
	XXInv=np.linalg.inv(XX)
	XY=np.dot(X.T,Y)
	beta=np.dot(XXInv,XY)
	return beta


def fill_matrix(x):
	n,k=x.shape
	for i in range(1,n):
		x[i]=(x[i]==0)*x[i-1]+(x[i]>0)*x[i]#ensuring obs for all stocks
	for i in range(k):
		if np.all(x[:,i]==0):
			x[:,i]=1
		nz=np.nonzero(x[:,i])[0]
		if len(nz)>0:
			x[:nz[0],i]=x[nz[0],i]	

def makeint(x,x_dict=None):
	n=len(x)
	unq=np.unique(x)
	m=len(unq)
	rng=np.arange(m)
	d=dict(zip(unq, rng))

	if not x_dict is None:
		n=np.max(d.values)
		for i in x_dict:
			if not i in d:
				n+=1
				d[i]=n
	integerx=np.array([d[k] for k in x],dtype=int)
	return integerx, rng,d,unq




