#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
import numpy as np
import Functions as fu
import datetime as dt
import Equity as eq

tbl_cols=['Date' ,'SecurityId','FundId','Symbol','ISIN' ,'Name','NAV','Dividends','CorpAdj','DividendAdj','lnDeltaNAV', 
          'lnDeltaOSEBX','lnDeltaOSEFX','lnDeltaOBX','NAVAdj','OSEBX','OSEFX','OBX']

tbl_name='mutualfunds'
db='OSEData'


def CreateTables():
	conn,crsr=DB.Connect(db)
	if True:
		DB.DropTable(tbl_name,conn,crsr,db)
		DB.createTable(tbl_name,conn,crsr)
	sl=DB.GetFundList(conn,crsr)
	i=0

	for sid,ISIN,fid,symbol,name in sl:	
		i+=1
		if  not DB.IsDone('SecurityId',sid,crsr,tbl_name):
			try:
				print ( 'Getting adjustment for %s (%s)' %(name,sid))
			except:
				print ( 'Getting adjustment for (%s)' %(sid,))
			p,Cadj,Dadj,d,r=GetAdjustments(sid,ISIN,conn,crsr)
			if not p is None:
				tbl=MakeDataSets(p,Cadj,Dadj,d,r,sid,fid,ISIN,symbol,name)
				print ( 'Appending to tables')
				DB.InsertTableIntoDB(conn,crsr,tbl_name,tbl_cols,tbl,db)
		else:
			print ( ISIN+ " done")
		#if i>10000:
		#	break
	#adjust_erronous(conn,crsr)
	conn.close()
	print ( 'Done ... ')

def excel_date(d):
	d=dt.datetime.strptime(d,'%Y-%m-%d')
	temp = dt.datetime(1899, 12, 30)
	delta = d - temp
	return float(delta.days) + (float(delta.seconds) / 86400)
	
def MakeDataSets(p,Cadj,Dadj,d,r,sid,fid,ISIN,symbol,name):
	n=len(r)
	if len(r)!=len(p):
		raise RuntimeError('price and query lenght do not match')
	tbl=[]
	
	prices0=np.zeros(4)
	for i in range(n):
		(Date,DateY,DateM,DateD,NAV,OSEBX,OSEFX,OBX)=r[i]
		NAVAdj=p[i][0]*Dadj[i][0]*Cadj[i][0]

		prices1=eq.checkprices([NAVAdj,OSEBX,OSEFX,OBX],prices0)	
		
		nons=np.array([i is None for i in prices1])
		prices1[nons]=0
		prices1=np.array(prices1,dtype=float)

		deltap=np.log(prices1+(prices1==0))-np.log(prices0+(prices0==0))
		deltap=np.array(deltap*(prices1!=0)*(prices0!=0),dtype=object)
		deltap[nons]=None

		variables=[Date,sid,fid,symbol,ISIN,name,p[i][0],d[i][0],Cadj[i][0],Dadj[i][0]]
		
		variables.extend(deltap)
		variables.extend(prices1)
		
		tbl.append(tuple(variables))
		prices0=prices1
	return tbl
	
	

def GetAdjustments(SecID,ISIN,conn,crsr):

	dtp,p,r=DB.GetFundPrices(conn,crsr,SecID)
	if len(dtp)==0:
		return None,None,None,None,None
	Cadj=GetCorpAdj(SecID,dtp,p,conn,crsr)
	Dadj,d=eq.GetDivAdj(SecID,ISIN,dtp,p,conn,crsr,1)
	return p,Cadj,Dadj,d,r

def GetCorpAdj(SecID,dtp,p,conn,crsr):
	"""Returns a corporate action adjustment factor with the same dimesion as the price vector"""
	dta,a=DB.GetFundAdjFacts(conn,crsr,SecID)
	dta,a=eq.removeOutsideDates(dta,a,dtp)
	if len(a)==0:
		return np.ones((len(dtp),1))
	dta,a=eq.AddEndStart(dta,a,dtp)
	cuma=eq.CumSum(a)
	adj=eq.IdentifyAdjustment(dtp,dta,cuma)
	#fu.SaveVar(adj)
	return adj


def adjust_erronous(conn,crsr):
	r=get_erronous_obs(crsr)
	errs=dict()
	for d,i in r:
		if i in errs:
			errs[i].append(d)
		else:
			errs[i]=[d]
	DB.DropTable(tbl_name+'2',conn,crsr,db)
	DB.CopyTable(conn,crsr,tbl_name,tbl_name+'2',db)
	cols=str(tbl_cols).replace('[','').replace(']','').replace("',",'],').replace("'",'[')
	cols=cols[:len(cols)-1]+']'	
	for i in errs.keys():
		f=DB.Fetch("""SELECT %s
			            FROM [OSEData].[dbo].[mutualfunds]
			            where [SecurityId]=%s
			            order by [Date]""" %(cols,i),crsr)		
		print ('correcting error for ' + str(i))
		h=np.array(DB.Fetch("""SELECT [NAVAdj],[CorpAdj],[lnDeltaNAV],[NAV]
	                        FROM [OSEData].[dbo].[mutualfunds]
	                        where [SecurityId]=%s
	                        order by [Date]""" %(i,),crsr))	
		DB.deleterows_byfieldval('SecurityId',i,tbl_name+'2',db,conn,crsr)
		for j in range(1,len(f)):
			if f[j][0] in errs[i]:
				a=h[j,0]/h[j-1,0]
				h[0:j,1]=h[0:j,1]*a#CorpAdj
				h[0:j,0]=h[0:j,0]*a#NAVAdj
		h=np.array(h,dtype=float)
		NAVAdj=h[:,0]
		NAVAdj_sh=fu.ShiftArray(h[:,0],-1)
		lnDeltaNAV=(np.log(NAVAdj+(NAVAdj==0))-np.log(NAVAdj_sh+(NAVAdj_sh==0)))*(NAVAdj_sh!=0)*(NAVAdj!=0)
		tbl=[]
		for j in range(len(f)):
			tbl.append(tuple(list(f[j][0:8])+[NAVAdj[j],f[j][9],h[j,1],f[j][11],lnDeltaNAV[j]]))
		DB.InsertTableIntoDB(conn,crsr,tbl_name+'2',tbl_cols,tbl,db)
			
		
			
					
					
				
			
		


def get_erronous_obs(crsr):
	return DB.Fetch("""select [Date],[SecurityId] from
	(
		select distinct [Date],[ISIN],[SecurityId],[Name],[lnDeltaNAV] from [OSE].[dbo].[mutualfund] as U1
		where abs([lnDeltaNAV])>0.5
	union
		select distinct [Date],[ISIN],[SecurityId],[Name],[lnDeltaNAV] from [OSE].[dbo].[mutualfund] as U2
		where [Date] in
		(select [Date] from
		(select count(*) as n,[Date] from
		(SELECT distinct
				[Date]
			  ,[Name]
			  ,[lnDeltaNAV]
		  FROM [OSEData].[dbo].[%s]
		  WHERE [lnDeltaNAV]<-0.05 and abs([lnDeltaNAV])<=0.5) as T0
		  group by [Date]) as T1
		  where n=1) and [lnDeltaNAV]<-0.20 and abs([lnDeltaNAV])<=0.5
	union
		select distinct [Date],[ISIN],[SecurityId],[Name],[lnDeltaNAV] from [OSE].[dbo].[mutualfund] as U3
		where [Date] in
		(select [Date] from
		(select count(*) as n,[Date] from
		(SELECT distinct
				[Date]
			  ,[Name]
			  ,[lnDeltaNAV]
		  FROM [OSEData].[dbo].[%s]
		  WHERE [lnDeltaNAV]>0.05 and abs([lnDeltaNAV])<=0.5) as T0
		  group by [Date]) as T1
		  where n=1) and [lnDeltaNAV]>0.20 and abs([lnDeltaNAV])<=0.5
	) as T0
	order by [Date],[SecurityId]""" %(tbl_name,tbl_name),
	crsr)

	
	