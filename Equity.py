#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
import numpy as np
import Functions as fu
import datetime as dates
import WorkingTables as WT
import datetime

def CreateTables():
	db='OSEData'
	conn,crsr=DB.Connect(db)
	tbl_name='equity'
	#WT.CreateTableAdjustments(conn,crsr)#Uncomment!!!
	if True:
		DB.DropTable(tbl_name,conn,crsr,db)
		DB.createTable(tbl_name,conn,crsr)
	sl=DB.GetSecurityList(conn,crsr)
	i=0
	for sid,cid,ISIN,stype,symbol,sname,curr in sl:	
		i+=1
		if  not (DB.IsDone('ISIN',ISIN,crsr,tbl_name) and DB.IsDone('SecurityID',sid,crsr,tbl_name)):# and ISIN=='NO0005898007': #
			snameprt=sname.encode('latin1','ignore')
			snameprt=snameprt.decode('utf-8','ignore')
			print ( 'Getting adjustment for %s (%s)' %(snameprt,ISIN))
			p,Cadj,Dadj,d,r=GetAdjustments(sid,ISIN,conn,crsr,curr)
			if not p is None:
				tbl2,cols2=MakeDataSets2(p,Cadj,Dadj,d,r,sid,cid,ISIN,stype,symbol,sname,curr)
				tbl,cols=MakeDataSets(p,Cadj,Dadj,d,r,sid,cid,ISIN,stype,symbol,sname,curr,tbl2)

				print ( 'Appending to tables')
				DB.InsertTableIntoDB(conn,crsr,tbl_name,cols,tbl,db)
		elif not ISIN is None:
			print ( ISIN+ " done")
		#if i>10000:
		#	break
	print ( 'Done ... ')


	conn.close()

def excel_date(d):
	d=dates.datetime.strptime(d,'%Y-%m-%d')
	temp = dates.datetime(1899, 12, 30)
	delta = d - temp
	return float(delta.days) + (float(delta.seconds) / 86400)
	
def MakeDataSets_old(p,Cadj,Dadj,d,r,sid,cid,ISIN,stype,symbol,sname,curr,tbl2):
	n=len(r)
	if len(r)!=len(p):
		raise RuntimeError('price and query lenght do not match')
	tbl=[]
	cols=['Date','SecurityId' ,'CompanyId' ,	'Symbol', 'ISIN',	 'Name', 'BestBidPrice' ,	 
	           'BestAskPrice' ,	'Open' , 'High' ,	 'Low' ,'LastTradedPrice' , 
	           'OfficialNumberOfTrades' ,  'OfficialVolume' , 'UnofficialNumberOfTrades' ,
	           'UnofficialVolume' ,'VolumeWeightedAveragePrice','Price' ,'Dividends' , 
	           'CorpAdj' ,'DividendAdj' ,'Currency','SecurityType',
	           'lnDeltaP','lnDeltaOSEBX','lnDeltaOBX',
	           'AdjustedPrice','OSEBX','OBX']
	padj0=0
	p0=0
	pdiv0=0
	pcorp0=0
	
	prices0=np.zeros(3)
	LastTradedPriceTot=None
	for i in range(n):
		(Date,DateY,DateM,DateD
		,BestBidPrice,BestAskPrice,Openp,High,Low,LastTradedPrice
		,OfficialNumberOfTrades,OfficialVolume,UnofficialNumberOfTrades
		,UnofficialVolume,VolumeWeightedAveragePrice,OSEBX,OBX)=r[i]
		padj=p[i][0]*Dadj[i][0]*Cadj[i][0]
		prices1=checkprices([padj,OSEBX,OBX],prices0)
		nons=np.array([i is None for i in prices1])
		prices1[nons]=0
		prices1=np.array(prices1,dtype=float)
		deltap=np.log(prices1+(prices1==0))-np.log(prices0+(prices0==0))
		deltap=np.array(deltap*(prices1!=0)*(prices0!=0),dtype=object)
		deltap[nons]=None
			
		variables=[Date,sid,cid,symbol,ISIN,sname,BestBidPrice, BestAskPrice, Openp,High,Low,
		 LastTradedPrice,OfficialNumberOfTrades,OfficialVolume,UnofficialNumberOfTrades, 
		 UnofficialVolume,VolumeWeightedAveragePrice,p[i][0],d[i][0],Cadj[i][0],Dadj[i][0],curr,stype]	
		
		variables.extend(deltap)
		variables.extend(prices1)	
		if tuple(variables)!=tbl2[i]:
			a=0
		tbl.append(tuple(variables))
		prices0=prices1

	return tbl,cols


def MakeDataSets(p,Cadj,Dadj,d,r,sid,cid,ISIN,stype,symbol,sname,curr):
	n=len(r)
	if len(r)!=len(p):
		raise RuntimeError('price and query lenght do not match')
	tbl=[]
	cols=['Date','SecurityId' ,'CompanyId' ,	'Symbol', 'ISIN',	 'Name', 'BestBidPrice' ,	 
	      'BestAskPrice' ,	'Open' , 'High' ,	 'Low' ,'LastTradedPrice' , 
	      'OfficialNumberOfTrades' ,  'OfficialVolume' , 'UnofficialNumberOfTrades' ,
	      'UnofficialVolume' ,'VolumeWeightedAveragePrice','Price' ,'Dividends' , 
	      'CorpAdj' ,'DividendAdj' ,'Currency','SecurityType',
	      'lnDeltaP','lnDeltaOSEBX','lnDeltaOBX',
	      'AdjustedPrice','OSEBX','OBX']


	(Date,DateY,DateM,DateD
     ,BestBidPrice,BestAskPrice,Openp,High,Low,LastTradedPrice
     ,OfficialNumberOfTrades,OfficialVolume,UnofficialNumberOfTrades
     ,UnofficialVolume,VolumeWeightedAveragePrice,OSEBX,OBX)=slicevar(r)
	padj=p*Dadj*Cadj
	prices,dprices=checkprices(np.concatenate([padj,OSEBX,OBX],1))
	
	c1=np.repeat([[sid,cid]],n,0).reshape(n,2)
	c2=np.repeat([[symbol,ISIN,sname]],n,0).reshape(n,3)
	c3=np.repeat([[curr]],n,0).reshape(n,1)
	c4=np.repeat([[stype]],n,0).reshape(n,1)
	data=[Date,c1,c2,BestBidPrice, BestAskPrice, Openp,High,Low,
               LastTradedPrice,OfficialNumberOfTrades,OfficialVolume,UnofficialNumberOfTrades, 
               UnofficialVolume,VolumeWeightedAveragePrice,p,d,Cadj,Dadj,c3,c4,dprices,prices]	
	data=np.concatenate(data,1)
	for i in data:
		tbl.append(tuple(i))

	return tbl,cols

def slicevar(r):
	#cnverts r to numpy and slices it:
	r=np.array(r)
	n,k=r.shape
	slices=[]
	for i in range(k):
		slices.append(r[:,i:i+1])
	return slices
	
	
def checkprices(prices):
	n,k=prices.shape
	dprices=np.repeat(0.0,n*k).reshape((n,k))#change to none!
	rng=np.arange(n)
	for i in range(k):
		p=prices[:,i:i+1]
		z=np.nonzero(np.equal(p,None))[0]
		start=0
		if len(z):
			z=z[np.nonzero(z==rng[:len(z)])]#only pick those that are consequitvely None from start
			if len(z):
				start=z[-1]+1
		if start>n-5:#no point with an index only for the five last days
			prices[:start,i:i+1]=0#Remove!!!
			return prices,dprices
		p2=np.array(p[start:],dtype=float)#working only whith prices after some positive price has been observed
		for j in range(n):
			if not np.any(np.equal(p2,None)):
				break
			z=np.nonzero(np.equal(p2,None))[0]
			p2[z]=p2[z-1]#using last observed price
		
		#obtaining the lagged price
		lp2=np.roll(p2,1)
		lp2[0]=p2[0]#first lagged equals next day price, so that first return is zero

		dp2=np.log(p2+(p2==0))-np.log(lp2+(lp2==0))		
		
		#updating
		prices[start:,i:i+1]=p2
		prices[:start,i:i+1]=0#Remove!!!
		dprices[start:,i:i+1]=dp2

	return prices,dprices


def GetAdjustments(SecID,ISIN,conn,crsr,curr='NOK'):

	dtp,p,r=DB.GetPrices(conn,crsr,ISIN)
	if len(dtp)==0:
		return None,None,None,None,None
	Cadj=GetCorpAdj(ISIN,dtp,p,conn,crsr)
	Dadj,d=GetDivAdj(SecID,ISIN,dtp,p,conn,crsr,curr=curr)
	return p,Cadj,Dadj,d,r

def GetCorpAdj(ISIN,dtp,p,conn,crsr):
	"""Returns a corporate action adjustment factor with the same dimesion as the price vector"""
	dta,a=DB.GetAdjFacts(conn,crsr,ISIN)
	dta,a=removeOutsideDates(dta,a,dtp)
	if len(a)==0:
		return np.ones((len(dtp),1))
	dta,a=AddEndStart(dta,a,dtp)
	cuma=CumSum(a)
	adj=IdentifyAdjustment(dtp,dta,cuma)
	#fu.SaveVar(adj)
	return np.array(adj)

def removeOutsideDates(dta,a,dtp):
	if len(dta)==0:
		return [],[]
	dta_ret=[]
	for i in range(len(dta)):
		if dta[i]<dtp[0]:	
			dta_ret.append(dtp[0])
		elif dta[i]>dtp[-1]:
			dta_ret.append(dtp[-1])
		else: 
			dta_ret.append(dta[i])

	return np.array(dta_ret),np.array(a)
	
	
def AddEndStart(dates,a,dtp):
	
	if dtp[-1]>dates[-1]:
		dates=np.append(dates,[dtp[-1]],0)
		a=np.append(a,[[1.0]],0)
	else:
		dates=np.append(dates,[dtp[-1]],0)
		a=np.append(a,[a[-1]],0)		
	dates=np.append([dtp[0]],dates,0)
	a=np.append([a[0]],a,0)
	return dates,a
	
	
	
def IdentifyAdjustment(dtp,dta,cumprod):
	n=len(dtp)
	k=len(dta)
	adj=np.zeros((n,1))
	sel=GetSelMatrix(dtp,dta)
	nz=np.nonzero(sel)
	for i in range(k-1):
		adj[nz[0][i]:nz[0][i+1]+1]=cumprod[i+1]
	return adj

def GetDivAdj(SecID,ISIN,dtp,p,conn,crsr,SecType=0,curr='NOK'):
	"""Returns a dividend adjustment factor and dividends with the same dimesion as the price vector"""
	if SecType==0:
		dtdiv,div=DB.GetDividends(conn,crsr,SecID,curr)
	else:
		dtdiv,div=DB.GetFundDividends(conn,crsr,SecID)
	dtdiv,div=removeOutsideDates(dtdiv,div,dtp)
	n=len(p)
	k=len(div)
	if k==0:
		adj=np.ones((n,1))
		divAtDates=adj*0
		return adj,divAtDates
	sel=GetSelMatrix(dtp,dtdiv)
	divAtDates=np.sum(sel*div.T,1).reshape((n,1))
	p0=fu.ShiftArray(p,-1)
	p1=fu.ShiftArray(p,1)
	d=fu.Concat((p0,p,p1,divAtDates))
	nz=np.nonzero(divAtDates)
	d2=d[nz[0]]
	d=d2[:,1:2]/(d2[:,3:4]+d2[:,1:2])
	sel=np.sum(sel,1)
	nz=np.nonzero(sel)
	dtdiv2=dtp[nz[0]]
	dta,d=AddEndStart(dtdiv2,d,dtp)
	cumdiv=CumSum(d)
	adj=IdentifyAdjustment(dtp,dta,cumdiv)
	#fu.SaveVar(adj,'tmpd')
	return adj,divAtDates
	


def CumSum(a):
	k=len(a)
	r=np.ones((k,1))
	for i in range(k-1):
		r[k-i-2][0]=r[k-i-1][0]*a[k-i-2][0]
	r[k-1]=a[k-1]
	return r
	
	
def GetSelMatrix(dtp,dta):
	n=len(dtp)
	k=len(dta)
	for i in range(n*k*100):
		sel=dtp==dta.T
		sumsel=np.sum(sel,0)
		if np.sum(np.sum(sel,0))==len(dta):#all adjustments have correspoinding dates
			break
		else:
			for j in range(k):
				if sumsel[j]==0:
					dta[j]=AddDay(dta[j][0])
					if dta[j]>dtp[n-1]:
						raise RuntimeError("can't find dates for adjustments")
	return sel
				
def AddDay(dateint):
	d=dates.datetime.strptime(str(dateint),'%Y%m%d')
	d=d + dates.timedelta(days=1)
	d=d.strftime('%Y%m%d')
	d=int(d)
	return d
	



	
	
	