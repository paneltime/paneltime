#!/usr/bin/python
# -*- coding: UTF-8 -*-

import Functions as fu
import urllib.request
import numpy as np
import xlrd
import DB
from datetime import timedelta, date, datetime
import zipfile




db='OSEData'


def import_forex():
	t_name='forex'
	conn,crsr=DB.Connect(db)
	DB.DropTable(t_name,conn,crsr,db)
	DB.createTable(t_name,conn,crsr)
	
	tbl=GetTable()

	DB.InsertTableIntoDB(conn,crsr,t_name,['Date','symbol','rate'],tbl,db)
	DB.CreateIndex(conn,crsr,t_name,'OSEData',True)
	pass

def AddInitialRates():
	tbl=[]
	start_date = date(1980, 1, 1)
	end_date = date(1981, 12, 10)
	symbol='NOK'
	for single_date in daterange(start_date, end_date):
		dt=single_date.strftime("%Y-%m-%d")
		tbl.append((dt,'NOK',1))
	return tbl

def GetTable():
	url='http://www.norges-bank.no/WebDAV/stat/valutakurser/xlsx/valutakurser_d.xlsx'
	fname=fu.DownloadFile(url,'../downloaded/')
	xl_workbook=xlrd.open_workbook(fname)
	xl_sheet = xl_workbook.sheet_by_index(0)
	heading_row=6
	tbl=AddInitialRates()
	dt,dtstr=reformat_xldate(xl_sheet.row_values(heading_row+1)[0])
	dtrng=daterange(date(1981, 12, 11), dt,reverse=True)
	for k in range(heading_row,xl_sheet.nrows):
		r=xl_sheet.row_values(k)
		if k>heading_row:
			dtrng=addrow(r,symbols,amounts,dtrng,n,tbl)
		else:
			n,symbols,amounts=format_heading(r)
		k+=1
	return tbl

def daterange(start_date, end_date,reverse=False):
	n=int ((end_date - start_date).days)
	rng=[]
	if not reverse:
		for i in range(n):
			rng.append(start_date + timedelta(i))
	else:
		for i in range(n):
			rng.append(end_date - timedelta(i))		
	return rng

def addrow(r,symbols,amounts,dtrng,n,tbl):
	for i in range(n):
		if r[i+1]=='':
			r[i+1]=None
	rates=np.array(r[1:],dtype=float)/amounts
	dt,dtstr=reformat_xldate(r[0])
	max_gap=10
	k=0
	for k in range(min((max_gap,len(dtrng)))):#ensuring that all dates are covered
		if dtrng[k]>dt:
			dt_missed=dtrng[k].strftime("%Y-%m-%d")
			tbl=addcols(n,r,dt_missed,symbols,rates,tbl)
		elif k==max_gap-1:
			raise RuntimeError('Found gap in currency data of more than 10 days')
		else:
			break
	dtrng=dtrng[k+1:]
	tbl=addcols(n,r,dtstr,symbols,rates,tbl)
	return dtrng
	
def reformat_xldate(xldatevalue):
	dtstr=xlrd.xldate_as_tuple(xldatevalue,0)
	dtstr='%s-%s-%s' %dtstr[0:3]
	dt=datetime.strptime(dtstr,"%Y-%m-%d").date()
	return dt,dtstr
	
	
def addcols(n,r,dt,symbols,rates,tbl):
	tbl.append((dt,'NOK',1))
	for i in range(n):
		if r[i+1]!=None:
			tbl.append((dt,symbols[i],rates[i]))	
	return tbl
	

def format_heading(r):
	n=len(r)-1
	amounts=np.ones(n)
	symbols=n*[None]
	for i in range(n):
		res=r[i+1].split(' ')
		if len(res)==2 and len(res[0])>0:
			amounts[i]=int(res[0])
			symbols[i]=res[1]
		else:
			symbols[i]=res[0]	
	return n,symbols,amounts


def get_FF_data():
	t_name='FamaFrench'
	conn,crsr=DB.Connect(db)
	DB.DropTable(t_name,conn,crsr,db)
	DB.createTable(t_name,conn,crsr)		
	get_FF_file('http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip','_US',t_name,conn,crsr,5,4)
	get_FF_file('http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Global_ex_US_3_Factors_Daily_CSV.zip','_World_ex_US',t_name,conn,crsr,7)

def get_FF_file(url,region,t_name,conn,crsr,strt,end=0):
	fname=fu.DownloadFile(url,'../downloaded/')	
	zipf=zipfile.ZipFile(fname)
	zipf.extractall('downloaded/')
	fname='downloaded/'+zipf.filelist[0].filename
	data=fu.GetCSVMatrixFile(fname,delimiter=',')
	heading=[i+region for i in data[strt-1]]
	if end>0:
		data=np.array(data[strt:-end],dtype=float)
	else:
		data=np.array(data[strt:],dtype=float)
	data=data[data[:,0]>19800000]
	data[:,1:]=np.log((data[:,1:]/100)+1) # FF uses arithmetic means in %, converting to geometric and decimal
	k=len(data[0])
	tbl=[]
	for i in range(len(data)):
		dt=str(data[i,0])
		dt="%s-%s-%s" %(dt[0:4],dt[4:6],dt[6:8])
		for j in range(1,k):
			tbl.append((dt,data[i,j],heading[j]))
	DB.InsertTableIntoDB(conn, crsr, t_name, ['Date','return','factor_name'], tbl, db)
	pass
	

	
	
	