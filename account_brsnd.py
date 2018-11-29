#!/usr/bin/python
# -*- coding: UTF-8 -*-

#A self running module that fetch accouting data from the net

import sys
sys.path.append('../../')
import urllib.request
import Functions as fu
import re
import datetime as dt
import csv
#import google
import time
import numpy as np
import datetime
import DB

db='OSEData'
maxyear=2014

def GetData():
	file=open('../accountingdata/org.csv','r')
	tbl='account_brnsnd'
	reader=csv.reader(file,quoting=csv.QUOTE_NONE,delimiter=';')
	conn,crsr=DB.Connect(db)
	#DB.createTable(tbl,conn,crsr)
	sqlstr=DB.GetSQLInsertStr(tbl,columns)
	k=0
	#fl = open('logfile.csv', 'wb')
	for r in reader:
		if k>0:#Assuming first line contains headers
			now=datetime.datetime.now()
			wakeup=datetime.datetime(now.year,now.month,now.day,8,3,27)
			gohome=datetime.datetime(now.year,now.month,now.day,17,58,36)				
			if now<wakeup or now>gohome:
				sleeptime=0
			else:
				sleeptime=0
			time.sleep(0+sleeptime*np.random.random())
			if len(r)>3:
				if r[3]=='':
					url=None
				else:
					url=r[3]
			else:
				url=None
			for corp in (True,False):
				time.sleep(1+np.random.random())
				print ( 'Appending %s   %s corp:%s...' %(r[0],r[1],corp),)
				AppendData(r[2],url,corp,r[0],r[1],conn,crsr,sqlstr)
				print ( ' done')
		k+=1
	DB.CreateIndex(conn,crsr,tbl,db)
	
def AppendData(ID,url,corp,compID,Name,conn,crsr,sqlstr):
	t,webName=GetDataFromID(ID,Name,url,corp)
	tbl=[]
	maxyear=max_year(corp,ID,crsr)
	if len(t)==0:
		#d=[corp,ID,compID,Name,webName,dt.datetime.now(),dt.datetime.now().year,'NoData',None,None]
		#crsr.execute(sqlstr,tuple(d))
		#conn.commit()	
		return
	for year ,description,value,DescrID in t:
		if year>0:
			d=[corp,ID,compID,Name,webName,dt.datetime.now()]#corporate accounting,ID (organization number), CompanyID,Name
			d.extend([year ,description,value,DescrID])#
			crsr.execute(sqlstr,tuple(d))
			conn.commit()
			pass



def GetDataFromID(ID,Name,url,corp=True):
	

	if corp:
		url="http://www.purehelp.no/company/corp/"#for corprate numbers
	else:
		url="http://www.purehelp.no/company/account/"
	url=url + str(ID)
	doc=fetchDoc(url)
	if doc==None:
		print ( "Error getting info from %s,%s,corp: %s,%s" %(Name,ID,corp,'purehelp'))
		return [],''
	docStr=read(doc)
	srchstr1='<table(.*?)</table>'
	srchstr2='<tr class(.*?)</tr>'
	srchstr3='(?<=>)([^\<\>\n]+)(?=<)'#Finds all text between '>' and '<' that does not include '>' or '<'. 
	webnamesrchstr='(?<=<title>)([^\<\>\n]+)(?=</title>)'

	tblset,webname=GetTable(url,docStr,srchstr1,srchstr2,srchstr3,webnamesrchstr)
	lngtptbl=[]
	ctr=0
	if len(tblset)>0:
		for i in range(2):
			ctr=LongFormatTable(tblset[i],lngtptbl,ctr)
	return lngtptbl,webname

def read(doc):
	docStr=doc.read()
	return docStr.decode('utf-8')

def LongFormatTable(tbl,lftbl,ctr):
	if len(tbl)==0:
		return []
	if tbl[0][0]!="År":
		raise RuntimeError("ikke år først")	
	yr=tbl[0]
	for i in tbl:
		if i[0]!="År":
			ctr+=1
			if len(i)!=len(yr):
				raise RuntimeError("length problem")
			for j in range(1,len(i)):
				
				r=[yr[j],i[0],i[j],ctr]
				lftbl.append(r)
	return ctr

	
	
def GetTable(url,docStr,srchstr1,srchstr2,srchstr3,webnamesrchstr):
	MandNot=False
	#f=open('docstr.txt','w')
	#f.write(docStr)
	tblstrings=re.findall(srchstr1,docStr)
	webname=re.findall(webnamesrchstr,docStr)[0]
	tblset=[]
	for i in tblstrings:
		rowstrings=re.findall(srchstr2,i)
		tbl=[]
		for j in rowstrings:
			row=re.findall(srchstr3,j)
			r=[]
			for k in row:
				v=convert(k)
				if v!=' ':
					r.append(v)
			if len(r)>1:
				tbl.append(r)
		tblset.append(tbl)
			
	return tblset,webname

def convert(s):
	s=s.replace('POS','')
	s=s.replace('NEG','')
	s=s.replace('.','')
	s=s.replace(',','.')
	isperc= '%' in s
	s=s.replace('%','')
	try:
		return convertperc(int(s),isperc)
	except:
		try:
			return convertperc(float(s),isperc)
		except:
			return s
	
def convertperc(f,isperc):
	if isperc:
		return f/100.0
	else:
		return f
		
	
	
	
def fetchDoc(DocStr):
	c=0
	while True:
		try:
			if c>5:
				return None
			f=urllib.request.urlopen(DocStr)
			break
		except:
			print ( 'waiting ...')
			time.sleep(10+np.random.random()*10)
			c+=1
	return f

def max_year(IsCorporateAccount,OrganizationID,crsr):
	#Retrieves the last year in the database
	SQLExpr="""SELECT distinct
      Year
	  FROM [dbo].[account_brnsnd]
	  where [IsCorporateAccount]='%s' and [OrganizationID]='%s' """ %(IsCorporateAccount,OrganizationID)
	crsr.execute(SQLExpr)
	f=crsr.fetchall()
	r=[]
	for i in range(len(f)):
		if not f[i][0] is None:
			r.append(f[i][0])
	if len(r)==0:
		r=0
	else:
		r=max(r)
	return r

def NotDone(IsCorporateAccount,OrganizationID,crsr,MinYear):
	SQLExpr="""SELECT distinct
      *
	  FROM [dbo].[account_brnsnd]
	  where [IsCorporateAccount]='%s' and [OrganizationID]='%s' and [YEAR]>=%s""" %(IsCorporateAccount,OrganizationID,MinYear)
	crsr.execute(SQLExpr)
	r=crsr.fetchall()
	return len(r)==0

def GetURLFromGoogle(orgnr):
	orgstr=str(orgnr)
	orgstr=orgstr[0:3]+" "+orgstr[3:6]+" "+orgstr[6:9]
	srchstr='"www.proff.no" "Sum salgsinntekter" "Sum+driftsinntekter" "Org nr %s"' %(orgstr,)
	slist=[]
	srch=google.search(srchstr)
	res=None
	for s in srch:
		time.sleep(5+5*np.random.random())
		print ( s)
		if 'regnskap' in s:
			res=s
			break
	return res

def save_to_file():

	t,webName=GetDataFromID(983790739,'Name',None,True,True)
	fu.WriteCSVMatrixFile('account',t,currpath=True)

	pass

columns="""[IsCorporateAccount],[OrganizationID] , [CompanyID],[Name],[webName] ,[FetchDate] ,[Year] , [Description] ,[Value] ,[DescrID]"""
GetData()