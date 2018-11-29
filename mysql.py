#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:filetype=python:
#
# Routines to communicate with the MySQL backend database
#


import pymysql
import DB



def copyall():
	myconn,mycrsr=connect()
	msconn,mscrsr=DB.Connect('OSE')
	r=mscrsr.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
	r=mscrsr.fetchall()
	for i in range(len(r)):
		copytable(myconn,mycrsr,mscrsr,r[i][2])
		a=0
	
	
	
	

def connect(): 

	myconn = pymysql.connect(host='titlon.uit.no', 
	                       user='OSEupdater', 
	                       passwd='FinansDB',
	                       port=3306, 
	                       db='OSE')		
	mycrsr=myconn.cursor()
	return myconn,mycrsr


def copytable(myconn,mycrsr,mscrsr,table):
	print('copying %s' %(table))
	createstr,selectstr,insertstr=getsqlstrings(mscrsr,table)
	createtable(mycrsr,myconn,table,createstr)
	insertinto(myconn, mycrsr, mscrsr, selectstr, insertstr)

	
def getsqlstrings(mscrsr,table):
	mscrsr.execute("""
	select [Column_name],[Data_type],
		[Character_maximum_length] 
	from [OSE].INFORMATION_SCHEMA.COLUMNS
	where [Table_name]='%s'""" %(table))
	fields=mscrsr.fetchall()
	ix=[]
	d=[]
	msn=[]	
	myn=[]	
	for name,datatype,characterlen in fields:
		if name in ['Date', 'SecurityId', 'CompanyId', 'Symbol', 'ISIN']:
			ix.append(name)
		if datatype=='varchar' or datatype=='nvarchar':
			datatype+=' (%s)' %(characterlen)
		if datatype!='identity':
			d.append(datatype)
			msn.append('['+name+']')
			myn.append('`'+name+'`')
			
	ixstr=','.join(ix)
	if ixstr!='':
		ixstr= ", \nINDEX ix (%s)" %(ixstr)	
	fieldtype=', \n\t'.join(['%s %s' %(myn[i],d[i]) for i in range(len(fields))])
	myfieldstr=', \n\t'.join(myn)
	msfieldstr=', \n\t'.join(msn)
	selectstr='SELECT \n\t%s \nFROM %s' %(msfieldstr,table)
	createstr="CREATE TABLE IF NOT EXISTS %s (\n\t%s%s)" %(table,fieldtype,ixstr)

	s=','.join(['%s']*len(myn))
	insertstr="INSERT INTO %s (\n\t%s) \nVALUES (%s)" %(table,myfieldstr,s)	
	return createstr,selectstr,insertstr


def createtable(mycrsr,myconn,table,createstr):

	mycrsr.execute(createstr)
	myconn.commit()	
	mycrsr.execute('TRUNCATE TABLE %s' %(table,))
	myconn.commit()		
	
def insertinto(myconn,mycrsr,mscrsr,selectstr,insertstr,batchsize=10000):
	mscrsr.execute(selectstr)
	m=0

	while True:
		m+=1
		print(m)
		row=mscrsr.fetchmany(batchsize)
		if row==None:break
		try:
			mycrsr.executemany(insertstr,row)
		except UnicodeEncodeError as e:
			for i in range(len(row)):
				row[i]=finderror(row[i])
			mycrsr.executemany(insertstr,row)
		myconn.commit()
		if len(row)<batchsize:break	

def finderror(row):
	row=list(row)
	for i in range(len(row)):
		if type(row[i])==str:
			try:
				s=row[i].encode('latin-1')
			except UnicodeEncodeError as e:
				s=row[i].encode('latin-1',errors='ignore')
				print ("Warning: Problems encoding '%s'" %(s,))
				row[i]=s
	return tuple(row)
	
def insertsqlstr(fields,table):

	return sqlstr

