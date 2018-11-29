#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import DBparser
import DB
import Functions as fu
from Functions import prntout
import time
import Equity as eq
import Funds as fnd
import WorkingTables as wt
import FinalTables as ft
import FileSync
import sendmail
import traceback
import mysql


db='OSEData'

def Main():
	UpdateTables()
	err=False
	flupdt=True
	if False:#True for debugging
		flupdt=Update()
	else:
		try:
			flupdt=Update()
			pass
		except:
			prntout(traceback.format_exc())
			err=True
	txt=open('../Output/output.txt','r').read()
	if not err:
		if flupdt:
			txt='Database updated successfully \r\n'+txt
		else:
			txt='No changes in directory \r\n'+txt
	else:
		txt='Update failed \r\n'+txt
	sendmail.SendMail(txt,'results from update')
	


def Update():
	prntout('****************Update procedure******************',True)
	#fu.RemoveCSV('parserec')
	#DB.DeleteAllTables(db)#DO NOT ENABLE UNLESS YOU KNOW WHAT YOU AR DOING. WILL DELETE ALL TABLES.
	prntout('FileSync:')
	if True:#set to false for debugging
		if True:#set to False for forced updating
			syncedFiles=FileSync.Sync()
			if len(syncedFiles)==0:
				return False
		prntout('Updating tables:') 
		UpdateTables()
		prntout('Adding indicies:') 
		UpdateIndexes()
		prntout('Splitting tables') 
		wt.SplitTable('Type','bondfeed',db)
		wt.SplitTable('Type','equityfeed',db)
	MakeTables()
	mysql.copyall()
	return True




def MakeTables():
	prntout('Creating index tables')
	indextbl()
	
	prntout('Creating equity table')
	eq.CreateTables()
	prntout('Creating fund table')
	fnd.CreateTables()
	prntout('Creating working tables') 
	wt.MakeTables()
	prntout('Creating final tables') 
	ft.MakeTables()

def indextbl():
	conn,crsr=DB.Connect(db)
	wt.CastLinkedIndicies(conn,crsr)
	wt.MakeIndexTable(conn,crsr)	
	
def UpdateTables():
	"""Discovers tables in the data-folder, and parses them to
	the database if not present in parserec.csv"""
	i=0
	rec=fu.GetCSVMatrixFile('parserec')
	conn,crsr=DB.Connect(db)
	if rec is None:
		rec=[['File name','sequence','time parsed','rows inserted']]
	for p,d,f in os.walk(FileSync.curdir+'/data'):
		i+=1
		prntout(i)
		if i>=0:
			ParseTables(p,f,i,rec,conn,crsr)
	conn.close()

def ParseTables(path,flst,i,rec,conn,crsr):
	frec=[r[0] for r in rec]
	for f in flst:
		tbl,floc,fname=GetDBName(path,f)
		if not tbl is None:
			prntout(fname) 
			if not fname in frec:
				ParseTable(tbl,floc,fname,i,f,rec,conn,crsr)
			else:
				prntout('%s has been parsed' %(fname,))
	prntout('Done')

def ParseTable(tbl,floc,fname,i,f,rec,conn,crsr):
	prntout('Parsing: %s' %(f,))
	m=0
	if ('.xml' in f):
		m=DBparser.MakeDBxml(tbl,floc,fname,db,conn,crsr)
	elif ('.txt' in f) or ('.csv' in f):
		m=DBparser.MakeDBtxt(tbl,floc,fname,db,conn,crsr)
	if m>0:
		t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		rec.append([fname,i,t,m])
		fu.WriteCSVMatrixFile('parserec.csv',rec,True)    


def HasBeenParsed(rec,fname):
	for i in range(1,len(rec)):
		if len(rec[i])>0:
			if rec[i][0]==fname:
				return True
	return False



def UpdateIndexes():
	rec=fu.GetCSVMatrixFile('parserec')
	conn,crsr=DB.Connect(db)
	if rec is None:
		rec=[['File name','sequence','time parsed','rows inserted']]
	for path,d,flst in os.walk(os.getcwd()+'/data'):
		for f in flst:
			tbl,floc,fname=GetDBName(path,f)
			if not tbl is None:
				prntout('Indexing: %s' %(f,))
				DB.CreateIndex(conn,crsr,tbl,db)
		prntout('Done')
	conn.close()

def GetDBName(path,fname):
	DBnames=['bondfeed',
		     'bondindex',
		     'bondprices',
		     'equity_pricedump',
		     'equityfeed',
		     'equityindex',
		     'equitypricedump',
		     'fund_adj_factors',
		     'fund_dividends',
		     'fund_prices',
		     'funds',
		     'futforw_prices',
		     'shareidx_prices',
		     'Lenkede_indekser',
	         'bills_historical',
	         'options_prices',
	         'gics',
	         'GICS_all',
	         'account_mapping',
	         'OSEBX_number_shares']
	path=CleanPath(path)
	for i in DBnames:
		if i in fname:
			d=FileSync.curdir+ '/data'
			d=path.replace(d,'')
			floc=path + '/' + fname
			fname=d + '/' + fname
			return i,floc,fname
	return None, None, None


def CleanPath(path):
	path=path.replace('\\','/')
	path=path.replace('//','/')
	path=path.replace('//','/')
	return path

Main()