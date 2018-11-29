#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../Modules')
from xml.dom import minidom
from xml import parsers
import linecache
import DB
import os
import csv
import shutil
import Functions as fu

fileloc='Files/'

#need to add the following to expatbuilder.py, line 214:
#except expat.ExpatError:
#	pass


			

	
	
def MakeDBxml(table,floc,fname,dbase,conn,crsr):
	if not DB.TableExists(dbase,table,crsr):
		DB.createTable(table,conn,crsr)
	else:
		DB.DeleteFileEntries(fname,dbase,conn,crsr)
		DB.DeleteIndex(table,dbase,conn,crsr)
	if DB.FileInDB(table,fname,crsr):
		print ( '%s all ready in table %s' (fname,table))
		return False
	xmldoc=ParseXML(floc,0)
	tbl=[]
	node=xmldoc.documentElement
	tblProps=[DB.GetColumnNames(crsr,table)]
	m=0
	for n in node.childNodes:
		if len(n.childNodes)>0:
			if table=='equityfeed' or table=='bondfeed':
				for k in n.childNodes:
					if len(k.childNodes)>0:
						keys,values=GetValues(k,table,'Type',k.nodeName)
						err=AddRow(conn,crsr,dbase,table,keys,values,tblProps,fname)
						m+=1
			else:
				keys,values=GetValues(n,table)
				err=AddRow(conn,crsr,dbase,table,keys,values,tblProps,fname)
				m+=1
			pass
	
	return m
	
def ParseXML(floc,n):
	try:
		print ( 'running xml parser')
		xmldoc=minidom.parse(floc,)
		print ( 'xml parser done')
	except parsers.expat.ExpatError as err:
		str=err.message
		if 'reference to invalid character number' in str:
			str=str.replace(',',' ')
			l,c=[int(s) for s in str.split() if s.isdigit()]
			print ( 'replacing character %s,%s - run %s' %(l,c,n))
			ReplaceCharacter(floc,l,c)
			print ( 'Done replacing')
			n+=1
			if n<50:
				xmldoc=ParseXML(floc,n)
			else:
				raise RuntimeError("more errors than allowed for parsing")
		else:
			raise parsers.expat.ExpatError(str)
	return xmldoc

def ReplaceCharacter(floc,row,col):
	f = open(floc, "r",encoding='utf-8')
	lines=f.readlines()
	for i in range(len(lines)):
		if i==row-1:
			print ( 'err in ' + lines[i][col-10:])
			#lines[i]=lines[i][:col-2]+'   '+lines[i][col+1:]
			n=lines[i][col:].find('<')
			lines[i]=lines[i][:col-1]+'   '+lines[i][col+n:]
			#lines[i]=lines[i].replace('&#11','')
			break
	f.close()
	f=open(floc, "w")
	f.writelines(lines)
	f.close()
	
	
	
	
				
def replaceInLine(filename, lineno):
	f=os.open(filename, os.O_RDWR,encoding='utf-8')
	m=mmap(f,0)
	p=0
	for i in range(lineno-1):
		p=m.find('\n',p)+1
	q=m.find('\n',p)
	m[p:q] = ' '*(q-p)
	os.close(f)
	
def OpenReader(file,delimiter):
	reader=csv.reader(file,quoting=csv.QUOTE_NONE,delimiter=delimiter)
	file.seek(0)
	try:
		r=next(reader)
	except StopIteration:
		return 0#empty file
	has_header=True
	for i in r:
		if i.replace('.','').replace(',','').isdigit():
			has_header=False
	if not has_header:
		file.seek(0)
		return len(r),None,reader
	return len(r),r,reader

def OpenFileReader(floc,delimiter='\t',encoding='cp865'):
	if encoding=='utf-8':
		encoding2='cp865'
	else:
		encoding2='utf-8'
	file=open(floc,"r",encoding=encoding)
	try:
		a=OpenReader(file,delimiter)
	except UnicodeDecodeError:
		file=open(floc,"r",encoding=encoding2)
		a=OpenReader(file,delimiter)
	if a==0:#file is empty
		return 0
	n,fcols,reader=a	
	if n==1:
		n,fcols,reader=OpenReader(file,';')	
	return n,fcols,reader,file
		

def MakeDBtxt(table,floc,fname,dbase,conn,crsr):
	a=OpenFileReader(floc)
	if a==0:
		return 0
	n,fcols,reader,file=a
	if not DB.TableExists(dbase,table,crsr):
		tblProps=DB.createTable(table,conn,crsr,fcols)
	else:
		DB.DeleteFileEntries(fname,dbase,conn,crsr)
		DB.DeleteIndex(table,dbase,conn,crsr)
		tblProps=DB.GetColumnNames(crsr,table)
	columns=tblProps[0]
	if DB.FileInDB(table,fname,crsr):
		print ( '%s all ready in table %s' (fname,table))
		return False
	tbl=[]
	
	if not columns[-1]=='FileName':
		columns.append('FileName')
	batchsize=20000
	m=0		
			
	for r in reader:
		for i in range(n):
			if 'date' in columns[i].lower():#The procedure recognizes any fieldname containng 'Date' as a date field, so dont call it that unless you want date!
				if not '-' in r[i]:
					if not ('/' in r[i]) or ('.' in r[i]):
						r[i]='%s-%s-%s' %(r[i][0:4],r[i][4:6],r[i][6:8])
					else:
						r[i]='%s-%s-%s' %(r[i][6:10],r[i][3:5],r[i][0:2])
			r[i]=r[i].replace(',','.')
			if tblProps[2][i]=='bigint':
				r[i]=r[i].replace(' ','')
		r.append(fname)
		tbl.append(tuple(r))
		if len(tbl)>batchsize:
			m+=batchsize
			DB.InsertTableIntoDB(conn,crsr, table, columns,tbl,dbase)
			print ( "%s rows inserted" %(m,))
			tbl=[]
	m+=len(tbl)
	DB.InsertTableIntoDB(conn,crsr, table, columns,tbl,dbase)
	file.close()
	return m


def AddRow(conn,crsr,dbase,table,columns,values,tblProps,fname):
	if len(columns)!=len(values):
		raise RuntimeError('not equal number of keys and values')
	if len(columns)>0:
		columns.append('FileName')
		values.append(fname)
		return DB.InsertIntoDB(conn,crsr,dbase,table,columns,values,tblProps)	
	return False
	
	
def GetValues(node,table,initkey=None,initval=None):
	#returns an array with keys and one with values
	keys=[]
	values=[]
	if not initkey is None:
		keys.append(initkey)
		values.append(initval)
	if len(node.childNodes)>0:
		for k in node.childNodes:
			if len(k.childNodes)>0:
				keys.append(k.nodeName)
				values.append(k.childNodes[0].nodeValue)
	return keys,values
	
	



