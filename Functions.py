#!/usr/bin/python
# -*- coding: UTF-8 -*-

#This module contains some useful functions

import csv
import numpy as np
import urllib.request
from datetime import datetime, timedelta
import os
import ssl
import win32com.client
import re

#import warnings
#warnings.simplefilter('error')
#np.seterr(all='raise')


def DateNDaysAgo(strdate,N,fmat='%Y-%m-%d'):
	date_N_days_ago = datetime.strptime(strdate,fmat) - timedelta(days=N)
	return date_N_days_ago.strftime(fmat)

def DayDifference(strdt0,strdt1,fmat='%Y-%m-%d'):
	d0 = datetime.strptime(strdt0,fmat)
	d1 = datetime.strptime(strdt1,fmat)
	delta = d0 - d1
	print ( delta.days	)

def DownloadFile(url,savedir='',showprogress=False):

	file_name = url.split('/')[-1]
	if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
		ssl._create_default_https_context = ssl._create_unverified_context
	u = urllib.request.urlopen(url)
	f = open(savedir+file_name, 'wb')

	if showprogress:
		meta = u.info()
		file_size = int(meta.getheaders("Content-Length")[0])
		print ( "Downloading: %s Bytes: %s" % (file_name, file_size))

	file_size_dl = 0
	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break
		f.write(buffer)
		if showprogress:
			file_size_dl += len(buffer)
			status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
			status = status + chr(8)*(len(status)+1)
			print ( status,)
	f.close() 
	return savedir+file_name


def det(Matrix):
	"""Rreturns the determinant of a matrix. In case of an
	exception in np.linalg 1e+308 is returned """
	with np.errstate(all='raise'):
		try: 
			dt=np.linalg.det(Matrix)
		except FloatingPointError:
			dt=1e+308
	return dt

def TimeFormat(TimeVal):
	"""Returns a string representing the time corresponding to a float variable in a '1.0=24 hours' time system"""
	Hours=int(TimeVal*24.0)
	TimeVal=TimeVal-(Hours/24.0)
	Minutes=int(TimeVal*60.0*24.0)
	TimeVal=TimeVal-(Minutes/(60.0*24.0))
	Seconds=int((TimeVal*60.0*60.0*24.0)+0.001)
	if Seconds>=60:
		Seconds=Seconds-60
		Minutes=Minutes+1
	if Minutes>=60:
		Minutes=Minutes-60
		Hours=Hours+1
	if Hours>=24:
		Hours=Hours-24
	if Hours<10:Hours="0"+str(Hours)
	if Minutes<10:Minutes="0"+str(Minutes)
	if Seconds<10:Seconds="0"+str(Seconds)
	TimeVal =str(Hours)+str(Minutes)+str(Seconds)

	return TimeVal

def run_excelmacro(modulename):
	xl=win32com.client.Dispatch("Excel.Application")
	xl.Application.Run(modulename)
	pass

def GetCSVMatrixFile(filename,delimiter=';',encoding='utf-8'):
	"""Returns the contents of FileName in the form of a matrix"""
	if not filename.lower()[-4:]=='.csv':
		filename=filename+'.csv'
	file = open(filename, "r",encoding=encoding)
	reader=csv.reader(file,quoting=csv.QUOTE_NONE,delimiter=delimiter)
	TempVar=[]
	for row in reader:
		TempVar.append(row)
	file.close()
	return TempVar

def WriteCSVMatrixFile(FileName,Variable,currpath=False):
	"""stores the contents of Variable to a file named FileName"""
	FileName=FileName.replace('.csv','')
	if "../Input/" in FileName or currpath:
		file = open("%s.csv" %(FileName), "w")
	else:
		file = open("../Output/%s.csv" %(FileName), "w")
	writer = csv.writer(file,delimiter=';',lineterminator='\n')
	writer.writerows(Variable)
	file.close()
	pass


def SaveSQLTable(conn,crsr,SQLstr,filename):
	"""Saves an SQL table as a csv file"""

	if SQLstr==None:
		SQLstr="SELECT * FROM %s;" % (Table)
	crsr.execute(SQLstr)
	r=crsr.fetchall()
	desc=transpose(crsr.description)
	r.insert(0,desc[0])

	WriteCSVMatrixFile(filename,r)
	print ( "... done")



def transpose(X):
	rows=len(X)
	cols=len(X[0])
	ret=[[0]*rows for i in range(cols)]
	for i in range(rows):
		for j in range(cols):
			ret[j][i]=X[i][j]
	return ret


def SaveVar(variable,name='tmp'):
	"""takes var and name and saves var with filname <name>.csv """	
	SaveVars(((variable,name),))

def SaveVars(varlist):
	"""takes a tuple of (var,name) pairs and saves numpy array var 
	with <name>.csv. Use double brackets for single variable."""		
	for var,name in varlist:
		if type(var)!=np.ndarray:
			var=np.array(var)
		name=name.replace('.csv','')
		if "../Input/" in name:
			np.savetxt("%s.csv" %(name),var,delimiter=";")
		else:
			np.savetxt("../Output/%s.csv" %(name),var,delimiter=";")

def LoadVars(varlist):
	"""takes a set of [var,name] pairs and loads the files with name into var for each pair. 
	Used to retrieve data saved with SaveVars."""
	retvars=[]
	for name in varlist:
		name=name.replace('.csv','')
		ret=np.loadtxt("../Output/%s.csv" %(name),delimiter=";")
		if ret.size==1:
			ret=np.array([ret])
		retvars.append(ret)
	return retvars

def CSVMatirxToNumpy(filename,delimiter=";"):
	"""Uses np.loadtxt() to load a csv-file to a numpy array"""
	filename=filename.replace('.csv','')
	filename=filename+'.csv'
	return np.loadtxt(filename,delimiter=delimiter)

def ShiftArray(npArray,nElements,EmptyVal=0):
	"""For nElements>0 (nElements<0) this function shifts the elements up (down) by deleting the top (bottom)
	nElements and inserting at botom (top) nElements with EmptyVal"""
	if nElements==0:
		return npArray
	n=len(npArray)
	if type(npArray[0])==np.ndarray:#is a matrix
		k=len(npArray[0])
		Fill=np.ones((abs(nElements),k),dtype=npArray.dtype)*EmptyVal
	else:
		Fill=np.ones(abs(nElements),dtype=npArray.dtype)*EmptyVal
	if nElements<0:
		return np.append(Fill,npArray[0:n+nElements],0)
	else:
		return np.append(npArray[nElements:],Fill,0)

def Unique(arr):
	"""takes a flat array and returns the indicies of the first uniqe elements in array, 
	assuming arr is sorted in ascending order"""
	unq=np.nonzero(ShiftArray(arr,-1,-arr[0])!=arr)[0]
	return unq

def MDeterm(Matr):
	"""Returns the determinant of MDeterm"""
	ub=len(Matr)
	L,U=LU(Matr)
	retvar=1
	for i in range(ub):
		retvar=retvar*L[i][i]#The product of the elements in the diagonal of the L matrix is the determinant
	return retvar

def LU(matr):
	"""This function makes a LU decomposition of the matrix matr for the MInverse function, so that matrix=L x U. This 
	function is used in many matrix operations such as inversion and eigen values.
	Global variables set: L (lower triangular matrix), U (upper triangular matrix)"""
	Sum=0.0
	ub=len(matr)
	L=np.diag(np.diag(np.ones((ub,ub))))   
	U=np.zeros((ub,ub)) 
	for k in range(ub):	
		for i in range(k,ub):
			Sum=0
			for m in range(k):
				Sum =Sum + L[i][m] * U[m][k]
			L[i][k] = matr[i][k] - Sum
		if abs(L[k][k]) < 1E-18:
			L[k][k] = 1E-18
		for j in range(k+1,ub):	
			Sum = 0
			for m in range(k):
				Sum=Sum+L[k][m] * U[m][j]
			U[k][j] = (matr[k][j] - Sum) / L[k][k]
	return L,U

def Split(array):
	"""Abbrevation for np.split(array,len(array[0]),1)"""
	ret=np.split(array,len(array[0]),1)
	return ret

def ConditionalCumsum(array,condition=0):
	"""Generates a conditional cumsum that restarts 
	everytime condition is met. Array is flattned"""
	array=array.flatten()*1
	n = array==condition
	a = ~n
	c = np.cumsum(a)
	d = np.diff(np.concatenate(([0], c[n])))
	array[n] = -d
	return np.cumsum(array)


def Concat(arrSec,dim=1):
	"""Abbrevation for np.concatenate(arrSec,1), 
	arrays are converted to (n,1) row vectors. Works only for dim={0,1}"""
	arr=[]
	for i in arrSec:
		if type(i)!=np.ndarray:
			i=np.array(i)
		if len(i.shape)==1:
			if dim==1:
				arr.append(i.reshape((len(i),1)))
			else:
				arr.append(i.reshape((1,len(i))))
		else:
			arr.append(i)
	return np.concatenate(arr,dim)

def is_number(s,less=None,greater=None):
	try:
		s=float(s)
		test=True
		if not less is None:
			test=s<less
		if not greater is None:
			test=(s>greater) and test
		return test
	except ValueError:
		return False

def Clean(string,split=','):
	"""Cleans the text for linfeed etc., and splits the text if split=True"""
	strtmp=string.replace('\n','').replace('\t','').replace(' ','')
	if split==False or split==None:
		return strtmp
	return strtmp.split(split)

def ListInsert(elem,listvar,pos,replace=False):
	"""Inserts the list elem as new elements in the list listvar, starting at position pos. 
	Hence, the length of the list is increased by len(elem). 
	If replace, the element at position pos in listvar is replaced by elem, and the lenght is increased
	by len(elem)-1"""
	if replace:
		listvar.pop(pos)
	return listvar[0:pos]+elem+listvar[pos:]

def RetArg(arg):
	"Don't ask"
	return arg

def FlattenList(listoflists):
	l=listoflists
	res=[]
	for i in listoflists:
		if type(i)==list:
			for j in i:
				res.append(j)
		else:
			res.append(i)
	return res

def GetStringBetween(text,startstr,endstr):
	
	result = re.search(startstr+'(.*)'+endstr, text)
	if result==None:
		return ''
	return result.group(1)


def prntout(txt,init=False):
	if init:
		file=open('../Output/output.txt','w+')
	else:
		file=open('../Output/output.txt','a')
	file.write(str(txt)+'\r\n')
	file.close()
	print ( txt)

def strInList(lst,string):
	for i in lst:
		if string in i:
			return True
	return False