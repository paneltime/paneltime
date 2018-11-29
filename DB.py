#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../Modules')
import Functions as fu
import pymssql 
import csv
import DBCreate
import DBIndicies
import numpy as np
#import MySQLdb
dbase='OSEData'




def GetMSSQLTables(MSSQL):
	MSSQL.execute("""EXEC sp_tables  @table_type = "'table'" """)
	tablelist=MSSQL.fetchall()
	tablelist=[listitm[2] for listitm in tablelist]
	return tablelist

def Connect(db):
	conn = pymssql.connect(host='titlon.uit.no', user='OSEupdater', 
		                   password='FinansDB', database=db)  
	crsr=conn.cursor()
	return conn,crsr

def FileInDB(table,fname,crsr):
	SQLExpr='SELECT [FileName] FROM dbo.%s GROUP BY [FileName]' %(table,)
	crsr.execute(SQLExpr)
	r=crsr.fetchall()
	for i in r:
		if fname==i:
			return True
	return False

def IsDone(FieldName,FieldValue,crsr,tbl):
	SQLExpr="""SELECT distinct
      [%s]
	  FROM [OSEData].[dbo].[%s]
	  where [%s]='%s'""" %(FieldName,tbl,FieldName,FieldValue)
	crsr.execute(SQLExpr)
	r=crsr.fetchall()
	return len(r)>0

def TableExists(db,table,crsr):
	SQLExpr="""SELECT Distinct TABLE_NAME 
                FROM %s.information_schema.TABLES
                where TABLE_NAME='%s'""" %(db,table)
	crsr.execute(SQLExpr)
	r=crsr.fetchall()
	return len(r)==1

def Fetch(sqlstr,crsr):
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	return r

def Execute(sqlstr,conn,crsr):
	crsr.execute(sqlstr)
	conn.commit()

def GetALLTables(db,conn,crsr):
	SQLExpr='SELECT Distinct TABLE_NAME FROM %s.information_schema.TABLES' %(db,)
	crsr.execute(SQLExpr)
	r=crsr.fetchall()
	return r

def DeleteAllTables(db,EnableCheck=True):
	if EnableCheck:
		var = raw_input("Delete all tables in %s? Enter 'Y' to confirm: " %(db,))
		if var!='Y':
			return
	conn,crsr=Connect(db)
	r=GetALLTables(db,conn,crsr)
	for i in r:
		DropTable(i[0],conn,crsr,db)


def DeleteFileEntries(filename,db,conn,crsr):
	SQLExpr="""Delete FROM %s
      where [FileName]='%s'"""
	r=GetALLTables(db,conn,crsr)
	n=0
	for i in r:
		c=GetColumnNames(crsr,i)
		if 'FileName' in c[0]:
			SQLExpr2=(SQLExpr %(i[0],filename))
			crsr.execute(SQLExpr2)
			n+=crsr.rowcount
			conn.commit() 
	print ( "%s rows affected by deleting" %(n,))


def deleterows_byfieldval(fieldname,fieldval,tbl,db,conn,crsr):
	SQLExpr="""Delete FROM [%s].dbo.%s
			where [%s] ='%s'""" %(db,tbl,fieldname,fieldval)
	crsr.execute(SQLExpr)
	n=crsr.rowcount
	conn.commit() 
	print ( "%s rows affected by deleting" %(n,))



def GetColumnNames(crsr,TblName):
	SQLstr="EXEC sp_columns @table_name = '%s';" %(TblName)
	crsr.execute(SQLstr)
	r=crsr.fetchall()
	r=fu.transpose(r)

	return r[3:8]



def InsertIntoDB(conn,crsr,dbase, table,columns,values,tblProps=None):
	columns=ColNameWrapper(columns)
	n=len(columns)
	sstr=',%s'*n
	sstr='('+sstr[1:]+')'
	SQLExpr='INSERT INTO dbo.%s ' %(table,)
	SQLExpr+=sstr %tuple(columns)
	SQLExpr+=' VALUES '+sstr
	return InsertWithColumnCreation(conn,crsr,table,SQLExpr,columns,values,tblProps)

def InsertTableIntoDB(conn,crsr, table, columns,datatable,db):
	columns=ColNameWrapper(columns)
	n=len(columns)
	sstr=',%s'*n
	sstr='('+sstr[1:]+')'
	SQLExpr='INSERT INTO [%s].dbo.%s ' %(db,table,)
	SQLExpr+=sstr %tuple(columns)
	SQLExpr+=' VALUES '+sstr
	if True:#for debugging
		for i in datatable:
			crsr.execute(SQLExpr,i)

	else:
		crsr.executemany(SQLExpr,datatable)
	conn.commit()
	#print ( 'Table inserted')
	pass

def ColNameWrapper(columns,smallcaps=False):
	"wraps all column names in []"
	if type(columns)==str:
		columns=fu.Clean(columns)
	n=len(columns)
	for i in range(n):
		x=columns[i].replace('[','').replace(']','')
		if smallcaps:
			columns[i]='['+x.lower()+']'
		else:
			columns[i]='['+x+']'
	return columns

def InsertWithColumnCreation(conn,crsr,table,SQLExpr,columns,values,tblProps):
	values=tuple(values)
	err=False
	try:
		crsr.execute(SQLExpr,values)
		conn.commit()
	except pymssql.ProgrammingError as inst:
		err=True
		if inst[0]==207:
			tblProps[0]=GetColumnNames(crsr,table)
			AddCols(conn,crsr,table,columns,tblProps,values)
			try:
				crsr.execute(SQLExpr,values)
				conn.commit()
			except pymssql.OperationalError as inst:
				InsertWithColumnExtension(conn,crsr,table,columns,tblProps,values,inst,SQLExpr)
		else:
			raise pymssql.ProgrammingError(inst)
	except pymssql.OperationalError as inst:
		err=True
		InsertWithColumnExtension(conn,crsr,table,columns,tblProps,values,inst,SQLExpr)
	return err


def InsertWithColumnExtension(conn,crsr,table,columns,tblProps,values,inst,SQLExpr):
	if inst[0]==8152:
		tblProps[0]=GetColumnNames(crsr,table)
		AddCols(conn,crsr,table,columns,tblProps,values)
		tblProps[0]=GetColumnNames(crsr,table)
		ExtendCols(conn,crsr,table,columns,tblProps,values)
		values=FixT24Problem(values)
		crsr.execute(SQLExpr,values)
		conn.commit()
	elif inst[0]==242:
		values=FixT24Problem(values)
		crsr.execute(SQLExpr,values)
		conn.commit()        
	else:
		raise pymssql.OperationalError(inst)    

def FixT24Problem(values):
	if 'T24' in values[0]:
		values=list(values)
		values[0]=values[0].replace('T24','T00')
		values=tuple(values)
	return values


def drop_col(tbl,colname,conn,crsr,dbase):

	sqlstr=	"""select 
		dobj.name as def_name
	from sys.columns col 
		left outer join sys.objects dobj 
		    on dobj.object_id = col.default_object_id and dobj.type = 'D' 
	where col.object_id = object_id('[%s].dbo.[%s]')  and col.name='%s'
	and dobj.name is not null""" %(dbase,tbl,colname)
	r=Fetch(sqlstr,crsr)
	if len(r)>0:
		for i in r:
			Execute('ALTER TABLE [%s].dbo.[%s] DROP CONSTRAINT %s' %(dbase,tbl,i[0]),conn,crsr)
	Execute('alter table [%s].dbo.[%s] drop column %s' %(dbase,tbl,colname),conn,crsr)
		
	

def createTable(tbl,conn,crsr,cols=None,tabledef=None):
	"""crating a generic table"""
	SQLStr=''
	try:
		if tabledef is None:
			SQLStr=vars(DBCreate)[tbl]
		else:
			SQLStr=vars(DBCreate)[tabledef]
	except KeyError:
		pass
	FromCreate=SQLStr!=''
	if SQLStr=='':
		SQLStr="""CREATE TABLE %s(ID bigint NOT NULL IDENTITY,
		[FileName] [varchar](100) NULL DEFAULT (NULL)) """ %(tbl,)
	crsr.execute(SQLStr)
	conn.commit()
	tblProps=[GetColumnNames(crsr,tbl)]
	if not FromCreate and not cols is None:
		AddCols(conn,crsr,tbl,cols,tblProps)
	AddPrimaryKey(crsr,conn,tbl)	
	return tblProps[0]

def CreateIndex(conn,crsr,tbl,dbase,createID=False,IndexFields=''):
	dbstr=Getdbstr(dbase)
	if not HasIndex(tbl,crsr):
		AddPrimaryKey(crsr,conn,tbl,dbase,createID)
		if IndexFields=='':
			try:
				IndexFields=vars(DBIndicies)[tbl]
			except KeyError:	
				return
		print ( 'creating index IX_%s ON %s.[%s]' %(tbl,dbstr,tbl))
		crsr.execute("""CREATE NONCLUSTERED INDEX IX_%s ON %s.[%s] (%s)""" %(tbl,dbstr,tbl,IndexFields))
		conn.commit()
	
		
def Getdbstr(db):
	if db!='':
		dbstr='[%s].[dbo]' %(db,)
	else:
		dbstr='[dbo]'
	return dbstr

def HasIndex(tbl,crsr):
	SQLstr="""SELECT * 
            FROM sys.indexes 
            WHERE name='IX_%s'""" %(tbl,)
	crsr.execute(SQLstr)
	r=crsr.fetchall()
	return len(r)>0 

def GetSQLInsertStr(tbl,columns):
	n=len(columns.split(','))
	sstr=',%s'*n
	sstr='('+sstr[1:]+')'
	SQLExpr='INSERT INTO dbo.%s (%s)' %(tbl,columns)
	SQLExpr+=' VALUES '+sstr
	return SQLExpr

def DeleteIndex(tbl,dbase,conn,crsr):
	dbstr=Getdbstr(dbase)
	if HasIndex(tbl,crsr):
		print ( 'deleting index IX_%s ON %s' %(tbl,tbl))
		crsr.execute("""DROP INDEX IX_%s ON %s.[%s]""" %(tbl,dbstr,tbl))
		conn.commit()	

def DropPrimaryKey(crsr,conn,tbl,db=''):
	dbstr=Getdbstr(db)
	SQLStr="""ALTER TABLE %s.[%s] DROP CONSTRAINT PK_%s """ %(dbstr,tbl,tbl)
	crsr.execute(SQLStr)
	conn.commit()	


def AddPrimaryKey(crsr,conn,tbl,db='',createID=False):
	dbstr=Getdbstr(db)
	if createID:
		try:
			crsr.execute("""ALTER TABLE %s.[%s] ADD ID INT IDENTITY""" %(dbstr,tbl))
			conn.commit()
		except:
			pass
	try:
		crsr.execute("""ALTER TABLE %s.[%s] ADD CONSTRAINT
	        PK_%s PRIMARY KEY CLUSTERED (ID)""" %(dbstr,tbl,tbl))
		conn.commit()	
	except:
		pass

def AddCols(conn,crsr,table,columns,tblProps,values=None):
	existingcols=tblProps[0][0]
	existingcols=ColNameWrapper(existingcols,True)
	columns=ColNameWrapper(columns)
	n=len(columns)
	for i in range(n):
		if not(columns[i].lower() in existingcols):
			if not values is None:
				AddColumn(crsr,conn,table,columns[i],max(2*len(values[i]),10))
			else:
				AddColumn(crsr,conn,table,columns[i],20)




def AddColumn(crsr,conn,table,column,length):
	if table=='newsdump' and column=='[text]':
		SQLstr="""ALTER TABLE %s ADD %s varchar(max) DEFAULT NULL""" %(table,column)
	else:
		SQLstr="""ALTER TABLE %s ADD %s varchar(%s) DEFAULT NULL""" %(table,column,max((length,1)))
	crsr.execute(SQLstr)
	conn.commit()


def ExtendCols(conn,crsr,table,columns,tblProps,values):
	columns=ColNameWrapper(columns)
	lenghts=tblProps[0][3]
	keys=ColNameWrapper(tblProps[0][0])
	existingdict = dict(zip(keys, lenghts))
	n=len(columns)
	for i in range(n):
		if existingdict[columns[i]]<len(values[i]):
			ExtendColumn(crsr,conn,table,columns[i],values[i])



def ExtendColumn(crsr,conn,table,column,value):
	if table=='newsdump' and column=='[text]':
		SQLstr="""ALTER TABLE %s ALTER COLUMN %s varchar(max)""" %(table,column)
	else:
		SQLstr="""ALTER TABLE %s ALTER COLUMN %s varchar(%s)""" %(table,column,max(2*len(value),10))
	crsr.execute(SQLstr)
	conn.commit()


def DropTable(table,conn,crsr,db):
	"Deletes a table"
	try:
		crsr.execute("DROP TABLE [%s].[dbo].[%s];" %(db,table))
		conn.commit()
	except:
		pass

def GetMSSQLTables(MSSQL):
	MSSQL.execute("""EXEC sp_tables  @table_type = "'table'" """)
	tablelist=MSSQL.fetchall()
	tablelist=[listitm[2] for listitm in tablelist]
	return tablelist

def GetAdjFacts(conn,crsr,ISIN):
	sqlstr="""SELECT distinct [EventDate],
			year([EventDate]),month([EventDate]),
	        day([EventDate]),[AdjustmentFactor],[NumberOfShares],[TotalNumberOfShares],[SubscriptionPrice],[MarketPrice],[EventId],[Type]
	        FROM [OSEData].[dbo].[Adjustments]
	        where [ISIN]='%s'
	        order by EventDate""" %(ISIN,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[]
	#d,adj=fixRightsIssueProblem(r)
	d,a=DateAndPrice(r,8)
	adj=a[:,0:1]
	return d,adj

def fixRightsIssueProblem(r):
	d,a=DateAndPrice(r,8)
	adj=a[:,0]
	nsh=a[:,1]
	tnsh=a[:,2]
	spr=a[:,3]
	mpr=a[:,4]
	for i in range(1,len(r)):
		if r[i][10]=='RightsIssue' and nsh[i-1]>0 and tnsh[i-1]>0 and tnsh[i]>nsh[i-1] and mpr[i]>0 and spr[i]>0:
			newsh=tnsh[i]-tnsh[i]
			oldsh=tnsh[i-1]
			adj[i]=(newsh*spr[i]+oldsh*mpr[i])/((newsh+oldsh)*mpr[i])
	return d,adj
	

def GetFundAdjFacts(conn,crsr,SecID):
	sqlstr="""SELECT distinct [Date], 
			year([Date]),month([Date]),
	        day([Date]),[AdjFactor]
	        FROM [OSEData].[dbo].[fund_adj_factors]
	        where [SecurityID]='%s'
	        order by [Date]""" %(SecID,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[]
	d,a=DateAndPrice(r)
	return d,a


def GetDividends(conn,crsr,SecID,curr):
	sqlstr="""SELECT DISTINCT [EventDate],
	year([EventDate]),month([EventDate]),
          day([EventDate]),[DividendInNOK],[DividendInForeignCurrency]
		  ,[DividendId]
          FROM [OSEData].[dbo].[equityfeed_Dividend]
          where [SecurityId]=%s and [DividendInForeignCurrency]!=0
          order by EventDate""" %(SecID,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[]
	d,dividends=DateAndPrice(r,5)
	if curr!='NOK':
		ds=dividends[:,1]
	else:
		ds=dividends[:,0]
	return d,ds

def GetFundDividends(conn,crsr,SecID):
	sqlstr="""SELECT DISTINCT [Date],
	year([Date]),month([Date]),
          day([Date]),[Dividend]
          FROM [OSEData].[dbo].[fund_dividends]
          where [SecurityId]=%s  and [Dividend]!=0
          order by Date""" %(SecID,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[]
	d,dividends=DateAndPrice(r)
	return d,dividends

def GetFundPrices(conn,crsr,SecID):
	sqlstr="""SELECT distinct fnd.[Date],
	            year(fnd.[Date]),month(fnd.[Date]),day(fnd.[Date])
	            ,[NAV]
				, osebx.[Last] as [LinkedOSEBXIndex]
				, osefx.[Last] as [OSEFXIndex]
				,obx.[Last] as [OBXIndex]

	        FROM [OSEData].[dbo].[fund_prices]  as fnd
	left join
	  (SELECT DISTINCT [Date] ,[SecurityId] ,[Last]
	   FROM [OSEData].[dbo].[equityindex_linked]
	   where [SecurityId]=2) as osebx
	   ON osebx.[Date]=fnd.[Date]

	left join
	  (SELECT DISTINCT [Date] ,[SecurityId] ,[Last]
	   FROM [OSEData].[dbo].[equityindex_linked]
	   where [SecurityId]=799) as osefx
	   ON osefx.[Date]=fnd.[Date]
	left join
	  (SELECT DISTINCT [Date] ,[SecurityId] ,[Last]
	   FROM [OSEData].[dbo].[equityindex_linked]
	   where [SecurityId]=9026) as obx
	   ON obx.[Date]=fnd.[Date]
	left join
	  (SELECT [Date],[index] FROM [OSEData].[dbo].[equity_titlon_index]) as tix
	   ON tix.[Date]=fnd.[Date]

where fnd.[SecurityID]=%s
order by [Date]""" %(SecID,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[],[]
	d,p=DateAndPrice(r)
	p,d,r=FillInnPriceGaps(p,d,r)
	return d,p,r

def FillInnPriceGaps(p,d,r):
	"uses last observed price to fill in price gaps"
	n=len(p)
	for i in range(n-2):
		if len(np.nonzero(p==0)[0])==0:
			break
		ps=fu.ShiftArray(p,-i-1)
		p=(p==0)*ps+(p!=0)*p
	k=np.nonzero(p)[0]
	if len(k)>0:
		k=k[0]
		p=p[k:]
		d=d[k:]
		r=r[k:]	
		return p,d,r
	else:
		return [],[],[]

def GetPrices(conn,crsr,ISIN):
	sqlstr="""SELECT DISTINCT eq.[Date],
          year(eq.[Date]),month(eq.[Date]),day(eq.[Date])
          ,[BestBidPrice]
          ,[BestAskPrice]
          ,[Open]
          ,[High]
          ,[Low]
          ,[LastTradedPrice]
          ,[OfficialNumberOfTrades]
          ,[OfficialVolume]
          ,[UnofficialNumberOfTrades]
          ,[UnofficialVolume]
          ,[VolumeWeightedAveragePrice]
		  ,osebx.[Last] as [LinkedOSEBXIndex]
		  ,obx.[Last] as [OBXIndex]

      FROM [OSEData].[dbo].[equitypricedump] as eq
	left join
	  (SELECT DISTINCT [Date] ,[SecurityId] ,[Last]
	   FROM [OSEData].[dbo].[equityindex_linked]
	   where [SecurityId]=2) as osebx
	   ON osebx.[Date]=eq.[Date]
	left join
	  (SELECT DISTINCT [Date] ,[SecurityId] ,[Last]
	   FROM [OSEData].[dbo].[equityindex_linked]
	   where [SecurityId]=9026) as obx
	   ON obx.[Date]=eq.[Date]
	left join
	  (SELECT [Date],[index] FROM [OSEData].[dbo].[equity_titlon_index]) as tix
	   ON tix.[Date]=eq.[Date]

  where [ISIN]='%s'
  order by [Date]""" %(ISIN,)
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	if len(r)==0:
		return [],[],[]
	d,pr=DateAndPrice(r,12)
	p=pr[:,5:6]
	m=(pr[:,3:4]*pr[:,4:5]>0)*(pr[:,3:4]+pr[:,4:5])/2#mean high low
	ab=(pr[:,0:1]*pr[:,1:2]>0)*(pr[:,0:1]+pr[:,1:2])/2#mean bid ask
	ab=((((pr[:,0:1]/(ab+1e-20)))+((pr[:,1:2]/(ab+1e-20))))<0.2)*ab
	p=(p==0)*m+(p!=0)*p
	p=(p==0)*ab+(p!=0)*p
	p,d,r=FillInnPriceGaps(p,d,r)
	return d,p,r

def DateAndPrice(r,lastcol=4):
	r=np.array([i[1:lastcol+1] for i in r],dtype=float)
	r=np.nan_to_num(r)
	d=np.array(r[:,0:3],dtype=int)
	d=d[:,0:1]*10000+d[:,1:2]*100+d[:,2:3]
	p=r[:,3:lastcol]	
	return d,p

def GetSecurityList(conn,crsr):
	sqlstr="""SELECT distinct
      t0.[SecurityId]
	  ,t1.[CompanyId]
      ,[ISIN]
      ,[SecurityType]
      ,[Symbol]
      ,[SecurityName]
      ,[Currency]
      FROM [OSEData].[dbo].[equityfeed_EquitySecurityInformation] as t0
	  left join
	  (select distinct [CompanyId],[SecurityId] from [OSEData].[dbo].[equityfeed_EquityInformation]) as t1
	  on t0.[SecurityId]=t1.[SecurityId]
	 


	  """
	crsr.execute(sqlstr)
	return crsr.fetchall()


def GetFundList(conn,crsr):
	sqlstr="""SELECT distinct
      [SecurityId]
      ,[ISIN]
      ,[FundId]
      ,[Symbol]
      ,[Name]
	  FROM [OSEData].[dbo].[funds]


"""
	crsr.execute(sqlstr)
	return crsr.fetchall()


def CopyTable(conn,crsr,fromtbl,totbl,fromdb,todb=None):
	if todb is None:
		todb=fromdb
	sqlstr="""SELECT * INTO [%s].[dbo].[%s]
            FROM [%s].[dbo].[%s]""" %(todb,totbl,fromdb,fromtbl)
	crsr.execute(sqlstr)
	conn.commit()

	
