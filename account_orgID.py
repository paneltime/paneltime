#!/usr/bin/python
# -*- coding: UTF-8 -*-

#for matching org id from brønøysundregistrene with company id
#procedure:
#1. download the file with all org ids from brønnøysund (currently 'enheter_alle.json')
#2. run this modue, and first make sure both conditions in main() are True (so that all the code runs) 
#3. inspect the database table account_unpunched. If there are entries there with organization ID, then
#   account dat can be downloaded, so add them to the org.csv file
#5. run account_brsnd.py to fetch online account data
#4. firms that are listed in account_unpunched without 



import csv
import DB
import DBparser
import datetime
import Functions as fu
import json
import re
db='OSEData'

def main():
	tbl_name_tmp='account_org_brnsnd_all'
	tbl_name='account_org_brnsnd'	
	
	if True:
		add_org_file('../accountingdata/org.csv') #adds org.csv to the database
	
	if False:#set to true when you have downloaded data from brønnøysund
		addbrnsund_org_ID(tbl_name_tmp,'../accountingdata/enheter_alle.json') #adds the brønnøysund file to the database
		addbrnsund_org_ID_csv(tbl_name_tmp,'../accountingdata/organisasjonsnummer.csv') #adds the brønnøysund file to the database

	#Filters out the uninteresting org-types
	conn,crsr=DB.Connect(db)
	if True:
		
		DB.DropTable(tbl_name,conn,crsr,db)
		DB.Execute("""
	    SELECT * INTO [OSEData].[dbo].[%s]
	    FROM [OSEData].[dbo].[%s]
	   where [orgKode]='AS' or [orgKode]='ASA' or [orgKode]='SPA'""" %(tbl_name,tbl_name_tmp), conn, crsr)
		conn.commit()
		DB.CreateIndex(conn, crsr, tbl_name, db,False,'[organisasjonsnummer],[navn]')
	#DB.DropTable(tbl_name_tmp,conn,crsr,db)
	
	
	#identifies the problamatic companies without orgID
	tbl='account_unpunched'
	DB.DropTable(tbl,conn,crsr,db)
	DB.Execute("""
    SELECT * INTO [OSEData].[dbo].[%s]
    FROM (%s) as maintbl
	ORDER BY [Name],[navn]""" %(tbl, sqlstr), conn, crsr)
	conn.commit()
	
	
def addbrnsund_org_ID_csv(tbl_name,fname):
	"""adds the entire brønnøysund data file to the database"""
	
	conn,crsr=DB.Connect(db)


	#DB.DropTable(tbl_name,conn,crsr,db)
	#DB.createTable(tbl_name,conn,crsr)	
	
	file=open(fname,encoding='utf-8')

	
	today=str(datetime.date.today())

	k=0
	c=['[%s]' %(i,) for i in cols_brnsdnd]
	
	for r in file.readlines():
		if k==0:
			h=r.replace('"','').replace('\n','').split(';')
		else:
			d=[]
			s=r.replace('"N"','0').replace('"J"','1')
			s=s.replace('"','').replace('\n','')			
			cols=s.split(";")
			if len(cols)!=len(h):
				quotes=re.findall('"(.*?)"',r)
				for q in quotes:
					if ';' in q:
						s=s.replace(q,q.replace(';',' - '))
			cols=s.split(";")
			for j in cols_csv:
				i=h.index(j)
				d.append(cols[i])
			d.append(fname)
			DB.InsertIntoDB(conn, crsr, 'OSEData',tbl_name , c, d)
		k+=1

	#DB.CreateIndex(conn, crsr, tbl_name, db,False,'[organisasjonsnummer],[navn]')


	
	
def addbrnsund_org_ID(tbl_name,fname):
	"""adds the entire brønnøysund data file to the database"""
	
	conn,crsr=DB.Connect(db)


	#DB.DropTable(tbl_name,conn,crsr,db)
	#DB.createTable(tbl_name,conn,crsr)	
	
	file=open(fname,encoding='utf-8')
	data=json.load(file)
	
	today=str(datetime.date.today())

	k=0
	c=['[%s]' %(i,) for i in cols_brnsdnd]
	
	for r in data:
		d=[]
		for j in cols_brnsdnd:
			if j!='orgKode':		
				if j in r:
					if type(r[j])==dict:
						d.append(str(r[j])[:200])
					else:
						if type(r[j])==str:	
							d.append(r[j][:200])
						else:
							d.append(r[j])
				elif j!='':
					d.append(None)
			else:
				d.append(r['organisasjonsform']['kode'])
		d.append(fname)
		DB.InsertIntoDB(conn, crsr, 'OSEData',tbl_name , c, d)

	#DB.CreateIndex(conn, crsr, tbl_name, db,False,'[organisasjonsnummer],[navn]')


	
def add_org_file(fname):
	"adds org.csv to the db"
	conn,crsr=DB.Connect(db)
	tbl_name='account_OID'
	DB.DropTable(tbl_name,conn,crsr,db)
	DB.createTable(tbl_name,conn,crsr)	
	file=open(fname,encoding='cp865')
	
	

	k=0
	for r in file.readlines():
		k+=1
		if k>1:
			r=r.replace('\n','')
			r_split=r.split(';')
			
			r_split.append(fname)
			DB.InsertIntoDB(conn, crsr, db, tbl_name, cols_OID, r_split)
			p=0

	DB.CreateIndex(conn, crsr, tbl_name, db,False,'[CID] ,[Name] ,[OrgID]')


cols_OID="""[CID] ,
[Name] ,
[OrgID],
[fname]"""

cols_brnsdnd=[ 'organisasjonsnummer' ,
 'navn' ,
 'naeringskode1',
 'registreringsdatoEnhetsregisteret' ,
 'registrertIMvaregisteret' ,
 'konkurs' ,
 'underAvvikling' ,
 'antallAnsatte',
 'registrertIForetaksregisteret',
 'forretningsadresse',
 'organisasjonsform',
 'orgKode', 
 'source']

cols_csv=[
    'organisasjonsnummer', 
    'navn', 
    'naeringskode1.kode', 
    'registreringsdatoEnhetsregisteret', 
    'registrertIMvaregisteret', 
    'konkurs', 
    'underAvvikling', 
    'antallAnsatte', 
    'registrertIForetaksregisteret', 
    'forretningsadresse.adresse', 
    'orgform.beskrivelse', 
    'orgform.kode', 
]

sqlstr="""


select * from

	(select T.* from
		(select T3.* from
			(SELECT
				[CompanyId]
                ,[ISIN]
				,[Name]
				,max(year([Date])) as maxyear
			FROM [OSE].[dbo].[equity]
			group by [CompanyId],[Name],[ISIN]) as T3
			where T3.[maxyear]> 2000) as T
			
	left join
	(SELECT distinct
		[CID]
		,[Name]
		,[OrgID]

	FROM [OSEData].[dbo].[account_OID]) as OID
	ON OID.[CID]=T.[CompanyID]
	WHERE OID.[CID] is NULL) as T2 /* filters out all  */

left join
	(SELECT  [organisasjonsnummer]
      ,[navn]
      ,[orgKode]
	FROM [OSEData].[dbo].[account_org_brnsnd]) as OID
	ON OID.[navn] like '%'+T2.[Name]+'%'


left join
	(SELECT * FROM
		(SELECT distinct
			[CompanyID] as cidf
		FROM [OSEData].[dbo].[account_foreign]) AS F
	) AS [foreign]
	ON T2.[CompanyID] =[foreign].[cidf]

left join
	(SELECT * FROM
		(SELECT distinct
			[CompanyID] as cidose
		FROM [OSEData].[dbo].[account_ose]) AS O
	) AS [OSE]
	ON T2.[CompanyID] =[OSE].[cidose]



WHERE (OID.[orgKode] is null or OID.[orgKode]='ASA'  or OID.[orgKode]='SPA') and [foreign].[cidf] is null and not (T2.[Name] like '%XACT%' or T2.[Name] like '%OBX%')
 and ([OSE].[cidose] is null or T2.[maxyear]>2010)

"""

main()