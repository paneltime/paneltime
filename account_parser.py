#!/usr/bin/python
# -*- coding: UTF-8 -*-

#Parser for importing accounting data from files 

import os
import DB
import Functions as fu
import datetime as dt

db='OSEData'


def parse(tbl,fdir,func,idfile=None):
	idtbl=fu.GetCSVMatrixFile(idfile,';','latin1')
	conn,crsr=DB.Connect(db)
	#DB.DropTable(tbl,conn,crsr,db) DO NOT UNCOMMENT unless you know what you ar doing!
	#DB.createTable(tbl,conn,crsr)
	sqlstr=DB.GetSQLInsertStr(tbl,columns[tbl])
	lables=None
	for fn in os.listdir(fdir):
		print ('parsing %s' %(fn))
		if fn.split('.')[1]=='csv':
			lables=func(fdir,fn,conn,crsr,sqlstr,lables,idtbl)
	DB.CreateIndex(conn,crsr,tbl,db)
		
	
	
def parse_ose(d,fn,conn,crsr,sqlstr,lables,idtbl):
	tbl=fu.GetCSVMatrixFile(d+'/'+fn,',')
	head=tbl[0]
	for i in range(1,len(tbl)):
		r=tbl[i]
		short_name,symb,long_name,compID,orgID,year,period,acc_type=r[0:8]
		print (long_name)
		for j in range(8,len(r)):
			h=head[j]
			h=h.replace('"','')
			h=h.replace('  ',' ')
			if h[0]=='.' or h[0]==' ':
				h=h[1:]
			if h[len(h)-1]=='.' or h[len(h)-1]==' ':
				h=h[:len(h)-1]
			if h in corrections:
				h=corrections[h].decode('utf-8')
			d=[short_name,symb,long_name,compID,orgID,year,period,acc_type,fn,h,tbl[i][j]]
			crsr.execute(sqlstr,tuple(d))
			conn.commit()			
	return lables


def parse_punched(d,fn,conn,crsr,sqlstr,lables,idtbl):
	if fn[-4:]!='.csv':
		return
	tbl=fu.GetCSVMatrixFile(d+'/'+fn,',','latin1')
	name=tbl[0][2]
	comp_id=findID(idtbl,name)
	for i in range(len(tbl)):
		for j in range(len(tbl[i])):
			if tbl[i][j]=='':
				tbl[i][j]=None
			else:
				tbl[i][j]=tbl[i][j].replace('|',',')
	lables=lables_check(tbl,lables)
	n=len(lables)
	corp=tbl[0]
	curr=tbl[1]
	thousands=tbl[2]
	year=tbl[3]
	for i in range(5,n):
		AppendData_punched(comp_id,tbl[i],lables[i][0],conn,crsr,sqlstr,corp,curr,thousands,year,d)
	return lables

def findID(idtbl,name):
	for cid,namei,isin in idtbl:
		if namei.lower().strip()==name.lower().strip():
			return cid
	a=0
	

def lables_check(tbl,lables):
	lbls=[[i[j] for j in range(4)] for i in tbl]
	lbls[0][2]=''
	
	if not lables is None:
		m=len(lables)
		if m<len(lbls):
			for i in range(m,len(lbls)):
				for j in lbls[i]:
					if j!=None:
						raise RuntimeError('extra information in sheet')
	else:
		m=len(lbls)
	for i in range(1,m):
		if lbls[i][0]==None:
			lbls[i][0]=lbls[i-1][0]
	if lables is None:
		lables=lbls
	else:
		if lables!=lbls[:m]:
			raise RuntimeError('lables do not correspond to the initial ones')
	return lables
	
def AppendData_punched(comp_id,data,acc_type,conn,crsr,sqlstr,corp,curr,thousands,year,d):
	n=len(data)
	if data[3]!=None:
		account,descriptionEN,description=data[1:4]
		for i in range(4,len(data)):
			if (corp[i]!='KONSERN' and corp[i]!=None) or  thousands[i]!=None:
				if int(thousands[i])!=1000 and int(thousands[i])!=1000000:
					raise RuntimeError('error in corp or thousand coding')
			if year[i]!=None:
				x=[corp[i]=='KONSERN',thousands[i]=='1000',curr[i],comp_id,account,acc_type,d,year[i],description,descriptionEN,data[i]]
				crsr.execute(sqlstr,tuple(x))
				conn.commit()
	pass
		
		
columns=dict()	
columns['account_ose']="""[short_name],[symbol] , [long_name],[companyID],[orgID] ,[Year] , [period] ,[account_type],[file_name],[Description],[Value]"""
columns['account_foreign']="""[IsCorporateAccount],[In1000],[Currency],[CompanyID],[account],[accountType],[Directory],[Year],[Description],[DescriptionEN],[Value]"""

corrections={"Admin.kostnader":"Administrasjonskostnader",
"Annen driftinntekt":"Annen driftsinntekt",
"andeler":"Andeler i ansvarlige selskaper",
"honnorarer":"honorarer",
"Annen opptjen egenkapital":"Annen opptjent egenkapital",
"Avskrivninger*":"Avskrivninger",
"Egne ikke-amotiserte sertifikater":"Egne ikke-amortiserte sertifikater",
"Eierinteresser kredittinstitusjoner":"Eierinteresser i kredittinstitusjoner",
"Endring av utsatt skatt":"Endring utsatt skatt",
"fly ol":"fly o.l",
"inventar":"inventar mm",
"Lån og innsk.fra kr.inst. u/avt løpetid":"Lån og innsk. fra kr.inst. u/avt løpetid",
"Netto gev/tap på aksjer_andre vp m/var.avk":"Netto gev/tap på aksjer & andre vp m/var.avk",
"Netto gev/tap på valuta og fin.derivater":"Netto gev/tap på valuta og fin. derivater",
"Netto innskudd i og utlån til fin. inst":"Netto innskudd i og utlån til fin.inst",
"Netto kontantstøm fra perioden":"Netto kontantstrøm fra perioden",
"obl og andre vp":"obl. og rentebærende verdipapir",
"obl og renteb. Vp":"obl. og rentebærende verdipapir",
"obl. og renteb. Vp":"obl. og rentebærende verdipapir",
"Obligasjoner og andre verdipapirer med fast avkastning":"obl. og rentebærende verdipapir",
"Obligasjoner ol":"obl. og rentebærende verdipapir",
"Opptj.ubet.innt.forsk.bet.ikke.pål.kostn":"Opptj. ubet. innt. forsk.bet. ikke pål.kostn",
"Pensjonsforpliktelse":"Pensjonsforpliktelser",
"Rentebærende vp utstedt av de offentlige":"Rentebærende vp utstedt av det offentlige",
"Sum av-og nedskr varige dr.m./immatr.eiend":"Sum av- og nedskr varige dr.m./immatr.eiend",
"Sum forskuddbet. Og opptjente inntekter":"Sum forskuddsbet. og opptjente inntekter",
"Sum rentekostnader o.l. kostn":"Sum rentekostnader o.l. kostnader",
"Sum utb. og andre innt. av vp m/var.avkastnin":"Sum utb. Og andre innt.av vp m/avkastning",
"Utlån til fordr på kr.inst.m/avtalt løpetid":"Utlån til & fordr på kr.inst.m/avtalt løpetid",
"Utlån til fordr på kr.inst.u/avtalt løpetid":"Utlån til & fordr på kr.inst.u/avtalt løpetid",
"Økning / Reduksjon av kortsiktig gjeld":"Økning / reduksjon av langsiktig gjeld"}



#parse('account_ose','/data/account/ose',parse_ose)
parse('account_foreign','../accountingdata/Punching 2018',parse_punched,'../accountingdata/utenlandske.csv')
		
