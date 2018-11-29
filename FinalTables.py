#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
import numpy as np
import Functions as fu

wrnt_info_tbl='WarrantsInfo'
wrnt_tbl='Warrants'
secinfo_tbl='SecuritiesInfo'
bondinfo_tbl='BondInfo'
index_tbl='equityindex'
futforw_tbl='FutureForwards'
fund_tbl='mutualfund'
equity_tbl='equity'
bond_tbl='bonds'
equityextended_tbl='equity_extended'
options='options'
allsecs_tbl='AllSecurities'
tblcol_tbl='AllFields'
bondix_tbl='bondindex'
dbase='OSE'
wdbase='OSEData'


def MakeTables():
	conn,crsr=DB.Connect(dbase)
	make_bondindex(conn,crsr)
	MakeWarrantInfoTbl(conn,crsr)
	MakeWarrantsTable(conn,crsr)
	MakeSecInfoTable(conn,crsr)
	MakeBondInfoTable(conn,crsr)
	MakeIndexTable(conn,crsr)
	FutForw(conn,crsr)
	MakeFundTable(conn,crsr)
	maketable_factors(conn,crsr)
	MakeEquityTable(conn,crsr)
	MakeBondPricesTable(conn,crsr)
	MakeAllSecuritiesTable(conn,crsr)
	MakeTableColDef(conn,crsr)
	MakeOptionsTable(conn, crsr)
	
	conn.close()

def FutForw(conn,crsr):
	sqlstr="""
SELECT [ID]
      ,[Date]
      ,T.[SecurityId]
	  ,EqInfo.[Name] as IssuerName
      ,[Symbol]
      ,[ISIN]
      ,[ContractSymbol]
      ,[Description]
      ,[ExDate]
      ,[IssuerSecurityId]
      ,[IssuerSymbol]
      ,[BestBidPrice]
      ,[BestAskPrice]
      ,[Open]
      ,[High]
      ,[Low]
      ,[LastTradedPrice]
      ,[NumberOfTrades]
      ,[Volume]
      ,[VolumeWeightedAveragePrice]
	  into [OSE].[dbo].[%s]
  FROM [OSEData].[dbo].[futforw_prices] T
  left join
  (SELECT DISTINCT
	              [SecurityId]
	              ,[Name]
	          FROM [OSEData].[dbo].[%s]) as EqInfo
	on T.[IssuerSecurityId]=EqInfo.[SecurityId]
	ORDER BY [Date],T.[SecurityId]
  """ %(futforw_tbl,equity_tbl)
	DB.DropTable(futforw_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,futforw_tbl,dbase,
	            IndexFields="""
	            [Date]
	            ,[SecurityId]
	            ,[ISIN]
	            ,[IssuerSecurityId]""")		
	
def make_bondindex(conn,crsr):
	sqlstr="""
SELECT distinct [Date]
      ,[SecurityId]
      ,[Symbol]
      ,[Name]
      ,[IndexType]
      ,[Open]
      ,[High]
      ,[Low]
      ,[Close]
      ,[OpenYield]
      ,[CloseYield]
      ,[OpenDuration]
      ,[CloseDuration]
	  into [OSE].[dbo].[%s]
  FROM [OSEData].[dbo].[%s] T
  """ %(bondix_tbl,bondix_tbl)
	DB.DropTable(bondix_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,bondix_tbl,dbase,
	            IndexFields="""
	   [Date]
      ,[SecurityId]
      ,[Symbol]
      ,[Name]
      ,[IndexType]""")		
	
	
def MakeWarrantInfoTbl(conn,crsr):
	sqlstr="""SELECT distinct
	[FromDate]
      ,[ToDate]
      ,t0.[SecurityId]
      ,t0.[Symbol]
      ,t0.[CompanyId]
      ,[CurrencyCode]
      ,[SubscriptionPrice]
      ,[WarrantType]
      ,[UnderlyingSecurity]
      ,[TermsShares]
      ,[WarrantsIssued]
      ,[TermsWarrants]
      ,[SecurityType]
      ,[ISIN]
      ,[SecurityName]
      ,[Currency]
      ,[Exchange]
      ,[EquityClassId]
      ,[Description]
	INTO [OSE].[dbo].[%s]
  FROM [OSEData].[dbo].[equityfeed_Warrants] as t0
  left join
(SELECT distinct
	 [SecurityId]
      ,[CompanyId]
      ,[SecurityType]
      ,[Symbol]
      ,[ISIN]
      ,[SecurityName]
      ,[Currency]
      ,[Exchange]
      ,[FromDate]
      ,[ToDate]
      ,[EquityClassId]
      ,[Description]
  FROM [OSEData].[dbo].[Securities]) as t1
  on t1.[SecurityId]=t0.[SecurityId]
  ORDER BY [SecurityId]""" %(wrnt_info_tbl,)
	DB.DropTable(wrnt_info_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,wrnt_info_tbl,dbase,True,
	            IndexFields="""
	            [SecurityId]
	            ,[CompanyId]
	            ,[CurrencyCode]
	            ,[ISIN]
	            ,[FromDate]""")	





def MakeBondInfoTable(conn,crsr):
	sqlstr="""SELECT distinct
		  [Type]
		  ,ISNULL(t1.[SecurityId], [sid])
		  ,t1.[SecurityType]
		  ,t1.[Symbol]
		  ,t1.[ISIN]
		  ,t1.[SecurityName]
	      ,[EqName],[EqSecurityID],[eqISIN]
	      ,t0.[CompanyId]
	      ,[Name] as [Issuer]
	      
		  ,[FromDate]
		  ,ISNULL([ToDate],[To_Date])
		  ,[IssueDate]
		  ,[MaturityDate]
		  ,[ActualMaturityDate]
		  ,[CouponDate]
		  ,[OpeningDate]
		  ,[ClosingDate]
		  ,[ExpirationDate]
		  ,[AnnouncementDate]
		  ,[InterestRegulationDate]
	          
		  ,t1.[CurrencyCode]
		  ,[CountryCode]
		  ,[SectorId]
		  ,[Exchange]
	      ,[ListingType]
		  ,[BondIssueType]
		  ,[BondType]
		  ,[AmortizationType]
		  ,[DownPaymentMethod]
	      ,[BondDescription]
	      
		  ,[InterestAsOf]
		  ,[FirstCouponPayment]
		  ,[FirstDownPayment]
		  ,[CouponRate]
	      ,[CouponInformation]
		  ,[CouponSpread]
		  ,[FirstCouponRate]
		  ,[SecondCouponRate]
	      
		  ,[RiskCategory]
		  ,[BondSector]
		  ,[CSD]
		  
		  ,[MaxAmount]
		  ,[DenominationAmount]
		  ,[CashFlowDate]
		  ,[CashFlowTypeId]
		  ,[Price]
		  ,[NominalAmountVPS]
		  ,[NominalAmountTotal]
		  ,[OptionType]

		  ,[TimeType]
		  ,[ExCount]
	      
	    INTO [OSE].[dbo].[%s]
	  FROM [OSEData].[dbo].[bondfeed] as t0
	  left join
	  (SELECT distinct
		  [SecurityId]
		  ,[SecurityType]
		  ,[Symbol]
		  ,[ISIN]
		  ,[SecurityName]
		  ,[CurrencyCode]
	  FROM [OSEData].[dbo].[bondfeed_SecurityIdentificationInformation]) as t1
	  on t1.[SecurityId]=t0.[SecurityId]
	  left join
	        (SELECT distinct *
	        FROM [OSEData].[dbo].[bondfeed_SecID]) as t2
	        on t2.[SecurityId]=t0.[SecurityId]
	    ORDER BY [Type],[SecurityId],[FromDate]"""	%(bondinfo_tbl,)
	DB.DropTable(bondinfo_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,bondinfo_tbl,dbase,True,
	            IndexFields="""
	            [Type],[SecurityId],[FromDate]
	            ,[ISIN]
	            ,[CompanyId]
	            ,[SectorId]
	            ,[ToDate]""")		
	
def MakeBondPricesTable(conn,crsr):

	sqlstr="""SELECT distinct
				[Date]
	            ,[FmDate]
	            ,[MaturityDate]
	            
	            ,t0.[SecurityId]
	            ,t1.[CompanyId]
	            ,[Name]
	            ,[LongName] as [Issuer]
	            ,[EqName],[EqSecurityID],[eqISIN]
	            ,[ISINSubCode]
	            
	            ,[BondType]
	            ,[Symbol]
	            ,[ISIN]
	            
	            ,[CouponRate]
	            ,[OpenPrice]
	            ,[High]
	            ,[Low]
	            ,[LastTradedPrice]
	            ,[OfficialVolume]
	            ,[UnofficialVolume]
	            ,[BestBidPrice]
	            ,[BestAskPrice]
	            
	            into  [OSE].[dbo].[%s]
	        FROM [OSEData].[dbo].[bondprices] as t0
	        left join
	        (SELECT distinct *
	        FROM [OSEData].[dbo].[bondfeed_SecID]) as t1
	        on t1.[SecurityId]=t0.[SecurityId]
	    order by [ISIN],[Date]""" %(bond_tbl,)
	DB.DropTable(bond_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,bond_tbl,dbase,True,
	            IndexFields="""
	            [ISIN],[Date]
	            ,[SecurityId]
	            ,[CompanyId]""")		


def MakeIndexTable(conn,crsr):
	columns="""[Date]
	            ,[SecurityId]
	            ,[Symbol]
	            ,[Name]
	            ,[Open]
	            ,[High]
	            ,[Low]
	            ,[Last]
	            """
	CID=DB.Fetch("""SELECT distinct
	            [SecurityID]
  				FROM [OSEData].[dbo].[equityindex_linked]""",
	        crsr)
	DB.DropTable(index_tbl+'2',conn,crsr,wdbase)
	DB.createTable(index_tbl+'2',conn,crsr)	
	for i in CID:
		oldtbl=DB.Fetch("""SELECT %s
			        FROM [OSEData].[dbo].[equityindex_linked]
		            where [SecurityID]=%s
		            order by [Date]""" %(columns,i[0]),
			    crsr)
		p=np.array(DB.Fetch("""SELECT [Last]
			            FROM [OSEData].[dbo].[equityindex_linked]
			            where [SecurityID]=%s
			            order by [Date]""" %(i[0],),
			                               crsr))
		lp=fu.ShiftArray(p,-1)
		deltap=(np.log(p+(p==0))-np.log(lp+(lp==0)))*(p!=0)*(lp!=0)
		newtbl=[]
		for j in range(len(oldtbl)):		
			newtbl.append(oldtbl[j]+(deltap[j][0],))
		DB.InsertTableIntoDB(conn,crsr,index_tbl+'2',columns+',[lnDelta]',newtbl,wdbase)
	DB.DropTable(index_tbl,conn,crsr,dbase)
	DB.CopyTable(conn,crsr,index_tbl+'2',index_tbl,wdbase,dbase)
	DB.CreateIndex(conn,crsr,index_tbl,dbase,True,
                IndexFields="""
                      [Name],[Date],[SecurityId],[Symbol]""")				
		
		
		
	
	
def MakeFundTable(conn,crsr):
	sqlstr="""SELECT
		  [ID]	
		  ,MF.[Date]
		  ,MF.[SecurityId]
		  ,[FundId]
		  ,MF.[Symbol]
		  ,[ISIN]
		  ,[Name]
		  ,[NAV]
		  ,[NAVAdj]
		  ,[Dividends]
		  ,[CorpAdj]
		  ,[DividendAdj]
	      ,[lnDeltaNAV]
	      ,[lnDeltaOSEBX]
	      ,[lnDeltaOSEFX]
	      ,[lnDeltaOBX]
	      INTO [OSE].[dbo].[%s]
	  FROM [OSEData].[dbo].[mutualfunds] as MF
	 
	  ORDER BY [FundId],[Date]""" %(fund_tbl,)
	DB.DropTable(fund_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,fund_tbl,dbase,
	            IndexFields="""
	             [Date]
	             ,[SecurityId]
	             ,[FundId]""")			
	
	
	

	
def MakeWarrantsTable(conn,crsr):
	sqlstr="""SELECT DISTINCT
                  [Date]
	              ,[SecurityId]
	              ,[CompanyId]
	              ,[Symbol]
	              ,[ISIN]
	              ,[Name]
	              ,[BestBidPrice]
	              ,[BestAskPrice]
	              ,[Open]
	              ,[High]
	              ,[Low]
	              ,[LastTradedPrice] as [Close]
	              ,[OfficialNumberOfTrades]
	              ,[OfficialVolume]
	              ,[UnofficialNumberOfTrades]
	              ,[UnofficialVolume]
	              ,[VolumeWeightedAveragePrice]
	              ,[Price]
	              ,[Currency]
	              ,[Description]
	              ,[CountryCode]
	              ,[NOKPerForex]
	              ,[Price]*[NOKPerForex] as [PriceNOK]
	          INTO [OSE].[dbo].[%s]  
	          FROM [OSEData].[dbo].[%s] as S
	        where [SecurityType]=5
	        order by [ISIN],[Date]
	        """ %(wrnt_tbl,equityextended_tbl+'2')
	DB.DropTable(wrnt_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,wrnt_tbl,dbase,
                IndexFields="""
                    [Date]
                    ,[SecurityId]
                    ,[CompanyId]
                    ,[ISIN]""")	
	
	
def MakeOptionsTable(conn,crsr):
	sqlstr="""SELECT distinct
					T.[Date]
	                ,T.[SecurityID]
	                ,T.[Symbol]
	                ,T.[ISIN]
	                ,T.[Name]
	                ,T.[ExpirationDate]
	                ,T.[UnderlyingSID]
	                ,T3.[ISIN] as [UnderlyingISIN]
	                ,ISNULL(T3.[SecurityName],T4.[Name]) as [UnderlyingName]
	                ,T.[UnderlyingSymbol]
	                ,T.[BestBidPrice]
	                ,T.[BestAskPrice]
	                ,T.[Last]
	                ,T.[Open]
	                ,T.[High]
	                ,T.[Low]
	                ,T.[Volum]+ISNULL(T.[void],0)  as [Volum]
	                ,T.[OpenInterest]                
	                ,iif((isnumeric(substring(T.[Name],StrStrt,iif(StrEnd-StrStrt>0,StrEnd-StrStrt,0)))=1) and StrEnd-StrStrt>0,
						cast(substring(T.[Name],StrStrt,StrEnd-StrStrt) as float),
						try_cast(iif(MidNumber>0,
							  iif(ltrsAtEnd2>0,
								  substring(T.[Symbol],MidNumber+2,len(T.[Symbol])-MidNumber-3),
								  iif(ltrsAtEnd1>0,
									  substring(T.[Symbol],MidNumber+2,len(T.[Symbol])-MidNumber-2),
									  substring(T.[Symbol],MidNumber+2,len(T.[Symbol])-MidNumber-1))
								  ),
							  substring(T.[Symbol],FirstNumber,len(T.[Symbol])-FirstNumber)
							  ) as float)
						) as Strike
	                ,PATINDEX('[A-L]',
	                      iif(MidNumber>0,
	                          substring(T.[Symbol],MidNumber+1,1),
	                          substring(T.[Symbol],len(T.[Symbol]),1)
	                      )
	                  ) as IsCall
	            INTO [OSE].[dbo].[%s]
	            FROM [OSEData].[dbo].[%s] AS T
	            left join
	                  (SELECT DISTINCT
	                    [Symbol]
	                    ,PATINDEX('%%[0-9][A-Z]',[Symbol]) as ltrsAtEnd1
	                    ,PATINDEX('%%[0-9][A-Z][A-Z]',[Symbol]) as ltrsAtEnd2
	                    ,PATINDEX('%%[0-9][A-Z][0-9]%%',[Symbol]) as MidNumber
	                    ,PATINDEX('%%[0-9]%%',[Symbol]) as FirstNumber
	                    FROM [OSEData].[dbo].[options_prices]) AS T1
	                    on T1.[Symbol]=T.[Symbol]	
				left join
	                  (SELECT DISTINCT
	                    [Name]
	                   ,PATINDEX('%%[0-9]%%',[Name]) as StrStrt
					   ,PATINDEX('%%[0-9][ ]%%',[Name]) as StrEnd
	                    FROM [OSEData].[dbo].[options_prices]							
						) AS T2
	                    on T2.[Name]=T.[Name]						
	            left join 
	                    (SELECT distinct
	                  [SecurityId]
	                  ,[ISIN]
	                  ,[SecurityName]
	               
	                    FROM [OSEData].[dbo].[Securities]) as T3
	                    on T.[UnderlyingSID]=T3.[SecurityId] 	
	            left join 
	                    (SELECT distinct
	                  [SecurityId]
	                  ,[Name]
	               
	                    FROM [OSEData].[dbo].[equityindex]) as T4
	                    on T.[UnderlyingSID]=T4.[SecurityId] 	                    
	            order by [ISIN],[Date]""" %(options,'options_prices')
	DB.DropTable(options,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,options,dbase,
                IndexFields="""
                    [Date]
                    ,[SecurityId]
                    ,[ISIN]
	                ,[ExpirationDate]
	                ,[UnderlyingSID]
	                ,[Strike]
	                ,[UnderlyingName]
	                ,[UnderlyingISIN]""")	
	conn.commit() 	
	
def MakeSecInfoTable(conn,crsr):
		sqlstr="""
                  SELECT distinct [ID]
		          ,[EventDate]
		          ,[ToDate]
		          ,[FromDate]
		          ,[CompanyId]
		          ,[ISIN]
		          ,[Name]
		          ,[Symbol]
		          ,[SecurityId]
		          ,[SecurityName]
		          ,[SecurityType]
		          ,[EquityClassId]
		          ,[EventId]
		          ,[DividendId]
		          ,[AdjustmentFactor]
		          ,[DividendInForeignCurrency]
		          ,[DividendInNOK]
		          ,[NumberOfDividendsPerYear]
		          ,[TypeOfDividend]
		          ,[SequenceNumber]
		          ,[Price]
		          ,[SubscriptionPrice]
		          ,[Description]
		          ,[CountryCode]
		          ,[Currency]
		          ,[CurrencyCode]
		          ,[Exchange]
		          ,[Type]
		          ,[NewShares]
		          ,[OldShares]
		          ,[NumberOfShares]
		          ,[CompanyOwnedShares]
		          ,[TotalCompanyOwnedShares]
		          ,[TotalNumberOfShares]
		          ,[NominalValue]
		          INTO [OSE].[dbo].[%s]
		          FROM [OSEData].[dbo].[%s]""" %(secinfo_tbl,secinfo_tbl)
		
		DB.DropTable(secinfo_tbl,conn,crsr,dbase)
		crsr.execute(sqlstr)
		conn.commit() 	
		DB.CreateIndex(conn,crsr,secinfo_tbl,dbase,
			        IndexFields="""[Type],[CompanyId],[EventDate]
	            ,[EquityClassId]
	            ,[ISIN]""")	
		
def MakeAllSecuritiesTable(conn,crsr):
		sqlstr="""
                  select * 
		          into [OSE].[dbo].[%s]
		          from (
		              select distinct
		                [SecurityId]
		                ,[SecurityType]
		                ,[ISIN]
		                ,[SecurityName] as  [Name]
		                ,[Issuer]
		                ,[CompanyId]
		                ,'BondInfo' as [Table]
		                ,NULL as [Symbol]
		                ,[EqSecurityID] as [IssuerSID]
		                ,[EqISIN] as [UnderlyingISIN]
		              FROM [OSE].[dbo].[BondInfo]
		              where not [ISIN] is NULL
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,2 as [SecurityType]
		                ,[ISIN]
		                ,[Name] 
		                ,[Issuer]
		                ,[CompanyId]
		                ,'bonds' as [Table]
		                ,[Symbol]
		                ,[EqSecurityID] as [IssuerSID]
		                ,[EqISIN] as [UnderlyingISIN]
		              FROM [OSE].[dbo].[bonds]
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,4 as [SecurityType]
		                ,NULL as [ISIN]
		                ,[Name]
		                ,NULL as [Issuer]
		                ,NULL as [CompanyId]
		                ,'equityindex' as [Table]
		                ,[Symbol]
		                ,NULL as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSE].[dbo].[equityindex]
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,5 as [SecurityType]
		                ,[ISIN]
		                ,[Symbol]+' '+[Description] as [Name]
		                ,NULL as [Issuer]
		                ,NULL as [CompanyId]
		                ,'FutureForwards' as [Table]
		                ,[Symbol]
		                ,[IssuerSecurityID] as [IssuerSID]
		                ,NULL [UnderlyingISIN]
		              FROM [OSE].[dbo].[FutureForwards]
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,1 as [SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,NULL as [Issuer]
		                ,[CompanyId]
		                ,'equity' as [Table]
		                ,[Symbol]
		                ,NULL as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSE].[dbo].[equity]
		              
		              
		              UNION
		              select distinct
		                [SecurityId]
		                ,[SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,NULL as [Issuer]
		                ,[CompanyId]
		                ,'warrants' as [Table]
		                ,[Symbol]
		                ,NULL as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSEData].[dbo].[equity]
		              where [SecurityType]=5
		              
		              UNION
		              select distinct
		                [SecurityId]
		                ,3 as [SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,NULL as [Issuer]
		                ,NULL as [CompanyId]
		                ,'mutualfund' as [Table]
		                ,[Symbol]
		                ,NULL as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSE].[dbo].[mutualfund]
		              where not [ISIN] is NULL
		          
		              UNION
		              select distinct
		                [SecurityID]
		                ,5 as [SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,[UnderlyingName] as [Issuer]
		                ,NULL as [CompanyId]
		                ,'options' as [Table]
		                ,[Symbol]
		                ,[UnderlyingSID] as [IssuerSID]
		                ,[UnderlyingISIN]
		              FROM [OSE].[dbo].[options]
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,[SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,NULL as [Issuer]
		                ,[CompanyId]
		                ,'SecuritiesInfo' as [Table]
		                ,[Symbol]
		                ,NULL as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSE].[dbo].[SecuritiesInfo]
		              where not [Name] is null
		          
		              UNION
		              select distinct
		                [SecurityId]
		                ,5 as [SecurityType]
		                ,[ISIN]
		                ,[SecurityName] as [Name]
		                ,NULL as [Issuer]
		                ,[CompanyId]
		                ,'WarrantsInfo' as [Table]
		                ,[Symbol]
		                ,[UnderlyingSecurity] as [IssuerSID]
		                ,NULL as [UnderlyingISIN]
		              FROM [OSE].[dbo].[WarrantsInfo]
		              ) as t
		              order by [Table],[Name],[ISIN],[SecurityId],[Symbol]
		              """ %(allsecs_tbl)
		
		DB.DropTable(allsecs_tbl,conn,crsr,dbase)
		crsr.execute(sqlstr)
		conn.commit() 	
		DB.CreateIndex(conn,crsr,allsecs_tbl,dbase,
			        IndexFields="""[SecurityId]
		                ,[SecurityType]
		                ,[ISIN]
		                ,[Name]
		                ,[Issuer]
		                ,[CompanyId]
		                ,[Table]
		                ,[Symbol]
		                ,[IssuerSID]
		                ,[UnderlyingISIN]""")	
		
def MakeColDefSQL(table,caption,inclfields=[]):
	s=	"""
			SELECT 
			[COLUMN_NAME] 
	        ,'%s' as [Table]
	        ,'%s' as [Caption]
	        FROM [OSE].[INFORMATION_SCHEMA].[Columns] 
	        WHERE [TABLE_NAME] = '%s'
	        """ %(table,caption,table)
	s2=''
	for i in inclfields:
		s2+=" [COLUMN_NAME] ='%s' OR" %(i)
	s2=s2[0:len(s2)-3]
	if len(s2):
		s=s + "AND (" + s2 + ') '
	return s
	
	
		
def MakeTableColDef(conn,crsr):
	s=MakeColDefSQL('equity','Stocks',['Date','ISIN','Name','AdjustedPrice'])
	s+='UNION'+MakeColDefSQL('equity','Stocks advanced')
	s+='UNION'+MakeColDefSQL('SecuritiesInfo','Stocks info')
	s+='UNION'+MakeColDefSQL('equityindex','Stocks indicies')
	s+='UNION'+MakeColDefSQL('Warrants','Warrants')
	s+='UNION'+MakeColDefSQL('Warrants','Warrants info')	
	s+='UNION'+MakeColDefSQL('FutureForwards','Futures and forwards')
	s+='UNION'+MakeColDefSQL('options','Options')
	s+='UNION'+MakeColDefSQL('bonds','Bonds')
	s+='UNION'+MakeColDefSQL('BondInfo','Bonds info')
	s+='UNION'+MakeColDefSQL('mutualfunds','Mutual funds')

	sqlstr="select * into [OSE].[dbo].[%s] from (" %(tblcol_tbl,)
	sqlstr=sqlstr+s+""") as t ORDER BY [Table],[Caption] """
	
	DB.DropTable(tblcol_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,tblcol_tbl,dbase,
                IndexFields="""[COLUMN_NAME] 
                    ,[Table]
                    ,[Caption]""")	
	
	
	

def maketable_factors(conn,crsr):
	sqlstr="""SELECT *,

(1.0/3.0)*(low_small + med_small + high_small) - (1.0/3.0)*(low_big + med_big + high_big) AS SMB
,
0.5*(low_small + low_big) - 0.5*(high_small + high_big) AS HML
, 
0.5*(high_big_mom + high_small_mom) - 0.5*(low_big_mom + low_small_mom) as MOM	 

INTO [OSE].[dbo].[factors]
FROM
(SELECT distinct
      tbl.[Date]
		,low_small.[return] as low_small
		,high_small.[return] as high_small
		,low_big.[return] as low_big
		,high_big.[return] as high_big
		,med_small.[return] as med_small
		,low_small_mom.[return] as low_small_mom
		,med_small_mom.[return] as med_small_mom
		,med_big.[return] as med_big
		,high_small_mom.[return] as high_small_mom
		,high_big_mom.[return] as high_big_mom
		,low_big_mom.[return] as low_big_mom
		,med_big_mom.[return] as med_big_mom

FROM [OSEData].[dbo].[factors_tmp] as tbl
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='low_small') as low_small on tbl.[Date]=low_small.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='high_small') as high_small on tbl.[Date]=high_small.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='low_big') as low_big on tbl.[Date]=low_big.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='high_big') as high_big on tbl.[Date]=high_big.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='med_small') as med_small on tbl.[Date]=med_small.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='low_small_mom') as low_small_mom on tbl.[Date]=low_small_mom.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='med_small_mom') as med_small_mom on tbl.[Date]=med_small_mom.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='med_big') as med_big on tbl.[Date]=med_big.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='high_small_mom') as high_small_mom on tbl.[Date]=high_small_mom.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='high_big_mom') as high_big_mom on tbl.[Date]=high_big_mom.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='low_big_mom') as low_big_mom on tbl.[Date]=low_big_mom.[Date] 
join (SELECT [Date],[return]  FROM [OSEData].[dbo].[factors_tmp] WHERE [factor_name]='med_big_mom') as med_big_mom on tbl.[Date]=med_big_mom.[Date] 
) as ftbl"""
	DB.DropTable('factors', conn, crsr,'OSE')
	crsr.execute(sqlstr)
	conn.commit()
	DB.CreateIndex(conn,crsr,'factors','OSE',True,
	               IndexFields="""[Date]""")		
	
	
	
	
def make_table_account(conn,crsr):
	sqlstr="""
	SELECT distinct N.[Name],N.[SecurityId],A.*
	INTO [OSE].[dbo].[account]
  FROM [OSEData].[dbo].[account] A

  join
  (select distinct
  [CompanyID],
  [SecurityId],
  [Name] from  [OSEData].[dbo].[equity_extended2]) N
  on N.[CompanyID]=A.[CompanyID]"""
	DB.DropTable('account',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,'account',dbase,
	               IndexFields="""[SecurityID]
                    ,[CompanyID]
	                ,[OrganizationID]
	                ,[Year]
	                ,[Description]
	                ,[Source]
	                ,[Corporation]""")		
	
	
def MakeEquityTable(conn,crsr):
	sqlstr="""SELECT DISTINCT
                  S.[Date]
	              ,S.[SecurityId]
	              ,S.[CompanyId]
	              ,S.[Symbol]
	              ,s.[ISIN]
	              ,[Name]
	              ,[BestBidPrice]
	              ,[BestAskPrice]
	              ,[Open]
	              ,[High]
	              ,[Low]
	              ,[LastTradedPrice] as [Close]
	              ,[OfficialNumberOfTrades]
	              ,[OfficialVolume]
	              ,[UnofficialNumberOfTrades]
	              ,[UnofficialVolume]
	              ,[VolumeWeightedAveragePrice]
	              ,[Price]
	              ,[AdjustedPrice]
	              ,[Dividends]
	              ,[CorpAdj]
	              ,[DividendAdj]
	              ,[Currency]
	              ,[Description]
	              ,[CountryCode]
	              ,[SumAnnualDividends]
	              ,[NumberOfShares]
	              ,[CompanyOwnedShares]
	              ,[NumberOfShares]-[CompanyOwnedShares] as OutstandingShares
	              ,[Exchange]
	              ,[NOKPerForex]
	              ,MC.[marketcap] 
	              ,MC.[mktshare]
	              ,MC.[alpha]
	              ,MC.[beta]
	              ,[DividendPriceRatio]
	              ,[lnDeltaP]
	              ,[lnDeltaOSEBX]
	              ,[lnDeltaOBX]
	              ,[lnDeltaTIX]
	              ,[bills_DayLnrate]
	              ,[Sector]
	          INTO [OSE].[dbo].[equity]  
	          FROM [OSEData].[dbo].[equity_extended2] as S
	          left join
	          (SELECT [Date]  ,[ISIN] ,[mktshare] ,[alpha] ,[beta] ,[marketcap]
	         	         FROM [OSEData].[dbo].[equity_mktshares]
	            ) as MC
	            on MC.[Date]=S.[Date] and MC.[ISIN] =S.[ISIN] 


	        where S.[SecurityType]=1
	        order by [ISIN],[Date]
	        """
	DB.DropTable('equity',conn,crsr,'OSE')
	crsr.execute(sqlstr)
	conn.commit()
	DB.CreateIndex(conn,crsr,'equity','OSE',
	               IndexFields="""
                    [Date]
                    ,[SecurityId]
                    ,[CompanyId]
                    ,[ISIN]""")		