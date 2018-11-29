#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
import download

dbase='OSEData'
sec_tbl='Securities'
bondfeed_eq_tbl='bondfeed_SecID'
linkedtmp_tbl='linkedindicies'
secinfo_tbl='SecuritiesInfo'
index_tbl='equityindex_linked'



def MakeTables():
	conn,crsr=DB.Connect(dbase)
	merge_gics(conn,crsr)
	MakeSecurityTable(conn,crsr)
	MakeSecInfoTable(conn,crsr)
	MakeBondfeedEqTable(conn,crsr)

	download.import_forex()
	download.get_FF_data()
	MakeBillsTable(conn,crsr)
	MakeExtendedEquityTable(conn, crsr)
	make_factor_criteria(conn, crsr)
	create_tbl_accounts1(conn,crsr)
	create_tbl_accounts2(conn,crsr)
	create_tbl_accounts3(conn,crsr)
	
	conn.close()




def MakeSecurityTable(conn,crsr):
	sqlstr="""

SELECT distinct
      t0.[SecurityId]
	  ,t2.[CompanyId]
      ,t0.[SecurityType]
      ,t0.[Symbol]
      ,t0.[ISIN]
      ,t0.[SecurityName]
      ,t0.[Currency]
	  ,t1.[Exchange]
	  ,t1.[FromDate]
	  ,t1.[ToDate]
	  ,t2.[EquityClassId]
	  ,t2.[Description]
	INTO [OSEData].[dbo].[%s]
FROM [OSEData].[dbo].[equityfeed_EquitySecurityInformation] as t0
left join 
(SELECT distinct
      [SecurityId]
      ,[Exchange]
      ,[FromDate]
      ,[ToDate]
  FROM [OSEData].[dbo].[equityfeed_ListingPeriod]) as t1
  on t0.[SecurityId]=t1.[SecurityId]
left join
   (select distinct
      [SecurityId]
      ,[Symbol]
      ,[CompanyId]
      ,[EquityClassId]
      ,[Description]
  FROM [OSEData].[dbo].[equityfeed_EquityInformation]) as t2
  on t0.[SecurityId]=t2.[SecurityId]
  ORDER BY [SecurityType],[SecurityId]"""%(sec_tbl,)
	DB.DropTable(sec_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,sec_tbl,dbase,True,
	               IndexFields="""
	            [SecurityType],[SecurityId]
	            ,[CompanyId]
	            ,[ISIN]""")

def MakeBillsTable(conn,crsr):
	tname='bills'
	sqlstr="""	SELECT * INTO [OSEData].[dbo].[%s] FROM
	                (SELECT [Date], AVG(-LOG([Price])/[DtoMat]) as bills_DayLnrate 
	                    FROM
	                    (SELECT distinct [Date],[ISIN],DATEDIFF(DAY, [Date], [MaturityDate]) As DtoMat,([BestBidPrice]+[BestAskPrice])*0.5/100 as Price
	                        FROM [OSEData].[dbo].[bondprices]
	                        where [Name] like '%sstatskasse%s') t1
	                where not [Price] is null
	                group by [Date]
	                UNION
	                    SELECT [Date],[bills_lnDayly] as bills_DayLnrate
	                    FROM [OSEData].[dbo].[bills_historical]	 
	                ) t2

	            ORDER by t2.[Date] 
	        """ %(tname,'%','%')
	DB.DropTable(tname,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,tname,dbase,True,
	               IndexFields='[Date]')


def MakeSecInfoTable(conn,crsr):
	sqlstr="""SELECT distinct
	    [EventDate]
	    ,t4.[ToDate]
	    ,t4.[FromDate]

		,t0.[CompanyId]
	    ,t3.[ISIN]
	    ,t1.[Name]
	    ,t2.[Symbol]
		,t0.[SecurityId]
		,t3.[SecurityName]
		,t3.[SecurityType]

		,t2.[EquityClassId]
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

	    ,t2.[Description]
		,t1.[CountryCode]
		,t3.[Currency]
		,[CurrencyCode]
	    ,t4.[Exchange]
	    ,[Type]

		,[NewShares]
	    ,[OldShares]
	    ,[NumberOfShares]
	    ,[CompanyOwnedShares]
		,[TotalCompanyOwnedShares]
		,[TotalNumberOfShares]
		,[NominalValue]

	    into [OSEData].[dbo].[%s]
	    FROM [OSEData].[dbo].[equityfeed] as t0

	    left join
	    (SELECT distinct
	        [CompanyId]
	        ,[CountryCode]
	        ,[Name]
	    FROM [OSEData].[dbo].[equityfeed_CompanyIdentificationList]) as t1
	    on t0.[CompanyId]=t1.[CompanyId]

	    left join 
	    (SELECT distinct
	        [SecurityId]
	        ,[Symbol]
	        ,[CompanyId]
	        ,[EquityClassId]
	        ,[Description]
	    FROM [OSEData].[dbo].[equityfeed_EquityInformation]) as t2
	    on t0.[CompanyId]=t2.[CompanyId]

	    left join
	    (SELECT distinct
	        [SecurityId]
	        ,[SecurityType]
	        ,[Symbol]
	        ,[ISIN]
	        ,[SecurityName]
	        ,[Currency]
	    FROM [OSEData].[dbo].[equityfeed_EquitySecurityInformation]) as t3
	    on t2.[SecurityId]=t3.[SecurityId]

	    left join 
	    (SELECT distinct
	        [SecurityId]
	        ,[Exchange]
	        ,[FromDate]
	        ,[ToDate]
	    FROM [OSEData].[dbo].[equityfeed_ListingPeriod]) as t4
	    on t2.[SecurityId]=t4.[SecurityId] 

	    where not t0.[CompanyId] is null
	    order by t0.[Type],t0.[CompanyId],t0.[EventDate]""" %(secinfo_tbl,)
	DB.DropTable(secinfo_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,secinfo_tbl,dbase,True,
	               IndexFields="""
	            [Type],[CompanyId],[EventDate]
	            ,[EquityClassId]
	            ,[ISIN]""")		
	

def MakeIndexTable(conn,crsr):
	sqlstr="""
		SELECT distinct
	          [StartDate]
	          ,[Date]
	          ,t0.[SecurityId]
	          ,[Symbol]
	          ,[Name]
	          ,[Open]
	          ,[High]
	          ,[Low]
	          ,[Last]
	    INTO [OSEData].[dbo].[%s]
	    FROM [OSEData].[dbo].[equityindex] as t0
		left join
		(SELECT
			[SecurityId],min([Date]) as StartDate FROM [OSEData].[dbo].[equityindex]
			group by [SecurityId]) as t1
		on t0.[SecurityId]=t1.[SecurityId]

	    union
	    (SELECT  '1983-01-03' as [StartDate],[Date]
			,2 as [SecurityId]
	        ,'OSEBX' as [Symbol]
	        ,'LinkedBenchmark' as [Name]
	        ,NULL AS [Open]
	        ,NULL AS[High]
	        ,NULL AS[Low]
	        ,[EquetyLinkedIndex] as [Last]
  		FROM [OSEData].[dbo].[linkedindicies]
	    where [Date]<'1996-01-02'
	    union
	    SELECT  '1983-01-03' as [StartDate],[Date]
	    	,3 as [SecurityId]
	        ,'OSEFX' as [Symbol]
	        ,'LinkedMutualFund' as [Name]
	        ,NULL AS [Open]
	        ,NULL AS[High]
	        ,NULL AS[Low]
	        ,[MutualFundLinkedIndex] as [Last]
	    FROM [OSEData].[dbo].[linkedindicies]
	    where [Date]<'1995-12-28'
	    union
	    SELECT  '1983-01-03' as [StartDate],[Date]
	    	,1 as [SecurityId]
	        ,'OSEAX' as [Symbol]
	        ,'LinkedAllShare' as [Name]
	        ,NULL AS [Open]
	        ,NULL AS[High]
	        ,NULL AS[Low]
	        ,[AllShareLinkedIndex] as [Last]
	    FROM [OSEData].[dbo].[linkedindicies] 
	    where [Date]<'1995-12-29'

	    union
	    SELECT  '1983-01-03' as [StartDate],[Date]
			,2 as [SecurityId]
	        ,'OSEBX' as [Symbol]
	        ,'LinkedBenchmark' as [Name]
	        ,[Open]
	        ,[High]
	        ,[Low]
	        ,[Last]
  		FROM [OSEData].[dbo].[equityindex]
	    where [Date]>='1996-01-02' and [Name]='Oslo Børs Benchmark Index_GI' 
	    union
	    SELECT  '1983-01-03' as [StartDate],[Date]
	    	,3 as [SecurityId]
	        ,'OSEFX' as [Symbol]
	        ,'LinkedMutualFund' as [Name]
	        ,[Open]
	        ,[High]
	        ,[Low]
	        ,[Last]
	    FROM [OSEData].[dbo].[equityindex]
	    where [Date]>='1995-12-28' and [Name]='Oslo Børs Mutual Fund Index_GI' 
	    union
	    SELECT  '1983-01-03' as [StartDate],[Date]
	    	,1 as [SecurityId]
	        ,'OSEAX' as [Symbol]
	        ,'LinkedAllShare' as [Name]
	        ,[Open]
	        ,[High]
	        ,[Low]
	        ,[Last]
	    FROM [OSEData].[dbo].[equityindex] 
	    where [Date]>='1995-12-29' and [Name]='Oslo Børs All-share Index_GI') 
	    order by [Name],[Date]
	""" %(index_tbl,)
	DB.DropTable(index_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	DB.CreateIndex(conn,crsr,index_tbl,dbase,True,
	               IndexFields="""
	              [Name],[Date]""")		


def MakeBondfeedEqTable(conn,crsr):
	"""Creates a table to link bondinformation with equity information"""
	sqlstr="""  SELECT distinct t0.[SecurityId],t0.[CompanyId],[EqName],[EqSecurityID],[eqISIN]
					into [OSEData].[dbo].[%s]
	            FROM [OSEData].[dbo].[bondfeed_Borrower] as t0
	            left join 
	            (SELECT distinct
	            	[CompanyId]
	            	,[SecurityId] as [EqSecurityID]
	            	,[SecurityName] as [EqName]
	            	,[ISIN] as [eqISIN]
	            FROM [OSEData].[dbo].[SecuritiesInfo]) as t1
	            on t0.[CompanyId]=t1.[CompanyId]
	            ORDER BY [SecurityId]""" %(bondfeed_eq_tbl,)
	DB.DropTable(bondfeed_eq_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,bondfeed_eq_tbl,dbase,True,
	               IndexFields="""
	            [SecurityId],[CompanyId],[EqSecurityID],[eqISIN]""")	


def CastLinkedIndicies(conn,crsr):
	sqlstr="""delete 
	FROM [OSEData].[dbo].[Lenkede_indekser]
	where isdate([Date]) = 0

	SELECT 
		CAST([FileName] AS varchar(100)) AS [FileName]
		,CAST([Date] as Date) AS [Date]
		,CAST([EquetyLinkedIndex] AS float) AS [EquetyLinkedIndex]
		,CAST([MutualFundLinkedIndex] AS float) AS [MutualFundLinkedIndex]
		,CAST([AllShareLinkedIndex] AS float) AS [AllShareLinkedIndex]
		into [OSEData].[dbo].[%s]
	FROM [OSEData].[dbo].[Lenkede_indekser]
	ORDER BY [Date]""" %(linkedtmp_tbl,)
	DB.DropTable(linkedtmp_tbl,conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	DB.CreateIndex(conn,crsr,linkedtmp_tbl,dbase,True,
	               IndexFields='[Date]')	

def CreateTableAdjustments(conn,crsr):
	tblname='Adjustments'
	tbltmp=tblname+'_tmp'
	DB.DropTable(tbltmp,conn,crsr,dbase)
	DB.DropTable(tblname,conn,crsr,dbase)
	tbls=[['SELECT','BonusIssue'],
	      ['INSERT','ConversionFromOtherEquityClass'],
	      ['INSERT','Demerger'],
	      ['INSERT','InitialNumberOfShares'],
	      ['INSERT','IssueOfConvertibleBond'],
	      ['INSERT','RightsIssue'],
	      ['INSERT','ScriptDividendAlternative'],
	      ['INSERT','StockSplit']]
	sqlstr="""%s v.[SecurityId]
      ,t.[Symbol]
	  ,t.[ISIN]
	  ,t.[SecurityName]
	  ,t.[SecurityType]
      ,v.[Type]
      ,[CompanyId]
      ,[EventId]
      ,[EventDate]
      ,[SequenceNumber]
      ,[AdjustmentFactor]
	  ,[NumberOfShares]
	  ,[TotalNumberOfShares]
	  ,[SubscriptionPrice]
	  ,[MarketPrice]
      %s 
      FROM [OSEData].[dbo].[equityfeed_%s] as v
      left join [OSEData].[dbo].[equityfeed_EquitySecurityInformation] as t
      on v.[SecurityId]=t.[SecurityId] and t.[SecurityType]=1"""

	for t in tbls:
		print ( '%s from %s' %tuple(t))
		if t[0]=='SELECT':
			s1='SELECT'
			s2='INTO [OSEData].[dbo].[%s]' %(tbltmp,)
		else:
			s1='INSERT INTO [OSEData].[dbo].%s SELECT' %(tbltmp,) 
			s2=''  
		sqlstr2=sqlstr %(s1,s2,t[1])
		if t[1]=='IssueOfConvertibleBond':
			sqlstr2=sqlstr2.replace(',[SequenceNumber]',',NULL as [SequenceNumber]')
			sqlstr2=sqlstr2.replace(',[NumberOfShares]',',NULL as [NumberOfShares]')
			sqlstr2=sqlstr2.replace(',[TotalNumberOfShares]',',NULL as [TotalNumberOfShares]')
		if t[1]=='BonusIssue':
			sqlstr2=sqlstr2.replace(',[SubscriptionPrice]',',CAST(NULL as float) as [SubscriptionPrice]')
			sqlstr2=sqlstr2.replace(',[MarketPrice]',',CAST(NULL as float) as [MarketPrice]')			
		if t[1]!='ScriptDividendAlternative' and t[1]!='RightsIssue':
			sqlstr2=sqlstr2.replace(',[SubscriptionPrice]',',0.0 as [SubscriptionPrice]')
			sqlstr2=sqlstr2.replace(',[MarketPrice]',',0.0 as [MarketPrice]')
		
		#print ( sqlstr2)
		crsr.execute(sqlstr2)
		conn.commit()
	sqlstr="""SELECT DISTINCT * INTO [OSEData].[dbo].[%s] 
          FROM [OSEData].[dbo].[%s]
          WHERE NOT [AdjustmentFactor] IS NULL
	      ORDER BY [CompanyId],[EventDate]"""  %(tblname,tbltmp)
	crsr.execute(sqlstr)
	conn.commit()    
	DB.DropTable(tbltmp,conn,crsr,dbase)
	DB.CreateIndex(conn,crsr,tblname,dbase,True,
	               IndexFields="""[CompanyId],[EventDate],
	            [SecurityId]
	            ,[ISIN]
	            ,[SecurityType]""")	

	print ( 'Done')


	

def SplitTable(splitfield,tbl,dbase):
	"""Splitting a table into several based on splitfield"""
	conn,crsr=DB.Connect(dbase)
	r=DB.GetColumnNames(crsr,tbl)
	n=len(r[0])
	SQLText="""SELECT [%s] FROM [%s] 
                group by [%s]""" %(splitfield,tbl,splitfield)
	crsr.execute(SQLText)
	newtbls=crsr.fetchall()
	newtbls=[i[0] for i in newtbls]
	sqlcols=(',COUNT([%s])'*n %tuple([i for i in r[0]]))[1:]
	for t in newtbls:

		SQLText="""SELECT %s FROM [%s] 
                        WHERE [%s]='%s'""" %(sqlcols,tbl,splitfield,t)        
		crsr.execute(SQLText)
		countcols=crsr.fetchall()[0]
		keepcols=[]
		for i in range(n):
			if countcols[i]>0:
				keepcols.append(r[0][i])
		k=len(keepcols)
		cols=(',[%s]'*k %tuple([i for i in keepcols]))[1:]
		newtblname=(tbl+'_'+t).replace('-','_')
		DB.DropTable(newtblname,conn,crsr,dbase)
		print ( 'Creating ' + newtblname)
		SQLText="""SELECT %s INTO %s FROM [%s] 
                WHERE [%s]='%s'""" %(cols,newtblname,tbl,splitfield,t)    
		crsr.execute(SQLText)
		conn.commit() 
		MakeIxFields(conn,crsr,keepcols,newtblname)




def MakeIxFields(conn,crsr,keepcols,newtblname):
	keep=[]
	ixflds=[""",[CompanyId]
				,[EquityClassId]
	            ,[EventDate]
	            ,[FromDate]
	            ,[ISIN]
	            ,[SecurityId]
	            ,[SecurityName]
	            ,[SecurityType]
	            ,[Type]
	            ,[FileName]
	            """]
	for i in keepcols:
		if i in ixflds:
			keep.append(i)
	ixflds=''
	if len(keep)>0:
		ixflds=(',[%s]'*k %tuple([i for i in keep]))[1:]
	DB.CreateIndex(conn,crsr,newtblname,dbase,IndexFields=ixflds)
	pass

def create_tbl_accounts1(conn,crsr):
	sqlstr="""	select * 
		INTO [OSEData].[dbo].[account_ID_yr]
		from
		(SELECT distinct
			[CompanyID]
			,[Year]
		FROM [OSEData].[dbo].[account_brnsnd]
		union
		SELECT 
			[CompanyID]
			,[Year]
		FROM [OSEData].[dbo].[account_foreign]
		union
		SELECT 
			[CompanyID]
			,[Year]
		FROM [OSEData].[dbo].[account_ose]) as t00
	where not [Year] is NULL"""
	DB.DropTable('account_ID_yr',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
	sqlstr="""select distinct t0.*
	  			,t1.[IsCorporateAccount]
	            ,t1.[OrganizationID]
	            ,t1.[Name]
	            ,t1.[Description]
	            ,t1.[Value] 
	        INTO [OSEData].[dbo].[account_tmp1]
	        from
	        [OSEData].[dbo].[account_ID_yr] AS T0

	        left join
	        (select       [IsCorporateAccount]
      			,[OrganizationID]
	            ,[CompanyID]
	            ,[Name]
	            ,[webName]
	            ,[FetchDate]
	            ,[Year]
	            ,[Description]
	            ,[Value]
   			from [OSEData].[dbo].[account_brnsnd]) as t1
   		on t0.[CompanyID]=t1.[CompanyID] and t0.[Year]=t1.[Year]
   		order by t0.[Year]"""
	DB.DropTable('account_tmp1',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 

def create_tbl_accounts2(conn,crsr):
	sqlstr="""SELECT T1.* 
				INTO [OSEData].[dbo].[account_tmp2]
				FROM
				(SELECT distinct [CompanyID]
	            ,[Year]
	            ,[Name]
	        FROM [OSEData].[dbo].[account_tmp1]
	        where [Name] is NULL) AS T0
	        left join
	        (SELECT [IsCorporateAccount]
	            ,[In1000]
	            ,[Currency]
	            ,[CompanyID]
	            ,[account]
	            ,[accountType]
	            ,[Directory]
	            ,[Year]
	            ,[Description]
	            ,[DescriptionEN]
	            ,[Value]
	        FROM [OSEData].[dbo].[account_foreign]) AS T1
	        ON T0.[CompanyID]=T1.[CompanyID] AND T0.[Year]=T1.[Year]"""
	dfs="""SELECT T1.* FROM
(SELECT distinct [CompanyID]
      ,[Year]
	  ,[Name]
  FROM [OSEData].[dbo].[account_tmp1]
  where [Name] is NULL) AS T0
  left join
  (SELECT 
	   [CompanyID]
	  ,[Year]
	  ,[IsCorporateAccount]

      ,NULL AS [OrganizationID]

      ,[account]
      ,[Description]
      ,[DescriptionEN]
	   ,[Currency]
      ,IIF([In1000]=1,[Value]*1000,[Value]) as [Value]
  FROM [OSEData].[dbo].[account_foreign]) AS T1
  ON T0.[CompanyID]=T1.[CompanyID] AND T0.[Year]=T1.[Year]"""
	DB.DropTable('account_tmp2',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 	
	
def create_tbl_accounts3(conn,crsr):
	sqlstr1="""

			SELECT * INTO [OSEData].[dbo].[account_tmp3] FROM
			(SELECT 
				[CompanyID]
	            ,[OrganizationID]
				,[Year]
				,map.[original] as [description]
	            ,map.[simplified_en] as [desc_simpl_en]
	            ,map.[simplified_no] as [desc_simpl_no]
	            ,map.[acc_no] as [account_number]
				,map.[type] as [account_type]
				,[Value]
				,1 as [DB_rank]
				,'Brønnøysund' as [source]
	            ,1 as [Corporation]
			FROM [OSEData].[dbo].[account_brnsnd] as brs
			left join (SELECT [original]
				,[simplified_en]
	            ,[simplified_no]
	            ,[acc_no]
				,[reverse_sign]
				,[from]
				,[type]
			FROM [OSEData].[dbo].[account_mapping]
			where [from]='bronnoysund') as map
			on brs.[Description]=map.[original]
			where map.[simplified_en] is not NULL
			and [IsCorporateAccount]=1
		
			UNION
		
			SELECT 
				[CompanyID]
	            ,[OrganizationID]
				,[Year]
				,map.[original] as [description]
	            ,map.[simplified_en] as [desc_simpl_en]
	            ,map.[simplified_no] as [desc_simpl_no]
	            ,map.[acc_no] as [account_number]
				,map.[type] as [account_type]
				,[Value]
				,2 as [DB_rank]
				,'Brønnøysund' as [source]
	            ,0 as [Corporation]
			FROM [OSEData].[dbo].[account_brnsnd] as brs
			left join (SELECT [original]
				,[simplified_en]
	            ,[simplified_no]
	            ,[acc_no]
				,[reverse_sign]
				,[from]
				,[type]
			FROM [OSEData].[dbo].[account_mapping]
			where [from]='bronnoysund') as map
			on brs.[Description]=map.[original]
			where map.[simplified_en] is not NULL
			and  [IsCorporateAccount]=0
			
			UNION
		
			SELECT 
				AF.[CompanyID]
	            ,NULL as [OrganizationID]
				,AF.[Year]
				,[DescriptionEN] AS [description]
				,[DescriptionEN] AS [desc_simpl_en]
				,[Description] AS [desc_simpl_no]
				,map.[acc_no] as [account_number]
				,map.[type] as [account_type]
				,[Value]*iif([In1000]=1,1000,1)*F.avgrate as [Value]
				,3 as [DB_rank]
				,'Foreign' as [source]
	            ,1 as [Corporation]
			FROM [OSEData].[dbo].[account_foreign] AF
			join (select year([Date]) as yr,symbol,avg(rate) as avgrate from [OSEData].[dbo].[forex]
				group by year([Date]),symbol) as F
			 on F.symbol=AF.[Currency] and AF.[Year]=F.yr
			join (select [CompanyID]
					,[Year]
					,[sumval]
					FROM (select [CompanyID]
					    ,[Year]
					    , sum([Value]) as sumval
			FROM [OSEData].[dbo].[account_foreign]
			group by [Year],[CompanyID]) ST
			where [sumval]>0
			) as zerofilter
			on zerofilter.[CompanyID]=AF.[CompanyID] and zerofilter.[Year]=AF.[Year]

			left join (SELECT distinct
				[original],
				[simplified_en]
	            ,[simplified_no]
	            ,[acc_no]
				,[type]
			FROM [OSEData].[dbo].[account_mapping]
			) as map
			on AF.[Description]=map.[original]

			where [IsCorporateAccount]=1
			
			UNION
		
			SELECT 
				AF.[CompanyID]
	            ,NULL as [OrganizationID]
				,AF.[Year]
				,[DescriptionEN] AS [description]
				,[DescriptionEN] AS [desc_simpl_en]
				,[Description] AS [desc_simpl_no]
				,map.[acc_no] as [account_number]
				,map.[type] as [account_type]
				,[Value]*iif([In1000]=1,1000,1000000)*F.avgrate as [Value]
				,4 as [DB_rank]
				,'Foreign' as [source]
	            ,0 as [Corporation]
			FROM [OSEData].[dbo].[account_foreign] AF
			join (select year([Date]) as yr,symbol,avg(rate) as avgrate from [OSEData].[dbo].[forex]
				group by year([Date]),symbol) as F
			 on F.symbol=AF.[Currency] and AF.[Year]=F.yr
			join (select [CompanyID]
					,[Year]
					,[sumval]
					FROM (select [CompanyID]
					    ,[Year]
					    , sum([Value]) as sumval
			FROM [OSEData].[dbo].[account_foreign]
			group by [Year],[CompanyID]) ST
			where [sumval]>0
			) as zerofilter
			on zerofilter.[CompanyID]=AF.[CompanyID] and zerofilter.[Year]=AF.[Year]

			left join (SELECT distinct
				[original],
				[simplified_en]
	            ,[simplified_no]
	            ,[acc_no]
				,[type]
			FROM [OSEData].[dbo].[account_mapping]
			) as map
			on AF.[Description]=map.[original]

			where [IsCorporateAccount]=0
		
			UNION
			
			SELECT 
				[companyID]
	            ,NULL as [OrganizationID]
				,[Year]
				,map.[original] as [description]
	            ,map.[simplified_en] as [desc_simpl_en]
	            ,map.[simplified_no] as [desc_simpl_no]
	            ,map.[acc_no] as [account_number]
				,map.[type] as [account_type]
				,iif([reverse_sign]=1,-[Value],[Value])*1000 AS [Value]
				,5 as [DB_rank]
				,'OSE' as [source]
	            ,NULL as [Corporation]
			FROM [OSEData].[dbo].[account_ose] ose
			left join (SELECT [original]
				,[simplified_en]
	            ,[simplified_no]
	            ,[acc_no]
				,[reverse_sign]
				,[from]
				,[type]
			FROM [OSEData].[dbo].[account_mapping]
			where [from]='OSE') as map
			on ose.[Description]=map.[original]
			where map.[original] is not null
			) AS T
		ORDER BY [companyID],[Year], [DB_rank]"""

	
	
	sqlstr2="""
	SELECT 
	    T2.[companyID]  
	    ,T2.[OrganizationID]
	    ,T2.[Year]
	    ,T2.[description]
	    ,T2.[desc_simpl_en]
	    ,T2.[desc_simpl_no]
	    ,T2.[account_number]
	    ,T2.[account_type]
	    ,T2. [Value]
	    ,T2.[source]
	    ,T2.[Corporation]
	    into [OSEData].[dbo].[account] FROM
	    (SELECT [CompanyID],[Year], MIN([DB_rank]) as [DB]
	  		FROM [OSEData].[dbo].[account_tmp3] 
	  		where [CompanyID]>0
	  		group by [CompanyID],[Year]) T1
	        

		join
			(SELECT * FROM [OSEData].[dbo].[account_tmp3]) T2
			ON T1.[CompanyID]=T2.[CompanyID] and T1.[Year]=T2.[Year] and T1.[DB]=T2.[DB_rank]
	    """
	
	DB.DropTable('account_tmp3',conn,crsr,dbase)
	DB.DropTable('account',conn,crsr,dbase)
	crsr.execute(sqlstr1)
	conn.commit() 
	crsr.execute(sqlstr2)
	conn.commit() 
		
	
	
	
def make_factor_criteria(conn,crsr):
	sqlstr="""SELECT	eq.[Year]
		,eq.[Month]
		,eq.[SecurityId]
		,eq.[CompanyId] 
		,eq.[ISIN] 
		,eq.[AvgMktCap]
		,iif(acc.[Value]<0,0,acc.[Value])/eq.[AvgMktCap] as BM
		,ret.[lnReturn]
	    INTO [OSEData].[dbo].[factor_criteria]
		From(select year([Date]) as [Year]
			,month([Date]) as [Month]
			,[SecurityId]
			,[CompanyId]
			,[ISIN]
			,avg([MktCap]) as AvgMktCap
			
		FROM [OSEData].[dbo].[equity_extended2]
	    where SecurityType=1 and not [CompanyId] is null
		group by year([Date]),month([Date]),[CompanyId],[SecurityId],[ISIN]) eq

		left join 	(select distinct [CompanyID],[Year]+1 as [Year],[Value],[source]
			FROM [OSEData].[dbo].[account] 
			where [desc_simpl_en]='Total equity') as acc
		on eq.[CompanyId]=acc.[CompanyId] and eq.[Year]=acc.[Year] 


		join

		(SELECT distinct
			T.[Year]
			,T.[Month]
			,T.[ISIN]
			,lst.[AdjustedPrice] as [LastPrice]
			,frst.[AdjustedPrice] as [FirstPrice]
			,log(lst.[AdjustedPrice])-log(frst.[AdjustedPrice]) as lnReturn
			 FROM
		(SELECT	
				year([Date]) as [Year]
				,month([Date]) as [Month]
				,min(day([Date])) as [FirstDay]
				,max(day([Date])) as [LastDay]
			  ,[ISIN]
		  FROM [OSEData].[dbo].[equity_extended2] 
	      where SecurityType=1 and not [CompanyId] is null
		  group by year([Date]),month([Date]),[ISIN]) AS T

		  join
			(SELECT	
					year([Date]) as [Year]
					,month([Date]) as [Month]
					,day([Date]) as [Day]
				  ,[ISIN]
				  ,[AdjustedPrice]
			  FROM [OSEData].[dbo].[equity_extended2]
	          where SecurityType=1 and not [CompanyId] is null) as lst
	          
			  on lst.[Year]=T.[Year] and lst.[Month]=T.[Month] and lst.[Day]=T.[LastDay] and lst.[ISIN]=T.[ISIN]

		  join
			(SELECT	
					year([Date]) as [Year]
					,month([Date]) as [Month]
					,day([Date]) as [Day]
				  ,[ISIN]
				  ,[AdjustedPrice]
			  FROM [OSEData].[dbo].[equity_extended2]
	          where SecurityType=1 and not [CompanyId] is null) as frst
	          
			  on frst.[Year]=T.[Year] and frst.[Month]=T.[Month] and frst.[Day]=T.[FirstDay] and frst.[ISIN]=T.[ISIN]
		) as ret
		on ret.[Year]=eq.[Year] and ret.[Month]=eq.[Month] and ret.[ISIN]=eq.[ISIN]"""
	
	DB.DropTable('factor_criteria',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit() 
		
		
		
def merge_gics(conn,crsr):
	
	sqlstr="""
	
	SELECT * INTO [OSEData].[dbo].[gics_merged]
	
	FROM
	((SELECT
      [sid]
      ,[symbol]
      ,[sec_name]
      ,[isin]
      ,[Sector]
      ,[fm_date]
      ,[to_date]
	  ,iif([FileName]='oa_gics.txt','OAX','OSE') as 'Exchange'
      ,[FileName]
  FROM [OSEData].[dbo].[gics]

union

  SELECT
		[sid]
      ,[symbol]
      ,[sec_name]
      ,[isin]
      ,[Sector]
      ,[fm_date]
      ,[to_date]
      ,[Exchange]
      ,[FileName]
  FROM [OSEData].[dbo].[gics_all])) AS g"""
	DB.DropTable('gics_merged',conn,crsr,dbase)
	crsr.execute(sqlstr)
	conn.commit()    
	DB.CreateIndex(conn,crsr,'gics_merged',dbase,True,
	               IndexFields="""      [sid]
      ,[symbol]
      ,[sec_name]
      ,[isin]""")		
	
	
	
	
	
def MakeExtendedEquityTable(conn,crsr):
	sqlstr1="""
	SELECT 
	  S.[Date]
      ,[SecurityId]
      ,[CompanyId]
      ,[Symbol]
      ,S.[ISIN]
      ,[Name]
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
      ,S.[Price]
	  ,[AdjustedPrice]
      ,S.[Dividends]
      ,[CorpAdj]
      ,S.[DividendAdj]
      ,[Currency]
      ,[SecurityType]
      ,[lnDeltaP]
	  ,[lnDeltaOSEBX]
      ,[lnDeltaOBX]
      ,[lnDeltaTIX]
	,d.[Dividends] as SumAnnualDividends,
	    (SELECT TOP 1 
	        [EventDate]
	        FROM [OSEData].[dbo].[SecuritiesInfo]
	        where [EventDate]<=S.[Date] and [ISIN]=S.[ISIN] and [TotalNumberOfShares]>0 and ([TotalNumberOfShares]!=[NumberOfShares])
	        order by [EventDate] desc) as NSharesDate,
	    (SELECT TOP 1 
	        [EventDate]
	        FROM [OSEData].[dbo].[SecuritiesInfo]
	        where [EventDate]>S.[Date] and [ISIN]=S.[ISIN] and [TotalNumberOfShares]>0 and ([TotalNumberOfShares]!=[NumberOfShares])
	        order by [EventDate]) as NSharesNextDate,
	    (SELECT TOP 1 
	        [Description]
	        FROM [OSEData].[dbo].[SecuritiesInfo]
	        where [ISIN]=S.[ISIN]) as Description,
	    (SELECT TOP 1 
	        [CountryCode]
	        FROM [OSEData].[dbo].[SecuritiesInfo]
	        where [ISIN]=S.[ISIN]) as CountryCode,
	    (SELECT TOP 1 
	        [bills_DayLnrate] 
	        FROM [OSEData].[dbo].[bills]
	        where [Date] BETWEEN  DATEADD(day,-10,S.[Date]) AND S.[Date]
	        order by [Date] DESC) as bills_DayLnrate
	into [OSEData].[dbo].[equity_extended1]
	FROM [OSEData].[dbo].[equity] S
	left join
	(select year([Date]) as [Year],[ISIN],sum([Dividends]) as [Dividends] from [OSEData].[dbo].[equity] 
	group by year([Date]),[ISIN]) d
	on d.[Year]=Year(S.[Date]) and d.[ISIN]=S.[ISIN]

	    """ 
	sqlstr2="""
	SELECT DISTINCT S.*,
	        ISNULL(E11.[NumberOfShares],E12.[NumberOfShares]) as [NumberOfShares],
	        ISNULL(ISNULL(E11.[CompanyOwnedShares],E12.[CompanyOwnedShares]),0) as [CompanyOwnedShares],
	        E3.[Exchange],
	        E4.[rate] as NOKPerForex,
	        S.[Price]*(ISNULL(E11.[NumberOfShares],E12.[NumberOfShares])-ISNULL(E11.[CompanyOwnedShares],E12.[CompanyOwnedShares])) as MktCap,
	        S.[SumAnnualDividends]/NULLIF(S.[Price],0) as DividendPriceRatio,
	        gics.[Sector]

	INTO [OSEData].[dbo].[equity_extended2]
	FROM [OSEData].[dbo].[equity_extended1] S
	left join (
		SELECT distinct
			[isin]
			,[Sector]
		FROM [OSEData].[dbo].[gics_merged]) as gics
		on gics.[isin]=S.[ISIN]
	left JOIN
	    (SELECT DISTINCT
	        [ISIN],
	        [EventDate],
	        MAX(ISNULL([TotalNumberOfShares],0)) as NumberOfShares,
	        MAX(ISNULL([TotalCompanyOwnedShares],0)) as CompanyOwnedShares
	        FROM [OSEData].[dbo].[SecuritIESInfo]
	        where not ((ISIN is null) or ([EventDate] is null))
	        group by [ISIN],[EventDate]) as E11
	    ON E11.[ISIN]=S.[ISIN] and E11.[EventDate]=S.[NSharesDate]
	left JOIN
	    (SELECT DISTINCT
	        [ISIN],
	        [EventDate],
	        MAX(ISNULL([TotalNumberOfShares],0)) as NumberOfShares,
	        MAX(ISNULL([TotalCompanyOwnedShares],0)) as CompanyOwnedShares
	        FROM [OSEData].[dbo].[SecuritIESInfo]
	        where not ((ISIN is null) or ([EventDate] is null))
	        group by [ISIN],[EventDate]) as E12
	    ON E12.[ISIN]=S.[ISIN] and E12.[EventDate]=S.[NSharesNextDate]	    
	left JOIN
	    (SELECT DISTINCT
	        [Date],
	        [ISIN],
	        iif([FileName] like '%%OAX%%','OAX','OB') as Exchange
	        FROM [OSEData].[dbo].[equitypricedump]
	        WHERE NOT [ISIN] IS NULL) as E3
	    ON E3.[ISIN]=S.[ISIN] and E3.[Date]=S.[Date]
	left JOIN
	    (SELECT DISTINCT
	        [Date],
	        [symbol],
	        [rate]
	        FROM [OSEData].[dbo].[forex]) as E4
	    ON E4.[Date]=S.[Date] and E4.[symbol]=S.[Currency]

	where S.[Price]>0
	    """
	DB.DropTable('equity_extended1',conn,crsr,'OSEData')
	DB.DropTable('equity_extended2',conn,crsr,'OSEData')
	crsr.execute(sqlstr1)
	conn.commit() 
	DB.CreateIndex(conn,crsr,'equity_extended1','OSEData',
	               IndexFields="""
                    [Date]
                    ,[ISIN]
                    ,[NSharesDate]
                    ,[NSharesNextDate]	
                    ,[Currency]""")		
	crsr.execute(sqlstr2)
	conn.commit() 
	DB.CreateIndex(conn,crsr,'equity_extended2','OSEData',
	               IndexFields="""
                    [Date]
                    ,[SecurityId]
                    ,[CompanyId]
                    ,[ISIN]
                    ,[SecurityType]""")	


