#!/usr/bin/python
# -*- coding: UTF-8 -*-
import DB
import Functions as fu

def MakeBillsTable():
	conn,crsr=Connect(db)
	sqlstr="""CREATE TABLE [OSEData].[dbo].[bills](
				[ID] [bigint] IDENTITY(1,1) NOT NULL,
		[Date] date(24) NULL,
		[SecurityId] bigint NULL,
		[ISIN] [varchar](24) NULL,
		[Date] date NULL,
		[NAV] float NULL,
		[FileName] [varchar](100) NULL DEFAULT (NULL))"""
	crsr.execute(sqlstr)
	conn.commit()
	sqlstr="""SELECT [Date], AVG(-LOG([Price])/[DtoMat]) as bills_DayLnrate FROM
			(SELECT distinct [Date],[ISIN],DATEDIFF(DAY, [Date], [MaturityDate]) As DtoMat,([BestBidPrice]+[BestAskPrice])*0.5/100 as Price
	                FROM [OSE].[dbo].[bonds]
	                where [Name] like '%statskasse%') b4
	        where not [Price] is null
	        group by [Date]
	        order by [Date]
	        """
	crsr.execute(sqlstr)
	r=crsr.fetchall()
	billstbl=[]
	