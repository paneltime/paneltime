#!/usr/bin/python
# -*- coding: UTF-8 -*-

forex="""
	[Date] ,
	[symbol] 
"""

options_prices="""
	[Date],
	[SecurityID],
	[Symbol] ,
	[ISIN],
    [Name] ,
    [ExpirationDate] ,
	[UnderlyingSID] ,
    [UnderlyingSymbol]"""


bondfeed="""
	[Type] ,
	[SecurityId] ,
	[SecurityType] ,
	[Symbol] ,
	[ISIN] ,
	[CompanyId] ,
	[SectorId] ,
    [Name],
    [FileName]
"""

bondindex="""
	[Date] ,
	[SecurityId] ,
	[Symbol] ,
	[Name],
    [IndexType],
    [FileName]"""


bondprices="""
	[Date] ,
	[SecurityId] ,
	[BondType] ,
	[Symbol] ,
	[ISIN],
    [Name],
    [FileName]  """


equity_pricedump="""
	[Date] ,
	[SecurityId] ,
	[Symbol] ,
	[ISIN] ,
	[Name],
    [FileName] """

equityfeed="""
	[Type] ,
	[SecurityId] ,
	[SecurityType] ,
	[Symbol] ,
	[ISIN] ,
	[SecurityName],
	[Currency],
	[CompanyId] ,
	[CountryCode],
	[Name],
    [FileName]"""


equityindex="""
	[Date] ,
	[SecurityId] ,
	[Symbol],
	[Name] ,
	[Open] ,
	[High] ,
	[Low] ,
	[Last],
    [FileName]"""

equitypricedump="""
	[Date] ,
	[SecurityId] ,
	[Symbol] ,
	[ISIN] ,
	[Name],
    [FileName] """


fund_adj_factors="""
	[Symbol] ,
	[SecurityId] ,
	[ISIN] ,
	[Date],
    [FileName] """


fund_dividends="""
	[Symbol] ,
	[SecurityId] ,
	[ISIN] ,
	[Date],
    [FileName] """


fund_prices="""
	[Symbol] ,
	[SecurityId] ,
	[ISIN] ,
	[Date],
    [FileName] """


funds="""
	[SecurityId] ,
	[ISIN] ,
	[FundId] ,
	[Symbol],
	[Name],
    [FileName] """


futforw_prices="""
	[Date] ,
	[SecurityId] ,
	[Symbol] ,
	[ISIN] ,
	[ContractSymbol] ,
	[Description] ,
	[ExDate] ,
	[IssuerSecurityId] ,
	[IssuerSymbol],
    [FileName] """


newsdump="""
	[publishTime] ,
	[issuer] ,
	[issuerStatus],
	[category],
	[market],
    [FileName]"""


shareidx_prices="""
	[Date] ,
	[SecurityId] ,
	[Symbol],
	[Name] ,
	[IndexType],
    [FileName]
    """

account_brnsnd="""
[IsCorporateAccount],
[OrganizationID],
[CompanyID] ,
[Name],
[FetchDate],
[Year] ,
[Description]
"""

account_ose="""
[short_name],
[symbol] , 
[long_name],
[companyID],
[orgID] ,
[Year] , 
[period] ,
[account_type],
[file_name],
[Description]"""


account_foreign="""
[IsCorporateAccount],
[CompanyID],
[account],
[accountType],
[Directory],
[Year],
[Description],
[DescriptionEN],
[Value]
"""

GICS_all="""
[sid],	
[symbol] ,	
[sec_name] ,	
[isin] ,	
[Sector] ,	
[fm_date] ,	
[to_date],	
[Exchange] """

gics="""
[sid] ,	
[symbol] ,	
[sec_name] ,	
[isin] ,	
[Sector] ,	
[fm_date],	
[to_date]"""

account_mapping="""[native_desc],	
[acc_no],
[simplified_en] ,	
[simplified_no] ,	
[reverse_sign] ,	
[from]"""

msciworld="""[Date],	
[index_value]"""