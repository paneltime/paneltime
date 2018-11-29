#!/usr/bin/python
# -*- coding: UTF-8 -*-

forex="""CREATE TABLE [OSEData].[dbo].[forex](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[symbol] [varchar](5) NULL,
	[rate] float NULL
	)"""


bondfeed="""CREATE TABLE [OSEData].[dbo].[bondfeed](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Type] [varchar](100) NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[SecurityType] int NULL DEFAULT (NULL),
	[Symbol] [varchar](16) NULL DEFAULT (NULL),
	[ISIN] [varchar](24) NULL DEFAULT (NULL),
	[SecurityName] [varchar](34) NULL DEFAULT (NULL),
	[CurrencyCode] [varchar](10) NULL DEFAULT (NULL),
	[CompanyId] bigint NULL DEFAULT (NULL),
	[CountryCode] [varchar](10) NULL DEFAULT (NULL),
	[Name] [varchar](34) NULL DEFAULT (NULL),
	[SectorId] int NULL DEFAULT (NULL),
	[Exchange] int NULL DEFAULT (NULL),
	[FromDate] date NULL DEFAULT (NULL),
	[ToDate] date NULL DEFAULT (NULL),
	[ListingType] int NULL DEFAULT (NULL),
    [BondIssueType] int NULL DEFAULT (NULL),
    [BondType] int NULL DEFAULT (NULL),
    [AmortizationType] int NULL DEFAULT (NULL),
    [DownPaymentMethod] int NULL DEFAULT (NULL),
    [IssueDate] date NULL DEFAULT (NULL),
    [MaturityDate] date NULL DEFAULT (NULL),
    [ActualMaturityDate] date NULL DEFAULT (NULL),
    [InterestAsOf] date NULL DEFAULT (NULL),
    [FirstCouponPayment] date NULL DEFAULT (NULL),
    [FirstDownPayment] date NULL DEFAULT (NULL),
    [CouponRate] float NULL DEFAULT (NULL),
    [RiskCategory] int NULL DEFAULT (NULL),
    [BondSector] int NULL DEFAULT (NULL),
    [BondDescription] [varchar](1000) NULL DEFAULT (NULL),
    [CSD] int NULL DEFAULT (NULL),
    [CouponInformation] int NULL DEFAULT (NULL),
    [CouponSpread] float NULL DEFAULT (NULL),
    [FirstCouponRate] date NULL DEFAULT (NULL),
    [SecondCouponRate] date NULL DEFAULT (NULL),
    [CouponDate] date NULL DEFAULT (NULL),
    [OpeningDate] date NULL DEFAULT (NULL),
    [ClosingDate] date NULL DEFAULT (NULL),
    [MaxAmount] float NULL DEFAULT (NULL),
    [DenominationAmount] float NULL DEFAULT (NULL),
    [CashFlowDate] date NULL DEFAULT (NULL),
    [CashFlowTypeId] int NULL DEFAULT (NULL),
    [Price] float NULL DEFAULT (NULL),
    [NominalAmountVPS] float NULL DEFAULT (NULL),
    [NominalAmountTotal] float NULL DEFAULT (NULL),
    [OptionType] [varchar](3) NULL DEFAULT (NULL),
    [ExpirationDate] date NULL DEFAULT (NULL),
    [AnnouncementDate] date NULL DEFAULT (NULL),
    [InterestRegulationDate] date NULL DEFAULT (NULL),
    [To_date] date NULL DEFAULT (NULL),
    [sid] bigint NULL DEFAULT (NULL),
    [TimeType] int NULL DEFAULT (NULL),
    [ExCount] int NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""

bondindex="""CREATE TABLE [OSEData].[dbo].[bondindex](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[Symbol] [varchar](16) NULL DEFAULT (NULL),
	[Name] [varchar](50) NULL DEFAULT (NULL),
	[IndexType] [varchar](10) NULL DEFAULT (NULL),
	[Open] float NULL DEFAULT (NULL),
	[High] float NULL DEFAULT (NULL),
	[Low] float NULL DEFAULT (NULL),
	[Close] float NULL DEFAULT (NULL),
	[OpenYield] float NULL DEFAULT (NULL),
	[CloseYield] float NULL DEFAULT (NULL),
	[OpenDuration] float NULL DEFAULT (NULL),
	[CloseDuration] float NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""

bondprices="""CREATE TABLE [OSEData].[dbo].[bondprices](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[BondType] int NULL DEFAULT (NULL),
	[Symbol] [varchar](16) NULL DEFAULT (NULL),
	[ISIN] [varchar](24) NULL DEFAULT (NULL),
	[Name] [varchar](100) NULL DEFAULT (NULL),
	[LongName] [varchar](100) NULL DEFAULT (NULL),
	[CouponRate] float NULL DEFAULT (NULL),
	[FmDate] date NULL DEFAULT (NULL),
	[MaturityDate] date NULL DEFAULT (NULL),
	[OpenPrice] float NULL DEFAULT (NULL),
	[High] float NULL DEFAULT (NULL),
	[Low] float NULL DEFAULT (NULL),
	[LastTradedPrice] float NULL DEFAULT (NULL),
	[OfficialVolume] float NULL DEFAULT (NULL),
	[UnofficialVolume] float NULL DEFAULT (NULL),
	[BestBidPrice] float NULL DEFAULT (NULL),
	[BestAskPrice] float NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


equity_pricedump="""CREATE TABLE [OSEData].[dbo].[equity_pricedump](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
	[Symbol] [varchar](16) NULL,
	[ISIN] [varchar](13) NULL,
	[Name] [varchar](50) NULL,
	[BestBidPrice] float NULL,
	[BestAskPrice] float NULL,
	[Open] float NULL,
    [High] float NULL,
    [Low] float NULL,
    [LastTradedPrice] float NULL,
    [OfficialNumberOfTrades] float NULL,
    [OfficialVolume] float NULL,
    [UnofficialNumberOfTrades] float NULL,
    [UnofficialVolume] float NULL,
    [VolumeWeightedAveragePrice] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""

options_prices="""CREATE TABLE [OSEData].[dbo].[options_prices](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityID] bigint NULL,
	[Symbol] [varchar](16) NULL,
	[ISIN] [varchar](13) NULL,
    [Name] [varchar](100) NULL,
    [ExpirationDate] date NULL,
	[UnderlyingSID] bigint NULL,
    [UnderlyingSymbol] [varchar](13) NULL,
	[BestBidPrice] float NULL,
	[BestAskPrice] float NULL,
	[Last] float NULL,
    [Open] float NULL,
    [High] float NULL,
    [Low] float NULL,
    [Volum] float NULL,
    [void] [varchar](100) NULL,
    [OpenInterest] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


equityfeed="""CREATE TABLE [OSEData].[dbo].[equityfeed](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Type] [varchar](100) NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[SecurityType] int NULL DEFAULT (NULL),
	[Symbol] [varchar](22) NULL DEFAULT (NULL),
	[ISIN] [varchar](24) NULL DEFAULT (NULL),
	[SecurityName] [varchar](50) NULL DEFAULT (NULL),
	[Currency] [varchar](10) NULL DEFAULT (NULL),
	[CompanyId] bigint NULL DEFAULT (NULL),
	[CountryCode] [varchar](10) NULL DEFAULT (NULL),
	[Name] [varchar](50) NULL DEFAULT (NULL),
	[Exchange] [varchar](10) NULL DEFAULT (NULL),
	[FromDate] date NULL DEFAULT (NULL),
	[ToDate] date NULL DEFAULT (NULL),
	[EquityClassId] int NULL DEFAULT (NULL),
	[Description] [varchar](40) NULL DEFAULT (NULL),
	[EventId] bigint NULL DEFAULT (NULL),
	[EventDate] date NULL DEFAULT (NULL),
	[SequenceNumber] int NULL DEFAULT (NULL),
	[PaymentDate] date NULL DEFAULT (NULL),
	[AdjustmentFactor] float NULL DEFAULT (NULL),
	[CompanyOwnedShares] float NULL DEFAULT (NULL),
	[TotalCompanyOwnedShares] float NULL DEFAULT (NULL),
	[NumberOfShares] float NULL DEFAULT (NULL),
	[TotalNumberOfShares] float NULL DEFAULT (NULL),
	[CurrencyCode] [varchar](10) NULL DEFAULT (NULL),
	[NominalValue] float NULL DEFAULT (NULL),
	[SubscriptionPrice] float NULL DEFAULT (NULL),
	[MarketPrice] float NULL DEFAULT (NULL),
	[NewShares] float NULL DEFAULT (NULL),
	[OldShares] float NULL DEFAULT (NULL),
	[EffectiveDate] date NULL DEFAULT (NULL),
	[ConversionDate] date NULL DEFAULT (NULL),
	[ConversionPrice] float NULL DEFAULT (NULL),
	[DividendId] bigint NULL DEFAULT (NULL),
	[TypeOfDividend] int NULL DEFAULT (NULL),
	[NumberOfDividendsPerYear] int NULL DEFAULT (NULL),
	[DividendInNOK] float NULL DEFAULT (NULL),
	[DividendInForeignCurrency] float NULL DEFAULT (NULL),
	[WarrantType] int NULL DEFAULT (NULL),
	[UnderlyingSecurity] bigint NULL DEFAULT (NULL),
	[TermsShares] float NULL DEFAULT (NULL),
	[WarrantsIssued] float NULL DEFAULT (NULL),
	[TermsWarrants] float NULL DEFAULT (NULL),
	[Price] float NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


equityindex="""CREATE TABLE [OSEData].[dbo].[equityindex](
	[ID] [bigint],
	[Date] date NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[Symbol] [varchar](22) NULL DEFAULT (NULL),
	[Name] [varchar](100) NULL DEFAULT (NULL),
	[Open] float NULL DEFAULT (NULL),
	[High] float NULL DEFAULT (NULL),
	[Low] float NULL DEFAULT (NULL),
	[Last] float NULL DEFAULT (NULL),
    [lnDelta] float NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


equityindex_final="""CREATE TABLE [OSE].[dbo].[equityindex2](
	[ID] [bigint],
	[Date] date NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[Symbol] [varchar](22) NULL DEFAULT (NULL),
	[Name] [varchar](100) NULL DEFAULT (NULL),
	[Open] float NULL DEFAULT (NULL),
	[High] float NULL DEFAULT (NULL),
	[Low] float NULL DEFAULT (NULL),
	[Last] float NULL DEFAULT (NULL),
    [lnDelta] float NULL DEFAULT (NULL))"""

equityindex2="""CREATE TABLE [OSEData].[dbo].[equityindex2](
	[ID] [bigint],
	[Date] date NULL DEFAULT (NULL),
	[SecurityId] bigint NULL DEFAULT (NULL),
	[Symbol] [varchar](22) NULL DEFAULT (NULL),
	[Name] [varchar](100) NULL DEFAULT (NULL),
	[Open] float NULL DEFAULT (NULL),
	[High] float NULL DEFAULT (NULL),
	[Low] float NULL DEFAULT (NULL),
	[Last] float NULL DEFAULT (NULL),
    [lnDelta] float NULL DEFAULT (NULL))"""

equitypricedump="""CREATE TABLE [OSEData].[dbo].[equitypricedump](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
	[Symbol] [varchar](16) NULL,
	[ISIN] [varchar](13) NULL,
	[Name] [varchar](50) NULL,
	[BestBidPrice] float NULL,
	[BestAskPrice] float NULL,
	[Open] float NULL,
    [High] float NULL,
    [Low] float NULL,
    [LastTradedPrice] float NULL,
    [OfficialNumberOfTrades] float NULL,
    [OfficialVolume] float NULL,
    [UnofficialNumberOfTrades] float NULL,
    [UnofficialVolume] float NULL,
    [VolumeWeightedAveragePrice] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


fund_adj_factors="""CREATE TABLE [OSEData].[dbo].[fund_adj_factors](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Symbol] [varchar](24) NULL,
	[SecurityId] bigint NULL,
	[ISIN] [varchar](24) NULL,
	[Date] date NULL,
	[AdjFactor] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


fund_dividends="""CREATE TABLE [OSEData].[dbo].[fund_dividends](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Symbol] [varchar](24) NULL,
	[SecurityId] bigint NULL,
	[ISIN] [varchar](24) NULL,
	[Date] date NULL,
	[Dividend] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


fund_prices="""CREATE TABLE [OSEData].[dbo].[fund_prices](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Symbol] [varchar](24) NULL,
	[SecurityId] bigint NULL,
	[ISIN] [varchar](24) NULL,
	[Date] date NULL,
	[NAV] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""

mutualfunds="""CREATE TABLE [OSEData].[dbo].[mutualfunds](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
	[FundId] bigint NULL,
	[Symbol] [varchar](24) NULL,
	[ISIN] [varchar](24) NULL,
    [Name] [varchar](100) NULL,
	[NAV] float NULL,
    [Dividends] float NULL,
    [CorpAdj] float NULL,
    [DividendAdj] float NULL, 
    [lnDeltaNAV] float NULL,
    [lnDeltaOSEBX] float NULL,
    [lnDeltaOSEFX] float NULL,
    [lnDeltaOBX]float NULL,
    [NAVAdj] float NULL,
    [OSEBX] float NULL,
    [OSEFX] float NULL,
    [OBX]float NULL
    )"""



funds="""CREATE TABLE [OSEData].[dbo].[funds](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[SecurityId] bigint NULL,
	[ISIN] [varchar](24) NULL,
	[FundId] bigint NULL,
	[Symbol] [varchar](20) NULL,
	[Name] [varchar](100) NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


futforw_prices="""CREATE TABLE [OSEData].[dbo].[futforw_prices](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
	[Symbol] [varchar](20) NULL,
	[ISIN] [varchar](24) NULL,
	[ContractSymbol] [varchar](10) NULL,
	[Description] [varchar](30) NULL,
	[ExDate] date NULL,
	[IssuerSecurityId] bigint NULL,
	[IssuerSymbol] [varchar](10) NULL,
	[BestBidPrice] float NULL,
	[BestAskPrice] float NULL,
	[Open] float NULL,
    [High] float NULL,
    [Low] float NULL,
    [LastTradedPrice] float NULL,
    [NumberOfTrades] float NULL,
    [Volume] float NULL,
    [VolumeWeightedAveragePrice] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


newsdump=""" CREATE TABLE [OSEData].[dbo].[newsdump](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[publishTime] datetime NULL DEFAULT (NULL),
	[issuer] [varchar](36) NULL DEFAULT (NULL),
	[issuerStatus] [varchar](10) NULL DEFAULT (NULL),
	[title] [varchar](204) NULL DEFAULT (NULL),
	[category] [varchar](82) NULL DEFAULT (NULL),
	[market] [varchar](28) NULL DEFAULT (NULL),
	[text] [varchar](max) NULL DEFAULT (NULL),
    [FileName] [varchar](100) NULL DEFAULT (NULL))"""


shareidx_prices="""CREATE TABLE [OSEData].[dbo].[shareidx_prices](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
	[Symbol] [varchar](20) NULL,
	[Name] [varchar](50) NULL,
	[IndexType] [varchar](10) NULL,
	[Open] float NULL,
	[High] float NULL,
	[Low] float NULL,
	[Close] float NULL,
    [FileName] [varchar](100) NULL DEFAULT (NULL))
    """


equity="""CREATE TABLE [OSEData].[dbo].[equity](
	[ID] [bigint] IDENTITY(1,1) NOT NULL,
	[Date] date NULL,
	[SecurityId] bigint NULL,
    [CompanyId] [bigint] NULL DEFAULT (NULL),
	[Symbol] [varchar](16) NULL,
	[ISIN] [varchar](13) NULL,
	[Name] [varchar](50) NULL,
	[BestBidPrice] float NULL,
	[BestAskPrice] float NULL,
	[Open] float NULL,
    [High] float NULL,
    [Low] float NULL,
    [LastTradedPrice] float NULL,
    [OfficialNumberOfTrades] float NULL,
    [OfficialVolume] float NULL,
    [UnofficialNumberOfTrades] float NULL,
    [UnofficialVolume] float NULL,
    [VolumeWeightedAveragePrice] float NULL,
    [Price] float NULL,
    [Dividends] float NULL,
    [CorpAdj] float NULL,
    [DividendAdj] float NULL,
    [Currency] [varchar](10) NULL,
    [SecurityType] [int],

    [lnDeltaP] float NULL,
    [lnDeltaOSEBX] float NULL,
    [lnDeltaOBX] float NULL,
    [lnDeltaTIX] float NULL,
    [AdjustedPrice] float NULL,
    [OSEBX] float NULL,
    [OBX] float NULL,
    [TIX] float NULL,
    )"""

account_brnsnd="""CREATE TABLE [OSEData].[dbo].[account_brnsnd](
        [ID] [bigint] IDENTITY(1,1) NOT NULL,
        [IsCorporateAccount] bit NULL,
        [OrganizationID] bigint NULL,
        [CompanyID] bigint NULL,
        [Name] [varchar](50) NULL,
        [webName] [varchar](200) NULL,
        [FetchDate] datetime NULL,
        [Year]  bigint NULL,
        [Description] [varchar](100) NULL,
        [Value] float NULL,
        [DescrID] bigint NULL
        )"""

account_ose="""CREATE TABLE [OSEData].[dbo].[account_ose](
        [ID] [bigint] IDENTITY(1,1) NOT NULL,
        [short_name] [varchar](100) NULL,
        [symbol] [varchar](20) NULL, 
        [long_name] [varchar](100) NULL,
        [companyID] bigint NULL,
        [orgID] bigint NULL,
        [Year]  bigint NULL, 
        [period] int NULL,
        [account_type] int NULL,
        [file_name] [varchar](100) NULL,
        [Description] [varchar](100) NULL,
        [Value] float NULL
        )"""

account_foreign="""CREATE TABLE [OSEData].[dbo].[account_foreign](
        [ID] [bigint] IDENTITY(1,1) NOT NULL,
        [IsCorporateAccount] bit NULL,
        [In1000] bit NULL,
        [Currency] [varchar](10) NULL,
        [CompanyID] bigint NULL,
        [account]  bigint NULL,
        [accountType]  [varchar](50) NULL,
        [Directory] [varchar](100) NULL,
        [Year]  bigint NULL,
        [Description] [varchar](100) NULL,
        [DescriptionEN] [varchar](100) NULL,
        [Value] float NULL
        )"""

GICS_all="""CREATE TABLE [OSEData].[dbo].[GICS_all](
 [ID] [bigint] IDENTITY(1,1) NOT NULL,
[sid] bigint NULL,	
[symbol] [varchar](50) NULL,	
[sec_name] [varchar](100) NULL,	
[isin] [varchar](50) NULL,	
[Sector] [varchar](50) NULL,	
[fm_date] date NULL,	
[to_date] date NULL,	
[Exchange]  [varchar](50) NULL,
[FileName] [varchar](100) NULL)"""

gics="""CREATE TABLE [OSEData].[dbo].[gics](
 [ID] [bigint] IDENTITY(1,1) NOT NULL,
[sid] bigint NULL,	
[symbol] [varchar](50) NULL,	
[sec_name] [varchar](100) NULL,	
[isin] [varchar](50) NULL,	
[Sector] [varchar](50) NULL,	
[fm_date] date NULL,	
[to_date] date NULL,
[FileName] [varchar](100) NULL)"""

account_mapping="""CREATE TABLE [OSEData].[dbo].[account_mapping](
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[original] [varchar](100) NULL,	
[acc_no] [bigint] NULL,
[simplified_en] [varchar](100) NULL,	
[simplified_no] [varchar](100) NULL,
[reverse_sign] [bit] NULL,
[from] [varchar](1000) NULL,
[type] [varchar](100) NULL,
[FileName] [varchar](100) NULL)
"""



factors_tmp="""CREATE TABLE [OSEData].[dbo].[factors_tmp](
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[Date] date NULL,	
[return] float NULL,
[factor_name] [varchar](100) NULL)
"""

FamaFrench="""CREATE TABLE [OSEData].[dbo].[FamaFrench](
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[Date] date NULL,	
[return] float NULL,
[factor_name] [varchar](100) NULL,

)
"""

error_adj="""CREATE TABLE [OSEData].[dbo].[error_adj](
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[Date] date NULL,
[ISIN] [varchar](20) NULL,
[adj] float NULL,
[return_adj] float NULL,
[price_adj] float NULL,
[price] float NULL

)
"""

OSEBX_recalc="""CREATE TABLE [OSEData].[dbo].[OSEBX_recalc](
[Date] date NULL,
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[index] float NULL,
[index_d] float NULL,
[index_c] float NULL,
[index_p] float NULL,
[adj_d] float NULL,
[adj_p] float NULL,
[return] float NULL,
[return_d] float NULL,
[return_c] float NULL,
[return_p] float NULL
)
"""

OSEBX_mktshares="""CREATE TABLE [OSEData].[dbo].[OSEBX_mktshares](
[ID] [bigint] IDENTITY(1,1) NOT NULL,
[Date] date NULL,
[ISIN] [varchar](20) NULL,
[mktshare] float NULL,
[alpha] float NULL,
[beta] float NULL,
[marketcap] float NULL,
)
"""



account_org_brnsnd_all="""CREATE TABLE [OSEData].[dbo].[account_org_brnsnd_all](

 [organisasjonsnummer] bigint,
 [navn] varchar(200),
 [naeringskode1] varchar(200),
 [registreringsdatoEnhetsregisteret] date,
 [registrertIMvaregisteret] smallint,
 [konkurs] smallint,
 [underAvvikling] smallint,
 [antallAnsatte] bigint,
 [registrertIForetaksregisteret] smallint,
 [forretningsadresse] varchar(200),
 [organisasjonsform] varchar(200),
 [orgKode] varchar(10)
 [source] varchar(200)

)
"""

account_OID="""CREATE TABLE [OSEData].[dbo].[account_OID](
[CID] bigint NULL,
[Name] varchar (200) NULL,
[OrgID] bigint NULL,
[fname] varchar (200) NULL
)
"""

OSEBX_number_shares="""CREATE TABLE [OSEData].[dbo].[OSEBX_number_shares](
[Date] date NULL,
[ixSymbol] varchar (10) NULL,
[Symbol] varchar (10)  NULL,
[Name] varchar (100) NULL,
[ISIN] varchar (20) NULL,
[NoShares] bigint NULL,
[FileName] varchar (200) NULL
)
"""