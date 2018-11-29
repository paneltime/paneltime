#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DBparser
import DB

import Functions as fu
import FinalTables as FT
import WorkingTables as WT
import factors_create as FC
import Equity
import Funds
import download
import CleanUpUsers as cuu
import FileSync
import sendmail

import mysql
import indexcalc

conn, crsr=DB.Connect('OSEData')
Equity.CreateTables()
indexcalc.make_index(conn,crsr)
download.get_FF_data()

FT.MakeTableColDef(conn, crsr)

#WT.MakeTables()

#FT.MakeIndexTable(conn,crsr)
#DBUpdater.MakeTables()
#cuu.main()
#import DBUpdater

#FT.MakeBondInfoTable(conn, crsr)
#FT.MakeBondPricesTable(conn, crsr)

#Funds.CreateTables()
#Equity.CreateTables()

#WT.MakeTables()
#FT.MakeTables()

v=0

#FT.MakeFundTable(conn, crsr)
#FT.MakeEquityTable(conn, crsr)

#Equity.calc_MSCI_returns(conn,crsr)

#WT.create_tbl_accounts3(conn, crsr)
#FT.make_table_account(conn, crsr)

#FT.MakeEquityTable(conn,crsr)

#download.import_forex()
#conn,crsr=DB.Connect('OSEData')
#FT.MakeEquityTable(conn, crsr)
#Equity.make_index_caps(conn,crsr)

#equity_db.MakeExtendedEquityTable(conn, crsr)
#Equity.make_index_caps(conn, crsr)
#Equity.make_index(conn, crsr)
#equity_db.MakeExtendedEquity3table(conn, crsr)
#equity_db.make_alpha_table(conn, crsr)
#conn.close()
#equity_db.MakeExtendedEquityTable(conn, crsr)
#equity_db.MakeExtendedEquity3table(conn, crsr)
#WT.make_alpha_table(conn, crsr)

#FT.FutForw(conn, crsr)
#download.get_FF_data()

#FC.calc_factors()

#ab.save_to_file()