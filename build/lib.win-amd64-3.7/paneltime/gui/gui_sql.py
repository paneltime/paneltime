#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from multiprocessing import pool

import numpy as np
from gui import gui_scrolltext
import paneltime
import functions as fu
from gui import gui_script_handling




	
class sql_query(tk.Toplevel):
	def __init__(self, window,parent):
		tk.Toplevel.__init__(self, window)
		self.geometry("%dx%d%+d%+d" % (900, 700, 100, 0))
		self.win=window
		self.parent=parent
		self.columnconfigure(0,weight=1)
		self.rowconfigure(0)
		self.rowconfigure(1)
		self.rowconfigure(2,weight=5)
		self.rowconfigure(3)
		self.rowconfigure(4,weight=5)
		self.rowconfigure(5)
		
		self.name_txt=tk.StringVar(self)
		self.name_entry=tk.Frame(self)

		self.name_entry_lbl=tk.Label(self.name_entry,height=2,text="Name of query:",justify=tk.LEFT)
		self.name_entry_field=tk.Entry(self.name_entry,textvariable=self.name_txt)
		self.name_txt.set('Query 1')
		self.name_entry_lbl.pack(side=tk.LEFT)
		self.name_entry_field.pack(side=tk.LEFT)
		
		self.label_conn=tk.Label(self,height=2,text='Connection string:',anchor='sw',justify=tk.LEFT)
		self.conn_str=gui_scrolltext.ScrollText(self)
		self.conn_str.insert('1.0',window.data.get('conn_str'))
		self.label_sql=tk.Label(self,height=2,text='SQL query:',anchor='sw',justify=tk.LEFT)
		self.sql_str=gui_scrolltext.ScrollText(self,format_text=False)
		self.sql_str.insert('1.0',window.data.get('sql_str'))
		self.OK_button=tk.Button(self,height=2,text='OK',command=self.ok_pressed)
		
		self.name_entry.grid(row=0,column=0,sticky='ew')
		self.label_conn.grid(row=1,column=0,sticky='ew')
		self.conn_str.grid(row=2,column=0,sticky=tk.NSEW)
		self.label_sql.grid(row=3,column=0,sticky='ew')
		self.sql_str.grid(row=4,column=0,sticky=tk.NSEW)
		self.OK_button.grid(row=5,column=0,sticky=tk.NSEW)
		
		self.transient(window) #set to be on top of the main window
		self.grab_set() #hijack all commands from the master (clicks on the main window are ignored)	
		self.protocol("WM_DELETE_WINDOW", self.on_closing)
	
	def ok_pressed(self,event=None):
		self.win.data['sql_str']='\n'+fu.clean_section(self.sql_str.get_all())
		self.win.data['conn_str']=fu.clean_section(self.conn_str.get_all())
		res=self.win.exec(self.win.data['conn_str'])
		if res==False:
			return		
		name=self.name_txt.get()
		sqlstr=f"\"\"\"{self.win.data['sql_str']}\"\"\""
		exe_str=f"""
from paneltime import *\n
data=dict()
data['{name}']=load_SQL(conn,{sqlstr})"""
		res=self.win.exec(exe_str)
		if res==False:
			return		
		if not name in self.win.locals['data']:
			return
		df=self.win.locals['data'][name]
		self.win.grab_set()
		tree=self.win.right_tabs.data_tree
		data_import_script=f"{self.win.data['conn_str']}\n{exe_str}"
		tree.datasets.add(tree,name,df,self.win.data['sql_str'],data_import_script)
		self.withdraw()
			
	def on_closing(self):
		self.win.grab_set()
		self.win.data['sql_str']=self.sql_str.get_all()
		self.win.data['conn_str']=self.conn_str.get_all()		
		self.withdraw()
		
		
	def show(self):
		self.win.grab_set()
		self.deiconify()