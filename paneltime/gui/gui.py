#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from multiprocessing import pool
import sys
from gui import gui_charts
from gui import gui_functions as guif
import functions as fu
import os
from gui import gui_buttons
import functions as fu
from gui import gui_right_tabs
from gui import gui_scrolltext
from gui import gui_main_tabs
import tempstore
import numpy as np
import traceback
FONT_SIZE=10	
FONT_WIDTH=FONT_SIZE*0.35	
LINE_HEIGHT=1.54

GRAPH_IMG_WIDTH=0.35
GRAPH_IMG_HEIGHT=0.85



class window(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.title("Paneltime")
		self.define_styles()
		self.mc=None
		self.geometry('%sx%s+%s+%s' %(self.winfo_screenwidth(),self.winfo_screenheight()-75,-5,0))
		if 'nt'==os.name:
			iconpath=os.path.join(fu.currentdir(),'paneltime.ico')
			self.iconbitmap(iconpath)	
			self.iconpath=iconpath
		else:
			self.iconpath=None
		#self.add_menu()
		self.add_panes()
		#self.main_pane=self
		self.main_pane.rowconfigure(0, weight=1)
		self.main_pane.columnconfigure(0, weight=80,uniform="fred")
		self.main_pane.columnconfigure(1)
		self.main_pane.columnconfigure(2, weight=20,uniform="fred")
		self.add_main_frames()
		self.add_delimiters()
		self.output = gui_scrolltext.ScrollText(self.frm_left,format_text=False)
		self.output.grid(row=2, column=0,sticky=tk.NSEW)
		self.data=datastore(self)
		self.autosave()
		self.right_tabs=gui_right_tabs.right_tab_widget(self)
		self.main_tabs=gui_main_tabs.main_tabs(self)
		self.locals=dict()
		self.globals={'window':self,'data':self.right_tabs.data_tree.datasets}
		sys.stdout=stdout_redir(self.output)
		sys.stderr=stdout_redir(self.output)
		self.protocol("WM_DELETE_WINDOW", self.on_closing)
		
	def autosave(self):
		if fu.get_idle_duration()>600:
			self.data.save()
		self.after(600000,self.autosave)
	
	def exec(self,source):
		try:
			exec(source,self.globals,self.locals)
			return True
		except Exception as e:
			traceback.print_exc()
		return False
	
	def define_styles(self):
		style = ttk.Style()
		style.configure("TFrame", background='white')
		style.configure("TNotebook",background='white')

	def add_panes(self):
		self.rowconfigure(0)
		self.rowconfigure(1,weight=1,uniform="fred")
		self.rowconfigure(2)
		self.columnconfigure(0,weight=1)
		
		self.button_pane=tk.Frame(self,height=10,background="white")
		#self.buttons=gui_buttons.buttons(self)
		self.main_pane=tk.Canvas(self,background='pink')
		self.bottom_bar=tk.Frame(self,background='white',height=25)
		self.about=tk.Label(self.bottom_bar,text='This sofware is a scientific work and should be cited if used for scientific purposes. Cite Sirnes (2020)',background='white')	
		self.about.grid(row=0,column=0,sticky=tk.W)
		self.button_pane.grid(row=0,column=0,sticky=tk.EW)	
		self.main_pane.grid(row=1,column=0,sticky=tk.NSEW)	
		self.bottom_bar.grid(row=2,column=0,sticky=tk.EW)
		
	def do_nothing(self,event):
		pass
		
	def add_delimiters(self):
		self.delimiter_v=tk.Frame(self.frm_left,background="dark grey",height=5,cursor='sb_v_double_arrow')
		self.delimiter_v.grid(row=1,column=0,sticky=tk.EW)
		self.delimiter_v.bind("<ButtonRelease-1>", self.vertical_resize)
		
		self.delimiter_h=tk.Frame(self.main_pane,background="dark grey",width=5,cursor='sb_h_double_arrow')
		self.delimiter_h.grid(row=0,column=1,sticky=tk.NS)
		self.delimiter_h.bind("<ButtonRelease-1>", self.horizontal_resize)	
		
	def add_main_frames(self):
		left_weight=80
		self.frm_left = tk.Frame(self.main_pane,background='green')
		
		self.frm_left.rowconfigure(0, weight=left_weight,uniform="fred")
		self.frm_left.rowconfigure(1)
		self.frm_left.rowconfigure(2, weight=100-left_weight,uniform="fred")
		self.frm_left.columnconfigure(0, weight=1)
		self.frm_left.grid(row=0,column=0,sticky=tk.NSEW)
		
		self.frm_right = tk.Frame(self.main_pane,background='white')
		self.frm_right.rowconfigure(0)
		self.frm_right.rowconfigure(1,weight=1)
		self.frm_right.columnconfigure(0,weight=1)
		self.frm_right.grid(row=0,column=2,sticky=tk.NSEW)	
		
		self.main_frames_weight=left_weight
		
		#self.box_border_frm=tk.Frame(self.frm_left,background="red",height=100)
	
	def vertical_resize(self,event):
		self.pack_propagate(0)
		new_y=self.output.winfo_y()+event.y
		y=int(100*(new_y)/(self.winfo_height()))
		self.frm_left.rowconfigure(0, weight=y,uniform="fred")
		self.frm_left.rowconfigure(2, weight=100-y,uniform="fred")	
		
	def horizontal_resize(self,event):
		self.pack_propagate(0)
		new_x=self.frm_right.winfo_x()+event.x
		x=int(100*new_x/(self.winfo_width()))
		self.main_pane.columnconfigure(0, weight=max((x,0)))
		self.main_pane.columnconfigure(2, weight=max((100-x,0)))	
		
	def add_menu(self):
		menubar = tk.Menu(self)
		filemenu = tk.Menu(menubar, tearoff=0)
		menubar.add_cascade(label="File", menu=filemenu)

		openmenu = tk.Menu(menubar, tearoff=0)
		filemenu.add_cascade(label="Open data", menu=openmenu)
		openmenu.add_command(label="Data text file", command=self.donothing)
		openmenu.add_command(label="Sql connection", command=self.donothing)
		filemenu.add_command(label="Open project", command=self.donothing)
		filemenu.add_separator()
		filemenu.add_command(label="Save project", command=self.donothing)
		filemenu.add_command(label="Save project as", command=self.donothing)
		filemenu.add_separator()
		filemenu.add_command(label="Settings", command=self.donothing)
		filemenu.add_separator()
		filemenu.add_command(label="Quit", command=self.donothing)		
		
		self.settingsmenu = tk.Menu(menubar, tearoff=0)
		menubar.add_cascade(label="Regression", menu=self.settingsmenu)
		self.settingsmenu.add_command(label="Run Ctrl+R", command=self.donothing)
		self.settingsmenu.add_command(label="Abort Ctrl+A", command=self.abort)
		filemenu.add_separator()
		self.settingsmenu.add_command(label="Settings", command=self.donothing)
		filemenu.add_separator()
		self.settingsmenu.add_command(label="Scatter plots raw", command=self.show_scatter,state="disabled")
		self.settingsmenu.add_command(label="Scatter plots normalized", command=self.show_scatter_norm,state="disabled")
		
		self.config(menu=menubar)
			
	def donothing(self):
		pass

		
	def abort(self):
		if self.btn_abort.cget('relief')==tk.RAISED:
			self.btn_abort.config(relief=tk.SUNKEN)
		else:
			self.btn_abort.config(relief=tk.RAISED)
		
	def done(self,x):
		sys.stdout=sys.__stdout__
		self.pool.terminate()		
		self.destroy()
		self.quit()
		
	def on_closing(self):
		if self.right_tabs.preferences.options.save_datasets.value:
			tempstore.save_obj(tempstore.fname_datasets,self.right_tabs.data_tree.datasets)		
		self.data.save()
		exit()			
	
	def get(self):
		return self.process.get()

		
class stdout_redir():
	def __init__(self,textbox):
		self.textbox = textbox

	def write(self,string):
		self.textbox.insert('end', string)
		self.textbox.see('end')
		

class datastore(dict):
	def __init__(self,window):
		d=tempstore.load_obj(tempstore.fname_window)
		if d is None or type(d)!=dict:
			dict.__init__(self)
		else:
			dict.__init__(self,d)
		self.dict_default=dict()
		self.dict_default['sql_str']="SELECT * FROM TABLE <table>"
		self.dict_default['conn_str']=def_conn_str
		self.win=window
		self['current path']=os.getcwd()
		self['current json path']=os.getcwd()

		
	def get(self,key,default=None):
		try:
			v=self[key]
			if v is None:
				self[key]=default
			return self[key]
		except:
			self[key]=default
		return self[key]
		
	def __getitem__(self,key):
		try:
			return dict.__getitem__(self,key)
		except:
			self[key]=self.dict_default[key]
			return dict.__getitem__(self,key)

		
	def save(self):
		self.win.main_tabs._tabs.save_all_in_temp()
		tempstore.save_obj(tempstore.fname_window,dict(self))
		tempstore.save_zip()

				
	
def_conn_str="""conn = pymysql.connect(host='<hostname>', \n
\t\t\tuser='<username>', 
\t\t\tpasswd='<password>', 
\t\t\tdb='<dbname>')	"""