#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from gui import gui_charts
from gui import gui_scrolltext
			
class output_tab(tk.Frame):
	def __init__(self,window):
		main_tabs=window.main_tabs
		name=main_tabs.gen_name('regression')
		tk.Frame.__init__(self,main_tabs)
		self.widget=tk.Frame(self)
		self.widget.columnconfigure(0,weight=1)
		self.widget.columnconfigure(1)
		self.widget.rowconfigure(0,weight=1)

		
		self.box= gui_scrolltext.ScrollText(self.widget,format_text=False,name='regression',window=window)
		self.charts = gui_charts.process_charts(window,self.widget)
		self.progress_bar=bar(self)
		
		self.progress_bar.grid(row=2, sticky=tk.EW)
		

		self.tab=main_tabs._tabs.add(self,name=name,top_text='Regression output',top_color='#fcf3d9')

	
		self.widget.grid(row=1,column=0,sticky=tk.NSEW)
		
		self.box.grid(column=0,row=0,sticky=tk.NSEW)
		self.charts.grid(column=1,row=0,sticky=tk.NS)			
		
		
		main_tabs.select(self)
		main_tabs.insert('end',main_tabs.add_tab)			
		self.isrunning=True
	

class bar(tk.Frame):
	def __init__(self,master):
		tk.Frame.__init__(self,master,background='white',height=25)
		self.tab=master
		self.text=tk.StringVar(self)
		self.text_lbl=tk.Label(self,textvariable=self.text,background='white')
		self.progress=tk.Frame(self,background='#9cff9d',height=5,width=0)
		self.text_lbl.grid(row=0,column=0,sticky=tk.W)
		self.progress.grid(row=1,column=0,sticky=tk.W)
		
	def set_progress(self,percent,text):
		total_width=self.winfo_width()
		self.progress.config(width=int(total_width*percent))
		self.progress.grid()
		self.text.set(text)
		return self.tab.isrunning