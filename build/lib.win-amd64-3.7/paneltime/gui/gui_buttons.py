#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
import os
import traceback
from multiprocessing import pool
from gui import gui_sql

class buttons:
	def __init__(self,win):
		self.win=win
		dirname=os.path.dirname(__file__)
		self.isevaluating=False
		imgs=[('run.png',self.run),
			  ('stop.png',self.stop),
			  ('statistics.png',self.statistics),
			  ('open_data.png',self.open),
			  ('open_data_sql.png',self.open_sql),
			  ('save.png',self.save),
			  ('save_as.png',self.save_as)]
		self.buttons={}
		for i in range(len(imgs)):
			name,action=imgs[i]
			short_name=name.replace('.png','')
			img = tk.PhotoImage(file =  os.path.join(dirname,'img',name),name=short_name,master=win.button_pane)
			b=tk.Button(win.button_pane, image = img,command=action, highlightthickness=0,bd=0,width=40)
			b.grid(row=0,column=i,sticky=tk.W)
			self.buttons[short_name]=[img,b,True]
		imgs=[['run_disabled.png',self.buttons['run'][1]],
			  ['stop_disabled.png',self.buttons['stop'][1]],
			  ['statistics_disabled.png',self.buttons['statistics'][1]]
			  ]
		for name,btn in imgs:#adding disabled-buttons
			img=tk.PhotoImage(file =  os.path.join(dirname,'img',name),name=name.replace('.png',''),master=win.button_pane)
			self.buttons[name.replace('.png','')]=[img,btn,False]
		self.gui_sql=None
	
	def run(self):
		self.win.data.save()
		if not self.buttons['run'][2] and False:
			return
		try:
			text=self.win.main_tabs.selected_tab_text()
			text.replace('start()','')
		except:
			return
		self.run_disable()
		self.pool = pool.ThreadPool(processes=1)
		self.process=self.pool.apply_async(self.win.exec, (text,),callback=self.run_enable)
		self.win.right_tabs.tabs.select(self.win.right_tabs.chart_tab)
		
	
	def stop(self):
		self.pool.close()
		
	def statistics(self):
		pass
	
	def open(self):
		p=self.win.data['current path']
		filename = filedialog.askopenfilename(initialdir=p,title="Open data file",
			filetypes = (("CSV", "*.csv")
	        ,("text", "*.txt")
			,("All files", "*.*") ))
		if not filename: 
			return
		p,f=os.path.split(filename)
		self.win.data['current path']=p
		exe_str=f"load('{filename}')"
		df=self.win.eval(exe_str)
		tree=self.win.right_tabs.data_tree
		tree.datasets.add(tree,f,df,filename,'df='+exe_str,self.win.main_tabs)

	

	
	def save(self):
		pass
	
	def save_as(self):
		pass
	
	def run_disable(self):
		self.new_image('run_disabled')
		self.new_image('stop')	
	
	def run_enable(self,events=None):
		self.new_image('run')
		self.new_image('stop_disabled')
		
	def new_image(self,img_name):
		img,buttn,enabled=self.buttons[img_name]
		self.buttons[img_name.replace('_disabled','')][2]=img_name[-9:]=='_disabled'
		buttn.configure(image=img)

		
		