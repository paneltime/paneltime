#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from gui import gui_scrolltext




class main_tabs:
	def __init__(self,window):
		self.win=window
		self.main_tabs = ttk.Notebook(window.frm_left)          # Create Tab Control
		self.main_tabs.bind("<<NotebookTabChanged>>", self.main_tab_pressed)
		self.stat_tab = tk.Frame(self.main_tabs)
		self.main_tabs.add(self.stat_tab, text='regression')      # Add the tab	
		self.stat_tab.rowconfigure(0, weight=1)
		self.stat_tab.columnconfigure(0, weight=1)
		self.main_tabs.grid(column=0,row=0,sticky=tk.NSEW)  # Pack to make visible	
		self.text_boxes=text_boxes()
		self.add_tab = tk.Frame(self.main_tabs)
		self.main_tabs.add(self.add_tab, text='...')
		self.box = gui_scrolltext.ScrollText(self.stat_tab,True)
		self.box.grid(row=0, column=0,sticky=tk.NSEW)
		self.box.insert('1.0','This tab is dedicated to the regression table. Use other tabs for running scripts')
		try:
			for i in window.data.get('editor_data'):
				self.add_editor(i,window.data.get('editor_data')[i]).focus()	
		except:
			self.add_editor('script').focus()		
				
			
	def main_tab_pressed(self,event):
		tab=self.current_editor()
		if tab.title()=="...":
			self.add_editor()
		if tab.title()=='regression':
			self.win.buttons.run_disable()
		else:
			self.win.buttons.run_enable()
			
	def insert_current_editor(self,chars):
		tb=self.current_editor(True)
		tb.write(chars)
		tb.text_box.focus()
		
	def current_editor(self,return_obj=False):
		selection = self.main_tabs.select()
		tab=self.main_tabs.tab(selection, "text")
		if not return_obj:
			return tab
		text=self.text_boxes.name_to_textbox[tab]
		return text
		
	def selected_tab_text(self):
		tb=self.current_editor(True)
		text=tb.get('1.0',tk.END)	
		return text
			
	def add_editor(self,name=None,text=None):
		for i in range(1000):
			if not name is None:
				break
			name=f'script {i+2}'
			if not name in self.text_boxes.name_to_textbox:
				break
		tf= tk.Frame(self.main_tabs)
		tf.rowconfigure(0, weight=1)
		tf.columnconfigure(0, weight=1)			
		self.main_tabs.add(tf, text=name)
		text_box=self.text_boxes.add(name,tf,text)
		self.main_tabs.select(tf)
		self.main_tabs.insert('end',self.add_tab)	
		return text_box
	
	
class text_boxes:
	def __init__(self):
		self.name_to_textbox=dict()
		self.obj_to_textbox=dict()
		self.name_to_textbox=dict()
		self.obj_to_name=dict()
		self.name_to_obj=dict()
	
	def add(self,name,frame,text=None):
		txtbox = gui_scrolltext.ScrollText(frame,text=text)
		txtbox.grid(row=0, column=0,sticky=tk.NSEW)
		self.name_to_textbox[name]=txtbox
		self.obj_to_textbox[frame]=txtbox
		self.obj_to_name[frame]=name
		self.name_to_obj[name]=frame
		return txtbox
		
	def remove(self, name=None,frame=None):
		pass