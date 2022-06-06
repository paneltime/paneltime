#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from paneltimegui import gui_charts
from paneltimegui import gui_data_objects
from paneltimegui import gui_options

	
class right_tab_widget:
	def __init__(self,window):
		self.win=window
		frm=tk.Frame(window.frm_right)
		self.sel_data=tk.StringVar(frm)
		lbl_desc=tk.Label(frm,text='Selected dataset:',background='white',font='Arial 9')
		lbl=tk.Label(frm,textvariable=self.sel_data,background='white',font='Arial 9 bold')
		
		self.tabs = ttk.Notebook(window.frm_right)          # Create Tab Control	
		
		self.optionset=gui_options.optionset(self.tabs,window)
		self.data_tree=gui_data_objects.data_objects(self.tabs,window,self)
		self.preferences,pref_frame=gui_options.add_preferences_tab(self.tabs,window)
		
		self.tabs.add(self.data_tree.main_frame, text='datasets')      # Add the tab
		self.tabs.add(self.optionset.main_frame, text='options')      # Add the tab	
		self.tabs.add(pref_frame, text='preferences')      # Add the tab		
		
		lbl_desc.grid(row=0,column=0)
		lbl.grid(row=0,column=1)
		frm.grid(row=0,column=0,sticky=tk.NW)
		self.tabs.grid(row=1,column=0,sticky=tk.NSEW)  # Pack to make visible	
		#self.tabs.bind("<<NotebookTabChanged>>", self.main_tab_pressed)
		self.sql_script=''
		
		
		
	def main_tabs_pressed(self,event):
		pass
		
		
		
		
