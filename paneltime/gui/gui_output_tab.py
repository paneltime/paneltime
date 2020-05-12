#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from gui import gui_charts
from gui import gui_scrolltext
from tkinter import ttk
import functions as fu
import output
import os
			
class output_tab(tk.Frame):
	def __init__(self,window,exe_tab,name,tabs,main_tabs,output_data):
		self.window=window
		name=tabs.gen_name(name)
		tk.Frame.__init__(self,main_tabs)
		
		self.widget=tk.Frame(self)
		self.widget.columnconfigure(0,weight=1)
		self.widget.columnconfigure(1)
		self.widget.rowconfigure(0,weight=1)

		
		self.box= gui_scrolltext.ScrollText(self.widget,format_text=False,name='regression',window=window,)
		self.charts = gui_charts.process_charts(window,self.widget,main_tabs,tabs)
		self.progress_bar=bar(self,exe_tab)
		
		self.progress_bar.grid(row=2, sticky=tk.EW)
		
		top_color='#fcf3d9'
		self.tab=tabs.add(self,name=name,top_text='Regression output',top_color=top_color,skip_buttons=True)
		main_tabs.select(self)
		main_tabs.insert('end',tabs.add_tab)			
		
		self.round_digits_lbl=tk.Label(self.tab.button_frame_L,text="Round to:",background=top_color)
		self.add_digits_box(self.tab.button_frame_L)
		self.round_digits_lbl.grid(row=0,column=1)
		self.round_digits.grid(row=0,column=2)
		
		self.button_img=dict()
		self.button_img['upper_info']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','upper_info.png']),master=self.tab.button_frame)
		self.button_upper_info=tk.Button(self.tab.button_frame_L, image = self.button_img['upper_info'],command=self.upper_info, 
									   highlightthickness=0,bd=0,height=22, anchor=tk.E,background=top_color)	
		self.button_upper_info.grid(row=0,column=3)
		self.button_img['statistics_info']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','statistics_info.png']),master=self.tab.button_frame)
		self.button_statistics_info=tk.Button(self.tab.button_frame_L, image = self.button_img['statistics_info'],command=self.statistics_info, 
									   highlightthickness=0,bd=0,height=22, anchor=tk.E,background=top_color)	
		self.button_statistics_info.grid(row=0,column=4)
		self.button_img['maximization_info']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','maximization_info.png']),master=self.tab.button_frame)
		self.button_maximization_info=tk.Button(self.tab.button_frame_L, image = self.button_img['maximization_info'],command=self.maximization_info, 
									   highlightthickness=0,bd=0,height=22, anchor=tk.E,background=top_color)	
		self.button_maximization_info.grid(row=0,column=5)

		
		self.tab.exe_tab=exe_tab
		self.tab.widget=self.box
	
		self.widget.grid(row=1,column=0,sticky=tk.NSEW)
		
		self.box.grid(column=0,row=0,sticky=tk.NSEW)
		self.charts.grid(column=1,row=0,sticky=tk.NS)
		
		if not output_data is None:
			self.table,chart_images=output_data
			self.box.table=self.table#for storing the editor
			self.charts.charts_from_stored(chart_images)
			self.print()			
		
	def maximization_info(self,event):
		pass
	def statistics_info(self,event):
		pass
	def upper_info(self,event):
		pass	
		
	def add_digits_box(self,master):	
		self.round_digits_text=tk.StringVar(master)
		self.round_digits = ttk.Combobox(master,textvariable=self.round_digits_text,  width=2)
		preferences=self.window.right_tabs.preferences.options
		self.round_digits.config(values=['']+list(range(6)))
		self.round_digits_text.set(preferences.n_round.value)
		self.round_digits.bind("<<ComboboxSelected>>",self.print)
		
	def set_output_obj(self,ll, direction,robustcov_lags, its, y_name, ci,incr):
		cols=['count','names', ['args','se_robust', 'sign_codes'],'dx_norm','tstat', 'tsign', 'multicoll','assco','set_to', 'cause']	
		self.table=output.output(ll, direction,robustcov_lags,cols,incr,its).table()
		self.box.table=self.table#for storing the editor
		self.box.chart_images=self.charts.get_images_for_storage()
		self.print()
		
	def print(self,event=None):
		if not hasattr(self,'table'):
			return
		n=int(self.round_digits_text.get())
		s=self.table.table(n)
		tab_stops=self.table.get_tab_stops(self.box.text_box.config()['font'][4])
		self.box.text_box.config(tabs=tab_stops)	
		
		s=self.table.heading+s+self.table.footer+self.table.statistics
		self.box.replace_all(s)	


class bar(tk.Frame):
	def __init__(self,master,exe_tab):
		tk.Frame.__init__(self,master,background='white',height=25)
		self.tab=master
		self.suffix=''
		self.exe_tab=exe_tab
		self.text=tk.StringVar(self)
		self.text_lbl=tk.Label(self,textvariable=self.text,background='white')
		self.progress=tk.Frame(self,background='#9cff9d',height=5,width=0)
		self.text_lbl.grid(row=0,column=0,sticky=tk.W)
		self.progress.grid(row=1,column=0,sticky=tk.W)
		
	def set_progress(self,percent,text):
		total_width=self.winfo_width()
		self.progress.config(width=int(total_width*percent))
		self.progress.grid()
		if len(self.suffix):
			text=self.suffix + ' - ' + text
		self.text.set(text)
		if not self.exe_tab is None:
			return self.exe_tab.isrunning
		else:
			return True
		
		
		
		

	