#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from gui import gui_scrolltext
from gui import gui_output_tab
import functions as fu
import os
from tkinter import filedialog
import types
import numpy as np
from multiprocessing import pool
import paneltime as pt
import traceback
from matplotlib import pyplot  as plt
import tempstore




ADD_EDITOR_NAME='...'
DEFAULT_GREY='#e6e6e6'

class main_tabs(ttk.Notebook):
	def __init__(self,window):
		self.win=window
		ttk.Notebook.__init__(self,window.frm_left) 
		self.last_active_tab=None
		self.output_obj=None
		self.grid(column=0,row=0,sticky=tk.NSEW)  # Pack to make visible	
		self._tabs=tabs(self)
		self.isdeleting=False	
		self.bind("<<NotebookTabChanged>>", self.main_tab_pressed)
			
	def main_tab_pressed(self,event):	
		s=self.current_editor(0)
		if s==ADD_EDITOR_NAME:
			self._tabs.add_editor()
		else:
			self._tabs.selection_change(self.select())
			
	def insert_current_editor(self,chars):
		tb=self.current_editor(2)
		tb.write(chars)
		tb.text_box.focus()
		
	def current_editor(self,return_obj=1):
		selection = self.select()
		if return_obj==1:
			return str(self._tabs[selection].frame)#returns tab object
		elif return_obj==2:
			return self._tabs[selection].widget#returns widget
		return self.tab(selection, "text")

		
	def selected_tab_text(self):
		tb=self.current_editor(2)
		text=tb.get_all()	
		return text
	

		
		
class tab:
	def __init__(self,tabs,frame,widget,name,top_text='',top_color=DEFAULT_GREY,path=None,skip_buttons=False):
		self.frame = frame
		self.position=tabs.count
		self.isrunning=False
		self.tabs=tabs
		self.name=name
		self.path=path	
		self.top_color=top_color
		self.frame.rowconfigure(0)
		self.frame.rowconfigure(1, weight=1)
		self.frame.columnconfigure(0, weight=1)	
		self.notebook=tabs.notebook
		self.locals=dict()
		self.globals={'window':self.notebook.win,'tab':self,'data':self.notebook.win.right_tabs.data_tree.datasets}		
		self.add_buttons(name,top_color,top_text,skip_buttons)
		self.notebook.add(self.frame,text=name)
		self.widget = widget	
		self.locals=dict()
		if not widget is None:
			widget.grid(row=1, column=0,sticky=tk.NSEW)

		
	def add_buttons(self,name,top_color,top_text,skip_buttons):
		size=22
		self.button_frame=tk.Frame(self.frame,height=size,background=top_color)
		self.button_frame_L=tk.Frame(self.button_frame,height=size,background=top_color)
		self.button_frame_R=tk.Frame(self.button_frame,height=size,background=top_color)
		self.top_text=tk.StringVar(value=top_text,master=self.button_frame)
		self.button_frame_label_M=tk.Label(self.button_frame,textvariable=self.top_text,background=top_color)

		self.display_name=tk.StringVar(self.button_frame)
		self.display_name.set(name)
		self.name_box=tk.Entry(self.button_frame_L,width=50,textvariable=self.display_name)
		self.name_box.bind('<KeyRelease>', self.tab_name_edited)
		self.button_img=dict()		
		self.button_img['save']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','save.png']),master=self.button_frame)
		self.button_save=tk.Button(self.button_frame_L, image = self.button_img['save'],command=self.save, 
								   highlightthickness=0,bd=0,height=size, anchor=tk.E,background=top_color)
		

		
		self.button_img['delete']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','delete_small.png']),master=self.button_frame)
		self.button_delete=tk.Button(self.button_frame_R, image = self.button_img['delete'],command=self.delete, 
								   highlightthickness=0,bd=0,height=15, anchor=tk.E,background=top_color)
		
		
		self.button_frame.grid(row=0,sticky=tk.EW)
		self.button_frame.columnconfigure(0,weight=1)
		self.button_frame.columnconfigure(1,weight=5)
		self.button_frame.columnconfigure(2,weight=1)
		self.button_frame_L.grid(row=0,column=0,sticky=tk.W)
		self.button_frame_label_M.grid(row=0,column=1,sticky=tk.EW)
		self.button_frame_R.grid(row=0,column=2,sticky=tk.E)
		
		self.name_box.grid(row=0,column=0)
		self.button_delete.grid(row=0,column=2)
		if skip_buttons:
			return
		self.button_img['run']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','run.png']),master=self.button_frame)
		self.button_run=tk.Button(self.button_frame_L, image = self.button_img['run'],command=self.run, 
								   highlightthickness=0,bd=0,height=size, anchor=tk.E,background=top_color)		
		
		self.button_img['stop']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','stop.png']),master=self.button_frame)
		self.button_stop=tk.Button(self.button_frame_L, image = self.button_img['stop'],command=self.stop, 
								   highlightthickness=0,bd=0,height=size, anchor=tk.E,background=top_color)				
		
		
		self.button_save.grid(row=0,column=1)
		self.button_run.grid(row=0,column=2)
		self.button_stop.grid(row=0,column=3)
		
	def stop(self):
		self.tabs.notebook.win.mc.master.quit()
		self.tabs.notebook.win.mc=None
		if hasattr(self,'exe_tab'):
			self.exe_tab.isrunning=False
			self.exe_tab.pool.close()
		else:
			self.isrunning=False
			self.pool.close()	
		
	
	def run(self):
		win=self.notebook.win
		win.data.save()
		self.pool = pool.ThreadPool(processes=1)
		text=self.widget.get_all()
		self.globals['exe_tab']=self
		self.isrunning=True
		self.process=self.pool.apply_async(self.exec, (text,))

	def exec(self,source):
		try:
			exec(source,self.globals,self.locals)
		except Exception as e:
			print("""
The following error occured in you script:
------------------------------------------
""")
			traceback.print_exc()
	
	def run_enable(self,event):
		pass
		
	def tab_name_edited(self,event=None):
		self.notebook.tab(self.frame,text=self.display_name.get())
		
	def save(self):
		p=self.notebook.win.data['current path']
		filename = filedialog.asksaveasfilename(initialdir=p,title="Save",
			filetypes = (("text", "*.txt"),("Rich Text Format", "*.rtf"),("Python file", "*.py")), defaultextension=True)
		if not filename: 
			return
		self.path=filename
		p,f=os.path.split(filename)
		self.notebook.win.data['current path']=p
		txt=self.widget.get_all()
		file = open(filename,'w')
		file.write(txt)
		file.close()	
	
	def delete(self):
		self.notebook.isdeleting=True
		self.tabs.pop(self.frame)
		
	
	
class tabs(dict):
	def __init__(self,notebook):
		dict.__init__(self)
		self.count=0
		self.names=dict()
		self.notebook=notebook
		self.sel_list=[]
		self.subplot=plt.subplots(1,figsize=(4,2.5),dpi=75)
		self.print_subplot=plt.subplots(1,figsize=(6,3.25),dpi=200)	
		self.add_tab = tk.Frame(self)
		self.add(self.add_tab,None,ADD_EDITOR_NAME)		
		self.load_all_from_temp()
	
	def add(self,frame,widget=None,name='',top_text='',top_color=DEFAULT_GREY,path=None,skip_buttons=False):
		
		tab_item=tab(self,frame, widget, name,top_text,top_color,path,skip_buttons=skip_buttons)
		dict.__setitem__(self,str(frame), tab_item)
		self.names[name]=self[frame]
		self.count+=1
		self.sel_list.append(str(frame))
		return self[frame]
	
	def save_all_in_temp(self):
		win=self.notebook.win
		win.data['editor_data']=dict()
		for i in self:
			t=self[i]
			try:
				text=t.widget.get('1.0',tk.END)
			except:
				text=''
			try:			
				output_data=t.frame.widget.stored_output_data
			except:
				output_data=None
			if not t.name==ADD_EDITOR_NAME:
				win.data['editor_data'][i]=(t.name,text,t.top_text.get(),t.top_color,t.path,output_data)
				
	def load_all_from_temp(self):
		editor_data=self.notebook.win.data.get('editor_data')
		used_imgs=self.get_image_refs(editor_data)
		self.img_tmp=tempstore.temp_image_manager(used_imgs)		
		if editor_data is None:
			return
		n=0
		for i in editor_data:
			try:
				name,text, top_text, top_color,path,output_data=editor_data[i]
				if not (name==ADD_EDITOR_NAME or (name[:6]=='script' and text=='')):
					if output_data is None:
						self.add_editor(name,text,top_text=top_text,top_color=top_color,path=path).frame.focus()
					else:
						self.add_output(name,output_data=output_data)
					n+=1
			except:
				print(f'Could not add tab {i}')
		if n==0:
			self.add_editor('script').frame.focus()	
			
	def get_image_refs(self,editor_data):
		if editor_data is None:
			return []
		used_imgs=[]
		for i in editor_data:
			try:
				name,text, top_text, top_color,path,output_data=editor_data[i]
				if not output_data is None:
					for imgpath,name in output_data.chart_images:
						used_imgs.append(imgpath)
			except:
				pass
		return used_imgs
						

			
	def add_editor(self,name=None,text=None,format_text=True,top_text='',top_color=DEFAULT_GREY,path=None):
		name=self.gen_name(name)
		frame=tk.Frame(self.notebook)
		widget= gui_scrolltext.ScrollText(frame,text=text,format_text=format_text,name=name,window=self.notebook.win)
		tab=self.add(frame,widget,name,top_text,top_color,path)
		frame.tab=tab
		self.notebook.select(frame)
		self.notebook.insert('end',self.add_tab)	
		self.remove_redundant_script(name)
		return tab
	
	def add_output(self,exe_tab=None,name='regression',output_data=None):
		output_tab=gui_output_tab.output_tab(self.notebook.win,exe_tab,name,self,self.notebook,output_data)
		#(added to tabs class in gui_output_tab.output_tab)
		return output_tab
	
	def __getitem__(self,k):
		k=str(k)
		ret=dict.__getitem__(self,k)
		return ret
		
	def __setitem__(self,k,v):
		raise RuntimeError("Can't assign values to the tabs object. Use the add method to create a new tab")
		
	def pop(self, k, d=None):
		if self[k].name==ADD_EDITOR_NAME:
			return
		add_new=False
		if self.count<=2:
			add_new=True
		self.select_valid_tab(k)		
		k=str(k)
		self.sel_list.pop(self.sel_list.index(k))
		try:
			self.names.pop(self[k].name)
		except:
			print(f"Could not delete {self[k].name} from {self.names}")
		self.notebook.forget(k)
		dict.pop(self,k, d)
		self.count-=1
		if add_new:
			self.add_editor('script')
			self.count+=1
		
	def selection_change(self,selection):
		i=self.sel_list.index(selection)
		self.sel_list.append(self.sel_list.pop(i))

	def select_valid_tab(self,current_tab):
		sel_list=self.sel_list
		for i in range(len(sel_list)):
			f=self[sel_list[-1-i]]
			if f.frame!=current_tab and f.name!='...':
				self.notebook.select(sel_list[-1-i])
				break
						
	def gen_name(self,core):
		if core is None:
			core='script'
		if 	not core in self.names:
			return core
		i=0
		n=list(core.split('.'))
		if len(n)==1:
			while True:
				i+=1
				name=f"{core} {i}"
				if 	not name in self.names:
					return name	
		else:
			while True:
				i+=1
				name=f"{'.'.join(n[:-1])} {i}.{n[-1]}"
				if 	not name in self.names:
					return name				
		
	def remove_redundant_script(self,name):
		if name[:6]=='script':
			return
		poptabs=[]
		for i in self:
			try:
				if self[i].name[:6]=='script' and self[i].widget.get_all().replace('\n','')=='':
					poptabs.append(i)
			except:
				pass
		for i in poptabs:
			self.pop(i)
	

		