#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from gui import gui_charts
import time
from gui import gui_scrolltext
import options as options_module
from gui import gui_scrolltext
from gui import gui_script_handling
import numpy as np
NON_NUMERIC_TAG='|~|'
font='Arial 9 '
tags=dict()
tags['option']={'fg':'#025714','bg':'#e6eaf0','font':font+'bold'}
tags['unselected']={'fg':'black','bg':'white','font':font}

class optionset(dict):
	def __init__(self,tabs,window):
		dict.__init__(self)
		#s = ttk.Style()
		#s.configure('new.TFrame', background='white',font=font)			
		self.tabs=tabs
		self.tabs.rowconfigure(0,weight=1)
		self.tabs.columnconfigure(0,weight=1)		
		self.win=window
		self.default_options=options_module.regression_options()
		self.main_frame=tk.Frame(tabs,background='white')	
		self.main_frame.rowconfigure(0,weight=1)
		self.main_frame.columnconfigure(0,weight=1)			
		self.main_frame.grid(row=0,column=0,sticky=tk.NSEW)
		self.frames=dict()
		self.frames['default']=tk.Frame(self.main_frame,background='white')
		self.default_msg=tk.Label(self.frames['default'],text='Please select a data set before editing options',background='white')
		self.frames['default'].grid()

		
		
	def new_option_frame(self,dataset):
		name=dataset.name
		self.frames[name]=tk.Frame(self.main_frame,background='white')
		self[name]=options_item(self.frames[name], self.win,dataset.options,self.default_options,True,dataset)
		#self.frames[name].grid(row=0,column=0,sticky=tk.NSEW)
		return self[name]
		
	def show_options(self,dataset):
		for i in self.frames:
			self.frames[i].grid_remove()
		if dataset.name in self:
			self[dataset.name].gridding()

	
	def delete(self,name):
		try:
			f=self.frames.pop(name)
			f.grid_remove()
			self.pop(name)
		except:
			return
		
def add_preferences_tab(tabs,window):
	tabs.rowconfigure(0,weight=1)
	tabs.columnconfigure(0,weight=1)		
	
	main_frame=tk.Frame(tabs,background='white')	
	main_frame.rowconfigure(0,weight=1)
	main_frame.columnconfigure(0,weight=1)			
	main_frame.grid(row=0,column=0,sticky=tk.NSEW)
	f=tk.Frame(main_frame,background='white')
	opt=options_module.application_preferences()
	opt_def=options_module.application_preferences()
	o=options_item(f, window,opt , opt_def, False)
	return o,main_frame
	
	
		

class options_item(ttk.Treeview):
		
	def __init__(self,frame,window,options,default_options,link_to_script_edit,dataset=None):
		self.win=window
		self.link_to_script_edit=link_to_script_edit
		self.options=options
		self.dataset=dataset
		self.default_options=default_options
		self.main_frame=frame
		self.canvas=tk.Canvas(self.main_frame,background='white')
		self.opt_frame=tk.Frame(self.main_frame,background='white')
		self.add_heading()
		ttk.Treeview.__init__(self,self.canvas)
		self.level__dicts=[dict(),dict(),dict()]
		
		self.yscrollbar = ttk.Scrollbar(self.canvas, orient="vertical", command=self.yview)
		self.configure(yscrollcommand=self.yscrollbar.set)
		
		self.xscrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", command=self.xview)
		self.configure(xscrollcommand=self.xscrollbar.set)
		self.tree_construction()
		self.gridding()
		
		self.binding()
		self.script=''
		
	def add_heading(self):
		self.name_frame=tk.Frame(self.main_frame,background='white')
		if self.dataset is None:
			name_lbl=tk.Label(self.name_frame,text='General preferences',background='white')
			name_lbl.grid()
		else:
			self.name=tk.StringVar(value=self.dataset.name)
			name_lbl1=tk.Label(self.name_frame,text='Options for:',background='white')
			name_lbl2=tk.Label(self.name_frame,textvariable=self.name,background='white',font='Arial 12 bold')	
			name_lbl1.grid(row=0,column=0)
			name_lbl2.grid(row=0,column=1)			
		
	def get_script(self):
		scripts=[]
		search_patterns=[]
		d=self.options.__dict__
		for i in d:
			if hasattr(d[i],'value'):
				if d[i].value!=self.default_options.__dict__[i].value:
					scripts.append(f'options.{i}.set({d[i].value})')
					search_patterns.append(fr'options.{i}.set\(([\s\S]*?)\)')
		return scripts,search_patterns
				
		
	def binding(self):
		self.bind('<Double-Button-1>',self.tree_double_click)	
		self.bind('<<TreeviewSelect>>',self.tree_click)	
		self.bind('<Key>',self.key_down)	
		self.bind('<KeyRelease>',self.key_up)		
		
	def tree_construction(self):
		self["columns"]=("one","two")
		self.column("#0", stretch=tk.YES)
		self.column("one", width=50,stretch=tk.YES)
		self.column("two", width=50,stretch=tk.YES)
		self.heading("#0",text="Option",anchor=tk.W)
		self.heading("one", text="value",anchor=tk.W)
		self.heading("two", text="type",anchor=tk.W)	
		self.alt_time=time.perf_counter()
		for k in tags:
			tag_configure(self,k,tags[k])	
		self.tree=dict()	
		self.add_options_to_tree()
		a=0
		
	def gridding(self):
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)
		xscrollbar,yscrollbar=self.xscrollbar,self.yscrollbar
		
		self.main_frame.rowconfigure(0)
		self.main_frame.rowconfigure(1,weight=7)
		self.main_frame.rowconfigure(2,weight=5)
		self.main_frame.columnconfigure(0,weight=1)
		self.canvas.rowconfigure(0,weight=1)
		self.canvas.columnconfigure(0,weight=1)		
		self.opt_frame.rowconfigure(0,weight=1)
		self.opt_frame.columnconfigure(0,weight=1)				
		
		self.main_frame.grid(row=0,column=0,sticky=tk.NSEW)
		self.name_frame.grid(row=0,column=0,sticky=tk.W)
		self.opt_frame.grid(row=2,column=0,sticky='nw')			
		self.canvas.grid(row=1,column=0,sticky=tk.NSEW)	
		xscrollbar.grid(row=1,column=0,sticky='ew')
		yscrollbar.grid(row=0,column=1,sticky='ns')			
		self.grid(row=0,column=0,sticky=tk.NSEW)
		
		
	def key_down(self,event):
		if event.keysym=='Alt_L' or  event.keysym=='Alt_R':
			self.configure(cursor='target')
			self.alt_time=time.perf_counter()
			
			
	def key_up(self,event):
		if event.keysym=='Alt_L' or  event.keysym=='Alt_R':
			self.configure(cursor='arrow')

		
	def tree_double_click(self,event):
		item = self.selection()[0]
		item=self.item(item)['text']
		self.win.main_tabs.insert_current_editor(item)
		
	def tree_click(self,event):
		item = self.selection()
		if len(item)==0:
			return
		item=item[0]
		levels=item.split(';')
		if len(levels)==3:
			parent_itm=';'.join(levels[:-1])
			fname,j,k=levels
			value,vtype=self.item(parent_itm)['values']
			self.item(parent_itm,values=(k,vtype))
			self.item(parent_itm,open=False)
			self.tree[fname][j].option.set(k)
		elif levels[1]!='':#not top level:
			i,j=levels
			if self.item(item)['open']:
				self.item(item,open=False)
			else:
				self.item(item,open=True)
			self.hide_all_frames()
			self.tree[i][j].grid(row=1,column=0)
		if self.link_to_script_edit:
			gui_script_handling.edit_options_script(self)						
			
	def close_all(self):
		for i in self.tree:
			for j in self.tree[i]:
				self.item(j,open=False)	
		
	def add_options_to_tree(self):
		for i in self.options.categories_srtd:
			self.insert('', 1,f"{i};", text=i)
			self.add_node(i,self.options.categories[i])
			self.item(f"{i};",open=True)
		a=0
		
	def hide_all_frames(self):
		for i in self.tree:
			for j in self.tree[i]:
				self.tree[i][j].grid_remove()

	def add_node(self,cat,options):
		d=dict()
		self.tree[cat]=d
		for j in options:
			value=displayvalue(j.value)
			self.insert(f"{cat};",2, f"{cat};{j.name}", text=j.name,values=(value,j.dtype_str))	
			d[j.name]=option_frame(self.opt_frame, j,self,f"{cat};{j.name}")
			self.add_options(j, cat)		

	def add_options(self,option,cat):
		if not option.selection_var:
			return
		for i in range(len(option.permissible_values)):
			desc=''
			if not option.value_description is None:
				desc=option.value_description[i]
			val= option.permissible_values[i]
			self.insert(f"{cat};{option.name}",i, f"{cat};{option.name};{i}",values=(val,), text=desc,tags=('option',))	
			
	def register_validation(self):
		for i in self.tree:
			for j in self.tree[i]:
				self.tree[i][j].register_validation()
			

def tag_configure(tree,name,d,value=None):
	
	tree.tag_configure(name, foreground=d['fg'])
	tree.tag_configure(name, background=d['bg'])
	tree.tag_configure(name, font=d['font'])	
	if not value is None:
		tree.item(name,value=value)
		
class option_frame(tk.Frame):
	def __init__(self, master, option,option_tree,node_name):
		tk.Frame.__init__(self,master,background='white')
		self.entries=dict()
		self.option_tree=option_tree
		self.node_name=node_name
		self.option=option
		self.lines=dict()
		desc=option.descr_for_vector_setting
		if not type(option.description)==list:
			desc+=option.description
		self.desc=tk.Label(self,text=desc,anchor='nw',justify=tk.LEFT,background='white')
		self.desc.grid(row=0,column=0,sticky='nw')		
		if option.is_inputlist:#
			self.cntrl=tk.Frame(self,background='white')
			for i in range(len(option.description)):
				self.add_control_multi(option,self.cntrl,i)
			self.cntrl.grid(row=1,column=0,sticky='nw')
		elif not option.selection_var:
			self.add_control_single(option)
			self.cntrl.grid(row=1,column=0,sticky='nw')
		
	def register_validation(self):
		for i in self.entries:
			self.entries[i].register_validation()
		
			
			
	def add_control_single(self,option):		
		if option.dtype==str:
			self.cntrl=gui_scrolltext.ScrollText(self)
			if not option.value is None:
				self.cntrl.insert('1.0',option.value)
		else:
			self.cntrl=managed_text(self,option.dtype,option,self.option_tree, self.node_name)
			self.cntrl.text.set(option.value)
			self.entries[self.node_name]=self.cntrl
		

	def add_control_multi(self,option,master,i):		
		line=tk.Frame(self.cntrl,background='white')
		name=self.node_name+str(i)
		line.columnconfigure(0,weight=1)
		line.columnconfigure(1,weight=1)
		desc=option.description[i]
		lbl=tk.Label(line,text=desc,anchor='nw',background='white')
		self.entries[name]=managed_text(line,option.dtype,option,self.option_tree,self.node_name,i)
		self.entries[name].text.set(str(option.value[i]))
		self.entries[name].grid(row=0,column=2,sticky='nw')
		lbl.grid(row=0,column=0,sticky='nw')
		line.grid(row=i,sticky='nw')
			
			
		
		
class managed_text(tk.Entry):
	def __init__(self, master,dtype,option,option_tree,node_name,i=None):
		self.text=tk.StringVar(master)
		tk.Entry.__init__(self,master,textvariable=self.text,validate="key")
		self.option_tree=option_tree
		self.node_name=node_name
		self.dtype=dtype
		self.option=option
		self.i=i
		self.master=master
		
	def register_validation(self):
		vcmd = (self.master.register(self.onValidate),'%d','%P')	
		self.configure(validatecommand=vcmd)
		self.bind('<Return>', self.onEnterKey)
		
	def onEnterKey(self,event):
		self.option_tree.opt_frame.focus()
			
	def onValidate(self,d,P):
		try:
			if P=='':
				return True
			elif P=='None':
				P=None
			dtype=self.option.dtype
			if not(type(dtype)==list or type(dtype)==tuple):
				dtype=[dtype]
			if int in dtype:
				P=int(P)
			elif float in dtype:
				P=float(P)
		except:
			return False
		ok=self.option.set(P,self.i)
		if not ok:
			return
		value=displayvalue(self.option.value)
		self.option_tree.item(self.node_name,values=(value,self.option.dtype_str))
		if self.option_tree.link_to_script_edit:
			try:
				gui_script_handling.edit_options_script(self.option_tree)
			except:
				pass
		return ok
	
def displayvalue(value):
	if type(value)==str:
		if '\n' in value:
			return ''	
	return value