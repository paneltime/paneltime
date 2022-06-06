#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import time
from paneltimegui import gui_scrolltext
from paneltimegui import gui_script_handling
import options as options_module
import numpy as np
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
		self.option_frame=options_item(self.frames['default'], self.win,self.default_options,'Options')
		self.option_frame.register_validation()


		
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
	o=options_item(f, window,opt_def, 'Preferences')
	return o,main_frame
	
	
		

class options_item(ttk.Treeview):
		
	def __init__(self,frame,window,options,heading):
		self.win=window
		self.options=options
		self.frame_heading=heading
		self.options=options
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
		
	def add_heading(self):
		self.name_frame=tk.Frame(self.main_frame,background='white')
		name_lbl=tk.Label(self.name_frame,text=self.frame_heading,background='white')
		name_lbl.grid()
			
		
	def get_script(self):
		scripts=[]
		search_patterns=[]
		d=self.options.__dict__
		for i in d:
			if hasattr(d[i],'value'):
				v=d[i].value
				if v!=self.default_options.__dict__[i].value:
					dtype=d[i].dtype
					if not type(dtype)==list:
						dtype=[dtype]
					if str in dtype:
						if ('\n' in v):
							v=f'"""{v}"""'
						elif '\'' in v:
							v=f'"{v}"'
						else:
							v=f"'{v}'"
					scripts.append(f'options.{i}.set({v})')
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
			self.insert(f"{cat};{option.name}",i, f"{cat};{option.name};{i}",values=(val,),text=desc, tags=('option',))	
			
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
		desc=option.description
		self.desc=tk.Label(self,text=desc,anchor='nw',justify=tk.LEFT,background='white')
		self.desc.grid(row=0,column=0,sticky='nw')		
		if option.is_inputlist:#
			self.cntrl=tk.Frame(self,background='white')
			for i in range(len(option.descr_for_input_boxes)):
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
			self.scrolltext_updater=update_options_scrolltext(option, self.cntrl, self.option_tree)
			self.cntrl.bind_to_key_release(self.scrolltext_updater.update)
		else:
			self.cntrl=managed_text(self,option.dtype,option,self.option_tree, self.node_name)
			self.cntrl.text.set(option.value)
			self.entries[self.node_name]=self.cntrl
		

	def add_control_multi(self,option,master,i):		
		line=tk.Frame(self.cntrl,background='white')
		name=self.node_name+str(i)
		line.columnconfigure(0,weight=1)
		line.columnconfigure(1,weight=1)
		desc=option.descr_for_input_boxes[i]
		lbl=tk.Label(line,text=desc,anchor='nw',background='white')
		self.entries[name]=managed_text(line,option.dtype,option,self.option_tree,self.node_name,i)
		if option.value is None:
			self.entries[name].text.set('None')
		else:
			self.entries[name].text.set(str(option.value[i]))
		self.entries[name].grid(row=0,column=2,sticky='nw')
		lbl.grid(row=0,column=0,sticky='nw')
		line.grid(row=i,sticky='nw')
			
			
class update_options_scrolltext:
	def __init__(self,option,scrolltext,option_tree):
		self.option=option
		self.scrolltext=scrolltext
		self.option_tree=option_tree
		
	def update(self):
		P=self.scrolltext.get_all()
		ok=self.option.set(P)
		if not ok:
			return False
		return True
		
		
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
		return self.update_options(P)
		
	def update_options(self,P):
		ok=self.option.set(P,self.i)
		if not ok:
			return
		value=displayvalue(self.option.value)
		self.option_tree.item(self.node_name,values=(value,self.option.dtype_str))
		return ok
	
def displayvalue(value):
	if type(value)==str:
		if '\n' in value:
			return ''	
	return value