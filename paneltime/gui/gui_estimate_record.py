#!/usr/bin/env python
# -*- coding: utf-8 -*-

#NOT FINISHED!! JUST A COPY OF gui_options
import tkinter as tk
from tkinter import ttk
import time
import estimates as estimates_module
import numpy as np

font='Arial 9 '
tags=dict()
tags['option']={'fg':'#025714','bg':'#e6eaf0','font':font+'bold'}
tags['unselected']={'fg':'black','bg':'white','font':font}

class estimate_set(dict):
	def __init__(self,tabs,window):
		dict.__init__(self)
		#s = ttk.Style()
		#s.configure('new.TFrame', background='white',font=font)			
		self.tabs=tabs
		self.tabs.rowconfigure(0,weight=1)
		self.tabs.columnconfigure(0,weight=1)		
		self.win=window
		self.main_frame=tk.Frame(tabs,background='white')	
		self.main_frame.rowconfigure(0,weight=1)
		self.main_frame.columnconfigure(0,weight=1)			
		self.main_frame.grid(row=0,column=0,sticky=tk.NSEW)
		self.frames=dict()
		self.frames['default']=tk.Frame(self.main_frame,background='white')
		self.default_msg=tk.Label(self.frames['default'],text='Please select a data set before editing estimates',background='white')
		self.frames['default'].grid()

		
		
	def new_estimate_frame(self,dataset):
		name=dataset.name
		self.frames[name]=tk.Frame(self.main_frame,background='white')
		self[name]=estimates_item(self.frames[name], self.win,dataset.estimates,True,dataset)
		#self.frames[name].grid(row=0,column=0,sticky=tk.NSEW)
		return self[name]
		
	def show_estimates(self,dataset):
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
	
class estimates_item(ttk.Treeview):
		
	def __init__(self,frame,window,estimates,dataset=None):
		self.win=window
		self.estimates=estimates
		self.dataset=dataset
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
			name_lbl1=tk.Label(self.name_frame,text='estimates for:',background='white')
			name_lbl2=tk.Label(self.name_frame,textvariable=self.name,background='white',font='Arial 12 bold')	
			name_lbl1.grid(row=0,column=0)
			name_lbl2.grid(row=0,column=1)			
				
		
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
		self.heading("#0",text="estimate",anchor=tk.W)
		self.heading("one", text="LL",anchor=tk.W)
		self.heading("two", text="type",anchor=tk.W)	
		for k in tags:
			tag_configure(self,k,tags[k])	
		self.tree=dict()	
		self.add_estimates_to_tree()
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
			
			
	def key_up(self,event):
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
			self.tree[fname][j].estimate.set(k)
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
		
	def add_estimates_to_tree(self):
		for i in self.estimates.categories_srtd:
			self.insert('', 1,f"{i};", text=i)
			self.add_node(i,self.estimates.categories[i])
			self.item(f"{i};",open=True)
		a=0
		
	def hide_all_frames(self):
		for i in self.tree:
			for j in self.tree[i]:
				self.tree[i][j].grid_remove()

	def add_node(self,cat,estimates):
		d=dict()
		self.tree[cat]=d
		for j in estimates:
			value=displayvalue(j.value)
			self.insert(f"{cat};",2, f"{cat};{j.name}", text=j.name,values=(value,j.dtype_str))	
			d[j.name]=estimate_frame(self.opt_frame, j,self,f"{cat};{j.name}")
			self.add_estimates(j, cat)		

	def add_estimates(self,estimate,cat):
		if not estimate.selection_var:
			return
		for i in range(len(estimate.permissible_values)):
			val= estimate.permissible_values[i]
			self.insert(f"{cat};{estimate.name}",i, f"{cat};{estimate.name};{i}",values=(val,),tags=('estimate',))	
			
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
		
class estimate_frame(tk.Frame):
	def __init__(self, master, estimate,estimate_tree,node_name):
		tk.Frame.__init__(self,master,background='white')
		self.entries=dict()
		self.estimate_tree=estimate_tree
		self.node_name=node_name
		self.estimate=estimate
		self.lines=dict()
		desc=estimate.descr_for_vector_setting
		if not type(estimate.description)==list:
			desc+=estimate.description
		self.desc=tk.Label(self,text=desc,anchor='nw',justify=tk.LEFT,background='white')
		self.desc.grid(row=0,column=0,sticky='nw')		
		if estimate.is_inputlist:#
			self.cntrl=tk.Frame(self,background='white')
			for i in range(len(estimate.description)):
				self.add_control_multi(estimate,self.cntrl,i)
			self.cntrl.grid(row=1,column=0,sticky='nw')
		elif not estimate.selection_var:
			self.add_control_single(estimate)
			self.cntrl.grid(row=1,column=0,sticky='nw')
		
	def register_validation(self):
		for i in self.entries:
			self.entries[i].register_validation()
		
			
			
	def add_control_single(self,estimate):		
		if estimate.dtype==str:
			self.cntrl=gui_scrolltext.ScrollText(self)
			if not estimate.value is None:
				self.cntrl.insert('1.0',estimate.value)
		else:
			self.cntrl=managed_text(self,estimate.dtype,estimate,self.estimate_tree, self.node_name)
			self.cntrl.text.set(estimate.value)
			self.entries[self.node_name]=self.cntrl
		

	def add_control_multi(self,estimate,master,i):		
		line=tk.Frame(self.cntrl,background='white')
		name=self.node_name+str(i)
		line.columnconfigure(0,weight=1)
		line.columnconfigure(1,weight=1)
		desc=estimate.description[i]
		lbl=tk.Label(line,text=desc,anchor='nw',background='white')
		self.entries[name]=managed_text(line,estimate.dtype,estimate,self.estimate_tree,self.node_name,i)
		self.entries[name].text.set(str(estimate.value[i]))
		self.entries[name].grid(row=0,column=2,sticky='nw')
		lbl.grid(row=0,column=0,sticky='nw')
		line.grid(row=i,sticky='nw')
			
			
		
		
class managed_text(tk.Entry):
	def __init__(self, master,dtype,estimate,estimate_tree,node_name,i=None):
		self.text=tk.StringVar(master)
		tk.Entry.__init__(self,master,textvariable=self.text,validate="key")
		self.estimate_tree=estimate_tree
		self.node_name=node_name
		self.dtype=dtype
		self.estimate=estimate
		self.i=i
		self.master=master
		
	def register_validation(self):
		vcmd = (self.master.register(self.onValidate),'%d','%P')	
		self.configure(validatecommand=vcmd)
		self.bind('<Return>', self.onEnterKey)
		
	def onEnterKey(self,event):
		self.estimate_tree.opt_frame.focus()

			
	def onValidate(self,d,P):
		try:
			if P=='':
				return True
			elif P=='None':
				P=None
			dtype=self.estimate.dtype
			if not(type(dtype)==list or type(dtype)==tuple):
				dtype=[dtype]
			if int in dtype:
				P=int(P)
			elif float in dtype:
				P=float(P)
		except:
			return False
		ok=self.estimate.set(P,self.i)
		if not ok:
			return
		value=displayvalue(self.estimate.value)
		self.estimate_tree.item(self.node_name,values=(value,self.estimate.dtype_str))
		if self.estimate_tree.link_to_script_edit:
			try:
				gui_script_handling.edit_estimates_script(self.estimate_tree)
			except:
				pass
		return ok
	
def displayvalue(value):
	if type(value)==str:
		if '\n' in value:
			return ''	
	return value