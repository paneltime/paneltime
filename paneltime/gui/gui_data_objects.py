#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from paneltimegui import gui_charts
import functions as fu
import time
from paneltimegui import gui_scrolltext
from paneltimegui import gui_script_handling
from paneltimegui import gui_tooltip
from paneltimegui import gui_sql
import options as options_module
from tkinter import filedialog
import json
import os
import tempstore
import time
import paneltime as pt
import numpy as np
import pandas as pd

font='Arial 9 '
tags=dict()
tags['dependent']={'fg':'#025714','bg':'#e6eaf0','font':font+'bold','short':'Y'}
tags['independent']={'fg':'#053480','bg':'#e6eaf0','font':font+'bold','short':'X'}
tags['time variable']={'fg':'#690580','bg':'#e6eaf0','font':font+'bold','short':'T'}
tags['id variable']={'fg':'#910101','bg':'#e6eaf0','font':font+'bold','short':'ID'}
tags['het.sc._factors']={'fg':'#029ea3','bg':'#e6eaf0','font':font+'bold','short':'HF'}
selected={'bg':'#ebfbfc'}
unselected={'fg':'black','bg':'white','font':font,'short':''}
#for correct sorting:
tags_list=['dependent','independent','time variable','id variable','het.sc._factors']

	
class data_objects(ttk.Treeview):
	def __init__(self,tabs,window,right_tabs):
		s = ttk.Style()
		s.configure('new.TFrame', background='white',font=font)	
		
		self.tabs=tabs
		self.clicked=False
		self.right_tabs=right_tabs
		self.win=window
		self.main_frame=tk.Frame(tabs)
		self.add_buttons()
		self.canvas=tk.Canvas(self.main_frame)
		ttk.Treeview.__init__(self,self.canvas,style='new.TFrame')
		self.dbase_img=self._img = tk.PhotoImage(file= fu.join(os.path.dirname(__file__),['img','database.png']),master=self)
		self.var_img=self._img = tk.PhotoImage(file= fu.join(os.path.dirname(__file__),['img','variable.png']),master=self)		
		self.level__dicts=[dict(),dict(),dict()]
		
		yscrollbar = ttk.Scrollbar(self.canvas, orient="vertical", command=self.yview)
		self.configure(yscrollcommand=yscrollbar.set)
		
		xscrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", command=self.xview)
		self.configure(xscrollcommand=xscrollbar.set)
		
		self.gridding(xscrollbar,yscrollbar)
		self.tree_construction()
		self.binding()
			

		self.load()
		self.script=''
		self.expand_time=time.perf_counter()
		self.gui_sql=None
		self.import_buttons=None

		
		
	def add_buttons(self):
		bgcolor='white'
		self.button_frame=tk.Frame(self.main_frame,height=22,background=bgcolor)
		self.button_img=dict()
		
		self.button_add=self.add_button(self.button_frame,'add.png',self.add,'Opens a new dataset',bgcolor)
		self.button_save=self.add_button(self.button_frame,'save_as.png',self.save,'Save current dataset as a csv file',bgcolor)
		self.button_delete=self.add_button(self.button_frame,'delete.png',self.delete_dataset,'Remove current dataset',bgcolor)
		self.button_edit_vars=self.add_button(self.button_frame,'edit_variables.png',self.edit_variables,'Edit variables in current data set',bgcolor)
		self.button_import_script=self.add_button(self.button_frame,'import_script.png',self.insert_load_script,'Display import script',bgcolor)
		
	def insert_load_script(self):
		dataset=self.get_selected_dataset()
		dataset.get_script_editor(self.win.main_tabs)

		
	def add_button(self,master,img,command,tooltip,bgcolor,size=22):
		self.button_img[img]= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img',img]),master=self.button_frame)
		btn=tk.Button(master, image = self.button_img[img],command=command, 
								   highlightthickness=0,bd=0,height=size,width=size,compound='left',background=bgcolor)
		gui_tooltip.CreateToolTip(btn,tooltip)
		return btn	

	def add(self):
		if self.import_buttons is None:
			self.add_import_buttons()
			return
		if self.import_buttons.state()=='normal':
			self.import_buttons.withdraw()
		else:
			self.import_buttons.deiconify()
		
	def add_import_buttons(self):
		bgcolor='white'
		x = y = 0
		x, y, cx, cy = self.button_add.bbox("insert")
		x += self.button_add.winfo_rootx() + 25
		y += self.button_add.winfo_rooty() + 25
		# creates a toplevel window
		self.import_buttons = tk.Toplevel(self.button_add)
		# Leaves only the label and removes the app window
		self.import_buttons.wm_overrideredirect(True)
		self.import_buttons.geometry('+%d+%d' % (x, y))
		frm = tk.Frame(self.import_buttons, background="#ffffff", relief='flat')
		self.button_add_sql=self.add_button(frm,'Table_sql.png',self.open_sql,'Add dataset from an SQL database',bgcolor,40)
		self.button_add_file=self.add_button(frm,'Table_File.png',self.open_file,'Open a data file',bgcolor, 40)
		self.button_add_json=self.add_button(frm,'Table_JSON.png',self.open_json,'Open a JSON file',bgcolor,40)
		self.button_add_copy=self.add_button(frm,'Table_copy.png',self.copy,'Copy current data set for running system regression',bgcolor,40)
		

		self.button_add_sql.grid(row=0,column=0,padx=10,pady=10)
		self.button_add_file.grid(row=1,column=0,padx=10,pady=10)
		self.button_add_json.grid(row=0,column=1,padx=10,pady=10)
		self.button_add_copy.grid(row=1,column=1,padx=10,pady=10)
		frm.grid()
		

	def open_json(self):
		p=self.win.data['current json path']
		filename = filedialog.askopenfilename(initialdir=p,title="Open data file",
			filetypes = (("JSON", "*.json"),) )
		if not filename: 
			return
		p,f=os.path.split(filename)
		self.win.data['current json path']=p		
		exe_str=f"from paneltime import *\nload_json('{filename}')"
		
		exe_str=f"""from paneltime import *\n
data=dict()\ndata['{f}']=load_json('{filename}')"""
		res=self.win.exec(exe_str)
		if res==False:
			return		
		df=self.win.locals['data'][f]
		tree=self.right_tabs.data_tree
		tree.datasets.add(tree,f,df,filename,exe_str,self.win.main_tabs)
	
	def copy(self):
		pass	
	
	def open_file(self):
		p=self.win.data['current path']
		filename = filedialog.askopenfilename(initialdir=p,title="Open data file",
			filetypes = (("CSV", "*.csv")
	        ,("text", "*.txt")
			,("All files", "*.*") ))
		if not filename: 
			return
		p,f=os.path.split(filename)
		self.win.data['current path']=p
		exe_str=f"""from paneltime import *\n
data=dict()\ndata['{f}']=load('{filename}')"""
		res=self.win.exec(exe_str)
		if res==False:
			return
		df=self.win.locals['data'][f]
		tree=self.right_tabs.data_tree
		tree.datasets.add(tree,f,df,filename,exe_str)
		
	def open_sql(self):
		if self.gui_sql is None:
			self.gui_sql=gui_sql.sql_query(self.win,self)
		else:
			self.gui_sql.show()
			
	def edit_variables(self):
		dataset=self.get_selected_dataset()
		dataset.get_data_editor(self.main_tabs)
		
	def load(self):
		self.datasets=tempstore.load_obj(tempstore.fname_datasets)
		if self.datasets is None:
			self.datasets=datasets()
			return
		self.datasets.make_trees(self)
		s=self.win.data.get('selected data')
		try:
			self.focus_set()
			self.focus(s)
		except:
			self.win.data['selected data']=';'
			return
			
	def save(self):
		p=self.win.data['current path']
		filename = filedialog.asksaveasfilename(initialdir=p,title="Save",
			filetypes = (("CSV", "*.csv"),), defaultextension=True)
		if not filename: 
			return
		self.path=filename
		p,f=os.path.split(filename)
		self.win.data['current path']=p
		dataset=self.get_selected_dataset()
		X=[]
		for i in dataset:
			if type(dataset[i])==np.ndarray:
				if len(dataset[i].shape)==2:
					X.append(np.concatenate(([[i]],dataset[i])))
		if len(X)==0:
			print('No data in dataset')
			return
		X=np.concatenate(X,1)
		np.savetxt(filename, X,fmt='%s',delimiter=',')
		print('dataset saved')
	
	
	def delete_dataset(self):
		df=self.get_selected_dataset()
		if df is None:
			return
		self.datasets.delete(df.name,self)
		
	def binding(self):
		self.bind('<Double-Button-1>',self.tree_double_click)	
		self.bind('<<TreeviewSelect>>',self.tree_select)	
		self.bind('<Button-1>',self.tree_click)	
		self.bind('<Key>',self.key_down)	
		self.bind('<KeyRelease>',self.key_up)		
		
	def tree_construction(self):
		self["columns"]=("one","two")
		self.column("#0", stretch=tk.YES)
		self.column("one", width=15,stretch=tk.YES)
		self.column("two", width=75,stretch=tk.YES)
		self.heading("#0",text="Name",anchor=tk.W)
		self.heading("one", text="",anchor=tk.W)
		self.heading("two", text="type",anchor=tk.W)	
		self.alt_time=time.perf_counter()
		for k in tags_list:
			self.tag_configure_node(k,tags[k],False)		
		
	def gridding(self,xscrollbar,yscrollbar):
		self.main_frame.rowconfigure(0)
		self.main_frame.rowconfigure(1,weight=1,uniform="fred")	
		self.main_frame.columnconfigure(0,weight=1)
		
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)
		self.tabs.rowconfigure(0,weight=1)
		self.tabs.columnconfigure(0,weight=1)
		
		
		self.button_frame.grid(row=0,sticky=tk.EW)	
		
		pad=8
		self.button_add.grid(row=0,column=0,padx=pad,pady=pad)
		self.button_delete.grid(row=0,column=1,padx=pad,pady=pad)
		self.button_edit_vars.grid(row=0,column=2,padx=pad,pady=pad)
		self.button_save.grid(row=0,column=3,padx=pad,pady=pad)
		self.button_import_script.grid(row=0,column=4,padx=pad,pady=pad)
		
		
		self.canvas.rowconfigure(0,weight=1)
		self.canvas.columnconfigure(0,weight=1)		
		self.main_frame.grid(row=0,column=0,sticky=tk.NSEW)
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
		levels=item.split(';')
		if levels[1]=='' or len(levels)==3:
			return
		ds=self.datasets[levels[0]]
		txt=self.item(item)['text']
		editor=ds.get_data_editor(self.win.main_tabs)
		editor.widget.write(txt)
		editor.frame.focus()
		
	def tree_click(self,event=None):
		item=self.selected_item()
		self.clicked=True
		
	def tree_select(self,event=None):
		
		item=self.selected_item()
		dname,vname,sel=(item.split(';')+[''])[:3]
		if (dname,vname,sel)==('', '', ''):
			return
		self.win.data['selected data']=dname+';'
		if vname!='':#not top level
			if sel!='':
				self.var_defined(dname,vname,sel)
			else:
				self.var_clicked(item)
			gui_script_handling.edit_exe_script(self.win,self.datasets[dname])
		else:
			self.expand_collaps_parent(item)
		self.select_selected(dname+';')
		self.clicked=False
		self.right_tabs.sel_data.set(dname)
		
	def expand_collaps_parent(self,item):
		if not self.clicked:
			return
		if not item.split(';')[1]=='':
			return
		if self.item(item)['open']:
			self.item(item,open=False)
		else:
			self.item(item,open=True)	
		

	def select_selected(self,parent):
		for i in self.get_children():
			self.tag_configure(i, background=unselected['bg'])
			for j in self.get_children(i):
				t=self.tag_configure(j)
				if t['background']==selected['bg']:
					self.tag_configure(j, background=unselected['bg'])
		self.tag_configure(parent, background=selected['bg'])
		for i in self.get_children(parent):
			t=self.tag_configure(i)
			if t['background']==unselected['bg'] or  t['background']=='':	
				self.tag_configure(i, background=selected['bg'])

					
	def var_clicked(self,item):
		item_obj=self.item(item)
		short,vtype=item_obj['values']
		t=self.tag_configure(item)	
		if item_obj['open']:
			if t['font']!=unselected['font']:
				self.tag_configure_node(item,unselected)
			else:
				self.close_all()
		else:
			if time.perf_counter()-self.alt_time<0.1:#alt pressed
				if short=='':
					self.tag_configure_node(item,tags['independent'])
				else:
					self.tag_configure_node(item,unselected)
			else:
				self.close_all()
				self.item(item,open=True)
				
	def var_defined(self,dname,vname,sel):
		parent_itm=dname+';'+vname
		short,vtype=self.item(parent_itm)['values']
		s=tags[sel]['short']
		if s=='Y' or s=='T' or s=='ID':#only one of these are allowed
			for i in self.datasets.desc[dname].nodes:
				short_i,vtype_i=self.item(i)['values']
				if s==short_i:
					self.tag_configure_node(i,unselected)
		if short==s:#Click on same
			self.tag_configure_node(parent_itm,unselected)
		else:
			self.tag_configure_node(parent_itm,tags[sel])
		self.item(parent_itm,open=False)
		
				
	def close_all(self):
		for i in self.datasets:
			for j in self.datasets.desc[i].nodes:
				self.item(j,open=False)
				
	def selected_item(self):
		item = self.selection()
		if len(item)==0:
			return
		if len(item)!=1:
			item=[self.win.data.get('selected data',item[0])]
		item=item[0]	
		return item
		
	def get_selected_dataset(self):
		item=self.selected_item()
		if item is None:
			return
		itm=item.split(';')[0]
		return self.datasets[itm]	
	
	def tag_configure_node(self,name,d,addvalue=True):
		self.tag_configure(name, foreground=d['fg'])
		self.tag_configure(name, background=d['bg'])
		self.tag_configure(name, font=d['font'])	
		if not addvalue:
			return
		short,dtype=self.item(name)['values']
		self.item(name,values=(d['short'],dtype))
		fname=name.split(';')[0]
		self.datasets.desc[fname].nodes[name]=d

def np_type(name,df):
	x=df[name]
	return x.dtype

class dataset_description():
	def __init__(self,datasets,name,source,import_script,tree):
		self.source=source
		self.data_import_script=import_script
		self.exe_editor=None
		self.edit_editor=None
		self.script_editor=None
		self.name=name
		self.nodes={}
		self.datasets=datasets
		self.options=options_module.regression_options()
		self.auto_update=True
		
		
	def generate_exe_script(self,tree):
		X=[]
		HF=[]
		d=dict()
		d['Y'],d['ID'],d['T'],d['HF']='','','',''
		for i in self.nodes:
			fname,j=i.split(";")
			if tree.item(i)['values'][0]=='X':
				X.append(j)
			elif tree.item(i)['values'][0]=='HF':
				HF.append(j)			
			else:
				d[tree.item(i)['values'][0]]=f'{j}'

		args=[f"'{d['Y']}~{'+'.join(X)}'\n\t",f"data['{self.name}']"]
		if len(HF):
			d['HF']=f"[{','.join(HF)}]"
		for i in ['ID','T','HF']:
			if d[i]!='':
				args.append(f"{i}='{d[i]}'")
		return f"execute({','.join(args)})"
		
			
class datasets(dict):
	def __init__(self):
		dict.__init__(self)
		self.desc={}
	
	def add(self,tree,name,data_dict,source,import_script):
		self[name]=pd.DataFrame(data_dict)
		self.desc[name]=dataset_description(self,name, source, import_script,tree)
		self.make_trees(tree)
		tree.item(f"{name};",open=True)
		
	def make_trees(self,tree):
		for i in self:
			self.make_tree(i,tree)	
			
	def make_tree(self,name,tree):
		try:
			tree.insert('', 1,f"{name};", text=' '+name,tags=(name+';',),image=tree.dbase_img)
		except tk.TclError:
			tree.delete(f"{name};")
			tree.insert('', 1,f"{name};", text=' '+name,tags=(name+';',),image=tree.dbase_img)
		self.add_nodes(tree,name)
		tree.main_frame.focus()
		tree.selection_add(f"{name};")
		tree.item(name+';',open=True)
		self.expand_check=[True,name+';']
		
	def delete(self,item,tree):
		self.__delitem__(item)
		tree.delete(item+';')
		
				
	def add_nodes(self,tree,name,d=None):
		df=self[name]
		nodes=self.desc[name].nodes
		item_list=list(df)
		if not d is None:
			if len(item_list)!=len(d.keys()):
				for i in d:
					df[i]=d[i]
		item_list.sort(key=lambda v: v.upper())
		n=len(item_list)
		n_tags=len(tags_list)
		for j in item_list:
			nptype=np_type(j,df)
			nodename=f"{name};{j}"
			tree.insert(f"{name};",n, nodename, text=' '+j,values=('',nptype),tags=(nodename,), image=tree.var_img)	
			if not nodename in nodes:
				nodes[nodename]=unselected
			tree.tag_configure_node(nodename,nodes[nodename])
			for k in tags_list:
				tree.insert(nodename,n_tags, f"{name};{j};{k}",values=('',tags[k]['short']), text=k,tags=(k,))
	
		

