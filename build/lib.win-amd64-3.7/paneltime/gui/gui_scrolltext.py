#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import keyword
import numpy as np
import re

font0="Courier 10"
ret_chr=['\r','\n']

class ScrollText(tk.Canvas):
	def __init__(self,master,readonly=False,text=None,format_text=True,name=None, window=None,height=None,x_bar=True):
		tk.Canvas.__init__(self,master,bg='white')
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)		
		
		self.yscrollbar = tk.Scrollbar(self)
		self.yscrollbar.set(0,0)			
		
		self.xscrollbar=None
		if x_bar==True:
			self.xscrollbar = tk.Scrollbar(self,orient='horizontal')
			self.xscrollbar.set(0,0)

		self.master=master
		self.text_box = CustomText(self, wrap = tk.NONE,xscroll = self.xscrollbar,
								   yscroll = self.yscrollbar,undo=True,format_text=format_text,
								   name=name,window=window,height=height)			
		self.text_box.config(tabs='1c')
		if not self.xscrollbar is None:
			self.xscrollbar.config(command = self.text_box.xview)
			self.xscrollbar.grid(row=1,column=0,sticky='ew')
		
		self.yscrollbar.config(command = self.text_box.yview)
		self.yscrollbar.grid(row=0,column=1,sticky='ns')
		
		self.text_box.grid(row=0,column=0,sticky=tk.NSEW,padx=(15,0),pady=(5,0))
		
		self.readonly=readonly
		if not text is None:
			self.replace_all(text)
		if readonly:
			self.text_box.configure(state='disabled')
			
	def bind_to_key_release(self,func):
		self.text_box.after_key_release=func

		
	def get(self,index1,index2):
		return self.text_box.get(index1,index2)
	
	def get_all(self):
		return self.get('1.0',tk.END)
		
	def delete(self,index1,index2):
		if self.readonly:
			self.text_box.configure(state='normal')	
		self.text_box.delete(index1,index2)
		if self.readonly:
			self.text_box.configure(state='disabled')

	def insert(self,index1,chars,index2=None):
		if self.readonly:
			self.text_box.configure(state='normal')
		if not index2 is None:
			self.text_box.delete(index1, index2)
		self.text_box.insert(index1,chars)
		self.text_box.replace
		if self.readonly:
			self.text_box.configure(state='disabled')
		self.text_box.key_released()
		
	def write(self,chars):
		if self.readonly:
			return
		self.insert(tk.INSERT,chars)
		
	def see(self,index):
		self.text_box.see(index)
		
	def replace_all(self,string):
		if string is None:
			string=""
		if self.readonly:
			self.text_box.configure(state='normal')		
		if not self.xscrollbar is None:
			x=self.xscrollbar.get()
		y=self.yscrollbar.get()
		lines=string.split('\n')
		n=int(self.text_box.index('end').split('.')[0])-1
		if len(lines)==n and False:#attemting to reduce flickering, but does not seem to help
			for i in range(n):
				s=f'{i+1}.0',f'{i+1}.end'
				self.text_box.delete(s[0],s[1])
				self.text_box.insert(s[0],lines[i])
		else:
			self.text_box.delete('1.0',tk.END)
			self.text_box.insert('1.0',string)	
		if not self.xscrollbar is None:
			self.xscrollbar.set(*x)
			if x[1]>x[0]:
				self.text_box.xview(tk.MOVETO,x[0]/(x[1]-x[0]))

		self.yscrollbar.set(*y)		
		if y[1]>y[0]:
			self.text_box.yview(tk.MOVETO,y[0]/(y[1]-y[0]))
			
		if self.readonly:
			self.text_box.configure(state='disabled')
			
		self.text_box.key_released()
			
	



class CustomText(tk.Text):

	def __init__(self,master, wrap,undo,xscroll=None,yscroll=None,format_text=True,name=None,window=None,height=None):
		font='Courier'
		size=10
		
		yscrollcommand=None
		xscrollcommand=None
		if yscroll is None: 
			yscrollcommand=yscroll.set
		if not xscroll is None:
			xscrollcommand=xscroll.set
		
		tk.Text.__init__(self, master,wrap=wrap, 
						 xscrollcommand=xscrollcommand,yscrollcommand=yscrollcommand,undo=undo,bd=0,height=height)	
		self.master=master
		self.configure(font=(font,size,'normal'))
		self.bind('<KeyRelease>', self.key_released)
		self.bind('<KeyPress>', self.key_pressed)
		self.tag_configure('quote', foreground='dark red')
		self.tag_configure('builtins', foreground='#51769e')
		self.tag_configure('keyword', foreground='#0a00bf')
		self.tag_configure('comment', foreground='#00a619')
		self.tag_configure('definition', foreground='#008a5a')
		self.tag_configure('bold', font=(font,size,'bold'))
		self.tag_configure('normal', font=(font,size,'normal'))
		self.tag_configure('black', foreground='black')
		self.define_keywords()
		self.format_text=format_text
		self.pressed_key=''
		self.released_ignore_key=False
		self.name=name
		self.win=window
		self.after_key_release=None
		
		
		
	def define_keywords(self):
		kwlist=np.array(keyword.kwlist)
		kwlensrt=np.array([len(i) for i in keyword.kwlist]).argsort()
		self.kwrds=list(kwlist[kwlensrt])
		builtins=list(dir(__builtins__))+builtin_functions
		b=[]
		for i in builtins:
			if not i in self.kwrds:
				b.append(i)
		self.builtins=b
		False 
				
		
	def key_pressed(self,event):
		try:
			self.pressed_key=event.keysym
		except:
			pass
		
	
	def key_released(self,event=None):
		if not self.format_text:return
		if self.pressed_key in ignore_press_keys:
			self.pressed_key=''
			return
		if not event is None:
			if ((self.released_ignore_key==True) 
				and (not event.keysym in ignore_press_keys)):
				self.released_ignore_key=False
				if event.keysym!='v':#format text after pasting
					return
			if event.keysym in ignore_press_keys:
				self.released_ignore_key=True
			if (event.keysym in ignore_keys):
				return
		for tag in self.tag_names():
			self.tag_remove(tag,'1.0','end')	
			
		self.highlight_pattern(r"\"\"\"(.*?)\"\"\"", 'quote')
		self.highlight_pattern(r"\"(.*?)\"", 'quote')	
		self.highlight_pattern(r"'(.*?)'", 'quote')
		self.highlight_pattern_multiline(r"\"\"\"([\s\S]*?)\"\"\"", 'quote')

		for i in self.kwrds:
			self.highlight_pattern(r"\m(%s)\M" %(i,), 'keyword',tag2='bold')		
		for i in self.builtins:
			self.highlight_pattern(r"\m(%s)\M" %(i,), 'builtins')				
		self.highlight_pattern(r"def (.*?)\(", 'definition',addstart=4,subtractend=1,tag2='bold')
		
		self.highlight_pattern(r"#(.*?)\r", 'comment',end='end-1c')
		self.highlight_pattern(r"#(.*?)\n", 'comment',end='end-1c')
		if not self.after_key_release is None:
			self.after_key_release()

		
	def highlight_pattern(self, pattern, tag, start="1.0", end="end",
	                      regexp=True,tag2=None,addstart=0,subtractend=0):


		indicies=self.search_pattern(pattern, start=start, end=end, regexp=regexp, addstart=addstart, subtractend=subtractend)
		if len(indicies)==0:
			return
		for index1,index2 in indicies:
			if not len(self.tag_names(index1)):
				self.tag_add(tag, index1, index2)
				if not tag2 is None:
					self.tag_add(tag2, index1, index2)
				
	def highlight_pattern_multiline(self, pattern, tag, start="1.0", end="end",
	                      regexp=True,tag2=None,addstart=0,subtractend=0):

		s=self.get('1.0',tk.END)
		start=0
		while True:
			m=re.search(pattern,s[start:])
			if m is None:break
			index1=self.pos_to_index(m.start()+start,s)
			index2=self.pos_to_index(m.end()+start,s)
			self.tag_add(tag, index1, index2)
			if not tag2 is None:
				self.tag_add(tag2, index1, index2)	
			start=m.end()+start
			
	def search_pattern(self, pattern, start="1.0", end="end",
	                      regexp=True,addstart=0,subtractend=0):


		start = self.index(start)
		end = self.index(end)
		self.mark_set("matchStart", start)
		self.mark_set("matchEnd", start)
		self.mark_set("searchLimit", end)

		count = tk.IntVar(master=self)
		indicies=[]
		while True:
			index1 = self.search(pattern, "matchEnd","searchLimit",
								count=count, regexp=regexp)
			if index1 == "": break
			n=count.get()-subtractend
			if n <= 0: break # degenerate pattern which matches zero-length strings
			index2="%s+%sc" % (index1, n)
			if addstart>0:
				index1=f"{index1}+{addstart}c"
			self.mark_set("matchStart", index1)
			self.mark_set("matchEnd", index2)			
			indicies.append((index1,index2))
			
			
		return indicies

			
		
	def pos_to_index(self,pos,string):
		s=str(string[:pos+1])
		if pos+1>len(s):
			pos=len(s)-1
		lines=s.split('\n')
		if len(lines)==1:
			return f"1.{len(s)}" 
		chars=lines[-1]
		index=f"{len(lines)}.{len(chars)}"
		return index
	
			

ignore_press_keys=['Up','Down','Right','Left','Prior','Next','Tab','Control_L','Control_R','Alt_R','Alt_L']
ignore_keys=ignore_press_keys+['Shift_L','Shift_R']


builtin_functions=["abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod", 
				   "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", 
				   "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", 
				   "input", "int", "isinstance", "issubclass", "iter", "len", "list", "locals", "map", "max", "memoryview", 
				   "min", "next", "object", "oct", "open", "ord", "pow", "print", "property", "range", "repr", "reversed", 
				   "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", 
				   "vars", "zip", "__import__", ]
