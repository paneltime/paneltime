#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import keyword
import numpy as np

font0="Courier 10"
ret_chr=['\r','\n']

class ScrollText(tk.Frame):
	def __init__(self,master,readonly=False,text=None):
		tk.Frame.__init__(self,master)
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)		
		
		xscrollbar = tk.Scrollbar(self,orient='horizontal')
		yscrollbar = tk.Scrollbar(self)
	
		self.text_box = CustomText(self, wrap = tk.NONE,xscrollcommand = xscrollbar.set,yscrollcommand = yscrollbar.set,undo=True)			
		xscrollbar.config(command = self.text_box.xview)
		yscrollbar.config(command = self.text_box.yview)
		
		xscrollbar.grid(row=1,column=0,sticky='ew')
		yscrollbar.grid(row=0,column=1,sticky='ns')
		
		
		self.text_box.grid(row=0,column=0,sticky=tk.NSEW)
		
		self.readonly=readonly
		if not text is None:
			self.replace_all(text)
		if readonly:
			self.text_box.configure(state='disabled')
			


		
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

	def insert(self,index1,chars):
		if self.readonly:
			self.text_box.configure(state='normal')
		self.text_box.insert(index1,chars)
		if self.readonly:
			self.text_box.configure(state='disabled')
		self.text_box.changed()

		
	def write(self,chars):
		if self.readonly:
			return
		self.insert(tk.INSERT,chars)
		
	def see(self,index):
		self.text_box.see(index)
		
	def replace_all(self,string):
		if self.readonly:
			self.text_box.configure(state='normal')		
		self.text_box.delete('1.0',tk.END)
		self.text_box.insert(tk.INSERT,string)
		if self.readonly:
			self.text_box.configure(state='disabled')
			
		self.text_box.changed()
			
	



class CustomText(tk.Text):

	def __init__(self,master, wrap, xscrollcommand,yscrollcommand,undo):
		font='Courier'
		size=12

		tk.Text.__init__(self, master,wrap=wrap, 
						 xscrollcommand=xscrollcommand,yscrollcommand=yscrollcommand,undo=undo)	
		self.configure(font=(font,size,'normal'))
		self.bind('<KeyRelease>', self.changed)
		self.tag_configure('quote', foreground='dark red')
		self.tag_configure('keyword', foreground='#0a00bf')
		self.tag_configure('comment', foreground='#00a619')
		self.tag_configure('definition', foreground='#008a5a')
		self.tag_configure('bold', font=(font,size,'bold'))
		self.tag_configure('normal', font=(font,size,'normal'))
		self.tag_configure('black', foreground='black')
		self.define_keywords()
		
	def define_keywords(self):
		kwlist=np.array(keyword.kwlist)
		kwlensrt=np.array([len(i) for i in keyword.kwlist]).argsort()
		self.kwrds=list(kwlist[kwlensrt])
		self.kwrds.append('print')		
		
		
	
	def changed(self,event=None):
		if not event is None:
			if event.keycode==65 or event.keycode==17:
				return
		for tag in self.tag_names():
			self.tag_remove(tag,'1.0','end')		
		self.highlight_pattern(r"\"(.*?)\"", 'quote')		
		self.highlight_pattern(r"'(.*?)'", 'quote')
		self.highlight_pattern(r"def (.*?)\(", 'definition',addstart=4,subtractend=1,tag2='bold')
		self.highlight_pattern(r"\"\"\"(.*?)\"\"\"", 'quote')
		self.highlight_pattern(r"#(.*?)\r", 'comment',end='end-1c')
		self.highlight_pattern(r"#(.*?)\n", 'comment',end='end-1c')
		for i in self.kwrds:
			self.highlight_pattern(r"\m(%s)\M" %(i,), 'keyword',tag2='bold')
		
	def highlight_pattern(self, pattern, tag, start="1.0", end="end",
	                      regexp=True,tag2=None,addstart=0,subtractend=0):


		start = self.index(start)
		end = self.index(end)
		self.mark_set("matchStart", start)
		self.mark_set("matchEnd", start)
		self.mark_set("searchLimit", end)

		count = tk.IntVar()
		while True:
			index = self.search(pattern, "matchEnd","searchLimit",
			                    count=count, regexp=regexp)
			if index == "": break
			n=count.get()-subtractend
			if n <= 0: break # degenerate pattern which matches zero-length strings
			index2="%s+%sc" % (index, n)
			if addstart>0:
				index=f"{index}+{addstart}c"
			self.mark_set("matchStart", index)
			self.mark_set("matchEnd", index2)
			self.tag_add(tag, "matchStart", "matchEnd")
			if not tag2 is None:
				self.tag_add(tag2, "matchStart", "matchEnd")
			
			
			
