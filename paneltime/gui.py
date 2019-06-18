#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import scrolledtext
import threading
import multi_core
from multiprocessing import pool


class ScrollText:
	def __init__(self,window,font,state):
		self.frame = tk.Frame(window)
		xscrollbar = tk.Scrollbar(self.frame,orient='horizontal')
		xscrollbar.pack(side = tk.BOTTOM, fill = tk.X)
		yscrollbar = tk.Scrollbar(self.frame)
		yscrollbar.pack(side = tk.RIGHT, fill = tk.Y)		
		self.text = tk.Text(self.frame, wrap = tk.NONE,state=state,
				    xscrollcommand = xscrollbar.set,yscrollcommand = yscrollbar.set)
		
		xscrollbar.config(command = self.text.xview)
		yscrollbar.config(command = self.text.yview)
		self.text.pack(fill='both', expand=True)
		self.frame.pack(fill='both', expand=True)
		
	def pack(self,side,fill=None):
		self.frame.pack(side=side,fill=fill)
		
	def get(self,index1,index2):
		self.text.get(index1,index2)
		
	def delete(self,index1,index2):
		self.text.delete(index1,index2)
		
	def insert(self,index1,chars):
		self.text.insert(index1,chars)		



class window:
	def __init__(self,title,iconpath=None,height=300,width=1000):
		self.win= tk.Tk()
		self.win.title(title)
		self.win.geometry('%sx%s' %(width,height))
		if not iconpath=='':
			self.win.iconbitmap(iconpath)
		xscroll=tk.Scrollbar(orient='horizontal')
		self.box = ScrollText(self.win, font=("Courier",10, "bold"), state='normal')
		self.box.pack(side=tk.TOP)
		bop = tk.Frame()
		bop.pack(side=tk.BOTTOM,fill=tk.X)
		tk.Button(bop, text='Print to stdout', command=self.print).pack(side=tk.LEFT,fill=tk.X)
		tk.Button(bop, text='End with current coefficients', command=self.finalize).pack(side=tk.LEFT)  
		tk.Button(bop, text='Display normality and AC statistics', command=self.norm_and_AC).pack(side=tk.LEFT)   
		tk.Button(bop, text='Copy to clipboard', command=self.copy).pack(side=tk.RIGHT)
		

				
		
	def print(self):
		print(self.box.get(1.0,tk.END))
		
	def norm_and_AC(self):
		if self.showNAC==True:
			self.showNAC=False
		else:
			self.showNAC=True
			
	def copy(self):
		self.win.clipboard_clear()
		self.win.clipboard_append(self.box.get(1.0,tk.END))
	
		
	def finalize(self):
		self.finalized=True

	def update(self,string):
		self.box.delete(1.0,tk.END)
		self.box.insert(tk.INSERT,string)
		#self.win.update()
		
	def close(self,x):
		self.win.quit()
		
	def run(self,func,args,close_when_finished=False):
		self.finalized=False
		self.showNAC=False
		p = pool.ThreadPool(processes=1)
		if close_when_finished:
			self.pool=p.apply_async(func, args,callback=self.close)
		else:
			self.pool=p.apply_async(func, args)
		self.win.mainloop() 
		try:
			self.win.destroy()
		except:
			pass
		
	def get(self):
		return self.pool.get()
	
	def save(self):
		print('todo')
		
		
		
		
		
def threadit(func,args):
	t = pool.ThreadPool(processes=1, initializer=func, initargs=args)
	a=t.apply_async(func, args)
	return a

a=0

"""import tkinter as tk
import time
import threading
from multiprocessing import pool



def main():
    top = tk.Tk()
    tex = tk.Text(master=top)
    tex.pack(side=tk.TOP)

    a=threadit(tex,dosomething,(tex,))
    top.mainloop()  
    a.get()

def cbc(string, tex):
    return lambda : callback(string, tex)


def callback(string, tex):
    tex.insert(tk.END, string)
    tex.see(tk.END)             # Scroll if necessary
    
    
def threadit(tex,func,args):
    t = pool.ThreadPool(processes=1, initializer=func, initargs=args)
    a=t.apply_async(func, args)
    return a
    
def dosomething(tex):
    for i in range(2):
        time.sleep(5)
        tex.insert(tk.END, "id: %s" %(i,))
    return 1,2



main()"""