#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import ImageTk, Image

def setbutton(parent,text,command,side=None,anchor=None,fill=None,bg=None,relief=None):
	btn=tk.Button(parent, text=text, command=command,anchor=anchor,bg=bg,relief=relief,bd=0)
	return btn


def save(subplot,save_file):
	fgr,axs=subplot
	fgr.savefig(save_file)
	axs.clear()
	
	
def display(chart,name,i,subplot,action=None,path=None):
	fgr,axs=subplot
	if path is None:
		path=chart.path
	f=open(path,'wb+')
	try:
		fgr.savefig(f)
	except:
		fgr.savefig(f)
	plot_to_chart(f,chart)
	axs.clear()
	f.close()
	
	chart.name=name
	chart.i=i
	chart.path=path
	chart.bind("<Button-1>", action)	

def display_from_img(chart,f,name,i,action=None):
	plot_to_chart(f,chart)
	chart.name=name
	chart.i=i
	chart.bind("<Button-1>", action)	
	

def plot_to_chart(f,chart):
	if hasattr(chart,'graph_file'):
		chart.graph_file.close()
	chart.graph_file=Image.open(f)
	img = ImageTk.PhotoImage(chart.graph_file,master=chart)
	chart.configure(image=img)
	chart.graph_img=img	
	
def fix_fname(s,i=None):
	if i is None:
		i=''
	else:
		i=str(i)
	if '.' in s:
		l=len(s.split('.')[-1])
		s=s[:-l]+i+s[-l:]
	else:
		s=s+i+'.jpg'
	return s


