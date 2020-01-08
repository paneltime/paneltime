#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import ImageTk, Image

def setbutton(parent,text,command,side=None,anchor=None,fill=None,bg=None):
	btn=tk.Button(parent, text=text, command=command,anchor=anchor,bg=bg)
	return btn


def save(subplot,save_file):
	fgr,axs=subplot
	fgr.savefig(save_file)
	axs.clear()
	
	
def display(panel,chart,name,i,subplot,action):
	fgr,axs=subplot
	f=panel.input.tempfile.TemporaryFile()
	fgr.savefig(f)
	plot_to_chart(f,chart)
	axs.clear()
	f.close()
	
	chart.name=name
	chart.i=i
	chart.bind("<Button-1>", action)	

	
	

def plot_to_chart(chart_file,chart_label):
	if hasattr(chart_label,'graph_file'):
		chart_label.graph_file.close()
	chart_label.graph_file=Image.open(chart_file)
	img = ImageTk.PhotoImage(chart_label.graph_file)
	chart_label.configure(image=img)
	chart_label.graph_img=img	
	
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


