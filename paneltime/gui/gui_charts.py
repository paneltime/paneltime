#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from multiprocessing import pool

import numpy as np
import stat_functions as stat
from scipy import stats as scstats
from gui import gui_functions as guif



class process_charts(ttk.Frame):
	def __init__(self,window,master,main_tabs,tabs):
		style = ttk.Style()
		style.configure("TFrame", background='white')		
		ttk.Frame.__init__(self,master,style='new.TFrame')
		self.window=window
		self.ll=None	
		self.initialized=False
		self.subplot=tabs.subplot
		self.print_subplot=tabs.print_subplot
		self.img_tmp=tabs.img_tmp
		
	def get_images_for_storage(self):
		charts=[]
		for i in self.charts:
			charts.append((i.path,i.name))
		return charts
		
	def charts_from_stored(self,charts):
		self.add_content()
		if charts is None:
			return
		for i in range(len(charts)):
			path,name=charts[i]
			guif.display_from_img(self.charts[i],path,name,i)
		
	def add_content(self):
		self.n_charts=3
		self.columnconfigure(0,weight=1)
		for i in range(self.n_charts+1):
			self.rowconfigure(i,weight=1)
			

		tk.Label(self,text='Charts on normalized residuals:',bg='white',font='Tahoma 10 bold').grid(row=0,column=0)			

		self.charts=[]
		for i in range(self.n_charts):
			frm=tk.Frame(self,background='white')
			frm.rowconfigure(0,weight=1)
			frm.rowconfigure(1)
			frm.columnconfigure(0,weight=1)
			self.charts.append(tk.Label(frm,background='white'))
			self.charts[i].grid(row=0,column=0)	
			self.charts[i].path=self.img_tmp.TemporaryFile()
			guif.setbutton(frm, 'Save image', lambda: self.save(self.n_charts-i-1),bg='white').grid(row=1,column=0)
			frm.grid(row=i+1)
		
	def save(self,i):
		if not hasattr(self.charts[i],'graph_file') or not hasattr(self,'panel'):
			print('No graphics displayed yet')
			return
		name=self.charts[i].name
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"{name}.jpg")		
		if f is None:
			return
		flst=[
			self.histogram,
			self.correlogram,
			self.correlogram_variance,
		]
		flst[i](self.ll,self.print_subplot,f)
		f.close()
		
	def initialize(self,panel):
		if not self.initialized:
			self.panel=panel
			self.add_content()
			self.initialized=True		
		
	def plot(self,ll):
		self.initialize(ll.panel)
		self.ll=ll
		self.histogram(ll,self.subplot)
		self.correlogram(ll,self.subplot)
		self.correlogram_variance(ll,self.subplot)	
		
		
	def histogram(self,ll,subplot,f=None):
		fgr,axs=subplot
		e=ll.e_st_centered[self.panel.included]
		N=e.shape[0]
		e=e.reshape((N,1))
		
		grid_range=4
		grid_step=0.05	
		h,grid=histogram(e,grid_range,grid_step)
		norm=scstats.norm.pdf(grid)*grid_step	
		
		axs.bar(grid,h,color='grey', width=0.025,label='histogram')
		axs.plot(grid,norm,'green',label='normal distribution')
		axs.legend(prop={'size': 6})
		name='Histogram - frequency'
		axs.set_title(name)
		if f is None:
			guif.display(self.charts[0],name,0,subplot)
		else:
			guif.save(subplot,f)

	def correlogram(self,ll,subplot,f=None):
		fgr,axs=subplot
		lags=20
		rho=stat.correlogram(self.panel, ll.e_st_centered,lags)
		x=np.arange(lags+1)
		axs.bar(x,rho,color='grey', width=0.5,label='correlogram')
		name='Correlogram - residuals'
		axs.set_title(name)
		if f is None:
			guif.display(self.charts[1],name,1,subplot)
		else:
			guif.save(subplot,f)
		
	def correlogram_variance(self,ll,subplot,f=None):
		fgr,axs=subplot
		lags=20
		e2=ll.e_st_centered**2
		e2=(e2-self.panel.mean(e2))*self.panel.included
		rho=stat.correlogram(self.panel, e2,lags)
		x=np.arange(lags+1)
		axs.bar(x,rho,color='grey', width=0.5,label='correlogram')
		name='Correlogram - squared residuals'
		axs.set_title(name)
		if f is None:
			guif.display(self.charts[2],name,2,subplot)
		else:
			guif.save(subplot,f)
	
def histogram(x,grid_range,grid_step):
	N,k=x.shape
	grid_n=int(2*grid_range/grid_step)
	grid=np.array([i*grid_step-grid_range for i in range(grid_n)]).reshape((1,grid_n))
	ones=np.ones((N,1))
	x_u=np.concatenate((ones,x>=grid),1)
	x_l=np.concatenate((x<grid,ones),1)
	grid=np.concatenate((grid.flatten(),[grid[0,-1]+grid_step]))
	histogram=np.sum((x_u*x_l),0)
	if int(np.sum(histogram))!=N:
		raise RuntimeError('Error in histogram calculation')
	return histogram/N,grid

	
