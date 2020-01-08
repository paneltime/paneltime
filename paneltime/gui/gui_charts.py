#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from multiprocessing import pool

import numpy as np
import stat_functions as stat
from scipy import stats as scstats
from matplotlib import pyplot  as plt
from gui import gui_functions as guif


class process_charts():
	def __init__(self,window,frame):
		self.window=window
		self.subplot=plt.subplots(1,figsize=(4,2.5),dpi=75)
		self.print_subplot=plt.subplots(1,figsize=(6,3.25),dpi=200)
		self.ll=None	
		self.frame=frame
		self.initialized=False
		
	def add_content(self):
		self.btn_stats		=	guif.setbutton(self.frame, 'Display normality and AC statistics', self.show_stats,bg='white')
		self.btn_stats.config(relief=tk.SUNKEN)		
		self.statistics= tk.StringVar()
		tk.Label(self.frame,textvariable=self.statistics,bg='white',anchor=tk.NW,justify=tk.LEFT).grid(row=0,column=0)		
		tk.Label(self.frame,text='Charts on normalized residuals',bg='white').grid(row=1,column=0)			

		
		self.n_charts=3
		self.charts=[]
		for i in range(self.n_charts):
			self.charts.append(tk.Label(self.frame))
			self.charts[i].grid(row=i*2+2,column=0)		
			guif.setbutton(self.frame, 'Save image', lambda: self.save(self.n_charts-i-1),bg='white').grid(row=i*2+3,column=0)
			
			
	def show_stats(self):
		
		if self.btn_stats.cget('relief')==tk.RAISED:
			self.btn_stats.config(relief=tk.SUNKEN)
		else:
			self.btn_stats.config(relief=tk.RAISED)	
		
	def save(self,event):
		i=event.widget.i
		if not hasattr(self.charts[i],'graph_file') or not hasattr(self,'panel'):
			print('No graphics displayed yet')
			return
		name=event.widget.name
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"{name}.jpg")		
		flst=[
			self.histogram,
			self.correlogram,
			self.correlogram_variance,
		]
		flst[i](self.ll,self.print_subplot,f)
		f.close()
		
	def initialize(self):
		if not self.initialized:
			self.add_content()
			self.initialized=True		
		
	def plot(self,ll):
		self.panel=self.window.panel
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
			guif.display(self.panel, self.charts[0],name,0,subplot,self.save)
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
			guif.display(self.panel, self.charts[1],name,1,subplot,self.save)
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
			guif.display(self.panel, self.charts[2],name,2,subplot,self.save)
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

	
class scatter_charts(tk.Toplevel):
	def __init__(self, master,panel,X,Y,iconpath,height=400,width=1000):
		tk.Toplevel.__init__(self, master,height=400,width=1000,)
		self.print_subplot= plt.subplots(1,figsize=(6,3.25),dpi=200)
		self.subplot=plt.subplots(1,figsize=(4,2.5),dpi=75)

		
		self.panel=panel
		self.X=X
		self.Y=Y
		self.title('Scatter charts')
		self.geometry('%sx%s' %(width,height))
		self.iconbitmap(iconpath)
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)			
		
		self.n_cols=3
		self.col_height=250
		self.n_plots=self.panel.input.X.shape[1]
		self.n_rows=int(self.n_plots/self.n_cols)
		self.n_rows+=(self.n_rows*self.n_cols<self.n_plots)
		
		self.yscrollbar = tk.Scrollbar(self)
			
		self.main_frame=tk.Frame(self)
		self.main_frame.grid(row=0,column=0)			
		
		self.btn_saveall = tk.Button(self.main_frame, text='Save all', command=self.saveall,bg='white')
		self.btn_saveall.grid(row=1,column=0)		
		
		self.canvas=tk.Canvas(self.main_frame,yscrollcommand = self.yscrollbar.set,scrollregion=(0,0,width,self.n_rows*self.col_height),height=height,width=width)
					

		
		
		self.wdgt_frame=tk.Frame(self.canvas,width=width,height=self.n_rows*self.col_height)
			
		
		self.yscrollbar.grid(row=0,column=1,sticky='ns')
		self.wdgt_frame.grid(row=0,column=0,sticky=tk.NSEW)
		self.canvas.grid(row=0,column=0,sticky=tk.NSEW)	
		self.plot_all()	
		
		
		
			
		
		self.canvas.create_window(0,0,window=self.wdgt_frame,anchor='nw')
		self.yscrollbar.config(command = self.canvas.yview)	

		self.transient(master) #set to be on top of the main window
		self.grab_set() #hijack all commands from the master (clicks on the main window are ignored)
		#master.wait_window(self) #pause anything on the main window until this one closes (optional)


	def plot_all(self):
		self.charts=dict()
		for i in range(self.n_rows):
			self.wdgt_frame.rowconfigure(i,weight=1)	
		for i in range(self.n_cols):
			self.wdgt_frame.columnconfigure(i,weight=1)			
		for row in range(self.n_rows):
			for col in range(self.n_cols):
				i=row*self.n_cols+col
				if i>=self.n_plots:
					break
				self.charts[(row,col)]=self.plot(i,row,col,bgframe=self.wdgt_frame)
		for i in self.charts:			
			self.charts[i].grid(row=i[0],column=i[1])	
			
				
		
				
						
		
	def plot(self,i,row,col,subplot=None,f=None,bgframe=None):
		if subplot is None:
			subplot=self.subplot
		fgr,axs=subplot
		x_name=self.panel.input.x_names[i]
		y_name=self.panel.input.y_name[0]
		x=self.X[:,i]
		y=self.Y[:,0]
		
		axs.scatter(x,y, alpha=.1, s=10)
		axs.yaxis.label.set_text(y_name)
		axs.yaxis.label.set_text(x_name)
		name=f'{y_name} - {x_name}'
		axs.set_title(name)	
		
		if f is None:
			w=int(fgr.get_figwidth()*fgr.get_dpi())
			h=int(fgr.get_figheight()*fgr.get_dpi())
			chart=tk.Label(bgframe,width=w,height=h)
			guif.display(self.panel, chart,name,i,subplot,self.on_scatter_click)
			return chart
		
		else:
		
			guif.save(subplot,f)
					
	def on_scatter_click(self,event):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"{event.widget.name}.jpg")	
		frame=tk.Label(self.wdgt_frame)	
		self.plot(event.widget.i,subplot=self.print_subplot, f=f)
		frame.grid(row=0,column=0)		
		f.close()		
		
	def saveall(self):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile="paneltime_scatter_plots.jpg")	
		fname=f.name
		f.close()
		for i in self.charts:
			f=open(gui.fix_fname(fname,i.name))
			self.plot(i.i, self.print_subplot,f)
			f.close()
			
	def on_closing(self):
		self.withdraw()
		
	def save(self,i):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"paneltime_scatter_plot_{i}.jpg")		
		self.plot(i, self.print_subplot,f)
		f.close()
		
			
			
		
class ResizingCanvas(tk.Canvas):
	def __init__(self,parent,**kwargs):
		tk.Canvas.__init__(self,parent,**kwargs)
		#print self.winfo_reqwidth(),self.winfo_reqheight() #>>>854, 404
		self.bind("<Configure>", self.on_resize)

	def on_resize(self,event):
		self.width = event.width   #>>>854
		self.height = event.height #>>>404
		self.config(width=self.width, height=self.height)