#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from gui import gui_functions as guif
import functions as fu
from shutil import copyfile

class scatter_window(tk.Toplevel):
	def __init__(self, master,x_names,y_name,X,Y,iconpath,tabs,height=400,width=1000):
		tk.Toplevel.__init__(self, master,height=height,width=width,)	
		self.title('Scatter charts')
		self.geometry('%sx%s' %(width,height))
		self.iconbitmap(iconpath)
		self.rowconfigure(0,weight=1)
		self.columnconfigure(0,weight=1)			
		
		self.main_frame=scatter_charts(self,tabs)
		self.main_frame.plot(x_names,y_name,X,Y)
		self.main_frame.grid(row=0,column=0)
		
		
		self.transient(master) #set to be on top of the main window
		self.grab_set() #hijack all commands from the master (clicks on the main window are ignored)
		#master.wait_window(self) #pause anything on the main window until this one closes (optional)		
	
class scatter_charts(tk.Frame):
	def __init__(self, master,tabs):
		tk.Frame.__init__(self, master)
		self.rowconfigure(0,weight=1)
		self.plotted=False
		self.columnconfigure(0,weight=1)	
		self.rowconfigure(1)
		self.columnconfigure(1)	
		self.rowconfigure(2)
		self.n_cols=3
		self.col_height=250		
		self.subplot=tabs.subplot
		self.print_subplot=tabs.print_subplot
		self.yscrollbar = tk.Scrollbar(self)
		self.xscrollbar = tk.Scrollbar(self,orient = tk.HORIZONTAL)
		#self.btn_saveall = tk.Button(self, text='Save all', command=self.saveall,bg='white')
		#self.btn_saveall.grid(row=1,column=0)	
		self.canvas=tk.Canvas(self,yscrollcommand = self.yscrollbar.set,xscrollcommand = self.xscrollbar.set,
							  width=2000,height=1000,bg='white')
		self.wdgt_frame=tk.Frame(self.canvas,bg='white')
		
		self.yscrollbar.config(command = self.canvas.yview)	
		self.xscrollbar.config(command = self.canvas.xview)	
		
		self.yscrollbar.grid(row=0,column=1,sticky='ns')
		self.xscrollbar.grid(row=2,column=0,sticky='ew')
		self.canvas.grid(row=0,column=0,sticky=tk.NSEW)		
		
	def plot(self,x_names,y_name,X,Y):
		self.X=X
		self.Y=Y		
		self.n_plots=X.shape[1]
		self.n_rows=int(self.n_plots/self.n_cols)
		self.n_rows+=(self.n_rows*self.n_cols<self.n_plots)
		self.canvas.configure(scrollregion=(0,0,1000,self.n_rows*self.col_height))
		self.wdgt_frame.configure(height=self.n_rows*self.col_height,width=1000)
		self.wdgt_frame.grid(row=0,column=0,sticky=tk.NSEW)
		self.plot_all(x_names,y_name)	
		self.canvas.create_window(0,0,window=self.wdgt_frame,anchor='nw')
		self.plotted=True
		
		
	def resize(self,event=None):
		print(self.winfo_width())
		


	def plot_all(self,x_names,y_name):
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
				self.charts[(row,col)]=self.plot_scatter(i,x_names,y_name,bgframe=self.wdgt_frame)
		for i in self.charts:			
			self.charts[i].grid(row=i[0],column=i[1])	
					
	def plot_scatter(self,i,x_names,y_name,subplot=None,f=None,bgframe=None):
		if subplot is None:
			subplot=self.subplot
		fgr,axs=subplot
		x=self.X[:,i]
		y=self.Y[:,0]
		
		axs.scatter(x,y, alpha=.1, s=10)
		axs.yaxis.label.set_text(y_name[0])
		axs.xaxis.label.set_text(x_names[i])
		name=f'{x_names[i]}'
		axs.set_title(name)	
		
		if f is None:
			w=int(fgr.get_figwidth()*fgr.get_dpi())
			h=int(fgr.get_figheight()*fgr.get_dpi())
			chart=tk.Label(bgframe,width=w,height=h)
			path=fu.obtain_fname(f'./output/{name}.png')
			guif.display(chart,name,i,subplot,self.on_scatter_click,path)
			return chart
		
		else:
		
			guif.save(subplot,f)
					
	def on_scatter_click(self,event):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"{event.widget.name}.jpg")	
		if f is None:
			return
		ch=open(event.widget.path,'rb')
		f.write(ch.read())
		ch.close()
		f.close()
		
		
	def saveall(self):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile="paneltime_scatter_plots.jpg")	
		fname=f.name
		f.close()
		for i in self.charts:
			f=open(gui.fix_fname(fname,i.name))
			self.plot_scatter(i.i, self.print_subplot,f)
			f.close()
			
	def on_closing(self):
		self.withdraw()
		
	def save(self,i):
		f = tk.filedialog.asksaveasfile(mode='bw', defaultextension=".jpg",initialfile=f"paneltime_scatter_plot_{i}.jpg")		
		self.plot_scatter(i, self.print_subplot,f)
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