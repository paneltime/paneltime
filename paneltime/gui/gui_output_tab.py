#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from gui import gui_charts
from gui import gui_scrolltext
from gui import gui_scatter_charts
from tkinter import ttk
from tkinter import font
import functions as fu
import stat_functions as st
import traceback
import output
import os
import numpy as np
from PIL import ImageGrab
import communication

#color map:
bg_normal='white'
bg_selected='dark blue'
fg_normal='black'
fg_disabled='grey'
bg_sub_menu="#fafafa"
			
class output_tab(tk.Frame):
	def __init__(self,window,exe_tab,name,tabs,main_tabs,output_data):
		top_size=22
		self.txt_line_zoom=0
		self.window=window
		name=tabs.gen_name(name)
		self.tabs=tabs
		self.name=name
		tk.Frame.__init__(self,main_tabs)
		
		self.widget=tk.Frame(self)

		self.widget.rowconfigure(0,weight=1)
		self.widget.columnconfigure(0,weight=1)			
		
		
		self.charts = gui_charts.process_charts(window,self.widget,main_tabs,tabs)
		self.progress_bar=bar(self,exe_tab)
		
		self.progress_bar.grid(row=2, sticky=tk.EW)
		
		#adding the output tab to tha main tabs notebook
		top_color='#fcf3d9'
		self.tab=tabs.add(self,name=name,top_text='Regression output',top_color=top_color,skip_buttons=True)
		self.tab.exe_tab=exe_tab
		
		self.scroll_text_frame=tk.Frame(self.widget)
		self.scroll_text_frame.rowconfigure(0)
		self.scroll_text_frame.rowconfigure(1,weight=1)
		self.scroll_text_frame.columnconfigure(0,weight=1)			
		self.selection_frame=selection_frame(self.scroll_text_frame)
		self.tab.widget= gui_scrolltext.ScrollText(self.scroll_text_frame,format_text=False,name='regression',window=window,)
		self.output_text= gui_scrolltext.ScrollText(self.scroll_text_frame,format_text=False,name='regression',window=window,)
		main_tabs.select(self)
		main_tabs.insert('end',tabs.add_tab)
		self.scatter=gui_scatter_charts.scatter_charts(self.scroll_text_frame,tabs)
		self.scatter_norm=gui_scatter_charts.scatter_charts(self.scroll_text_frame,tabs)

		self.statistics=None
		self.reg_table=None
		self.widget.stored_output_data=stored_output()
		if not output_data is None:
			if hasattr(output_data, 'chart_images'):
				self.widget.stored_output_data=output_data
				try:
					self.charts.charts_from_stored(output_data.chart_images)
				except Exception as e:
					print(e)
					print('Chart images not found in temp folder. Change settings to\n'
						  'clean up the temp folder less frequently, in order ot avoid\n'
						  'this problem again\n')
				
		if not 'menu_selections' in self.window.data:
			self.window.data['menu_selections']=dict()
		self.add_buttons(top_size,top_color)
		self.add_formats()
		
		
	def add_formats(self):
		t=self.tab.widget.text_box
		t.configure(font=('Garamond', 12))
		t.tag_configure('bold_G', font=('Garamond', 12,'bold'))
		t.tag_configure('bold_G_underline', font=('Garamond', 12,'bold','underline'))
		t.tag_configure('10p', font=('Garamond', 10))		
		t.tag_configure('space_line', font=('Garamond', 7))
		
	def add_lines(self,zoom):		
		dirname=os.path.dirname(__file__)	
		if zoom==self.txt_line_zoom:
			return
		self.txt_line = tk.PhotoImage(file =  os.path.join(dirname,'img','line.png'))
		self.txt_line=self.txt_line.zoom(zoom,1)
		self.txt_double_line = tk.PhotoImage(file =  os.path.join(dirname,'img','double_line.png'))
		self.txt_double_line=self.txt_double_line.zoom(zoom,1)	
		self.txt_line_zoom=zoom
		
	
	def add_buttons(self,top_size,top_color):
		
		self.button_img=dict()
		
		self.button_img['save']= tk.PhotoImage(file =  fu.join(os.path.dirname(__file__),['img','save.png']),master=self.tab.button_frame)
		self.button_save=tk.Button(self.tab.button_frame_L, image = self.button_img['save'],command=self.save, 
								   highlightthickness=0,bd=0,height=top_size, anchor=tk.E,background=top_color)	
		self.button_save.grid(row=0,column=1)
		
		#selection frame buttons:
		self.font = font.Font(family='Helvetica', size=11, weight=font.BOLD)	
		self.font_small = font.Font(family='Helvetica', size=9, weight=font.BOLD)	

		
		self.menu_buttons={}
		i=0
		
		btnlist=[
			# Main button caption	Sub captions										command								click		sub 				Default
			#																												type main	groups				enabled sub							
			['REGRESSION',			['DIR','CONSTRNTS',['( )', '[ ]', 'disabled:( )'],
										['FLAT', 'STACKED'],
										['NORMAL', 'HTML', 'LATEX', 'RTF','INTERNAL'],
										['JOINED', 'JOINED LONG', 'disabled:JOINED']],	self.print_regression,				'group',	[[2,3,4]],			[0,1,2,3,4]],
			['SCATTER PLOTS',		['NORMALIZED'],										self.show_scatter,					'group', 	None,				[1]],
			[' CORREL ',			['NORMALIZED',[' CORREL ',' COVAR '],
										['VARIABLES','ESTIMATES','RAW MOMENTS']],		self.print_correl,					'group', 	[[1],[2]],			[1,2]],
			['DIAGNOSTICS',			['SAMP.SIZE','TESTS'],								self.print_stats,					'group',	[None],				[None]],
			['DESCRIPTIVES',		['EXP','LN','ORIGINAL'],							self.print_descriptive_statistics,	'group', 	[[0,1,2]],			[2]],
			['DIGITS',				list(range(10))+[16]+['SCI'],						self.print,							None,		[range(12)],		[5]],
			['DISTRIBUTION CHARTS',	None,												self.show_dist_charts,				'toggle',	None,				[None]]
		]
		#"disabled:" must come last in the sublist
		for caption, captions_sub,cmd,click_type,group_type_sub,def_sub in btnlist:
			self.menu_buttons[caption]=menu_button(self, i, caption, cmd,captions_sub,click_type,group_type_sub,def_sub)	
			i+=1

		self.no_print_menus=['DIGITS','FORMAT']
		self.widget.grid(row=1,column=0,sticky=tk.NSEW)
		self.tab.widget.grid(column=0,row=1,sticky=tk.NSEW)
		self.scroll_text_frame.grid(column=0,row=0,sticky=tk.NSEW)
		self.selection_frame.grid(0,0)
		self.charts.grid(column=1,row=0,sticky=tk.NS)
		self.get_stored()
		self.menu_buttons['REGRESSION'].click()
		self.store_options()
	
	
	def get_stored(self):
		"Obtains stored states of the buttons"
		d=self.window.data['menu_selections']
		for i in self.menu_buttons:
			self.menu_buttons[i].button_main['text']  =   d[i].button_main['text']
			self.menu_buttons[i].button_main['bg']    =   d[i].button_main['bg']
			self.menu_buttons[i].button_main['fg']    =   d[i].button_main['fg']
			if i in ['DIGITS', 'REGRESSION']:
				for j in self.menu_buttons[i].buttons_sub:
					if not j in ["['JOINED', 'JOINED LONG', 'disabled:JOINED']",
											   "['NORMAL', 'HTML', 'LATEX', 'RTF', 'INTERNAL']"]:				
						self.menu_buttons[i].buttons_sub[j]['text'] = d[i].buttons_sub[j]['text']
						self.menu_buttons[i].buttons_sub[j]['fg']   = d[i].buttons_sub[j]['fg']
						self.menu_buttons[i].buttons_sub[j]['bg']   = d[i].buttons_sub[j]['bg']

					
	def store_options(self):
		d=dict()
		self.window.data['menu_selections']=d
		for i in self.menu_buttons:
			d[i]=sub_botton_store(self.menu_buttons[i])
			for j in self.menu_buttons[i].buttons_sub:
				d[i].buttons_sub[j]=dict()
				d[i].buttons_sub[j]['text'] = self.menu_buttons[i].buttons_sub[j]['text']
				d[i].buttons_sub[j]['bg']   = self.menu_buttons[i].buttons_sub[j]['bg']
				d[i].buttons_sub[j]['fg']   = self.menu_buttons[i].buttons_sub[j]['fg']
			
	def save(self):
		"Saves the table part of the screen"
		#Problem: don't save all if table is bigger than screen. possible soloution, a popup window containing only the filled part of the text widget
		t=self.tab.widget.text_box
		img=ImageGrab.grab((t.winfo_rootx(),t.winfo_rooty(),t.winfo_width(),t.winfo_height()))
		img.save(os.path.join(self.window.data['current path'],'screen.png'))
		

				
			
			
		
	def data_doesnt_exist(self):
		"tests if data exists"
		if not hasattr(self.widget.stored_output_data,'data'):
			return True
		if self.widget.stored_output_data.data is None:
			print("Data not found")
			return True
		return False

	def set_output_obj(self,ll,panel, computation,main_msg):
		"sets the outputobject in the output" 
		self.output=output.output(ll,panel, computation,main_msg)
		
		
	def get_digits(self):
		"returns the selected number of digits"
		b=self.menu_buttons['DIGITS']
		for i in b.buttons_sub:
			if b.buttons_sub[i]['fg']==fg_normal:
				try:
					return int(i)
				except:
					return i

	def get_active(self):
		"returns the active menu button"
		for i in self.menu_buttons:
			b=self.menu_buttons[i]
			if b.button_main['bg']==bg_selected:
				return i	
		
	#****UPDATE CALLS********
		
	def update_after_direction(self,computation,its):
		self.output.update_after_direction(computation,its)
		self.reg_table=self.output.reg_table()
		self.print()
		
	def update_after_linesearch(self,computation,ll,panel,incr):
		self.output.update_after_linesearch(computation,ll,incr)
		if self.menu_buttons['DIAGNOSTICS'].button_main['bg']==bg_selected:
			self.statistics=self.output.statistics()
			self.widget.stored_output_data.statistics=self.statistics
		self.reg_table=self.output.reg_table()
		self.widget.stored_output_data.reg_table=self.reg_table		
		
		if self.menu_buttons['DISTRIBUTION CHARTS'].button_main["fg"]==fg_normal:
			self.charts.plot(ll,panel)
		self.widget.stored_output_data.chart_images=self.charts.get_images_for_storage()#for storing the editor
		self.widget.stored_output_data.data=stored_data(ll,panel,computation,self.reg_table)#for storing the editor		
		self.print()
		
	#****END UPDATE CALLS********
		

			
	#********MENU METHODS***********
		
	def print_regression(self):
		if self.reg_table is None:
			self.reg_table=self.widget.stored_output_data.reg_table
			if self.reg_table is None:return
		#for storing the editor
		n=self.get_digits()
		stacked=self.menu_buttons['REGRESSION'].buttons_sub["['FLAT', 'STACKED']"]['text']!='STACKED'
		if self.menu_buttons['REGRESSION'].buttons_sub[ "['( )', '[ ]', 'disabled:( )']"]['fg']==fg_disabled:
			bracket=''
		elif self.menu_buttons['REGRESSION'].buttons_sub[ "['( )', '[ ]', 'disabled:( )']"]['text']=='[ ]':
			bracket='['
		else:
			bracket='('
		fmt=self.menu_buttons['REGRESSION'].buttons_sub["['NORMAL', 'HTML', 'LATEX', 'RTF', 'INTERNAL']"]['text']
		joined=self.menu_buttons['REGRESSION'].buttons_sub["['JOINED', 'JOINED LONG', 'disabled:JOINED']"]
		if joined['fg']!=fg_normal:
			self.print_regression_single(n,stacked, bracket, fmt)
		else:
			self.join_tables(n, stacked,bracket, fmt,joined['text'])
			
	def show_scatter(self):
		if self.data_doesnt_exist():return
		d=self.widget.stored_output_data.data	
		#gui_scatter_charts.scatter_window(self,d.X_names,d.Y_name,d.X_st,d.Y_st,self.window.iconpath,self.tabs,700,1000)	
		if self.menu_buttons['SCATTER PLOTS'].buttons_sub['NORMALIZED']['fg']!=fg_disabled:
			if d.changed_since_last_scatter:
				self.scatter_norm.plot(d.X_names,d.Y_name,d.X_st,d.Y_st)
				d.changed_since_last_scatter=False
			self.scatter_norm.grid(column=0,row=1,sticky=tk.NSEW)
			self.scatter.grid_forget()
		else:
			if self.scatter.plotted==False:
				self.scatter.plot(d.X_names,d.Y_name,d.X,d.Y)
			self.scatter.grid(column=0,row=1,sticky=tk.NSEW)
			self.scatter_norm.grid_forget()
		self.tab.widget.grid_forget()

	def print_correl(self):
		if self.data_doesnt_exist():return
		d=self.widget.stored_output_data.data
		is_covar=self.menu_buttons[' CORREL '].buttons_sub["[' CORREL ', ' COVAR ']"]['text']==' COVAR '
		X,Y=d.X,d.Y
		if self.menu_buttons[' CORREL '].buttons_sub["NORMALIZED"]['fg']==fg_normal:
			X,Y=d.X_st,d.Y_st		
		if self.menu_buttons[' CORREL '].buttons_sub["['VARIABLES', 'ESTIMATES', 'RAW MOMENTS']"]['text']=='ESTIMATES':
			if d.hessian is None: return
			if is_covar:
				c=d.hessian
			else:
				n=len(d.hessian)
				s=np.maximum(np.diag(-d.hessian),0)**0.5
				s=s.reshape(n,1)*s.reshape(1,n)
				s[s==0]=d.hessian[s==0]
				c=-d.hessian/s
			names=d.args_names
		elif self.menu_buttons[' CORREL '].buttons_sub["['VARIABLES', 'ESTIMATES', 'RAW MOMENTS']"]['text']=='VARIABLES':
			c=st.correl_2dim(np.concatenate((Y,X),1),covar=is_covar)	
			names=d.all_names
		else:#RAW MOMENTS (remove?)
			V=np.concatenate((Y,X),1)
			c=np.dot(V.T,V)
			names=d.all_names
		correl=[['']+names]
		for i in range(len(c)):
			correl.append([names[i]]+list(c[i]))
		s=self.generate_table(correl)
		formatting={'single_line':[1],
					'double_line':[3],
					'bold':[2],
					'bold_col':2,
					'line_length':[len(c[0])]}		
		self.print_text(s,correl,formatting)
		
	def print_stats(self):
		if self.statistics is None:#in that case, this is an active output tab
			self.statistics=self.output.statistics()
			self.widget.stored_output_data.statistics=self.statistics			
			if self.statistics is None:return
		#for storing the editor
		n=self.get_digits()

		#tab_stops=self.reg_table.get_tab_stops(self.tab.widget.text_box.config()['font'][4])
		#self.tab.widget.text_box.config(tabs=tab_stops)	
		formatting={'bold_underline':[1]}
		tab_stops=('30',tk.LEFT,'60',tk.LEFT,'290',tk.RIGHT,'450',tk.NUMERIC)
		if self.menu_buttons['DIAGNOSTICS'].buttons_sub['SAMP.SIZE']['fg']==fg_normal:
			s="SAMPLE SIZE:\n\n"
			self.print_text(s+self.statistics.df_str,formatting=formatting,tab_stops=tab_stops)	
		elif self.menu_buttons['DIAGNOSTICS'].buttons_sub['TESTS']['fg']==fg_normal:
			s="DIAGNOSTICS:\n\n"
			s+=self.statistics.gen_mod_fit(n)
			s+=self.statistics.adf_str(n)
			self.print_text(s,formatting=formatting,tab_stops=tab_stops)
		else:
			s="ALL:\n\n"
			s+=self.statistics.df_str
			s+="\n  DIAGNOSTICS:"
			s+=self.statistics.gen_mod_fit(n)
			s+=self.statistics.adf_str(n)			
			self.print_text(s,formatting=formatting,tab_stops=tab_stops)	
			
	def print_descriptive_statistics(self,exp=False):
		if self.data_doesnt_exist():return
		d=self.widget.stored_output_data.data
		data=np.concatenate((d.Y,d.X),1)
		if self.menu_buttons['DESCRIPTIVES'].buttons_sub['EXP']['fg']!=fg_disabled:
			data=np.exp(data)
		elif self.menu_buttons['DESCRIPTIVES'].buttons_sub['LN']['fg']!=fg_disabled:
			data=np.log(data)
		n,k=data.shape
		stat=[['']+d.all_names]
		for operator,name in [
			(np.mean,'Mean:'),
			(np.median,'Median:'),
			(np.std,'S.D.:'),
			(np.min,'Minimum:'),
			(np.max,'Maximum:')]:
			stat.append([name]+list(operator(data,0)))
		stat_t=[]
		for i in range(len(stat[0])):
			stat_t.append([])
			for j in range(len(stat)):
				stat_t[i].append(stat[j][i])
		s=self.generate_table(stat_t)
		formatting={'single_line':[1],
					'double_line':[3],
					'bold':[2],
					'bold_col':2,
					'line_length':[len(stat_t[0])]}			
		self.print_text(s,stat_t,formatting)
		
		
	def print(self):
		for c in self.menu_buttons:
			if not (c in self.no_print_menus):
				b=self.menu_buttons[c]
				if b.button_main['bg']==bg_selected:
					b.command()
					return
				
			
	def show_dist_charts(self):
		b=self.menu_buttons['DISTRIBUTION CHARTS']
		if b.button_main['fg']==fg_disabled:
			self.charts.grid_forget()
		else:
			self.charts.grid(column=1,row=0,sticky=tk.NS)


	#********END MENU METHODS***********
		
	#******PRINTING METHODS*************
		
	def join_tables(self,digits,stacked,bracket,fmt,caption):
		"prints a table joined with previoius results"
		if self.widget.stored_output_data.data is None: return
		t=self.widget.stored_output_data.data.join_table
		if t is None:return
		s,X=t.make_table(stacked, bracket,digits,caption)	
		formatting=None
		n=len(t.names_v)
		if fmt=='NORMAL':
			formatting={'double_line':[4],
						'bold':[3],
						'bold_col':0,
						'line_length':[len(t)*(2-stacked)+3]
						}
			if stacked:
				formatting['10p']=[6+i*3 for i in range(len(t.names_v))]
				formatting['space_line']=[7+i*3 for i in range(len(t.names_v))]
				formatting['single_line']=[2,5+n*3,9+n*3]
			else:
				formatting['single_line']=[2,5+n,9+n]
		self.print_text(s,X,formatting)		
		
	def print_regression_single(self,digits,stacked,bracket,fmt):
		"prints a single regression"
		s,llength=self.reg_table.table(digits,bracket,fmt,stacked,
							   self.menu_buttons['REGRESSION'].buttons_sub['DIR']['fg']==fg_normal,
							   self.menu_buttons['REGRESSION'].buttons_sub['CONSTRNTS']['fg']==fg_normal)
		formatting=None
		if fmt=='NORMAL':
			formatting={'single_line':[5,9+self.reg_table.n_variables*3],
						'double_line':[7],
						'bold':[4,5,6],
						'bold_col':0,
						'line_length':[llength]
						}
			if stacked:
				formatting['10p']=[7+i*3 for i in range(self.reg_table.n_variables)]		
				formatting['space_line']=[8+i*3 for i in range(self.reg_table.n_variables)]	
		self.print_text(s,self.reg_table.X,formatting)
		
	def print_text(self,text,X=None,formatting=None,tab_stops="1c"):
		"prints the argument text"
		self.tab.widget.replace_all(text)	
		if not formatting is None:
			self.format_text_widget(formatting,text)	
		else:
			self.tab.widget.text_box.configure(font=("Courier", 10))	
		if not X is None:
			tab_stops=output.get_tab_stops(X,self.tab.widget.text_box.config()['font'][4])
		self.tab.widget.text_box.config(tabs=tab_stops)			
		self.scatter.grid_forget()
		self.scatter_norm.grid_forget()
		self.tab.widget.grid(column=0,row=1,sticky=tk.NSEW)
		

		
	def format_text_widget(self,formatting,text):
		"formats the output widget accoring to formatting"
		t=self.tab.widget.text_box
		t.configure(font=('Garamond', 12))
		if 'line_length' in formatting:
			self.add_lines(formatting['line_length'])
			for i in formatting['single_line']:
				t.image_create(f'{i}.0',image=self.txt_line)
			for i in formatting['double_line']:
				t.image_create(f'{i}.0',image=self.txt_double_line)	
		if 'bold' in formatting:
			for i in formatting['bold']:
				t.tag_add('bold_G',f'{i}.0' , f'{i}.end')
		if 'bold_underline' in formatting:
			for i in formatting['bold_underline']:
				t.tag_add('bold_G_underline',f'{i}.0' , f'{i}.end')		
		if '10p' in formatting:
			for i in formatting['10p']:
				t.tag_add('10p',f'{i}.0' , f'{i}.end')		
		if 'space_line' in formatting:
			for i in formatting['space_line']:
				t.tag_add('space_line',f'{i-1}.end' , f'{i+1}.0')
		if 'bold_col' in formatting:
			if formatting['bold_col']>0:
				for i in range(formatting['double_line'][-1]+1,text.count('\n')+1):
					ix0=f'{i}.0'
					for j in range(formatting['bold_col']):
						ix=t.search('\t',ix0,f'{i}.end')
						t.tag_add('bold_G',ix0 ,ix)
						ix0=f"{i}.{int(ix.split('.')[1])+1}"
						
	def generate_table(self,tbl):
		"generates a table as a text string from tbl"
		n=self.get_digits()
		for i in range(len(tbl)):
			for j in range(len(tbl[0])):
				try:
					tbl[i][j]=np.round(tbl[i][j],n)
				except:
					tbl[i][j]=tbl[i][j]
		s="\n"
		for i in range(len(tbl)):
			if i in [1]:
				s+='\n'			
			s+="\t"
			for j in range(len(tbl[0])):
				s+=f"{tbl[i][j]}\t"	
			s+="\n"
		return s	
						
		#******END PRINTING METHODS*************
		
		
		
class sub_botton_store(dict):
	def __init__(self,btn_main):
		self.button_main=dict()
		self.button_main['text']=btn_main.button_main['text']
		self.button_main['bg']=btn_main.button_main['bg']
		self.button_main['fg']=btn_main.button_main['fg']
		self.buttons_sub=dict()
	
class menu_button:
	"Class for the top level menu buttons"
	def __init__(self,output_tab,loc,caption_main,command,captions_sub,click_type,group_type_sub,sub_enabled):
		self.tab=output_tab	
		self.loc=loc
		self.button_frame=tk.Frame(self.tab.selection_frame,bg="white")
		if click_type is None:
			self.button_main=tk.Label(self.button_frame,text=caption_main, 
											   highlightthickness=0,bd=0, anchor=tk.E,bg=bg_normal,fg=fg_normal,font=self.tab.font)
		else:
			self.button_main=tk.Button(self.button_frame,text=caption_main,command=self.click, 
									   highlightthickness=0,bd=0, anchor=tk.E,bg=bg_normal,fg=fg_normal,font=self.tab.font)
		self.button_main.bind("<Motion>", self.mouseover)
		self.button_main.bind("<Leave>", self.button_hide_sub)
		self.button_main.grid(row=0,column=0)
		tk.Label(self.button_frame,text="|",bg=bg_normal,font=self.tab.font).grid(row=0,column=1)
		self.button_frame.grid(sticky='w',row=0,column=loc)	
		self.command=command
		self.captions_sub=captions_sub
		self.click_type=click_type
		self.group_type_sub=group_type_sub
		self.caption=caption_main
		self.sub_enabled=sub_enabled
		self.add_sub_menu()
		if not caption_main in self.tab.window.data['menu_selections']:
			self.tab.window.data['menu_selections'][caption_main]=dict()
		self.selections=self.tab.window.data['menu_selections'][caption_main]
		
	def select(self,withdraw=True):
		for i in self.tab.menu_buttons:
			if not i==self.caption and self.click_type=='group':
				self.tab.menu_buttons[i].alter_state(bg_normal,fg_normal)#reset all menus to unselected
			if withdraw:#If withdraw, then all button_frames are hidden
				self.tab.menu_buttons[i].button_box.withdraw()
		self.alter_state(bg_selected,bg_normal)	#set current menu to selected

		
	def toggle(self):
		if self.button_main['fg']==fg_normal:
			self.button_main.configure(fg=fg_disabled)
		else:
			self.button_main.configure(fg=fg_normal)			
		
	def click(self,runcommand=True):
		if self.click_type=='group':
			self.select()
		elif self.click_type=='toggle':
			self.toggle()
		if (self.command is None) or (self.click_type is None):
			return
		try:
			self.command()
		except Exception as e:
			traceback.print_exc()	
			
	def mouseover(self,event):
		if self.n_sub>0:
			self.show_sub()			
		
	def alter_state(self,bg,fg):
		self.button_main.configure(fg=fg)
		self.button_main.configure(bg=bg)
		
	def show_sub(self):
		x, y, cx, cy = self.button_main.bbox("insert")
		x += self.button_main.winfo_rootx()
		y += self.button_main.winfo_rooty()+self.button_frame.winfo_height()
		w=self.button_frame.winfo_width()
		h=(self.button_frame.winfo_height()+5)*self.n_sub
		self.button_box.geometry('%dx%d+%d+%d' % (150,h,x,y))
		for c in self.buttons_sub:
			self.buttons_sub[c].configure(widt=self.button_main.winfo_width())
		self.button_box.deiconify()	
		
	def hide_sub(self,event=None):
		w,h,x,y=split_geometry_string(self.button_box.geometry())
		if event.widget._name=='!toplevel' and event.y>0:
			self.button_box.withdraw()	
			
	def button_hide_sub(self,event):
		if event.y<self.button_main.winfo_height():
			self.button_box.withdraw()
		
	def click_sub(self,caption,i,execute=True):
		self.tab.store_options
		if type(self.captions_sub[i])==list:
			self.toggle_list(self.captions_sub[i],i)
			self.selections[str(caption)]=i
		elif self.sub_group[caption] is None:
			if self.group_type_sub==[None]:
				r=self.toggle_sub_select(caption)
			else:
				r=self.toggle_sub(caption)
			self.selections[caption]=r
		else:
			self.select_sub(caption)
			self.selections[str(self.sub_group[caption])]=caption
		if not execute:
			return
		if self.click_type=='group':
			self.select(False)
		if self.command is None:
			return
		try:
			self.command()
		except Exception as e:
			traceback.print_exc()
			
	def select_sub(self,caption):
		if self.buttons_sub[caption]['fg']==fg_disabled:
			for i in self.sub_group[caption]:
				self.buttons_sub[i].configure(fg=fg_disabled)
			self.buttons_sub[caption].configure(fg=fg_normal)
			
	def toggle_sub_select(self,caption):
		r=self.toggle_sub(caption)
		if self.buttons_sub[caption]['fg']==fg_normal:
			for i in self.buttons_sub:
				if not i==caption:
					self.buttons_sub[i].configure(fg=fg_disabled)
		return r
			
	def toggle_sub(self,caption):
		if self.buttons_sub[caption]['fg']==fg_disabled:
			self.buttons_sub[caption].configure(fg=fg_normal)
			return True
		else:
			self.buttons_sub[caption].configure(fg=fg_disabled)		
			return False
			
	def toggle_list(self,lst,i):
		caption=str(lst)
		if self.buttons_sub[caption]['fg']==fg_disabled:
			k=0
		else:
			current=self.buttons_sub[caption]['text']
			k=lst.index(current)+1
			if k==len(lst):
				k=0
		if 'disabled:' in lst[k]:
			self.buttons_sub[caption].configure(text=lst[k].split(':')[1])
			self.buttons_sub[caption].configure(fg=fg_disabled)
		else:
			self.buttons_sub[caption].configure(text=lst[k])
			self.buttons_sub[caption].configure(fg=fg_normal)
	
	def add_sub_menu(self):
		# creates a toplevel window
		self.button_box = tk.Toplevel(self.button_main,bg=bg_sub_menu)
		self.button_box.withdraw()
		self.button_box.bind("<Leave>", self.hide_sub)
		# Leaves only the label and removes the app window
		self.button_box.wm_overrideredirect(True)
		frm = tk.Frame(self.button_box, background=bg_sub_menu, relief='flat')
		if self.captions_sub is None:
			self.n_sub=0
		else:
			self.n_sub=len(self.captions_sub)
			
		self.buttons_sub={}
		self.captions_sub_str=[]
		fg=fg_disabled
		if self.sub_enabled=='all':
			fg=fg_normal
		for i in range(self.n_sub):
			c=str(self.captions_sub[i])
			self.captions_sub_str.append(c)
			if type(self.captions_sub[i])==list:
				cmd=eval(f"lambda: self.click_sub({c},{i})",{'self':self})
				text=self.captions_sub[i][0]
			else:
				cmd=eval(f"lambda: self.click_sub('{c}',{i})",{'self':self})
				text=c
			self.buttons_sub[c]=tk.Button(frm, text = text,command=cmd,
														font=self.tab.font, highlightthickness=0,bd=0,bg=bg_sub_menu,fg=fg,anchor="w")
			if (not self.sub_enabled is None) and i in self.sub_enabled:
				self.buttons_sub[c].configure(fg=fg_normal)
			else:
				if 'disabled:'==c[:9]:
					self.buttons_sub[c].configure(text=c[9:])
			self.buttons_sub[c].grid(row=i,column=0,sticky=tk.E)
		self.define_sub_grups()
		frm.grid(padx=10)

	def define_sub_grups(self):
		if self.captions_sub is None:
			return
		if self.group_type_sub=='all':
			self.group_type_sub=[self.captions_sub_str]
			self.sub_group={str(i):None for i in self.captions_sub_str}
			return
		self.sub_group={str(i):None for i in self.captions_sub_str}
		if not type(self.group_type_sub)==list:
			return
		g=[]
		n=len(self.group_type_sub)
		for i in range(n):
			g.append([])
			group=self.group_type_sub[i]
			if group is None:
				g[i]=None
			else:
				if not (type(group)==list or type(group)==range):
					raise RuntimeError("group_type_sub needs to be a list of lists or ranges")
				for j in group:
					g[i].append(self.captions_sub_str[j])
				for j in group:
					self.sub_group[self.captions_sub_str[j]]=g[i]
		self.group_type_sub=g
		

class stored_output:#for storing the editor
	def __init__(self):
		self.chart_images=None
		self.data=None
		self.reg_table=None
		self.statistics=None
		
		

class stored_data:
	def __init__(self,ll,panel,computation,reg_table):
		N,T,k=panel.X.shape
		self.X=panel.input.X
		self.Y=panel.input.Y
		if not hasattr(ll,'X_st'):
			ll.standardize(panel)
		self.X_st=ll.X_st[panel.included[2]]
		self.Y_st=ll.Y_st[panel.included[2]]		
		self.X_names=panel.input.X_names
		self.Y_name=panel.input.Y_names
		self.args_names=panel.args.names_v
		self.all_names=panel.input.Y_names+list(panel.input.X_names)
		self.args=ll.args
		self.LL=ll.LL
		self.changed_since_last_scatter=True
		self.hessian=computation.H
		self.gradient=computation.g
		self.descr=panel.input.descr
		self.reg_stats=reg_table.d
		self.add_join_table(panel,ll)
		
	def add_join_table(self,panel,ll):
		if panel.input.join_table is None:
			self.join_table=None
			return
		elif len(panel.input.join_table)==0:
			panel.input.join_table.append(output.join_table(ll.args,panel))
		elif type(panel.input.join_table[0])==str:
			if len(panel.input.join_table)>1:
				varnames=list(panel.input.join_table)
			else:
				s=panel.input.join_table
				varnames=s.replace(',','+').split('+')
			while len(panel.input.join_table):
				panel.input.join_table.pop()#removes all elements
			panel.input.join_table.append(output.join_table(ll.args,panel,varnames))
		self.join_table=panel.input.join_table[0]
		self.join_table.update(ll,self.reg_stats,self.descr,panel)
		
class bar(tk.Frame):
	def __init__(self,master,exe_tab):

		tk.Frame.__init__(self,master,background=bg_normal,height=75)
		self.tab=master
		self.exe_tab=exe_tab
		self.text_frame=tk.Frame(self,bg=bg_normal)
		self.text_frame_L=tk.Frame(self.text_frame,bg=bg_normal)
		self.text_frame_R=tk.Frame(self.text_frame,bg=bg_normal)
		self.text_frame_R.grid(row=0,column=1,sticky=tk.E)
		self.task_text=tk.StringVar(self)
		self.task_lbl=tk.Label(self.text_frame_R,textvariable=self.task_text,background=bg_normal,anchor='e', justify=tk.RIGHT,width=15)
		self.info_txt=tk.Text(self.text_frame_L,height=3,width=120,borderwidth = 0, highlightthickness = 0)
		self.info_txt.grid(row=0,column=0,sticky=tk.EW)
		self.info_txt.tag_configure('bold', font="Courier 10 bold")
		
		
		
		self.task_lbl.grid(row=0,column=0,sticky=tk.E)
		#self.text_frame_L.columnconfigure(0,weight=1)
		self.text_frame.columnconfigure(0,weight=1)
		
		self.columnconfigure(0,weight=1)
		
		self.text_frame_L.grid(row=0,column=0,sticky=tk.EW)
		self.text_frame.grid(row=0,column=0,sticky=tk.EW)
		self.progress=tk.Frame(self,background='#9cff9d',height=5,width=0)
		self.progress.grid(row=1,column=0,sticky=tk.W)
		
	def set_progress(self,percent=None,text="",task=''):
		total_width=self.winfo_width()
		if not percent==None:
			self.progress.config(width=int(total_width*percent))
			self.progress.grid()
		if len(task):
			self.set_task(task)
		if len(text):
			self.print(text)
		if not self.exe_tab is None:
			return self.exe_tab.isrunning
		else:
			return True
		
	def set_task(self,task):
		if task=='linesearch':
			self.progress.config(bg='#9cff9d')
			self.task_lbl.config(bg='#9cff9d')
			self.task_text.set('Linesearch')
			
		if task=='gradient':
			self.progress.config(bg='#8dd68e')
			self.task_lbl.config(bg='#8dd68e')
			self.task_text.set('Calculating gradient')	
			
		if task=='hessian':
			self.progress.config(bg='#82bd83')
			self.task_lbl.config(bg='#82bd83')
			self.task_text.set('Calculating hessian')	
			
		if task=='done':
			self.progress.config(bg='#9cff9d')
			self.task_lbl.config(bg='#9cff9d')
			self.task_text.set('Done')		
			
		if task=='err':
			self.progress.config(bg='red')
			self.task_lbl.config(bg='red')
			self.task_text.set('No convergence')			
			
		
		
	def print(self,text):
		n=max((int(self.info_txt.index('end').split('.')[0]),1))
		text='>'+text
		if len(self.info_txt.get('1.0','end')):
			text='\n'+text
		self.info_txt.insert('end',text)
		self.info_txt.tag_add('bold',f'{n}.0' , f'{n}.end')
		self.info_txt.tag_remove('bold','1.0' , f'{n-1}.end')
		self.info_txt.see('end')

				
		
		
		
class selection_frame(tk.Frame):
	def __init__(self, master,):
		tk.Frame.__init__(self,master,background=bg_normal)
		
	def grid(self,column,row):
		super().grid(column=column,row=row,sticky=tk.EW)

	
def split_geometry_string(s):
	s,x,y=s.split('+')
	w,h=s.split('x')
	return int(w),int(h),int(x),int(y)


		
