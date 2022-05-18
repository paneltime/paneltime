#!/usr/bin/env python
# -*- coding: utf-8 -*-

import IPython
import webbrowser
import output
import os
import charts
import shutil
import numpy as np


WEB_PAGE='paneltime.html'
TMP_PAGE='tmphtml'
pic_num=[1]


def get_channel(window,exe_tab,panel, console_output):
	if console_output:
		return console(panel)
	if not window is None:#tkinter gui
		return tk_widget(window,exe_tab,panel)
	try:
		n=IPython.get_ipython().__class__.__name__
		if n=='ZMQInteractiveShell':
			return web_output(True,panel)
	except:
		pass
	try:
		return web_output(False,panel)
	except:
		pass
	return console(panel)
		
class web_output:
	def __init__(self,Jupyter,panel):
		self.panel=panel	
		self.Jupyter=Jupyter
		if not Jupyter:
			self.f = open(TMP_PAGE, "w")
			self.save_html(get_web_page('None', 'None', 'None', '', True))
			webbrowser.open(WEB_PAGE, new = 2)
		self.charts=charts.process_charts(panel)
			
			
		
	def set_progress(self,percent=None,text="",task=''):
		return True
		
	def set_output_obj(self,ll, direction,main_msg):
		"sets the outputobject in the output" 
		self.output=output.output(ll,self.panel, direction,main_msg)
		
	def update_after_direction(self,direction,its):
		if not hasattr(direction,'ll'):
			return	
		self.its=its
		self.output.update_after_direction(direction,its)
		self.reg_table=self.output.reg_table()
		tbl,llength=self.reg_table.table(4,'(','HTML',True,
							   show_direction=True,
							   show_constraints=True)		
		web_page=get_web_page(direction.ll.LL, 
							  direction.ll.args.args_v, 
							  direction.dx_norm,
							  tbl,
							  self.Jupyter==False)
		if self.Jupyter:
			IPython.display.clear_output(wait=True)
			display(IPython.display.HTML(web_page))
		else:
			self.save_html(web_page)
		
	def update_after_linesearch(self,direction,ll,incr):
		if not hasattr(direction,'ll'):
			return			
		self.output.update_after_linesearch(direction,ll,incr)
		self.reg_table=self.output.reg_table()
		tbl,llength=self.reg_table.table(4,'(','HTML',True,
							   show_direction=True,
							   show_constraints=True)		
		web_page=get_web_page(ll.LL, 
							  ll.args.args_v, 
							  direction.dx_norm,
							  tbl,
							  self.Jupyter==False)
		if self.Jupyter:
			IPython.display.clear_output(wait=True)
			display(IPython.display.HTML(web_page))
		else:
			self.save_html(web_page)
		#self.charts.save_all(ll)
		
	def save_html(self,htm_str):
		self.f.truncate(0)
		self.f.write(htm_str)
		self.f.flush()
		fpath=os.path.realpath(self.f.name).replace(self.f.name,'')
		shutil.copy(fpath+TMP_PAGE, fpath+WEB_PAGE)

		
		
	

		
class console:
	def __init__(self,panel):
		self.panel=panel
		
	def set_progress(self,percent=None,text="",task=''):
		if task=='done':
			print(text)
		#perc = f'{int(percent*100)}%'.ljust(5)
		#print(f"{perc} - {task}: {text}")
		return True
		
	def set_output_obj(self,ll, direction,msg_main):
		self.output=output.output(ll,self.panel, direction,msg_main)
		
	def update_after_direction(self,direction,its):
		pass
		
	def update_after_linesearch(self,direction,ll,incr):
		pass
				
class tk_widget:
	def __init__(self,window,exe_tab,panel):
		self.panel=panel
		self.tab=window.main_tabs._tabs.add_output(exe_tab)
		self.set_progress=self.tab.progress_bar.set_progress

		
	def set_output_obj(self,ll, direction,msg_main):
		self.tab.set_output_obj(ll,self.panel, direction,msg_main)
		
	def update_after_direction(self,direction,its):
		self.tab.update_after_direction(direction,its)
		
	def update_after_linesearch(self,direction,ll,incr):
		self.tab.update_after_linesearch(direction,ll,self.panel,incr)
		


def get_web_page(LL, args, direction,tbl,auto_update):
	au_str=''
	if auto_update:
		au_str="""<meta http-equiv="refresh" content="1" >"""
	img_str=''
	pic_num[0]+=1
	if os.path.isfile('img/chart0.png'):
		img_str=(f"""<img src="img/histogram.png"?{pic_num[0]}   ><br>\n"""
				f"""<img src="img/correlogram.png?{pic_num[0]}"   ><br>\n"""
				f"""<img src="img/correlogram_variance.png?{pic_num[0]}"   >""")
	return f"""
<meta charset="UTF-8">
{au_str}
<head>
<title>paneltime output</title>
</head>
<style>
p {{
  margin-left: 60px;
  max-width: 980px;
  font-family: "verdana";
  text-align: left;
  color:#063f5c;
  font-size: 12;
}}
h1 {{
  margin-left: 20px;
  max-width: 980px;
  font-family: "verdana";
  text-align: left;
  color:black;
  font-size: 16;
}}
</style>
<body>
<div style='position:absolute;float:right;top:0;right:0'>
{img_str}
</div>
{tbl}
</body>
</html> """	

