#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module handle interfacing for various output paltforms

import IPython
import webbrowser
import output
import os
import charts
import shutil
import numpy as np
import time


WEB_PAGE='paneltime.html'
TMP_PAGE='tmphtml'
pic_num=[1]


class callback:
	def __init__(self,window, exe_tab, panel, console_output):
		self.channel = get_channel(window, exe_tab, panel, console_output)
		self.panel = panel
		self.set_progress = self.channel.set_progress
		
	def set_computation(self, computation, msg_main, _print=True):
		self.computation = computation
		self.msg_main = msg_main
		self._print = _print

	def print(self, msg, its, incr, ll, percent , update_type, task, dx_norm):
		if not self._print:
			return
		if not self.channel.output_set:
			self.channel.set_output_obj(ll, self.computation, self.msg_main, dx_norm)
		ok = self.channel.set_progress(percent ,msg ,task=task)
		if update_type>0:
			self.channel.update_after_direction(self.computation,its, dx_norm)
		elif update_type==0 or update_type==2:
			self.channel.update_after_linesearch(self.computation,ll,incr, dx_norm)
		return ok
	
	def print_final(self, msg, fret, conv, t0, xsol):
		self.channel.print_final(msg, fret, conv, t0, xsol)
	


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
		self.output_set = False
			
			
		
	def set_progress(self,percent, text, task):
		return True
		
	def set_output_obj(self,ll, comput,main_msg, dx_norm):
		"sets the outputobject in the output" 
		self.output=output.output(ll,self.panel, comput,main_msg)
		self.output_set = True
		
	def update_after_direction(self,comput,its, dx_norm):
		if not hasattr(comput,'ll'):
			return	
		self.its=its
		self.output.update_after_direction(comput,its, dx_norm)
		self.reg_table=self.output.reg_table()
		tbl,llength=self.reg_table.table(4,'(','HTML',True,
							   show_direction=True,
							   show_constraints=True)		
		web_page=get_web_page(comput.ll.LL, 
							  comput.ll.args.args_v, 
							  dx_norm,
							  tbl,
							  self.Jupyter==False)
		if self.Jupyter:
			IPython.display.clear_output(wait=True)
			display(IPython.display.HTML(web_page))
		else:
			self.save_html(web_page)
		
	def update_after_linesearch(self,comput,ll,incr, dx_norm):
		if not hasattr(comput,'ll'):
			return			
		self.output.update_after_linesearch(comput,ll,incr, dx_norm)
		self.reg_table=self.output.reg_table()
		tbl,llength=self.reg_table.table(4,'(','HTML',True,
							   show_direction=True,
							   show_constraints=True)		
		web_page=get_web_page(ll.LL, 
							  ll.args.args_v, 
							  dx_norm,
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

		
	def print_final(self, msg, fret, conv, t0, xsol):
		print(msg)
		print(f"LL={fret}  success={conv}  t={time.time()-t0}")
		print(xsol)	
		
		
	

		
class console:
	def __init__(self,panel):
		self.panel=panel
		self.output_set = False
		
	def set_progress(self,percent,text, task):
		if task=='done':
			print(text)
		#perc = f'{int(percent*100)}%'.ljust(5)
		#print(f"{perc} - {task}: {text}")
		return True
		
	def set_output_obj(self,ll, comput,msg_main, dx_norm):
		self.output=output.output(ll,self.panel, comput,msg_main, dx_norm)
		self.output_set = True
		
	def update_after_direction(self,comput,its, dx_norm):
		pass
		
	def update_after_linesearch(self,comput,ll,incr, dx_norm):
		print(ll.LL)
		
	def print_final(self, msg, fret, conv, t0, xsol):
		print(msg)
		print(f"LL={fret}  success={conv}  t={time.time()-t0}")
		print(xsol)		
				
class tk_widget:
	def __init__(self,window,exe_tab,panel):
		self.panel=panel
		self.tab=window.main_tabs._tabs.add_output(exe_tab)
		self.set_progress=self.tab.progress_bar.set_progress
		self.output_set = False

		
	def set_output_obj(self,ll, comput,msg_main, dx_norm):
		self.tab.set_output_obj(ll,self.panel, comput,msg_main, dx_norm)
		self.output_set = True
		
	def update_after_direction(self,comput,its, dx_norm):
		self.tab.update_after_direction(comput,its, dx_norm)
		
	def update_after_linesearch(self,comput,ll,incr, dx_norm):
		self.tab.update_after_linesearch(comput,ll,self.panel,incr, dx_norm)
		
	def print_final(self, msg, fret, conv, t0, xsol):
		print(msg)
		print(f"LL={fret}  success={conv}  t={time.time()-t0}")
		print(xsol)	


def get_web_page(LL, args, comput,tbl,auto_update):
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

