#!/usr/bin/env python
# -*- coding: utf-8 -*-

import IPython
import webbrowser
import output
import os

WEB_PAGE='paneltime.html'



def get_channel(window,exe_tab):
	if not window is None:#tkinter gui
		return tk_widget(window,exe_tab)
	try:
		n=IPython.get_ipython().__class__.__name__
		if n=='ZMQInteractiveShell':
			return web_output(True)
	except:
		pass
	try:
		return web_output(False)
	except:
		pass
	return console()
		
class web_output:
	def __init__(self,Jupyter):
		self.Jupyter=Jupyter
		if not Jupyter:
			save_html(get_web_page('None', 'None', 'None', '', True))
			webbrowser.open(WEB_PAGE, new = 2)
			
			
		
	def set_progress(self,percent=None,text="",task=''):
		return True
		
	def set_output_obj(self,ll, direction,main_msg):
		"sets the outputobject in the output" 
		self.output=output.output(ll, direction,main_msg)
		
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
			save_html(web_page)
		
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
			save_html(web_page)

		
class console:
	def __init__(self):
		pass
		
	def set_output_obj(self,ll, direction,msg_main):
		pass
		
	def update_after_direction(self,direction,its):
		pass
		
	def update_after_linesearch(self,direction,ll,incr):
		pass
				
class tk_widget:
	def __init__(self,window,exe_tab):
		self.tab=window.main_tabs._tabs.add_output(exe_tab)
		self.set_progress=self.tab.progress_bar.set_progress

		
	def set_output_obj(self,ll, direction,msg_main):
		self.tab.set_output_obj(ll, direction,msg_main)
		
	def update_after_direction(self,direction,its):
		self.tab.update_after_direction(direction,its)
		
	def update_after_linesearch(self,direction,ll,incr):
		self.tab.update_after_linesearch(direction,ll,incr)
		






def get_web_page(LL, args, direction,tbl,auto_update):
	au_str=''
	if auto_update:
		au_str="""<meta http-equiv="refresh" content="1" >"""
	img_str=''
	if os.path.isfile('img/chart0.png'):
		img_str=("""<img src="img/chart0.png"><br>\n"""
				"""<img src="img/chart1.png"><br>\n"""
				"""<img src="img/chart2.png">""")
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


def save_html(htm_str):
	f = open(WEB_PAGE, "w")
	f.write(htm_str)
	f.close()