#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module handle interfacing for various output paltforms


try:
  get_ipython()#fails if not in IPython environment
  import IPython
except NameError as e:
  IPython = None

from pydoc import importfile
import os
path = os.path.dirname(__file__)
from . import output

import os
import numpy as np
import time



WEB_PAGE='paneltime.html'
TMP_PAGE='tmphtml'
pic_num=[1]





def get_channel(window,exe_tab,panel, console_output):
  if console_output:
    return console(panel)
  if not window is None:#tkinter gui
    return tk_widget(window,exe_tab,panel)
  if not IPython is None:
    n=IPython.get_ipython().__class__.__name__
    if n=='ZMQInteractiveShell':
      return web_output(True,panel)
  try:
    return web_output(False,panel)
  except:
    pass
  return console(panel)

class web_output:
  def __init__(self,Jupyter,panel):
    global webbrowser
    import webbrowser
    self.panel=panel	
    self.Jupyter=Jupyter
    if not Jupyter:
      self.save_html(get_web_page('None', 'None', 'None', '', True))
      if panel.options.web_open_tab:
        webbrowser.open(WEB_PAGE, new = 2)
    self.output_set = False
    self.t0 = time.time()
    #from . import charts
    #self.charts = charts.ProcessCharts(panel)

  def set_output_obj(self,ll, comput, dx_norm):
    "sets the outputobject in the output" 
    if self.output_set:
      return		
    self.output=output.Output(ll,self.panel, dx_norm)
    self.output_set = True


  def update(self,comput, its, ll, incr, dx_norm):
    if self.panel.options.supress_output:
      return
    self.output.update(its, ll, incr, dx_norm, time.time() - self.t0)
    self.reg_table = output.RegTableObj(self.panel, ll, comput.g, comput.H, comput.G, comput.constr, dx_norm, self.output.model_desc) 
    tbl,llength=self.reg_table.table(4,'(','HTML',True,
                                                 show_direction=True,
                                                           show_constraints=True)		
    web_page=get_web_page(ll.LL, 
                                      ll.args.args_v, 
                                                          dx_norm,
                                                          tbl,
                                                          self.Jupyter==False)
    #self.charts.save_all(ll)
    if self.Jupyter:
      IPython.display.clear_output(wait=True)
      display(IPython.display.HTML(web_page))
    else:
      self.save_html(web_page)


  def save_html(self,htm_str):
    self.f = open(WEB_PAGE, "w")
    self.f.write(htm_str)
    self.f.close()



class console:
  def __init__(self,panel):
    return

  def set_output_obj(self,ll, comput, dx_norm):
    return

  def update(self, compute, its,ll,incr, dx_norm):
    return


class tk_widget:
  def __init__(self,window,exe_tab,panel):
    self.panel=panel
    self.tab=window.main_tabs._tabs.add_output(exe_tab)
    self.output_set = False

  def set_output_obj(self,ll, comput, dx_norm):
    if self.output_set:
      return		
    self.tab.set_output_obj(ll,self.panel, comput, dx_norm)
    self.output = self.tab.output
    self.output_set = True

  def update(self,comput, its, ll, incr, dx_norm):
    if self.panel.options.supress_output:
      return
    self.tab.update(self.panel, comput,its, ll, incr, dx_norm)


def get_web_page(LL, args, comput,tbl,auto_update):
  au_str=''
  if auto_update:
    au_str="""<meta http-equiv="refresh" content="1" >"""
  img_str=''
  pic_num[0]+=1
  if os.path.isfile('img/chart0.png'):
    img_str=(f"""<img src="img/histogram.png"><br>\n"""
                         f"""<img src="img/correlogram.png"   ><br>\n"""
                                f"""<img src="img/correlogram_variance.png"   >""")
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
               font-family: "Serif";
               text-align: left;
               color:#063f5c;
               font-size: 16;
}}
h1 {{
               margin-left: 20px;
               max-width: 980px;
               font-family: "Serif";
               text-align: left;
               color:black;
               font-size: 25;
               font-weight: bold;
}}

table.head {{
               font-family: "Serif";
               text-align: right;
               color:black;
               padding-left: 0px;
               font-size: 16;
}}
td.h:nth-child(odd) {{
               background: #CCC
}}
td {{
               padding-left: 0px;
}}
table {{
               font-family: "Serif";
               text-align: right;
               color:black;
               font-size: 16;
}}

th {{
border-collapse: collapse;
               border-bottom: double 3px;
               padding-left: 0px;
               white-space: nowrap;
}}
</style>
<body>
<div style='position:absolute;float:right;top:0;right:0'>
{img_str}
</div>
{tbl}
</body>
</html> """	

