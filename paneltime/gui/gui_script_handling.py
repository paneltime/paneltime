
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import re
		
	
def edit_exe_script(window,dataset):
	#Finds and edits an exe editor after click
	#to be removed when fixing this functionality
	editor=dataset.get_exe_editor(window.main_tabs,new_name='script')
	return
	#original script
	editor=dataset.get_exe_editor(window.main_tabs,False)
	if dataset is None:
		return
	editor=dataset.get_exe_editor(window.main_tabs,False)
	script=editor.widget.get('1.0',tk.END)
	#insert execution part:
	exe_script=dataset.generate_exe_script(window.right_tabs.data_tree)
	m=re.search(r"execute\(([\s\S]*?)\)", script)
	if m is None:
		script=script+'\n\n'+exe_script
	else:
		script=script[:m.start()]+exe_script+script[m.end():]	
	imprt='from paneltime import *'
	if not imprt in ' '.join(script.split()):
		script=imprt+'\n'+script
	editor.widget.replace_all(script)
	window.data.save()
	return script
	
def edit_options_script(options):
	editor=options.dataset.get_exe_editor(options.win.main_tabs,False).widget
	script=editor.get('1.0',tk.END)
	#insert execution part:
	optns,search_patterns=options.get_script()
	m=re.search(r"execute\(([\s\S]*?)\)", script)
	if m is None:
		script=edit_exe_script(options.win,options.dataset)
	end=re.search(r"execute\(([\s\S]*?)\)", script).start()
	if script[end-2:end]!='\n\n':
		script=script[:end]+'\n\n'+script[end:]
	for i in range(len(optns)):
		m=re.search(search_patterns[i], script[:end])
		if m is None:
			script=script[:end-2]+'\n'+optns[i]+ script[end-2:]
		else:
			script=script[:m.start()]+optns[i]+script[m.end():]		
	editor.replace_all(script)
	options.win.data.save()


def get_start_end(lines,identstr1,identstr2):
	start=-1
	end=-1
	for i in range(len(lines)):
		if lines[i]==identstr1[:-1] and start==-1:
			start=i
		elif lines[i]==identstr2[:-1] and start>-1:
			end=i
			break
	return start,end
		
def nicify(string):
	repl_list= [['\n\n\n','\n\n'],['\n\r\n','\n\n'],['\n\n\r','\n\n'],['\r\n\n','\n\n'],['  ',' ']]
	for s,r in repl_list:	
		while True:
			if s in string:
				string=string.replace(s,r)
			else:
				break
	return string
