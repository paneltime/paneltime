#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
try:
	import win32clipboard
except:
	mswin=False

def copy_wondows():
	CF_RTF = win32clipboard.RegisterClipboardFormat("Rich Text Format")
	rtf = bytearray("{\\rtf1\\ansi\\deff0 {\\pard This is a test\\par}}", 'utf8')
	win32clipboard.OpenClipboard(0)
	win32clipboard.EmptyClipboard()
	win32clipboard.SetClipboardData(CF_RTF, rtf)
	win32clipboard.CloseClipboard()
	
def copy_linux(string):
	#could this work for windows too?
	if str(type(string)) == "<class 'str'>":
		string = bytearray(string, 'utf8')
	subprocess.Popen(['xclip', '-selection', 'clipboard', '-t', 'text/rtf'], stdin=subprocess.PIPE).communicate(string)
	
text = r"{\rtf1\ansi\deff0 {\b This} is some text\row}"
copy_rtf(text)	