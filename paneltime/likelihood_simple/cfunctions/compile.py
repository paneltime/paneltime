#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path
import re


def main():
  if os.name == 'nt':
    compile_windows()
  else:
    compile_linux()
    
def compile_windows():
  #THIS PATH MUST BE CHANGED: Locate your VS Code installation path
  vsc_loc = 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat'
  commands = ["cl /LD /O2 /fp:fast test.cpp", 
              #"cl /LD /O2 /fp:fast ctypes.cpp"
              ]
  p = str(Path(__file__).parent.absolute())
  print(p)
  cwd = get_windows_drive(p)
  print(cwd)
  for c in commands:
    full_cmd = f'"{vsc_loc}" && {c}'
    subprocess.run(full_cmd, shell=False, cwd=cwd)    

  
def compile_linux():
  #os.chdir(path) #change path if neccessary
  command = "g++ -shared -o ctypes.so -fPIC ctypes.cpp*/"
  os.system(command)
  

  
def get_windows_drive(network_drive):
    if ':\\' in network_drive:
      return network_drive
    result = subprocess.run(['net', 'use'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
  
    drives = {}
    for line in lines:
      line = re.sub('  +', '\t', line).split('\t')
      if len(line)>2:
        windrive = line[2].replace(' ', '').lower()
        network_drive = network_drive.lower()
        if windrive in network_drive:
          windrive = f"{line[1]}\\{network_drive.replace(windrive,'')}"
          return windrive
  
main()