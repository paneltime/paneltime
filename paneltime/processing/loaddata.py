#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
from datetime import date
from . import model_parser
import pandas as pd

forbidden_names=['tobit_low','tobit_high',model_parser.DEFAULT_INTERCEPT_NAME,model_parser.CONST_NAME]

def load(filepath_or_buffer,sep=None, header="infer", 
         names=None, **kwargs):
  print ("opening file ...")
  sep=get_sep(filepath_or_buffer, sep)
  data=pd.read_csv(filepath_or_buffer, sep, sep,header, 
                         names, **kwargs)
  print ("... done")
  load_data_printout(data)
  return data

def load_json(path_or_buf=None, orient=None, typ="frame", **kwargs):
  return pd.read_json(path_or_buf, orient, typ, **kwargs)


def load_data_printout(data):
  print ("The following variables were loaded:"+', '.join(data.keys()))


def append(d,key,i):
  if key in d:
    d[key].append(i)
  else:
    d[key]=[i]

def load_SQL(sql_string,conn,**kwargs):
  return pd.read_sql(sql_string, conn, **kwargs)




def get_sep(fname,sep):
  f=open(fname,'r')
  r=[]
  sample_size=20
  for i in range(sample_size):
    r.append(f.readline())	
  f.close()
  d={}
  for i in [sep,';',',','\t',' ']:#checks whether the separator is consistent
    len0=len(r[0].split(i))
    err=False
    for j in r[1:]:
      rlen=len(j.split(i))
      if rlen!=len0:
        err=True
        break
    if not err and rlen>1:
      d[i]=rlen
  maxlen=max([d[i] for i in d])
  if ',' in d:
    if d[',']==maxlen:
      if len(d)>0:
        d.pop('.')
        maxlen=max([d[i] for i in d])
  for i in d:
    if d[i]==maxlen:
      return i

SQL_type_dict={0: float,
               1: int,
 2: int,
 3: float,
 4: float,
 5: float,
 6: float,
 8: int,
 9: int,
 16: int,
 246: int
 }