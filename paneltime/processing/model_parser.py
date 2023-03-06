#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEFAULT_INTERCEPT_NAME='Intercept'
VAR_INTERCEPT_NAME='log_variance_constant'
INSTRUMENT_INTERCEPT_NAME='instrument_intercept'
CONST_NAME='one'
NUMERIC_TAG="_numeric"

import numpy as np
import pandas as pd

def get_variables(ip,df,model_string,IDs,timevar,heteroscedasticity_factors,instruments,settings,pool=(None,'mean')):
  print ("Analyzing variables ...")
  if not type(df)==pd.DataFrame:
    raise RuntimeError('The dataframe supplied is not a pandas dataframe. Only pandas dataframes are supported.')
  if CONST_NAME in df:
    print(f"Warning: The name {CONST_NAME} is reserved for the constant 1."
                      f"The variable with this name will be overwritten and set to 1")
  df=pool_func(df,pool)

  df[CONST_NAME]=1
  df[DEFAULT_INTERCEPT_NAME]     = df[CONST_NAME]
  df[VAR_INTERCEPT_NAME]         = df[CONST_NAME]
  df[INSTRUMENT_INTERCEPT_NAME]  = df[CONST_NAME]

  IDs     =get_names(IDs, 'IDs')
  timevar =get_names(timevar, 'timevar')
  IDs_num,     IDs      = handle_IDs(ip,df,IDs)
  timevar_num, timevar  = handle_time(ip,df,timevar)	
  sort=[i for i in [IDs,timevar] if not i==[]]
  if len(sort):
    df=df.sort_values([IDs_num[0],timevar[0]])
  pd_panel=df.groupby(IDs_num)

  W=get_names(heteroscedasticity_factors,'heteroscedasticity_factors',True, VAR_INTERCEPT_NAME)
  Z=get_names(instruments,'instruments',True,INSTRUMENT_INTERCEPT_NAME)

  try:
    Y,X=parse_model(model_string, settings)
  except:
    RuntimeError("The model_string must be on the form Y~X1+X2+X3")	
  if X==[] or Y==['']:
    raise RuntimeError("No independent or dependent variables specified")		

  x = IDs_num+IDs+timevar+timevar_num+W+Z+Y+X
  x = list(dict.fromkeys(x))
  df,ip.max_lags=eval_variables(x, df, pd_panel)
  df=df[x]
  n=len(df)
  df=df.dropna()
  ip.lost_na_obs=(n-len(df))-ip.max_lags
  const={}
  for x,add_intercept,num in [
          ('IDs_num',False,True),('timevar_num',False,True),
                                ('IDs',False,False),('timevar',False,False),
                                ('W',True,True),('Z',True,True),('Y',False,True),
                                  ('X',settings.add_intercept.value,True)]:
    ip.__dict__[x], const[x]= check_var(df,locals()[x],x,add_intercept,num)
    ip.__dict__[x+"_names"]=list(ip.__dict__[x].columns)
  ip.dataframe=df
  ip.has_intercept=const['X']



def pool_func(df,pool):
  x,operation=pool
  if x is None:
    return df
  x=get_names(x, 'pool')
  df=df.groupy(x).agg(operation)
  return df

def check_var(df,x,inputtype,add_intercept,numeric):
  if len(x)==0:
    return None,None
  dfx=df[x]
  if not numeric:
    return dfx,None
  const_found=False
  for i in x:
    if ' ' in i:
      raise RuntimeError(f'Spaces are not allowed in variables, but found in the variable {i} from {inputtype}')
    try:
      v=np.var(dfx[i])
    except TypeError as e:
      raise TypeError(f"All variables except time and id must be numeric. {e}")
    if v==0 and const_found:
      if dfx[i].iloc[0]==0:
        print(f"Warning: All values in {i} from {inputtype} are zero, variable dropped")
      else:
        print(f"Warning: {i} from {inputtype} is constant. Variable dropped.")
      dfx=dfx.drop(i,1)
    elif v==0 and not const_found:
      if inputtype=='Y':
        raise RuntimeError('The dependent variable is constant')	
      const_found=True
  return dfx,const_found

def eval_variables(x,df,pd_panel):
  lag_obj=lag_object(pd_panel)
  d={'D':lag_obj.diff,'L':lag_obj.lag,'np':np}	
  for i in df.keys():
    d[i]=df[i]
  for i in x:
    df[i]=eval(i,d)
  return df,lag_obj.max_lags

class lag_object:
  def __init__(self,panel):
    self.panel=panel
    self.max_lags=0

  def lag(self,variable,lags=1):
    x=self.panel[variable.name].shift(lags)
    self.max_lags=max((self.max_lags,lags))
    return x

  def diff(self,variable,lags=1):
    x=self.panel[variable.name].diff(lags)
    self.max_lags=max((self.max_lags,lags))
    return x

def parse_model(model_string,settings):
  for i in ['~','=']:
    if i in model_string:
      split=i
      break
  Y,X=model_string.split(split)
  X=[i.strip() for i in X.split('+')]
  if X==['']:
    X=[]
  if settings.add_intercept.value:
    X=[DEFAULT_INTERCEPT_NAME]+X
  return [Y],X

def get_names(x,inputtype,add_intercept=False,intercept_name=None):
  if x is None:
    r=[]
  elif type(x)==str:
    r=[x]
  elif type(x)==pd.DataFrame:		
    r=x.columns
  elif type(x)==pd.Series:
    r=[x.name]
  else:
    raise RuntimeError(f"Input for {inputtype} needs to be a list or tuple of strings, a pandas DataFrame object or a pandas Series object")
  if add_intercept:
    r=[intercept_name]+r
  return r

def handle_time(ip,df,x):
  if x==[]:
    return [],[]
  x=x[0]
  try:
    x_dt=pd.to_datetime(df[x])
  except ValueError as e:
    try:
      x_dt=pd.to_numeric(x_dt)
    except ValueError as e:
      raise ValueError(f'Expected date or numeric for {inputtype}, but {x} is not recognized as a date or numeric variable by pandas.')
  x_dt=pd.to_numeric(x_dt)/(24*60*60*1000000000)
  x_int=x_dt.astype(int)
  if len(pd.unique(df[x]))==len(pd.unique(x_int)):
    x_dt=x_int
  df[x+NUMERIC_TAG]=x_dt
  return [x+NUMERIC_TAG],[x]


def handle_IDs(ip,df,x):
  if x==[]:
    return [],[]
  x=x[0]
  ids, ip.IDs_unique = pd.factorize(df[x],True)
  df[x+NUMERIC_TAG]=ids
  #both these are true before next assignment:
  #np.all(ip.IDs_unique[ids]==df[x])
  #np.all(np.arange(len(ids))[ids]==ids)
  return [x+NUMERIC_TAG],[x]
