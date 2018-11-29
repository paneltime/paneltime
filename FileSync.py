#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
sys.path.append('../Modules')
sys.path.append('F:/Modules')
from ftplib import FTP
import Functions as fu
from Functions import prntout
import numpy as np

curdir=os.path.dirname(os.path.realpath(__file__))
curdir='/'.join(curdir.split('\\')[:-1])

def Sync():
    ftp=FTP('ft.md.oslobors.no','uit','OWX1NHzv')
    s='/data/'
    ftpdir=GetDirTree(ftp)
    addedfiles=[]
    for d,f in ftpdir:
        Syncronize(ftp,d,f,addedfiles)
    ftp.quit()
    return addedfiles

def MakeDir(directory):
    if not os.path.exists(directory):
        if directory!='':
            os.makedirs(directory)	

def Syncronize(ftp,ftpdir,flst,addedfiles):
    MakeDir(curdir+ftpdir)
    for f in flst:
        fpath=ftpdir+'/'+f
        lpath=curdir+fpath
        if IsNewFile(ftp,fpath,lpath) and NotExcepted(lpath):
            fl=open(lpath,'wb')
            prntout('Saving %s' %(fpath,))
            ftp.retrbinary('RETR %s' %(fpath,), fl.write)
            addedfiles.append(fpath)
        else:
            #prntout('%s exists or is excepted' %(fpath,))
            pass


def NotExcepted(fname):
    exceptlst=['2014-2H/equity_pricedump_adjusted.txt',
               '2014-2H/equity_pricedump_adjusted_axess.txt'                 
               ]
    fname=CleanPath(fname)
    if fname[-4:].lower()=='.pdf':
        return False
    for i in exceptlst:
        if i in fname:
            return False
    return True

def CleanPath(path):
    path=path.replace('\\','/')
    path=path.replace('//','/')
    path=path.replace('//','/')
    return path



def IsNewFile(ftp,fpath,lpath):
    if not os.path.isfile(lpath):
        return True
    ls=os.path.getsize(lpath)
    fs=FileSize(ftp,fpath)
    if ls!=fs:
        raise RuntimeError("""FileSync: A version with the same name but different size all ready exist (path: %s).
                            Move or rename the local file""" %(lpath,))
    else:
        return False



def FileSize(ftp,dirstr):
    ftp.sendcmd("TYPE i")    # Switch to Binary mode
    s=ftp.size(dirstr)
    return s

def GetDirTree(ftp):
    dirlst=[]
    GetDirContents(ftp,'',dirlst)
    #fu.WriteCSVMatrixFile('dirlist',dirlst)
    return dirlst


def GetDirContents(ftp,dirstr,dirlist):
    flist=[]
    fl=ftp.nlst(dirstr)
    if not fl is None:
        for f in fl:
            d=dirstr+'/'+f
            if IsFile(ftp,d):
                flist.append(f)
            else:
                GetDirContents(ftp,d,dirlist)
    dirlist.append([dirstr,flist])



def IsFile(ftp,dirstr):
    try:
        f=ftp.size(dirstr)
    except:
        f=None
    return not f is None