#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:filetype=python:

import DB
import numpy as np

#used to generate a password to user, but this has been replaced by FEIDE authentication.  

def Main():
	conn,crsr=DB.Connect('OSE')
	pw=PasswordGen()

	crsr.execute("""use [master]""")
	crsr.execute("""ALTER LOGIN [OSEuserTest] with  
			PASSWORD='%s'""" %(pw,))
	conn.commit()
	
	MakeNewHTML(pw)
	#MakeTxtFile(pw)
	
	
def MakeNewHTML(pw):
	pw=HTMLCode.replace('PWstrinG',pw)
	topath='C:\inetpub\wwwroot\SAML\www\\pw.php'
	file = open(topath, "wb")
	file.write(pw)
	file.close()
	

def MakeTxtFile(pw):
	topath='C:\inetpub\wwwroot\\rwsd.txt'
	file = open(topath, "wb")
	file.write(pw)
	file.close()
	
	
	
	
def PasswordGen():
	chrset=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
	        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
	        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '$', '%', '&', 
	        '!', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
	        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
	        'y', 'z']
	n=len(chrset)
	pw=''.join([chrset[np.random.randint(n)] for i in range(15)])
	return pw



HTMLCode="""<?php
require_once('../lib/_autoload.php');
$as = new SimpleSAML_Auth_Simple('default-sp');
$isAuth = $as->isAuthenticated();
$attributes = $as->getAttributes();
if ($isAuth) {
	echo 'PWstrinG';
} else {
	echo 'Error';
}
?>""" 

Main()
