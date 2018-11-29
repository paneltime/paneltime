#!/usr/bin/python
# -*- coding: UTF-8 -*-

import DB
#removes all temporary users from database

def main():
	conn,crsr=DB.Connect('OSE')
	sqlstr="""SELECT name AS Login_Name
	FROM sys.server_principals """
	crsr.execute(sqlstr)
	logins=crsr.fetchall()	
	sqlstr="""SELECT session_id,login_name
	FROM  sys.dm_exec_sessions"""	
	crsr.execute(sqlstr)
	procs=crsr.fetchall()
	
	#killing processes
	for session_id,login_name in procs:
		if 'tmp' in login_name:
			try:
				crsr.execute('kill %' %(session_id,))
			except:
				pass			
		
	#removing logins
	for i, in logins:
		if 'tmp:' in i:
			try:
				dropstr="""DROP LOGIN [%s] """ %(i,)
				crsr.execute(dropstr)
				conn.commit()
			except:
				pass
			
			try: 
				dropstr="""DROP USER [%s] """ %(i,)
				crsr.execute(dropstr)
				conn.commit()
			except:
				pass
	pass
			
			
	
	





main()