#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
me='update@titlon.uit.no'
you='espen.sirnes@uit.no'

def SendMail(text,subject):
	msg = MIMEText(text)
	msg['Subject'] = subject
	msg['From'] = me
	msg['To'] = you
	s = smtplib.SMTP(host='smtp-relay.ad.uit.no')
	s.connect(host='smtp-relay.ad.uit.no')
	a=s.sendmail(me, [you], text)
	s.quit()
