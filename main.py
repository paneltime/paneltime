#!/usr/bin/env python
# -*- coding: utf-8 -*-

#executable




import sys
sys.path.append('src/')
import paneltime


args=sys.argv

fname=args[1]
model_string=args[2]
arimagarch=args[3]
arimagarch=arimagarch[1:-1]
arimagarch=','.arimagarch
p, d, q, m, k=[int(i) for i in arimagarch]
group=args[4]
t=args[5]
descr=args[6]



dataframe=paneltime.load(fname=fname)
panel,g,G,H,ll=paneltime.execute(dataframe,model_string,
                                     p, d, q, m, k,group,t,
                                     descr=descr)
paneltime.diagnostics(panel,g,G,H,ll)


