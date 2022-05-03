import numpy as np
import time
import calculus
import calculus_ll as cll
import loglikelihood as logl
import pickle



#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.


ITMAX=1000 
EPS=3.0e-16 
TOLX=(4*EPS) 
STPMX=100.0 


def lnsrch(xold, fold, g, p, stpmax, func, step=1):

	ALF=1.0e-3     #Ensures sufficient decrease in function value.
	TOLX=1.0e-14  #Convergence criterion on fx.
	check=0 
	n=len(xold)
	rev=False

	summ=np.sum(p**2)**0.5
	if summ > stpmax:
		p=p*stpmax/summ 
	slope=np.sum(g*p)					#Scale if attempted step is too big.
	if slope <= 0.0:
		print( "Warning: Roundoff problem in lnsrch")
		p=-p
		slope=np.sum(g*p)
		rev=True
	test=0.0 															#Compute lambda min.
	for i in range(0,n): 
		temp=abs(p[i])/max(abs(xold[i]),1.0) 
		if (temp > test): test=temp 
	alamin=TOLX/test 
	#*******CUSTOMIZATION
	for i in range(1000):#Setting alam so that the largest step is valid. Set func to return None when input is invalid
		alam=0.5**i*step #Always try full Newton step first.
		x=xold+alam*p
		ret=func(x) 
		if ret!=None: break
	f=ret
	#*************************
	f2=0
	alam2=alam
	alamstart=alam#***********CUSTOMIZATION
	for k in range (0,1000):			#Start of iteration loop.
		x=xold+alam*p
		if k>0: f=func(x) 
		if (alam < alamin):   #Convergence on delta p. For zero finding,the calling program should verify the convergence.
			x=xold*1 
			check=1 
			return fold,x,check,k,alam, rev
		elif (f >= fold+ALF*alam*slope): 
			return f ,x,check,k	, alam, rev							#Sufficient function decrease
		else:  															#Backtrack.
			if (alam == alamstart):#***********CUSTOMIZATION  alam == 1.0
				tmplam = -slope/(2.0*(f-fold-slope))  	#First time.
			else:  														#Subsequent backtracks.
				rhs1 = f-fold-alam*slope 
				rhs2=f2-fold-alam2*slope 
				a=(rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2) 
				b=(-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2) 
				if (a == 0.0):
					tmplam = -slope/(2.0*b)  
				else:  
					disc=b*b-3.0*a*slope 
					if (disc < 0.0):
						tmplam=0.5*alam  
					elif (b >= 0.0):
						tmplam=-(b+(disc)**0.5)/(3.0*a) 
					else:
						tmplam=slope/(-b+(disc)**0.5)
				if (tmplam > 0.5*alam): 
					tmplam=0.5*alam   								#  lambda<=0.5*lambda1
		alam2=alam 
		f2 = f 
		alam=max(tmplam,0.1*alam)								#lambda>=0.1*lambda1
		if alamstart<1.0:#*************CUSTOMIZATION
			alam=min((alam,alamstart*0.9**k))
	return f,x,check,k, alam, rev





def dfpmin(p, func, dfunc, callback=None, hessin=None,ddfunc=None,gtol=0,Print=False, step=1):
	"""Given a starting point p[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization"""

	n=len(p)	
	ptmp=np.zeros(n)

	fp=func(p) 										#Calculate starting function value and gradient,
	g=dfunc(p)	
	if ddfunc!=None:
		hessin=ddfunc(p)
	elif hessin==None:
		hessin=-np.diag(np.ones(n))
	stpmax=1000 
	xi=g
	xi, p, fp, g, hessin = calc_init_dir(g, dfunc, func, p)
	summ=np.sum(p**2)											#Initial line direction.
	stpmax=STPMX*max((abs(summ))**0.5,float(n)) 
	k=1
	for its in range(0,ITMAX):  						#Main loop over the iterations.
		fret,x,check, k, alam, rev=lnsrch(p,fp,g,xi,stpmax,func, step) 
		msg, x, fret = callback(fret, x, hessin)
		if msg=='abort': break
		if Print: print( fret)
		pnew=x*1							#The new function evaluation occurs in lnsrch  save the function value in fp for the
		xsol=x*1							#next line search. It is usually safe to ignore the value of check.
		fp = fret 
		xi=pnew-p 							#Update the line direction,
		p=pnew*1 								#and the current point.
		test=np.max(np.abs(xi)/np.maximum(np.abs(ptmp),1.0)) 
		if (test < TOLX):  
			print( "Warning: Convergence on delta p; the gradient is incorrect or the tolerance is set too low")
			Convergence=0
			callback(fret, x, hessin,'finished')
			return fret,xsol,hessin,its, Convergence #FREEALL
		dg=g*1   					#Save the old gradient,
		g=dfunc(p) 										#and get the new gradient.
		test=0.0 											#Test for convergence on zero gradient.
		den=abs(fret)+1e-12
		test=np.max(np.abs(g)*np.abs(p)/den )
		print(f"alam:{alam} k:{k} f: {fret}")
		if (test < gtol):  
			print( "Convergence on zero gradient; local or global minimum identified")
			Convergence=1
			callback(fret, x, hessin,'finished')
			return fret,xsol,hessin,its,Convergence #FREEALL
		hessin=hessin_num(hessin, g, dg, xi)
		#Now calculate the next direction to go,
		xi=-(np.dot(hessin,g.reshape(n,1))).flatten()
		a=0												#and go back for another iteration.
	print( "No convergence within %s iterations" %(ITMAX,))
	Convergence=2
	callback(fret, x, hessin,'finished')
	return fret,xsol,hessin,its,Convergence									#too many iterations in dfpmin				
															#FREEALL

def hessin_num(hessin, g, dg, xi):
	dg=g-dg 				#Compute difference of gradients,
	n=len(g)
	#and difference times current matrix:
	hdg=(np.dot(hessin,dg.reshape(n,1))).flatten()
	fac=fae=sumdg=sumxi=0.0 							#Calculate dot products for the denominators. 
	fac = np.sum(dg*xi) 
	fae = np.sum(dg*hdg)
	sumdg = np.sum(dg*dg) 
	sumxi = np.sum(xi*xi) 
	if (fac < (EPS*sumdg*sumxi)**0.5):  					#Skip update if fac not sufficiently positive.
		fac=1.0/fac 
		fad=1.0/fae 
														#The vector that makes BFGS different from DFP:
		dg=fac*xi-fad*hdg   
		#The BFGS updating formula:
		hessin+=fac*xi.reshape(n,1)*xi.reshape(1,n)
		hessin-=fad*hdg.reshape(n,1)*hdg.reshape(1,n)
		hessin+=fae*dg.reshape(n,1)*dg.reshape(1,n)		
		
	return hessin


class Function:
	def __init__(self,args, panel, f_write, f_read, id):
		self.f_read = f_read
		self.f_write = f_write
		if (not f_read is None) and (args is None):
			command, LL, args, H = self.read()
		if args is None:
			raise RuntimeError('Either f_read or args, or both, must be supplied')
		self.panel = panel
		self.LL(args)
		self.args_init = args
		self.init_LL=self.ll.LL		
		self.gradient = calculus.gradient(panel,set_progress)
		self.t = time.time()
		self.i = id
		
	def read(self):
		
		for i in range(30):
			try:
				f=open(self.f_read, 'rb')
				command, LL, args, H = pickle.load(f)
				break
			except EOFError:
				pass
			f.close()
		f.close()
		print(i)
		return command, LL, args, H 
		
	def write(self, response, fret, x, hessin):
		if time.time() - self.t < 1:
			return
		f=open(self.f_write, 'wb')
		pickle.dump((response, fret, x, hessin), f)
		f.flush()
		f.close()
		self.t = time.time()
		
		
	def LL(self,x, fargs=()):
		self.ll = logl.LL(x, self.panel)
		if self.ll is None:
			return self.init_LL
		elif self.ll.LL is None:
			return self.init_LL
		return self.ll.LL
	
	def jac(self, x, fargs=()):
		dLL_lnv, DLL_e = cll.gradient(self.ll , self.panel)
		return self.gradient.get(self.ll,DLL_e,dLL_lnv)
	
	def callback(self, fret, x, hessin, msg=''):
		if self.f_read is None:
			return '', fret, x
		command, fret_master, x_master, H_master = self.read()
		if command=='abort':
			self.f_read.close()
			self.f_write.close()
			return 'abort', fret, x

		if fret_master>fret:
			x=x_master
			fret=fret_master

		self.write((msg, self.id), fret, x, hessin)
		return '', x, fret
		
		
	
def maximize(panel, args=None, step=1, gtol=0.001, f_write=None, f_read=None, id=0):
	fun=Function(args, panel, f_write, f_read, id)
	t0=time.time()
	fret,xsol,hessin,its,Convergence=dfpmin(fun.args_init,fun.LL, fun.jac, fun.callback, gtol=gtol, step=step)
	print(f"LL={fret}  success={Convergence}  t={time.time()-t0}")
	print(xsol)
	return fret,xsol,hessin,its,Convergence
	
	
def calc_init_dir(g0,dfunc,func,p0):
	"""Calculates the initial direction"""
	n=len(p0)
	xi=np.zeros(n)
	r=np.arange(n)
	d=0.000001
	fps=np.zeros(n)
	ps=[]
	gs=[]
	dgs=[]
	xis=[]
	dgi=np.zeros(n)
	for i in range(len(p0)):
		e = (r==i)*d
		fp = func(p0+e)
		g = dfunc(p0+e)
		dg = (g-g0)/d
		if dg[i]!=0:
			xi[i]  = -g0[i]/dg[i]
			dgi[i] = dg[i]
		fps[i]=fp
		ps.append(p0+e)
		gs.append(g)
		dgs.append(dg)
		xis.append(e)
		#print(f"dg: {dg[i]} g:{g[i]}")
	i = np.argsort(fps)[-1]
	p = ps[i]
	fp = fps[i]
	fp=func(p)
	g = gs[i]
	hessin = 1/(dgi-(dgi==0))
	hessin = np.diag(hessin)
	hessin = hessin*0.75 - np.identity(n)*0.25
	xi=-np.dot(g,hessin)
	return xi,p, fp,g, hessin

	
def set_progress(self,percent=None,text="",task=''):
	return True