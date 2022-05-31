import numpy as np
import time
import calculus
import calculus_ll as cll
import loglikelihood as logl
import computation
import direction



#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
STPMX=100.0 


class LineSearch:
	def __init__(self, step, x, func):
		self.alf = 1.0e-3     #Ensures sufficient decrease in function value.
		self.tolx = 1.0e-14  #Convergence criterion on fx.		
		self.step = step
		self.stpmax = STPMX*max((abs(np.sum(x**2)	))**0.5,len(x))
		self.func = func

	def lnsrch(self, x, f, g, dx):
		
		#(x, f, g, dx) 

		self.check=0
		self.g = g
		n=len(x)
		self.rev = False
		if f is None:
			raise RuntimeError('f cannot be None')
	
		summ=np.sum(dx**2)**0.5
		if summ > self.stpmax:
			dx = dx*self.stpmax/summ 
		slope=np.sum(g*dx)					#Scale if attempted step is too big.
		if slope <= 0.0:
			print( "Warning: Roundoff problem in lnsrch")
			dx=-dx
			slope=np.sum(g*dx)
			self.rev = True
		test=0.0 															#Compute lambda min.
		for i in range(0,n): 
			temp=abs(dx[i])/max(abs(x[i]),1.0) 
			if (temp > test): test=temp 
		alamin = self.tolx/test 
		#*******CUSTOMIZATION
		for i in range(1000):#Setting alam so that the largest step is valid. Set func to return None when input is invalid
			self.alam = 0.5**i*self.step #Always try full Newton step first.
			self.x = x + self.alam * dx
			self.f = self.func(self.x) 
			if self.f != None: break
		#*************************
		f2=0
		alam2 = self.alam
		alamstart = self.alam#***********CUSTOMIZATION
		for self.k in range (0,1000):			#Start of iteration loop.
			self.x = x + self.alam * dx			
			if self.k > 0: self.f=self.func(self.x) 
			if self.f is None:
				print('The function returned None')
				self.f = f
			if (self.alam < alamin):   #Convergence on delta dx. For zero finding,the calling program should verify the convergence.
				self.x = x*1 
				self.check = 1
				self.f = f
				return
			elif (self.f >= f+self.alf*self.alam*slope): 
				return							#Sufficient function decrease
			else:  															#Backtrack.
				if (self.alam == alamstart):#***********CUSTOMIZATION  alam == 1.0
					tmplam = -slope/(2.0*(self.f-f-slope))  	#First time.
				else:  														#Subsequent backtracks.
					rhs1 = self.f-f-self.alam*slope 
					rhs2 = f2-f-alam2*slope 
					a=(rhs1/(self.alam**2)-rhs2/(alam2*alam2))/(self.alam-alam2) 
					b=(-alam2*rhs1/(self.alam**2)+self.alam*rhs2/(alam2*alam2))/(self.alam-alam2) 
					if (a == 0.0):
						tmplam = -slope/(2.0*b)  
					else:  
						disc=b*b-3.0*a*slope 
						if (disc < 0.0):
							tmplam = 0.5*self.alam  
						elif (b >= 0.0):
							tmplam=-(b+(disc)**0.5)/(3.0*a) 
						else:
							tmplam=slope/(-b+(disc)**0.5)
					if (tmplam > 0.5*self.alam): 
						tmplam = 0.5*self.alam   								#  lambda<=0.5*lambda1
			alam2 = self.alam 
			f2 = self.f
			self.alam = max(tmplam, 0.1*self.alam)								#lambda>=0.1*lambda1
			if alamstart<1.0:#*************CUSTOMIZATION
				self.alam = min((self.alam, alamstart*0.9**self.k))





def dfpmin(x, comput, gtol=0, step=1, tolx=TOLX, max_iter=1e+300):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

	comput.LL(x)
	x, f, g, hessin = comput.calc_init_dir(x)

	H = None
	its = 0
	while True:  	#Main loop over the iterations.		
		dx, dx_norm = direction.get(g, x, H, None, hessin, simple=True)
		ls = LineSearch(step, x, comput.LL)
		ls.lnsrch(x, f, g, dx) 
		
		dx = ls.x - x
		x, f, hessin, H, g = comput.exec(dx, hessin, H, its, ls, ls.f - f, False,  '')
		x, f, hessin, H, g = comput.slave_callback(x, f, hessin, H, g, False)	
		
		test=np.max(np.abs(dx)) 
		if (test < TOLX):  
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
			x, f, hessin, H, g = comput.slave_callback(x, f, hessin, H, g, True)
			return f,x,hessin,its, 0 #FREEALL
		if its>=max_iter:
			msg = f"Maximum iterations of {its} reached"
			x, f, hessin, H, g = comput.slave_callback(x, f, hessin, H, g, True)
			return f,x,hessin,its, 0 #FREEALL			

		test=np.max(np.abs(g)*np.abs(x)/(abs(f)+1e-12) )
		if (test < gtol):  
			msg = "Convergence on zero gradient; local or global minimum identified"
			x, f, hessin, H, g = comput.slave_callback(x, f, hessin, H, g, True)
			return f,x,hessin,its,1 #FREEALL
		
		its += 1											#and go back for another iteration.
	print( "No convergence within %s iterations" %(max_iter,))
	return f,x,hessin,its,2									#too many iterations in dfpmin				
															#FREEALL


		

	
def maximize(panel, args, callback=None, step=1, gtol=0, id=0, tolx=TOLX, 
			 max_iter=1e+300, ll=None, mp_callback=None):
	
	
	comput = computation.Computation(panel, callback, id, mp_callback) 

	t0=time.time()
	fret,xsol,hessin,its,Convergence=dfpmin(args,comput, gtol=gtol, step=step, tolx=tolx,  max_iter=max_iter)
	print(f"LL={fret}  success={Convergence}  t={time.time()-t0}")
	print(xsol)
	return fret,xsol,hessin,its,Convergence, comput.ll
	
def test(i, callback, msg):
	d_master=None
	for i in range(5):
		time.sleep(np.random.rand()*5)
		d ={'message':msg, 'node':i, 'rand':np.random.rand(), 'master':d_master}
		d_master  = callback(d)
	
	
	
	
def calc_init_dir(g0,dfunc,func,p0):
	"""Calculates the initial computation"""
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