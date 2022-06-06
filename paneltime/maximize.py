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
	def __init__(self, x, func, step = 1):
		self.alf = 1.0e-3     #Ensures sufficient decrease in function value.
		self.tolx = 1.0e-14  #Convergence criterion on fx.		
		self.step = step
		self.stpmax = STPMX * max((abs(np.sum(x**2)	))**0.5,len(x))
		self.func = func

	def lnsrch(self, x, f, g, dx):
		
		#(x, f, g, dx) 

		self.check=0
		self.g = g
		self.msg = ""
		n=len(x)
		self.rev = False
		if f is None:
			raise RuntimeError('f cannot be None')
	
		summ=np.sum(dx**2)**0.5
		if summ > self.stpmax:
			dx = dx*self.stpmax/summ 
		slope=np.sum(g*dx)					#Scale if attempted step is too big.
		if slope <= 0.0:
			self.msg = "Roundoff problem"
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
			self.f, self.ll = self.func(self.x) 
			if self.f != None: break
		#*************************
		f2=0
		alam2 = self.alam
		alamstart = self.alam#***********CUSTOMIZATION
		max_iter = 1000
		for self.k in range (0,max_iter):			#Start of iteration loop.
			self.x = x + self.alam * dx			
			if self.k > 0: self.f, self.ll = self.func(self.x) 
			if self.f is None:
				print('The function returned None')
				self.f = f
			if (self.alam < alamin):   #Convergence on delta dx. For zero finding,the calling program should verify the convergence.
				self.x = x*1 
				self.check = 1
				self.f = f
				self.msg = "Convergence on delta dx"
				return
			elif (self.f >= f+self.alf*self.alam*slope): 
				self.msg = "Sufficient function increase"
				return							#Sufficient function increase
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
				
			self.msg = f"No function increase after {max_iter} iterations"



def dfpmin(x, comput, print_func):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

	comput.LL(x)
	x, f, g, hessin, H = comput.calc_init_dir(x)

	its = 0
	max_iter = 1000
	for its in range(max_iter):  	#Main loop over the iterations.		
		dx, dx_norm = direction.get(g, x, H, comput.constr, hessin, simple=False)
		ls = LineSearch(x, comput.LL)
		ls.lnsrch(x, f, g, dx) 
		
		dx = ls.x - x
		incr = ls.f - f
		
		print_func(ls.msg, its, incr, ls.ll, 1.0 , 0, 'Line search', dx_norm)
		
		x, f, hessin, H, g, conv = comput.exec(dx, hessin, H, its, ls ,incr , False)
		
		print_func("Done", its, dx, ls.ll, 1.0 , 1, 'Derivatives', dx_norm)
	
		test=np.max(np.abs(dx)) 
		if (test < TOLX):  
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
			return f,x,hessin,its, 0, ls, msg #FREEALL


		if conv:  
			msg = "Convergence on zero gradient; local or global minimum identified"
			return f,x,hessin,its,1, ls, msg #FREEALL
	
	msg = "No convergence within %s iterations" %(max_iter,)
	return f,x,hessin,its,2, ls, msg								#too many iterations in dfpmin				
															#FREEALL


		

	
def maximize(panel, args,callback = None, msg_main = ""):
	
	
	comput = computation.Computation(panel, callback.set_progress, 1e-10, TOLX) 
	callback.set_computation(comput, msg_main)
	
	t0=time.time()
	fret,xsol,hessin,its, conv, ls, msg=dfpmin(args,comput, callback.print)
	callback.print_final(msg, fret, conv, t0, xsol)
	return ls.ll, comput, conv




class printout:
	def __init__(self,channel,panel,computation,msg_main,_print=True):
		self._print=_print
		self.channel = channel
		self.computation = computation
		self.msg_main = msg_main

	def print(self, msg, its, incr, ll, percent , update_type, task, dx_norm):
		if not self._print:
			return
		if not self.channel.output_set:
			self.channel.set_output_obj(ll, self.computation, self.msg_main, dx_norm)
		ok = self.channel.set_progress(percent ,msg ,task=task)
		if update_type>0:
			self.channel.update_after_direction(self.computation,its, dx_norm)
		elif update_type==0 or update_type==2:
			self.channel.update_after_linesearch(self.computation,ll,incr, dx_norm)
		return ok
	
	def print_final(self, msg, fret, conv, t0, xsol):
		self.channel.print_final(msg, fret, conv, t0, xsol)
	
