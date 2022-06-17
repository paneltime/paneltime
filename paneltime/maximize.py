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



def dfpmin(x, comput, callback):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

	comput.LL(x)
	x, ll, f, g, hessin, H = comput.calc_init_dir(x)

	its = 0
	max_iter = 1000
	for its in range(max_iter):  	#Main loop over the iterations.		
		dx, dx_norm = direction.get(g, x, H, comput.constr, hessin, simple=False)
		ls = LineSearch(x, comput.LL)
		ls.lnsrch(x, f, g, dx) 
		
		dx = ls.x - x
		incr = ls.f - f


		x, f, hessin, H, g, conv = comput.exec(dx, dx_norm,  hessin, H, ls.f, ls.x, ls.g, incr, ls.rev, ls.alam, its, ls.ll)
		
		
		callback(msg = ls.msg, dx_norm = dx_norm, f = f, x = x, H = H, g = g, 
				hessin = hessin, dx = dx, incr = incr, rev = ls.rev, alam = ls.alam, 
				its = its, constr = comput.constr, perc = 1.0, task = 'Line search')			
		
		test=np.max(np.abs(dx)) 
		if (test < TOLX):  
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
			return f,x,H,its, 0, ls, msg, dx_norm, incr #FREEALL

		if conv:  
			msg = "Convergence on zero gradient; local or global minimum identified"
			return f,x,H,its,1, ls, msg, dx_norm, incr #FREEALL
	
	msg = "No convergence within %s iterations" %(max_iter,)
	return f,x,H,its,2, ls, msg								#too many iterations in dfpmin				
															#FREEALL


		

	
def maximize(panel, args, callback, comput = None):
	
	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	if comput is None:
		comput = computation.Computation(panel, callback, 1e-10) 
	callback(conv = False, done = False)
	f, x, H, its, conv, ls, msg, dx_norm, incr = dfpmin(args,comput, callback)
	callback(f = f, x = x, H = H, its = its, conv = conv, msg = msg, done = True)
	panel.input.args_archive.save(ls.ll.args,True)
	
	return comput, ls, msg, its, conv, dx_norm, incr


def run(panel, args, callback, mp):
	t0=time.time()
	
	comm  = Comm(panel, args, callback, mp)
	
	callback.print_final(comm.msg, comm.its, comm.incr, comm.f, 1, 'Done', comm.conv, comm.dx_norm, t0, comm.x, comm.ll)
	
	
	return comm.ll, comm.conv, comm.comput.H, comm.comput.g, comm.comput.G



class Comm:
	def __init__(self, panel, args, callback, mp ):
		self.callback = callback
		self.comput = computation.Computation(panel, gtol = -1, tolx = TOLX) 
		callback.set_computation(self.comput)
		self.mp = mp
		self.panel = panel
		self.listen = None
		if not mp is None:
			self.listen = self.mp.listen(
				[f"maximize.maximize(panel, {list(args)}, callback)"])
			self.start_listening()
		else:
			comput, ls, msg, its, conv, dx_norm, incr = maximize(panel, args, callback.generic, self.comput)
			self.msg = msg
			self.f = ls.f
			self.conv = conv
			self.x = ls.x
			self.ll = ls.ll
			self.its = its
			self.incr = incr
			self.dx_norm = dx_norm

	
	def start_listening(self):
		t0 = time.time()
		while not self.listen.done[0]:
			if  time.time()-t0>0.5:
				self.print()
		self.print()

			
	def print(self):
		d = self.listen.inbox[0]
		if not 'g' in d:
			return
		(f, msg , its, incr, x, perc,task, 
		 dx_norm, dx, H, hessin, g, alam, rev, msg, conv, constr) = (d['f'], d['msg'], d['its'], d['incr'], 
						d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
						d['H'], d['hessin'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'])
		
		self.ll = logl.LL(x, self.panel, d['constr'])
		self.comput.exec(dx, dx_norm, hessin, H, f, x, g, incr, rev, alam, its, self.ll, False)
		
		if self.ll.LL!=f:
			raise RuntimeError("thats strange")
		
		self.msg = msg
		self.f = f
		self.conv = conv
		self.x = x
		self.its = its
		self.incr = incr
		self.dx_norm = dx_norm
		
		self.callback.print(msg, its, incr, self.ll, perc , task, dx_norm)



class printout:
	def __init__(self,channel,panel,computation,msg_main,_print=True):
		self._print=_print
		self.channel = channel
		self.computation = computation
		self.msg_main = msg_main


	
	def print_final(self, msg, fret, conv, t0, xsol):
		self.channel.print_final(msg, fret, conv, t0, xsol)
	





