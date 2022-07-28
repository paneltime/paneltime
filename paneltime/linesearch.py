import numpy as np
import loglikelihood as logl

STPMX=100.0 

class LineSearch:
	def __init__(self, x, func, mp, step = 1):
		self.alf = 1.0e-3     #Ensures sufficient decrease in function value.
		self.tolx = 1.0e-14  #Convergence criterion on fx.		
		self.step = step
		self.stpmax = STPMX * max((abs(np.sum(x**2)	))**0.5,len(x))
		self.func = func
		self.mp = mp

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
		#multithread:
		if False:
			self.ll, self.alam = lnsrch_master(mp, x, f, g, dx, self.panel, constr)
			if self.alam < alamin and (not self.ll is None):
					self.x = self.ll.args.args_v
					self.check = 1
					self.f = self.ll.LL
					self.msg = "Convergence on delta dx"
					return		
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
			if (self.alam < alamin):   #Convergence on delta x. For zero finding,the calling program should verify the convergence.
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



		
def solve_square_func(f0,l0,f05,l05,f1,l1):
	try:
		b=-f0*(l05+l1)/((l0-l05)*(l0-l1))
		b-=f05*(l0+l1)/((l05-l0)*(l05-l1))
		b-=f1*(l05+l0)/((l1-l05)*(l1-l0))
		c=((f0-f1)/(l0-l1)) + ((f05-f1)/(l1-l05))
		c=c/(l0-l05)
		if c<0 and b>0:#concave increasing function
			return -b/(2*c)
		else:
			return None
	except:
		return None
	

	
def lnsrch_master(mp,  x, f, g, dx, panel, constr):
	start=0
	end=2.0
	max_its = 10

	for i in range(max_its):
		delta=(end-start)/(mp.master.cpu_count-1)
		lls, args, lmbds = get_likelihoods(x, dx, panel, mp,delta,start,constr)	
		if len(lls)>0:
			if lls[0]>f:
				break
		else:
			start=delta/mp.master.cpu_count
			end=delta
	if len(lls) == 0:
		return None, None
	if lls[0] <= f:
		return None, None
	if len(lls) > 2:
		try:
			lmda = solve_square_func(lls[0], lmbds[0], lls[1], lmbds[1],lls[2], lmbds[2])
			1/lmda#Throws an exception if zero or None
			ll2 = logl.LL(args+lmda*dx,panel,constr)
			if ll2.LL<f:
				raise RuntimeError("Couldn't find max")
		except:
			ll2 = logl.LL(args[0],panel,constr)
			return ll2, lmbds[0]
	return ll2,lmda

def remove_nan(res):
	r=[]
	for i in res:
		if not np.isnan(i[0]):
			r.append(i)
	return np.array(r)	
	
				
def get_likelihoods(args, dx,panel,mp,delta,start, constr):
	lamdas=[]
	a=[]
	
	for i in range(mp.master.cpu_count):
		lmda=start+i*delta
		a.append(list(args+lmda*dx))
		lamdas.append(lmda)
	lls, args, lambdas=likelihood_spawn(a,lamdas,mp, constr)
	return lls, args, lambdas

	
def likelihood_spawn(args, lmds, mp, constr):
	"Returns a list of a list with the LL of the args and return_info, sorted on LL"
	expr=[]	
	n=len(args)
	mp.send_dict({'constr':constr})
	for i in range(n):
		expr.append([
			"try:\n"
			f"	f{i}=lgl.LL({list(args[i])}, panel,constr)\n"
			f"	LL{i}=f{i}.LL\n"
			"except:\n"
			f"	LL{i}=None\n"
			])
		a=0
	d = mp.execute(expr)
	lls = []
	for i in range(n):
		key=f'LL{i}'
		if not d[key] is None:
			#appends LL, node
			lls.append(d[key]['LL'])
	if len(lls)==0:
		return [], []
	srt=np.argsort(lls)[::-1]
	return lls[srt], args[srt], lmbds[srt]

	
	