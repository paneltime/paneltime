import numpy as np
import loglikelihood as logl

STPMX=100.0 

class LineSearch:
	def __init__(self, x, comput, mp, panel, step = 1):
		self.alf = 1.0e-3     #Ensures sufficient decrease in function value.
		self.tolx = 1.0e-14  #Convergence criterion on fx.		
		self.step = step
		self.stpmax = STPMX * max((abs(np.sum(x**2)	))**0.5,len(x))
		self.mp = mp
		self.comput = comput
		self.panel = panel

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
		if True:#self.comput.num_hess_count == 0:
			self.ll, self.alam, self.msg = lnsrch_master(self.mp, x, f, dx, self.panel, self.comput.constr, self.alf, slope)
			if not self.ll is None:
				self.x = self.ll.args.args_v
				self.check = 1
				self.f = self.ll.LL
				self.msg = "Convergence on delta dx"
				#print(f'LL mc:{self.ll.LL}')
				self.ll2 = self.ll
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
			if self.k > 0: 
				self.f, self.ll = self.func(self.x) 
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
				#print(f'LL single:{self.ll.LL}')
				#self.ll = self.ll2
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

	def func(self,x):
		ll = logl.LL(x, self.panel, self.comput.constr)
		if ll is None:
			return None, None
		elif ll.LL is None:
			return None, None
		return ll.LL, ll

	
def lnsrch_master(mp,  x, f, dx, panel, constr, alf, slope):
	MAX_ITS = 20
	rec = LikelihoodRecord(mp.master.cpu_count, 0.5, f, x)

	for i in range(MAX_ITS):
		lls, args, lmbds = get_likelihoods(x, dx, f, mp, constr, rec, panel)
		
		if len(lls) > 0:
			if (lls[0] >= f + alf*lmbds[0]*slope):
				break

	if len(lls) == 0:
		ll = logl.LL(x, panel, constr)
		return ll, 0, f'No valid likelihoods found after {MAX_ITS} iterations'
	if lls[0] <= f:
		ll = logl.LL(x, panel, constr)
		return ll, 0, f'No increase in function after {MAX_ITS} iterations and lambda = {min(lmbds)}'

	
	lmd_max = rec.find_max(lmbds[0])
	
	s0 = False
	try:
		ll = logl.LL(x + lmd_max * dx, panel, constr)
		if ll.LL >= lls[0]:
			return ll, lmd_max, f'LL: {ll.LL}'	
	except:
		ll = logl.LL(args[0], panel, constr)
		s0 = True
	if lls[0] > ll.LL:
		ll = logl.LL(args[0], panel, constr)
		if s0:
			raise RuntimeError("This was unexpected")
	return ll, lmbds[0], f'LL: {ll.LL}'		

				
def get_likelihoods(x, dx, f, mp, constr, rec, panel):
	
	lmbds = rec.get_lmbds()
	args = []
	for i in range(len(lmbds)):
		args.append(x + lmbds[i] * dx)
	lls = likelihood_spawn(args, lmbds, mp, constr, f, x, panel)
	rec.add(lls, args, lmbds)
	
	lls, args, lmbds, r = rec.get_sorted_np_arrays()
	return lls, args, lmbds

	
def likelihood_spawn(args, lmds, mp, constr, f, x, panel):
	"Returns a list of a list with the LL of the args and return_info, sorted on LL"
	expr=[]	
	n=len(args)
	mp.send_dict({'constr':constr})
	for i in range(n):
		expr.append(
			"try:\n"
			f"	LL{i}=logl.LL({list(args[i])}, panel,constr).LL\n"
			"except:\n"
			f"	LL{i}=None\n"
			)
		a=0
		
	d = mp.execute(expr)
	
	lls = []
	for i in range(n):
		lls.append(d[f'LL{i}'])

	return lls



	
	
class LikelihoodRecord:
	def __init__(self, n_cores, center, f, x):		
		self.lls = [f]
		self.args = [x]
		self.lmbds = [0]
		self.n_cores = n_cores
		self.center = center
		self.initiated = False
		self.f = f

		
	def set_range(self):
		lls, args, lmbds, r = self.get_sorted_np_arrays()

		if len(lls) == 0 or max(lls) < self.f:
			if self.initiated:
				self.center = self.center/self.n_cores
			else:
				self.initiated = True

		elif len(lls) < 3:
			self.center = lmbds[0]
		else:
			self.center = self.find_max(lmbds[0])
			
			
	def get_lmbds(self):
		self.set_range()
		lmbds = []
		if not 1 in self.lmbds:
			lmbds.append(1)
		if not self.center in self.lmbds:
			lmbds = [self.center]
		for i in range(20):
			d = 2 * self.center/(self.n_cores + i)
			for s in [1, -1]:
				r = np.arange(self.center + s * d, self.center + s*self.center + (s>0) * s * d , s*d)
				r = np.round(r, 
							 - int( np.log10(max(np.abs(r))) ) + 12
							 )
				if not len(np.unique(r)) == len(r):
					ix, v = identify_none_unique(r)
					raise RuntimeError(f"None-unique lamdas {v} at position {ix} has been added")
				for j in r:
					if not ((j in self.lmbds) or (j in lmbds)):
						lmbds.append(j)						
			if len(lmbds) >= self.n_cores:
				lmbds = np.array(lmbds)
				srt = np.argsort(np.abs(lmbds-self.center))
				lmbds = lmbds[srt[:self.n_cores]]
				all_lbmds = list(lmbds) + list(self.lmbds)
				if not len(np.unique(all_lbmds)) == len(all_lbmds):
					ix, v = identify_none_unique(all_lbmds)
					raise RuntimeError(f"None-unique lamdas {v} at position {ix} has been added")
				return lmbds
		raise RuntimeError("Something went wrong!")
			
	def get_sorted_np_arrays(self, offset = 1):
		a = []
		r = np.arange(len(self.lmbds))
		for i in [self.lls[offset:], self.args[offset:], self.lmbds[offset:], r[offset:]]:
			a.append(np.array(i, dtype=float))
		valid = (~np.isnan(a[0]))*(a[0]>self.f)
		for i in range(len(a)):
			a[i] = a[i][valid]
		srt = np.argsort(a[0])[::-1]
		for i in range(len(a)):
			a[i] = a[i][srt]
			
		lls, args, lmbds, r = a
		r = np.array(r, dtype=int)
		return lls, args, lmbds, r
		
		
	def add(self, lls, args, lmbds):
		if (len(lls) != len(args)) or (len(lls) != len(lmbds)):
			raise RuntimeError("Lists have unequal length")		
		for i in range(len(lls)):
			self.lls.append(lls[i])
			self.args.append(args[i])
			self.lmbds.append(lmbds[i])
			
		lmbds = np.array(self.lmbds, dtype=float)
		if not len(np.unique(lmbds)) == len(lmbds):
			raise RuntimeError("None-unique lamdas has been added")

			
			
	def find_max(self, default):
		lls, args, lmbds, r = self.get_sorted_np_arrays(0)
		maxlmd = max((lmbds[:3]))
		minlmd = min((lmbds[:3]))
		
		upper = get_adjacent(r, 1)
		lower = get_adjacent(r, -1)
		
		if (upper is None) or (lower is None):
			upper = 1
			lower = 2

		if (len(lls) < 3):
			return default
		
		f0, l0, f05, l05, f1, l1 = (
			lls[0], lmbds[0], lls[lower], lmbds[lower], lls[upper], lmbds[upper])
		try:
			b=-f0*(l05+l1)/((l0-l05)*(l0-l1))
			b-=f05*(l0+l1)/((l05-l0)*(l05-l1))
			b-=f1*(l05+l0)/((l1-l05)*(l1-l0))
			c=((f0-f1)/(l0-l1)) + ((f05-f1)/(l1-l05))
			c=c/(l0-l05)
			if c<0 and b>0:#concave increasing function
				lmbda = -b/(2*c)
				if lmbda > maxlmd:
					return maxlmd
				elif lmbda < minlmd:
					return minlmd
				return lmbda
			else:
				return default
		except:
			return default	
			

def get_adjacent(r, d):
	if not r[0] + d in r:
		adj = None
	else:
		adj = np.nonzero(r[0] + d == r)[0][0]
	return adj
	
	
def identify_none_unique(a):
	v, ix, c = np.unique(a, return_counts = True,  return_index = True)
	ix = ix[np.nonzero(c>1)[0][0]]
	return ix, v[ix]