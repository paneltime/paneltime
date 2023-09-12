import numpy as np
cimport numpy as cnp
cnp.import_array()

DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

cdef extern from "math.h":
	double log(double)

cdef void inverse(long n,  cnp.ndarray x_args, long nx, cnp.ndarray b_args, long nb, 
				cnp.ndarray a, cnp.ndarray ab):
	
	cdef long j,i
	
	cdef double sum_ax
	cdef double sum_ab
	
	for i in range(n):
		a[i]=0.0
	a[0]=1.0
	ab[0] = b_args[0]

	for i in range(1,n):
		sum_ax=0
		sum_ab=0
		for j in range(i):
			if j>=nx: break
			sum_ax+=x_args[j]*a[i-j-1]
		a[i]=-sum_ax;
		for j in range(i+1):
			if j>=nb: break
			sum_ab+=b_args[j]*a[i-j]

		ab[i]=sum_ab;

def armas(cnp.ndarray parameters,cnp.ndarray   lmbda,cnp.ndarray   rho,
		cnp.ndarray   gamma, cnp.ndarray  psi,
		cnp.ndarray  AMA_1, cnp.ndarray  AMA_1AR, 
		cnp.ndarray  GAR_1, cnp.ndarray  GAR_1MA, 
		cnp.ndarray u, cnp.ndarray e, cnp.ndarray var,  
		cnp.ndarray h, cnp.ndarray W, cnp.ndarray T_array):
				
	cdef double sum, esq
	cdef long k,j,i

	cdef long N = <long> parameters[0]
	cdef long T = <long> parameters[1]
	cdef long nlm = <long> parameters[2]
	cdef long nrh = <long> parameters[3]
	cdef long ngm = <long> parameters[4]
	cdef long npsi = <long> parameters[5]
	cdef long egarch = <long> parameters[6]
	cdef long lost_obs = <long> parameters[7]
	cdef double h_add = parameters[8]
	cdef long rw

	inverse(T, lmbda, nlm, rho, nrh, AMA_1, AMA_1AR);

	inverse(T, gamma, ngm, psi, npsi, GAR_1, GAR_1MA);
	

	for k in range(N):#individual dimension
		for i in range(<long> T_array[k]):#time dimension
			sum = 0;
			for j in range(i+1): #time dimesion, back tracking
				sum += AMA_1AR[j]*u[(i-j) + k*T]
			e[i + k*T] = sum
			#GARCH:
			if(i>=lost_obs):
				h[i + k*T] = sum*sum
				h[i + k*T] += h_add
				if(egarch):
					h[i + k*T] = log((h[i + k*T]) + (h[i + k*T]==0)*1e-18);
				
			
			sum =0;
			for j in range(i+1):#//time dimension, back tracking
				sum += GAR_1[j] * W[(i-j) + k*T] + GAR_1MA[j]*h[(i-j) + k*T]
			var[i + k*T] = sum

	return 0

	