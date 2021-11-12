import numpy as np
import ctypes as ct
import cfunctions as c
import time

gamma=np.array([0.5,-0.5])
lambda_=np.array([0.5,-0.5])
rho=np.array([0.5,-0.5])
psi=np.array([0.5,-0.5])
rho=np.insert(-rho,0,1)
psi=np.insert(psi,0,0) 


n=5000
k=1000

M=np.random.rand(n,k)
X=np.zeros((n,k))


AMA_1m,AMA_1ARm,GAR_1m,GAR_1MAm=(
	np.diag(np.ones(n)),
	np.zeros((n,n)),
	np.diag(np.ones(n)),
	np.zeros((n,n))
)


t0=time.time()
c.bandinverse(lambda_,rho,gamma,psi,n,AMA_1m,AMA_1ARm,GAR_1m,GAR_1MAm,0)
print(f"old_inv:{time.time()-t0}")
t0=time.time()
X_s=np.dot(AMA_1m, M)
print(f"old_mult:{time.time()-t0}")
a=0

AMA_1,AMA_1AR,GAR_1,GAR_1MA=(
	np.append([1],np.zeros(n-1)),
	np.zeros(n),
	np.append([1],np.zeros(n-1)),
	np.zeros(n),
)
t0=time.time()
c.bandinverse(lambda_,rho,gamma,psi,n,AMA_1,AMA_1AR,GAR_1,GAR_1MA,1)
print(f"new_inv:{time.time()-t0}")
t0=time.time()
A=AMA_1.reshape((n,1))

A=[np.roll(A,i) for i in range(n)]
X_d=np.dot(np.tril(np.concatenate(A,1)), M)
print(f"np_mult:{time.time()-t0}")


a=np.arange(n)
A=np.tril(np.array([np.roll(a,i) for i in range(n)]).T)
nz=np.nonzero(A+np.diag(np.ones(n)))
R=np.zeros((n,n))
az=A[nz]
t0=time.time()
R[nz]=AMA_1[az]
Rc=np.array(R)
X_c=np.dot(Rc, M)
print(f"new_mult with copy:{time.time()-t0}")
t0=time.time()
R[nz]=AMA_1[az]
X_c=np.dot(R, M)
print(f"new_mult:{time.time()-t0}")
t0=time.time()
R[nz]=AMA_1[az]
Rc=np.array(R)
print(f"create copy:{time.time()-t0}")
a=0