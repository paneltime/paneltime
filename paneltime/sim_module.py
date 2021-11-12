test_script="""
import paneltime as pt
import scipy
import numpy as np

#Defining sample
N=100
T=1000
k=5
I=np.diag(np.ones(T))
zero=I*0

#Creating conversion matrices:
def lag_matr(L,args):
	import numpy as np
	k=len(args)
	L=1*L
	r=np.arange(len(L))
	for i in range(k):
		d=(r[i+1:],r[:-i-1])
		L[d]=args[i]
	return L

M_AR=-lag_matr(-I,[0.5,-0.5]) #AR means process
M_MA=lag_matr(I,[0.5,-0.5]) #MA means process
M_MA_1=np.linalg.inv(M_MA)
M_MA_1AR=np.dot(M_MA_1,M_AR)
M_AR_1MA=np.linalg.inv(M_MA_1AR)

V_AR=-lag_matr(-I,[0.5,-0.5]) #MA variance process
V_MA=lag_matr(zero,[0.5,-0.5]) #AR variance process
V_AR_1=np.linalg.inv(V_AR)
V_AR_1MA=np.dot(V_AR_1,V_MA)

#Creating random variables:
def RE_Het_AC_ify(X):
	"Returns X after RE, ARIMA and GARCH "
	import numpy as np
	X_RE_t=np.random.normal(size=(T,1,1))
	X_RE_i=np.random.normal(size=(1,N,1))
	X_RE=X+X_RE_t+X_RE_i
	X_RE_Het=X_RE*sigma
	X_RE_Het_AC=np.array([
            np.dot(M_MA_1AR,X_RE_Het[:,i,:]) 
            for i in range(N)
            ])
	X_RE_Het_AC=np.swapaxes(X_RE_Het_AC, 0,1)
	return X_RE_Het_AC

#residuals after FE/RE:
epsilon=np.random.normal(size=(T,N,1))

#GARCH volatility:
e_var=np.random.normal(size=(T,N,1))**2
ln_sigma2=np.array([
            np.dot(M_MA_1AR,
                   np.log(e_var[:,i,:]+1e-10)) 
            for i in range(N)
            ])
ln_sigma2=np.swapaxes(ln_sigma2, 0,1)
sigma=np.exp(ln_sigma2)**2
X=RE_Het_AC_ify(np.random.normal(size=(T,N,k)))
u=RE_Het_AC_ify(np.random.normal(size=(T,N,1)))
#Assuming unit betas
Y=np.sum(X,2).reshape((T,N,1))+u
IDs=np.arange(N).reshape((1,N,1))*np.ones((T,1,1))
dates=np.arange(T).reshape((T,1,1))*np.ones((1,N,1))

Y=np.concatenate(Y,0).reshape((N*T,1))
X=np.concatenate(X,0).reshape((N*T,k))
IDs=np.concatenate(IDs,0).reshape((N*T,1))
dates=np.concatenate(dates,0).reshape((N*T,1))

#running the estimation:
#pt.options.loadargs.set(1)
#pt.options.variance_RE_norm.set(0.000005)
pt.options.fixed_random_variance_eff.set(0)
df={'X0':X[:,0],'X1':X[:,1],'X2':X[:,2],'X3':X[:,3],'X4':X[:,4], 
    'Y':Y.flatten(), 'IDs':IDs.flatten(), 'dates':dates.flatten()}
df=pd.DataFrame(df)
pt.execute("Y~X1+X2+X3+X4",df,T='dates',ID="IDs")
"""
