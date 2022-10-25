/* File : cfunctions.cpp */




static double* to_c_array(PyObject *pyobj,long* n_ref) {
	
	PyArrayObject *arr=  (PyArrayObject *) PyArray_FROM_OTF(pyobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	
	long n= PyArray_DIM(arr, 0);
	double* a=new double[n];

	for(long i=0;i<n;i++){
		a[i]=*(double *) PyArray_GETPTR1(arr, i);
		
		};
	*n_ref=n;
	return a;

}


double min(double a, double b){
	if(a>b){
		return b;
		}
		else
		{
		return a;
			}
	}
	



void inverse(long n, double *x_args, long nx, double *b_args, long nb, 
				double *a, double *ab) {
	
	long q,k,j,i;
	
	double sum_ax;
	double sum_ab;
	
	a[0]=1.0;
	
	ab[0] = b_args[0];
	*(double *) PyArray_GETPTR1(*ab_matr, 0)=ab[0];

	for(i=1;i<n;i++){
		sum_ax=0;
		sum_ab=0;
		for(j=0;j<min(nx,i);j++){
			sum_ax+=x_args[j]*a[i-j-1];
		}
		a[i]=-sum_ax;
		for(j=0;j<min(nb,i+1);j++){
			sum_ab+=b_args[j]*a[i-j];
		}
		ab[i]=sum_ab;
	}
	
}

;	
	
__declspec(dllexport) int  armas(long *shape, double *lambda, double *rho, double *gamma, double *psi,
				double *AMA_1, double *AMA_1AR, 
				double *GAR_1, double *GAR_1MA, 
				double *u, double *e, double *lnv) {
				
	double x, y, sum;
	long k,j,i, n, q, h;

    n = shape[0];
    nlm = shape[1];
    nrh = shape[2];
    ngm = shape[3];
    npsi = shape[4];
	
	//ofstream printfile("cproblems.txt");

	
	//printfile<<q;
	//printfile<<';';
	//printfile<<h;
	//printfile<<';';
	double* a_A=new double[n];
	double* ab_A=new double[n];
	inverse(lambda, rho, n, AMA_1,AMA_1AR, a_A, ab_A);
	double* a_G=new double[n];
	double* ab_G=new double[n];
	inverse(gamma, psi, n, GAR_1,GAR_1MA, a_G, ab_G);
	
	double* uarr = new double[n];
	double** earr = new double*[h];
	for(k=0;k<h;k++){
		earr[k] =  new double[n];
		};
	
	for(k=0;k<h;k++){//individual dimension
		for(i=0;i<n;i++){//time dimension
			//ARMA:
			if(k==0){
				uarr[i]=(*(double *) PyArray_GETPTR2(*u, k, i));
				}
			sum = 0;
			for(j=0;j<=i;j++){//time dimesion, back tracking
				sum += ab_A[j]*uarr[i-j];
				}
			earr[k][i] = sum;
			*(double *) PyArray_GETPTR2(*e,k,i) = sum;

			//GARCH:
			sum =0;
			for(j=0;j<=i;j++){//time dimension, back tracking
					sum += ab_G[j]*earr[k][i-j]*earr[k][i-j];
				}
			*(double *) PyArray_GETPTR2(*lnv, k,i) = sum;
			}
		}
	//printfile.close();
	for(k=0;k<h;k++){
		delete earr[k];
	};
	delete a_A, ab_A, a_G, ab_G, uarr, earr;
    return 0;
}
	