/* File : cfunctions.cpp */
#include "stdlib.h"
#include "stdio.h"
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
//#include "cfunctions.h"


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

void inverse(PyObject *py_x_args,PyObject *py_b_args,long n, 
				PyArrayObject** a_matr,PyArrayObject** ab_matr,long vector) {
	
	long q,k,j,i;
	
	double* x_args= to_c_array(py_x_args,&q);
	double* b_args= to_c_array(py_b_args,&k);

	double* a=new double[n];
	double* ab=new double[n];
	double sum_ax;
	double sum_ab;
	
	a[0]=1.0;

	for(i=0;i<n;i++){
		sum_ax=0;
		sum_ab=0;
		if(i>0){
			for(j=0;j<min(q,i);j++){
				sum_ax+=x_args[j]*a[i-j-1];
			}
			a[i]=-sum_ax;
		}
		for(long j=0;j<min(k,i+1);j++){
			sum_ab+=b_args[j]*a[i-j];
			ab[i]=sum_ab;
		}
		if (vector==0){
			for(j=0;j<n-i;j++){
				*(double *) PyArray_GETPTR2(*a_matr,  i+j, j)=a[i];
				*(double *) PyArray_GETPTR2(*ab_matr, i+j, j)=ab[i];
				}
		}else{
			*(double *) PyArray_GETPTR1(*a_matr,  i)=a[i];
			*(double *) PyArray_GETPTR1(*ab_matr, i)=ab[i];
		}
	}
		

	delete a,ab,b_args,x_args;
	
}



static PyObject *bandinverse(PyObject *self, PyObject *args) {
	
	
	PyObject *rho,*lambda,*psi,*gamma;
	PyObject* AMA_1_in;
	PyObject* AMA_1AR_in;
	PyObject* GAR_1_in;
	PyObject* GAR_1MA_in;

	long n,vector;
	
	
	if (!PyArg_ParseTuple(args, "OOOOlOOOOl", &lambda,&rho,&gamma,&psi,&n,&AMA_1_in,&AMA_1AR_in,&GAR_1_in,&GAR_1MA_in,&vector)){
		PyErr_SetString(PyExc_ValueError,"Error in parsing arguments");
		return NULL;
		};
	
	PyArrayObject* AMA_1 	=  (PyArrayObject *) PyArray_FROM_OTF(AMA_1_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* AMA_1AR =  (PyArrayObject *) PyArray_FROM_OTF(AMA_1AR_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* GAR_1 	=  (PyArrayObject *) PyArray_FROM_OTF(GAR_1_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* GAR_1MA =  (PyArrayObject *) PyArray_FROM_OTF(GAR_1MA_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	
	inverse(lambda,rho,n,&AMA_1,&AMA_1AR,vector);
	inverse(gamma,psi,n,&GAR_1,&GAR_1MA, vector);	
	
	Py_DECREF(AMA_1);
	Py_DECREF(AMA_1AR);
	Py_DECREF(GAR_1);
	Py_DECREF(GAR_1MA);

	Py_DECREF(rho);
	Py_DECREF(lambda);
	Py_DECREF(psi);
	Py_DECREF(gamma);
	
	
	return Py_BuildValue("i",1);
		
}



static PyObject *banddot(PyObject *self, PyObject *args) {
	

	PyObject* band_in;
	PyObject* M_in;
	PyObject* X_in;

	double sum;
	long i,j,m;
	
	
	if (!PyArg_ParseTuple(args, "OOO", &band_in,&M_in,&X_in)){
		PyErr_SetString(PyExc_ValueError,"Error in parsing arguments");
		return NULL;
		};
		
	PyArrayObject* X =  (PyArrayObject *) PyArray_FROM_OTF(X_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* band =  (PyArrayObject *) PyArray_FROM_OTF(band_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* M =  (PyArrayObject *) PyArray_FROM_OTF(M_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	long n= PyArray_DIM(M, 0);
	long k= PyArray_DIM(M, 1);

	
	for(i=0;i<k;i++){
		for(j=0;j<n;j++){
			sum=0;
			for(m=0;m<=j;m++){
				sum+=(*(double *) PyArray_GETPTR1(band, j-m))*(*(double *) PyArray_GETPTR2(M, j, i));
			}
			*(double *) PyArray_GETPTR2(X, j, i)=sum;
		}
	}
	


	
	Py_DECREF(band);
	Py_DECREF(M);
	Py_DECREF(X);

	
	return Py_BuildValue("i",1);
		
}



static PyObject *dot(PyObject *self, PyObject *args) {
	//less efficient than numpy, so not in use
	//Must delete objects to use
	PyObject* A_in;
	PyObject* B_in;
	PyObject* d_in;
	PyObject* ret_in;

	long i,j,n,t,k,m,q,p,s;
	
	if (!PyArg_ParseTuple(args, "OOOO", &A_in,&B_in,&d_in,&ret_in)){
		PyErr_SetString(PyExc_ValueError,"Error in parsing arguments");
		return NULL;
		};
	
	PyArrayObject* A 	=  (PyArrayObject *) PyArray_FROM_OTF(A_in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* B	=  (PyArrayObject *) PyArray_FROM_OTF(B_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* d 	=  (PyArrayObject *) PyArray_FROM_OTF(d_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* ret 	=  (PyArrayObject *) PyArray_FROM_OTF(ret_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	n=(long)*(double *) PyArray_GETPTR1(d, 0);
	t=(long)*(double *) PyArray_GETPTR1(d, 1);
	k=(long)*(double *) PyArray_GETPTR1(d, 2);
	m=(long)*(double *) PyArray_GETPTR1(d, 3);
	

	double sum;

	for(i=0;i<n;i++){
		for(j=0;j<k;j++){
			for(q=0;q<m;q++){
				for(p=0;p<t;p++){
					sum=0;
					for(s=0;s<t;s++){
						sum+=(*(double *) PyArray_GETPTR2(A, p, s))*(*(double *) PyArray_GETPTR4(B, i, s, j,q));
						
					}
					*(double *) PyArray_GETPTR4(ret, i, p, j,q)=sum;
				}
			}
		}
	}
	return Py_BuildValue("l",1);
		
}




static PyMethodDef cmethods[] = {

	{"bandinverse",  bandinverse, METH_VARARGS,
	"Execute a shell command."},
	{"dot",  dot, METH_VARARGS,
	"Execute a shell command."},
	{"bdot",  banddot, METH_VARARGS,
	"Execute a shell command."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"cfunctions", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	cmethods
};


PyMODINIT_FUNC PyInit_cfunctions(void)
{
	import_array();
	return PyModule_Create(&moduledef);
	

}

