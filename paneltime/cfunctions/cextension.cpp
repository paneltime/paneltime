/* File : cextension.cpp */
#include "stdlib.h"
#include "stdio.h"
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <iostream>
#include <fstream>
using namespace std;



using namespace std;
extern "C" 

{int multiply(int *arr_in, int factor, int *arr_out, unsigned int *shape) {

unsigned int num_rows, num_cols, row, col;

num_rows = shape[0];
num_cols = shape[1];

for (row=0; row<num_rows; row++) {
	for (col=0; col<num_cols; col++) {
	arr_out[row*num_cols + col] = factor*arr_in[row*num_cols + col];
	}
}

return 0;
};

}



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
				PyArrayObject** a_matr,PyArrayObject** ab_matr, 
				double *a, double *ab) {
	
	long q,k,j,i;
	
	double* x_args= to_c_array(py_x_args,&q);
	double* b_args= to_c_array(py_b_args,&k);

	double sum_ax;
	double sum_ab;
	
	a[0]=1.0;
	
	ab[0] = b_args[0];
	*(double *) PyArray_GETPTR1(*ab_matr, 0)=ab[0];

	for(i=1;i<n;i++){
		sum_ax=0;
		sum_ab=0;
		for(j=0;j<min(q,i);j++){
			sum_ax+=x_args[j]*a[i-j-1];
		}
		a[i]=-sum_ax;
		for(j=0;j<min(k,i+1);j++){
			sum_ab+=b_args[j]*a[i-j];
		}
		ab[i]=sum_ab;
		
		*(double *) PyArray_GETPTR1(*a_matr,  i)=a[i];
		*(double *) PyArray_GETPTR1(*ab_matr, i)=ab[i];

	}
		
	delete b_args,x_args;
	
}

;	
	
void armas(long n, PyObject *lambda,PyObject *rho, PyObject *gamma,PyObject *psi,
				PyArrayObject** AMA_1,PyArrayObject** AMA_1AR, 
				PyArrayObject** GAR_1,PyArrayObject** GAR_1MA, 
				PyArrayObject** u, PyArrayObject** e, PyArrayObject** lnv) {
				
	double x, y, sum;
	long k,j,i;

	
	//ofstream printfile("cproblems.txt");

	long q = PyArray_DIM(*u, 1);
	long h = PyArray_DIM(*u, 0);

	
	//printfile<<q;
	//printfile<<';';
	//printfile<<h;
	//printfile<<';';
	double* a_A=new double[n];
	double* ab_A=new double[n];
	inverse(lambda,rho,n,AMA_1,AMA_1AR, a_A, ab_A);
	double* a_G=new double[n];
	double* ab_G=new double[n];
	inverse(gamma,psi,n,GAR_1,GAR_1MA, a_G, ab_G);
	
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
}
	
	

static PyObject *arma_arrays(PyObject *self, PyObject *args) {
	
	
	PyObject *rho,*lambda,*psi,*gamma;
	PyObject* AMA_1_in;
	PyObject* AMA_1AR_in;
	PyObject* GAR_1_in;
	PyObject* GAR_1MA_in;
	PyObject* u_in;
	PyObject* e_in;
	PyObject* lnv_in;

	long n;
	
	
	if (!PyArg_ParseTuple(args, "OOOOlOOOOOOO", &lambda,&rho,&gamma,&psi,&n,&AMA_1_in,&AMA_1AR_in,&GAR_1_in,&GAR_1MA_in,
												&u_in, &e_in, &lnv_in
												)){
		PyErr_SetString(PyExc_ValueError,"Error in parsing arguments");
		return NULL;
		};
	
	PyArrayObject* AMA_1 		=  (PyArrayObject *) PyArray_FROM_OTF(AMA_1_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* AMA_1AR 		=  (PyArrayObject *) PyArray_FROM_OTF(AMA_1AR_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* GAR_1 		=  (PyArrayObject *) PyArray_FROM_OTF(GAR_1_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* GAR_1MA 		=  (PyArrayObject *) PyArray_FROM_OTF(GAR_1MA_in, 	NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* u 			=  (PyArrayObject *) PyArray_FROM_OTF(u_in, 		NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* e 			=  (PyArrayObject *) PyArray_FROM_OTF(e_in, 		NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject* lnv 			=  (PyArrayObject *) PyArray_FROM_OTF(lnv_in, 		NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	
	armas(n,lambda,rho, gamma,psi,
				&AMA_1, &AMA_1AR, 
				&GAR_1, &GAR_1MA, 
				&u, &e, &lnv);
	
	Py_DECREF(AMA_1);
	Py_DECREF(AMA_1AR);
	Py_DECREF(GAR_1);
	Py_DECREF(GAR_1MA);
	Py_DECREF(u);
	Py_DECREF(e);
	Py_DECREF(lnv);

	Py_DECREF(rho);
	Py_DECREF(lambda);
	Py_DECREF(psi);
	Py_DECREF(gamma);
	
	
	
	return Py_BuildValue("i",1);
		
}




static PyMethodDef cmethods[] = {

	{"arma_arrays",  arma_arrays, METH_VARARGS,
	"Execute a shell command."}
	,
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"cextension", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	cmethods
};


PyMODINIT_FUNC PyInit_cextension(void)
{
	import_array();
	return PyModule_Create(&moduledef);
	

}

