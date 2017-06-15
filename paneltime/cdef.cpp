/* File : cdef.c */
#include <Python.h>
#include <stdio.h>
#include "numpy/arrayobject.h"
#include "cfunctions.h"
using namespace std;



static PyObject *cfunctions_ManipulateMatrix(PyObject *self, PyObject *args) {
	//printf("Output: %f \n",2.0);//for printing
	PyArrayObject *array1;
	PyObject *input1;
	PyArrayObject *array2;
	PyObject *input2;
	if (!PyArg_ParseTuple(args, "OO", &input1,&input2))
		return NULL;
	array1 = (PyArrayObject	*)	 PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
	array2 = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
	if ((array1 == NULL) or (array2 == NULL))
		return NULL;
	return ManipulateMatrix(array1,array2);
}

static PyObject *cfunctions_Trading(PyObject *self, PyObject *args) {
	int i,nMatr=6,nInts=3;
	long* inputI=new long[nInts];
	PyObject** inputM=new PyObject*[nMatr];
	PyArrayObject** array=new PyArrayObject*[nMatr];
	if (!PyArg_ParseTuple(args, "OOOOOOlll", &inputM[0],&inputM[1],&inputM[2],&inputM[3],&inputM[4],&inputM[5],
											&inputI[0],&inputI[1],&inputI[2]))
		return NULL;
	for(i=0;i<nMatr;i++){
		array[i] = (PyArrayObject *) PyArray_ContiguousFromObject(inputM[i], PyArray_DOUBLE, 2, 2);
		if (array[i] == NULL)
			return NULL;
	}
	return Trading(array[0],array[1],array[2],array[3],array[4],array[5],
				inputI[0],inputI[1],inputI[1]);
}





static PyMethodDef cfunctionsMethods[] = {
	{"ManipulateMatrix",  cfunctions_ManipulateMatrix, METH_VARARGS, "Execute a shell command."},
	{"Trading",  cfunctions_Trading, METH_VARARGS, "Execute a shell command."},
	{NULL, NULL, 0, NULL}        /* Sentinel */

};

PyMODINIT_FUNC initcfunctions(void)
{
	import_array()
	(void) Py_InitModule("cfunctions", cfunctionsMethods);
}

