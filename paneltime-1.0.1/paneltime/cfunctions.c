/* File : cfunctions.c */
#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"



static PyObject * simpleinverse(PyObject *self, PyObject *args) {
	int i,nMatr=2,nInts=1;
	long* inputI=new long[nInts];
	PyObject** inputM=new PyObject*[nMatr];
	PyArrayObject** array=new PyArrayObject*[nMatr];
	if (!PyArg_ParseTuple(args, "OOl", &inputM[0],&inputM[1],
											&inputI[0]))
		return NULL;
	for(i=0;i<nMatr;i++){
		array[i] = (PyArrayObject *) PyArray_ContiguousFromObject(inputM[i], PyArray_DOUBLE, 2, 2);
		if (array[i] == NULL)
			return NULL;
	}
	double a=Get(array[0],2,3);
	return a;

}


static PyMethodDef cmethods[] = {

	{"simpleinverse",  simpleinverse, METH_VARARGS,
	"Execute a shell command."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef cfunctions =
{
	PyModuleDef_HEAD_INIT,
	"cfunctions", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	cmethods
};

PyMODINIT_FUNC PyInit_cfunctions(void)
{
	return PyModule_Create(&cfunctions);
}


