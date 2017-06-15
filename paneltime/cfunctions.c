/* File : cfunctions.c */
#include <Python.h>
#include <stdio.h>
#include "numpy/arrayobject.h"


float GetMoments(double **matrix) {//char s){
	//*matrix=2.0;
	return matrix[0][0];//*matrix;

}

static PyObject *cfunctions_GetMoments(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	PyObject *input;
	double **data;
	double *itm;
	double number;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type,&array))
		return NULL;
	//array = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 1, 1);
	//if (array == NULL)
	//	return NULL;
	data=(double**)PyArray_DATA(array);
	*itm=data[0];
	//number=itm[0];
	printf("Output: %d \n",itm);
	//Py_DECREF(array);
	//*(double *)(array->data + 0*array->strides[0]+0*array->strides[1])=8.0;
	return PyFloat_FromDouble(number);

}

/*
static PyObject *cfunctions_GetMoments(PyObject *self, PyObject *args) {
	PyArrayObject *array;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type,&array))
		return NULL;
	*(double *)(array->data + 0*array->strides[0])=8.0;
	return PyFloat_FromDouble(2.0);

}
*/
/*
static PyObject *cfunctions_GetMoments(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	float *data;
	float sts;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type,&array))
		return NULL;
	data=array->data;
	sts = GetMoments(data);
	return Py_BuildValue("f", sts);

}*/

static PyMethodDef cfunctionsMethods[] = {
	{"GetMoments",  cfunctions_GetMoments, 
		METH_VARARGS, "Execute a shell command."},
	{NULL, NULL, 0, NULL}        /* Sentinel */

};

PyMODINIT_FUNC initcfunctions(void)
{
	import_array()
	(void) Py_InitModule("cfunctions", cfunctionsMethods);
}


