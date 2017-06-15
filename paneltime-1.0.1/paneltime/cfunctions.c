/* File : cfunctions.c */
#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"



static PyObject * test(PyObject *self, PyObject *args) {
	

}


static PyMethodDef cmethods[] = {

	{"test",  test, METH_VARARGS,
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


