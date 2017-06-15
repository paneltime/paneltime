/* File : cfunctions.c */
#include <Python.h>
#include <stdio.h>
#include "numpy/arrayobject.h"


void Set(PyArrayObject *array,long i, long j,double value){
	*(double *)(array->data + i*array->strides[0]+j*array->strides[1])=value;//for multidimensional referencing
	return;
	}
	
double Get(PyArrayObject *array, long i, long j){
	double value=87.222;
	value= *(double *)(array->data + i*array->strides[0]+j*array->strides[1]);//for multidimensional referencing
	return value;
	}
	
PyObject *ManipulateMatrix(PyArrayObject *array1,PyArrayObject *array2) {
	double v=Get(array1,3,1);
	Set(array1,0,3,v);
	Set(array2,0,1,v);
	return PyFloat_FromDouble(1.0);;//*matrix;
}


PyObject *Trading(PyArrayObject *BidP,PyArrayObject *AskP,PyArrayObject *BidV,PyArrayObject *AskV,
						PyArrayObject *DPPred,PyArrayObject *TradeRec,
						long n,long depth,long IsSelling){
	long i;
	double* AccountingP=new double[n];
	double* AccountingV=new double[n];
	
	return PyFloat_FromDouble(1.0);//*matrix;
}

double* extend(double* original,long oldlenght,long extension, bool before){
	long i;
	long n=oldlenght+extension;
	double* ret=new double[n];
	if(before){
		for(i=0;i<oldlenght;i++){
			ret[i+extension]=original[i];
		}
	}
	else{
		for(i=0;i<oldlenght;i++){
			ret[i]=original[i];
		}		
	}
}

double* insert(double* AccP,double* AccV,double PriceItem,double VolItem,long position,long &n){
	long i,j;
	double tmp1,tmp2;
	if(AccP[0]>PriceItem){
		AccP=extend(AccP,n,100,true);
		AccV=extend(AccV,n,100,true);
	}else if(AccP[n]<PriceItem){
		AccP=extend(AccP,n,100,false);
		AccV=extend(AccV,n,100,false);
	}

	for(i=0;i<n;i++){
		if(PriceItem>AccP[i]){
			tmp1=PriceItem;
			for(j=i;i<n;i++){
				tmp2=AccP[i];
				AccP[i]=tmp1;
				tmp1=tmp2;
			}
			break;
		}
	}
}

//(PyArrayObject *BidP,PyArrayObject *AskP,PyArrayObject *BidV,PyArrayObject *AskV,
//					PyArrayObject *VolAccounting,PyArrayObject *TradingRec,PyArrayObject *DPPred)
//					long n,long depth)