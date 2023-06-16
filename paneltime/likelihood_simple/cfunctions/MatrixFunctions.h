#include <math.h>
CMatrix cU,cL,cTmpMatr,cI,cInverted,cTarget;
//PRIVATE FUNCTIONS ***************************
bool LU(CMatrix &cmatr)
{//This function makes a LU decomposition of the matrix matr for the MInverse function
	//The procedure assumes cU and cL are zero matrixes
    double Sum;
	long k,i,j,m,ub=cmatr.c;
	cU=CMatrix(ub,ub);
	cL=CMatrix(ub,ub);
	if(1<0||1>=ub||cmatr.r<ub||cU.r<ub||cU.c<ub||cL.r<ub||cL.c<ub){return false;};//Testing that the matrices are OK
    for(k=1;k<=ub;k++){ cU.v[k][k]=1;};
    for(k=1;k<=ub;k++)
	{
        for (i=k;i<=ub;i++)
		{
            Sum = 0;
            for (m=1;m<=k-1;m++)
			{
                Sum =Sum + cL.v[i][m] * cU.v[m][k];
            };
            cL.v[i][k] = cmatr.v[i][k] - Sum;
        };
        if(abs(cL.v[k][k]) < 1E-18) 
		{
			cL.v[k][k] = 1E-18;
			//return false;
		};
        for (j=k+1;j<=ub;j++)
		{
            Sum = 0;
            for (m=1;m<=k-1;m++)
			{
                Sum += cL.v[k][m] * cU.v[m][j];
            };
            cU.v[k][j] = (cmatr.v[k][j] - Sum) / cL.v[k][k];
        };
	};

    return true;

}

CMatrix MInverse(CMatrix &cmatr)
{
	long ub=cmatr.c;

	CMatrix cInverted(ub,ub);
	CMatrix cTmpMatr(ub,ub);
	CMatrix cI(ub,ub);
	if(cmatr.r!=ub){return cInverted;};//Testing that the matrices are OK
	
    double Sum;
	long k;
	long i;
	long j;
	
	bool OK=LU(cmatr);//LU decomposition
    for(i=1;i<=ub;i++){cI.v[i][i] = 1;}
    for(k=1;k<=ub;k++)
	{
        cTmpMatr.v[1][1] = cI.v[1][k] / cL.v[1][1];
        for(i=1+1;i<=ub;i++)
		{
            Sum = 0;
            for(j=1;j<=i-1;j++)
			{
                Sum = Sum + cL.v[i][j] * cTmpMatr.v[j][1];
            };
            cTmpMatr.v[i][1] = (cI.v[i][k] - Sum) / cL.v[i][i];
		};
        cInverted.v[ub][k] = cTmpMatr.v[ub][1];
        for(i=ub-1;i>=1;i--)
		{
            Sum = 0;
            for(j=i;j<=ub;j++)
			{
                Sum = Sum + cU.v[i][j] * cInverted.v[j][k];
            };
            cInverted.v[i][k] = cTmpMatr.v[i][1] - Sum;
        };
    };
    return cInverted;

}
void MInverseFast(CMatrix &cmatr)
{//All matrixes supplied must be NxN matrixes
	long ub=cmatr.c,k,i,j,m;
    double Sum;
	if(cL.r!=ub&&cL.c!=ub){cL=CMatrix(ub,ub);};
	if(cU.r!=ub&&cL.c!=ub){cU=CMatrix(ub,ub);};
	if(cInverted.r!=ub&&cInverted.c!=ub){cInverted=CMatrix(ub,ub);};
	if(cI.r!=ub&&cI.c!=ub){cI=CMatrix(ub,ub);};
	if(cTmpMatr.r!=ub&&cTmpMatr.c!=ub){cTmpMatr=CMatrix(ub,ub);};


	for(i=1;i<=ub;i++){
		for (j=1;j<=ub;j++){
			cU.v[i][j]=0.0;
			cL.v[i][j]=0.0;
			cTmpMatr.v[i][j]=0.0;
			cI.v[i][j]=0.0;
			cInverted.v[i][j]=0.0;
		}
		cI.v[i][i] = 1.0;
		cU.v[i][i] = 1.0;
	}
	//LU decomposition
    for(k=1;k<=ub;k++)
	{
        for (i=k;i<=ub;i++)
		{
            Sum = 0;
            for (m=1;m<=k-1;m++)
			{
                Sum =Sum + cL.v[i][m] * cU.v[m][k];
            };
            cL.v[i][k] = cmatr.v[i][k] - Sum;
        };
        if(abs(cL.v[k][k]) < 1E-18) 
		{
			cL.v[k][k] = 1E-18;
			//return false;
		};
        for (j=k+1;j<=ub;j++)
		{
            Sum = 0;
            for (m=1;m<=k-1;m++)
			{
                Sum += cL.v[k][m] * cU.v[m][j];
            };
            cU.v[k][j] = (cmatr.v[k][j] - Sum) / cL.v[k][k];
        };
	};
	
	//Continuing
	Sum=0.0;
	for(k=1;k<=ub;k++)
	{
        cTmpMatr.v[1][1] = cI.v[1][k] / cL.v[1][1];
        for(i=1+1;i<=ub;i++)
		{
            Sum = 0;
            for(j=1;j<=i-1;j++)
			{
                Sum = Sum + cL.v[i][j] * cTmpMatr.v[j][1];
            };
            cTmpMatr.v[i][1] = (cI.v[i][k] - Sum) / cL.v[i][i];
		};
        cInverted.v[ub][k] = cTmpMatr.v[ub][1];
        for(i=ub-1;i>=1;i--)
		{
            Sum = 0;
            for(j=i;j<=ub;j++)
			{
                Sum = Sum + cU.v[i][j] * cInverted.v[j][k];
            };
            cInverted.v[i][k] = cTmpMatr.v[i][1] - Sum;
        };
    };
    return;

}
CMatrix MMult(CMatrix &cMa,CMatrix &cMb) 
{//Returns the result of matrix multiplying cMa and cMb
	
	long ubRows=cMa.r;
	long ubRwCol=cMa.c;
	long ubCols=cMb.c;

	CMatrix cTarget(ubRows,ubCols);
	if(ubRwCol!=cMb.r){return cTarget;};
	if(ubRows<1||ubRwCol<1||ubCols<1){return cTarget;};
	double **Ma=cMa.v;//Reference to the double** member of the CMatrix object
	double **Mb=cMb.v;
	double **Target=cTarget.v;

	for(long i=1;i<=ubRows;i++)
	{
		for(long j=1;j<=ubCols;j++)
		{
			Target[i][j]=0;
			for(long k=1;k<=ubRwCol;k++)
			{
				Target[i][j]+=Ma[i][k]*Mb[k][j];
			};
		};
	};
	return cTarget;
} 
void MMultFast(CMatrix &cMa,CMatrix &cMb) 
{//Returns the result of matrix multiplying cMa and cMb
	
	long ubRows=cMa.r;
	long ubRwCol=cMa.c;
	long ubCols=cMb.c;
	if(cTarget.r!=cMa.r&&cTarget.r!=cMb.c){cTarget=CMatrix(cMa.r,cMb.c);};

	for(long i=1;i<=ubRows;i++)
	{
		for(long j=1;j<=ubCols;j++)
		{
			cTarget.v[i][j]=0;
			for(long k=1;k<=ubRwCol;k++)
			{
				cTarget.v[i][j]+=cMa.v[i][k]*cMb.v[k][j];
			};
		};
	};
	return;
} 

CMatrix MMultTra(CMatrix &cTrMa,CMatrix &cMb) 
{//Returns the result of matrix multiplying cMa and cMb
	if(cTarget.r!=cTrMa.r&&cTarget.r!=cMb.c){cTarget=CMatrix(cTrMa.r,cMb.c);};
	
	long ubRows=cTrMa.c;
	long ubRwCol=cTrMa.r;
	long ubCols=cMb.c;

	for(long i=1;i<=ubRows;i++)
	{
		for(long j=1;j<=ubCols;j++)
		{
			cTarget.v[i][j]=0;
			for(long k=1;k<=ubRwCol;k++)
			{
				cTarget.v[i][j]+=cTrMa.v[k][i]*cMb.v[k][j];
			};
		};
	};
	return cTarget;
} 


double MDeterm(CMatrix &cMatr)
{
	long ub=cMatr.c;
    long i;

	CMatrix cU(ub,ub);//Creates CMatrix matrices
	CMatrix cL(ub,ub);
	bool OK=LU(cMatr);

	if(cMatr.r!=ub){return 0;};//Testing that the matrices are OK
	

	double retvar=1;	
	for(i=1;i<=ub;i++)
	{
		retvar*=cL.v[i][i];//The product of the elements in the diagonal of the L matrix is the determinant
	};

	return retvar;
}

CMatrix EigenValues(CMatrix &cMatr, long iter)
{
	//cMatr is the matrix of the EigenValues
	//iter ist the number of iterations
	//chk is the tolerance
    long k;
	long ubk;
	long i;
	long j;
	long m;
	long ub=cMatr.c;
	double Sum1;
	double Sum2;

	CMatrix cEV(1,cMatr.c);
	if(cMatr.r!=ub){return cEV;};//Testing that the matrices are OK
	
	CMatrix cL(ub,ub);
	CMatrix cU(ub,ub);
	CMatrix cTmpMatr(ub,ub);
	for(i=1;i<=ub;i++)
	{
		for(j=1;j<=ub;j++){
			cTmpMatr.v[i][j]=cMatr.v[i][j];
		}
	}
	for(m=1;m<=iter;m++)
	{
		for(k=1;k<=ub;k++){ cU.v[k][k]=1;};
		for(k=1;k<=ub;k++)
		{
			for (i=k;i<=ub;i++)
			{
				Sum1 = 0;
				Sum2 = 0;
				for (j=1;j<=k-1;j++)
				{
					Sum1 =Sum1 + cL.v[i][j] * cU.v[j][k];
					Sum2 =Sum2 + cL.v[k][j] * cU.v[j][i];
				};
				cL.v[i][k] = cTmpMatr.v[i][k] - Sum1;
				cU.v[k][i] = (cTmpMatr.v[k][i] - Sum2) / (cL.v[k][k]+1E-300);
			};
		};
		for(i=1;i<=ub;i++)
		{
			for(j=1;j<=ub;j++)
			{
				cTmpMatr.v[i][j]=0;
				if(i>j){ubk=i;}else{ubk=j;};
				for(k=ubk;k<=ub;k++){
					cTmpMatr.v[i][j]=cTmpMatr.v[i][j]+cU.v[i][k]*cL.v[k][j];
				};
			};
		};
	};
	for(i=1;i<=ub;i++){cEV.v[1][i]=cTmpMatr.v[i][i];};

	return cEV;
}

CMatrix EigenValues(CMatrix &cMatr){
	return EigenValues(cMatr,0);
}
CMatrix EigenValuesL(CMatrix &cMatr, long iter){
	long i;
	CMatrix EVs=EigenValues(cMatr,iter);
	CMatrix EVI(EVs.c,EVs.c);
	for(i=1;i<=EVs.c;i++){EVI.v[i][i]=EVs.v[1][i];};
	return EVI;
}
CMatrix EigenVector(CMatrix &cMatr,CMatrix &cEV)

{
	long i=0,j=0,mk=0,mi=0,k=0,ub=cMatr.c;
	double SSq;

	CMatrix cU(ub,ub);//Creates CMatrix matrices
	CMatrix cL(ub,ub);
	CMatrix cTmpMatr1(ub-1,ub-1);
	CMatrix cTmpMatr2;
	CMatrix cRetMatr(ub,ub);
	CMatrix cNorm(ub-1,1);
	if(cEV.c<ub||cEV.r<1){return cRetMatr;};//Default for NewMATR is to set 1=0, so changing to current 1
	if(cMatr.r!=ub){return cRetMatr;};//Testing that the matrices are OK
	
	double **EV=cEV.v;
	double **Matr=cMatr.v;
	double **TmpMatr1=cTmpMatr1.v;
	double **RetMatr=cRetMatr.v;
	double **Norm=cNorm.v;
	for(j=1;j<=ub;j++){
		mi=0;
		for(i=1;i<=ub-1;i++){
			if(i==j){mi=1;};
			Norm[i][1]=-Matr[i+mi][j];
			mk=0;
			for(k=1;k<=ub-1;k++){
				if(k==j){mk=1;};
				if(i==k){
					TmpMatr1[i][k]=Matr[i+mi][k+mk]-EV[1][j];
				}else{
					TmpMatr1[i][k]=Matr[i+mi][k+mk];
				}
			};
		};
		cTmpMatr2=MInverse(cTmpMatr1);
		cTmpMatr2=MMult(cTmpMatr2,cNorm);
		SSq=0;
		for(i=1;i<=ub-1;i++){SSq+=cTmpMatr2.v[i][1]*cTmpMatr2.v[i][1];}
		SSq+=1;SSq=sqrt(SSq);
		mi=0;
		for(i=1;i<=ub-1;i++) {
			if(EV[1][j]==0){
				if(i==j){mi=1;};
				RetMatr[i+mi][j]=0;
			}else{
				if(i==j){mi=1;};
				RetMatr[i+mi][j]=cTmpMatr2.v[i][1]/SSq;
			};
		};
		RetMatr[j][j]=1.0/SSq;
	};
	return cRetMatr;
}
CMatrix Transpose(CMatrix &cMatr)
{
    long i;
	long j;
	long c=cMatr.c;
	long r=cMatr.r;
	CMatrix cTransposed(c,r);
	if(1<0||1<0||1>r||1>c){return cTransposed;};//Testing that the matrices are OK


	for(i=1;i<=r;i++)
	{
		for(j=1;j<=c;j++)
		{
			cTransposed.v[j][i]=cMatr.v[i][j];
		};
	};
	return cTransposed;
}

void Transpose(CMatrix &cMatr,CMatrix &cTransposed)
{
    long i;
	long j;
	long c=cMatr.c;
	long r=cMatr.r;
	cTransposed=CMatrix(c,r);
	if(1>r||1>c){return;};//Testing that the matrices are OK


	for(i=1;i<=r;i++)
	{
		for(j=1;j<=c;j++)
		{
			cTransposed.v[j][i]=cMatr.v[i][j];
		};
	};
	return;
}


#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void QuickSort(CMatrix &cMatr)
{
 //Here M is the size of subarrays sorted by straight insertion and NSTACK is the required auxiliary storage.
//Sorts an array arr[1..n] into ascending numerical order using the Quicksort algorithm. n is
//input; arr is replaced on output by its sorted rearrangement.

	long n=cMatr.c;
	double *arr=cMatr.v[1];
	long M=7;
	long NSTACK=50;
	long i,ir=n,j,k,l=1;
	long jstack=0;
	double a,temp;
	long* istack=IntNewVec(NSTACK);
	for (;;) {								//Insertion sort when subarray small enough.
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				a=arr[j];
				for (i=j-1;i>=l;i--) {
					if (arr[i] <= a) break;
					arr[i+1]=arr[i];
				}
				arr[i+1]=a;
			}
			if (jstack == 0) break;
			ir=istack[jstack--]; //Pop stack and begin a new round of partitioning.
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1; //Choose median of left, center, and right elements as partitioning element a. Also rearrange so that a[l]  a[l+1]  a[ir].
			SWAP(arr[k],arr[l+1]);
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
			}
			i=l+1; //Initialize pointers for partitioning.
			j=ir;
			a=arr[l+1];					//Partitioning element.
			for (;;) {					//Beginning of innermost loop.
				do i++; while (arr[i] < a); //Scan up to nd element > a.
				do j--; while (arr[j] > a); //Scan down to nd element < a.
				if (j < i) break;			//Pointers crossed. Partitioning complete.
				SWAP(arr[i],arr[j]);	//Exchange elements.
			}							//End of innermost loop.
			arr[l+1]=arr[j]; //Insert partitioning element.
			arr[j]=a;
			jstack += 2;
//Push pointers to larger subarray on stack, process smaller subarray immediately.
			if (jstack > NSTACK){
				return; //("NSTACK too small in sort.");
			};
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	delete [] istack; 
}
void QuickSortMatrix(CMatrix &arr,long col)
{
//The matrix Need to be transposed if the sorting dimesion is the first row dimension, because it allways sort the 
//last dimension for simlicity. 
 //Here M is the size of subarrays sorted by straight insertion and NSTACK is the required auxiliary storage.
//Sorts an array arr[1..n] into ascending numerical order using the Quicksort algorithm. n is
//input; arr is replaced on output by its sorted rearrangement.

	if(col>arr.c)col=arr.c;
	long M=7;
	long NSTACK=500;
	//long i,ir=n,j,k,l=1;
	long i,ir=arr.r,j,k,l=1;
	long jstack=0;
	double* a;
	double* temp;
	long *istack=IntNewVec(NSTACK);

	for (;;) {								//Insertion sort when subarray small enough.
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				a=arr.v[j];
				for (i=j-1;i>=l;i--) {
					if (arr.v[i][col] <= a[col]) break;
					arr.v[i+1]=arr.v[i];
				}
				arr.v[i+1]=a;
			}
			if (jstack == 0) break;
			ir=istack[jstack--]; //Pop stack and begin a new round of partitioning.
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1; //Choose median of left, center, and right elements as partitioning element a. Also rearrange so that a[l]  a[l+1]  a[ir].
			SWAP(arr.v[k],arr.v[l+1]);
			if (arr.v[l][col] > arr.v[ir][col]) {
				SWAP(arr.v[l],arr.v[ir])
			}
			if (arr.v[l+1][col] > arr.v[ir][col]) {
				SWAP(arr.v[l+1],arr.v[ir])
			}
			if (arr.v[l][col] > arr.v[l+1][col]) {
				SWAP(arr.v[l],arr.v[l+1])
			}
			i=l+1; //Initialize pointers for partitioning.
			j=ir;
			a=arr.v[l+1];					//Partitioning element.
			for (;;) {					//Beginning of innermost loop.
				do i++; while (arr.v[i][col] < a[col]); //Scan up to nd element > a.
				do j--; while (arr.v[j][col] > a[col]); //Scan down to nd element < a.
				if (j < i) break;			//Pointers crossed. Partitioning complete.
				SWAP(arr.v[i],arr.v[j]);	//Exchange elements.
			}							//End of innermost loop.
			arr.v[l+1]=arr.v[j]; //Insert partitioning element.
			arr.v[j]=a;
			jstack += 2;
//Push pointers to larger subarray on stack, process smaller subarray immediately.
			if (jstack > NSTACK) return; //("NSTACK too small in sort.");
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	delete [] istack;
}
void QuickSort(CMatrix &cMatr,long Ascending)
{
//Reverses quick sort ascending if required

	QuickSort(cMatr);
	if(Ascending){
		long c=cMatr.c;
		double temp;
		long i;
		long mc=long(c/2);
		for(i=1;i<=mc;i++){
			SWAP(cMatr.v[1][i],cMatr.v[1][c-i+1]);
		}
	}
}
//PUBLIC FUNCTIONS ****************************
__declspec(dllexport) double WINAPI MDetermDLL(SAFEARRAY** saMatr)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
    return MDeterm(cMatr);
}
__declspec(dllexport) double WINAPI EigenValuesDLL(SAFEARRAY** saMatr,SAFEARRAY** saEigenValues,long iter)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
	CMatrix cEigenValues=EigenValues(cMatr,iter);
    return (double)CToSafeArray(cEigenValues,saEigenValues);
}

__declspec(dllexport) double WINAPI EigenVectorDLL(SAFEARRAY** saMatr,SAFEARRAY** saEigenVectors,long iter)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
	CMatrix cEigenValues=EigenValues(cMatr,iter);
	CMatrix cEigenVector=EigenVector(cMatr,cEigenValues);
    return (double)CToSafeArray(cEigenVector,saEigenVectors);
}
__declspec(dllexport) double WINAPI EigenVector2DLL(SAFEARRAY** saMatr,SAFEARRAY** saEV,SAFEARRAY** saEigenVectors)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
	CMatrix cEigenValues=SafeArrayToC(saEV);
	CMatrix cEigenVector=EigenVector(cMatr,cEigenValues);
    return (double)CToSafeArray(cEigenVector,saEigenVectors);
}
__declspec(dllexport) double WINAPI TransposeDLL(SAFEARRAY** saMatr,SAFEARRAY** saTransposed)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
	CMatrix cTransposed=Transpose(cMatr);
    return (double)CToSafeArray(cTransposed,saTransposed);
}
__declspec(dllexport) double WINAPI MInvDLL(SAFEARRAY** saMatr,SAFEARRAY** saInverted)
{
	CMatrix cMatr=SafeArrayToC(saMatr);
	CMatrix cInverted=MInverse(cMatr);
    return (double)CToSafeArray(cInverted,saInverted);
}
__declspec(dllexport) double WINAPI MMultDLL(SAFEARRAY** saMa, SAFEARRAY** saMb, SAFEARRAY** saTarget) 
{
	CMatrix cMa=SafeArrayToC(saMa);
	CMatrix cMb=SafeArrayToC(saMb);
	CMatrix cTarget=MMult(cMa,cMb);
    return (double)CToSafeArray(cTarget,saTarget);
} 

__declspec(dllexport) double WINAPI QuickSortDLL(SAFEARRAY** saArr,long col) 
{
	CMatrix cArr=SafeArrayToC(saArr);
	QuickSortMatrix(cArr,col);
	//_CrtDumpMemoryLeaks();
    return (double)CToSafeArray(cArr,saArr);
} 

__declspec(dllexport) double WINAPI QuickSortSimpleDLL(SAFEARRAY** saArr,long col) 
{
	CMatrix cArr=SafeArrayToC(saArr);
	QuickSort(cArr);
	//_CrtDumpMemoryLeaks();
    return (double)CToSafeArray(cArr,saArr);
} 