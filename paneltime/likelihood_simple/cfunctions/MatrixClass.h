#include <oleauto.h>
#include <iostream>

#define T "T"

class CMatrix
{
public:
	long r; //These dimetions are public and thus may be changed. It does not affect memory usage or the actual
	long c; //dimention of the CMatrix, but changes how operators work. 
    CMatrix::CMatrix();  // Default constructor
	CMatrix::CMatrix(long Rows,long Cols);//Alternate Constructor with lower bound set to zero
	CMatrix::CMatrix(long Rows,long Cols,double** newMatr);//Constructor when a matrix all ready exists
	CMatrix::CMatrix(long Cols,double* newVec);//Constructor when a vector all ready exists
	~CMatrix();           //  and destructor.
	CMatrix(const CMatrix& old);  // Declare copy constructor
	CMatrix& CMatrix::operator=(const CMatrix& old);//Assignment operator definition
	CMatrix& CMatrix::operator&(CMatrix& old);//Address-of operator definition
	CMatrix CMatrix::operator*(const CMatrix& B);//Matrix multiplication operator
	CMatrix CMatrix::operator^(char *power);//Power operator
	CMatrix CMatrix::operator^(double power);//Power operator
	double*& CMatrix::operator[](long subscr);//subscript operator definition
	CMatrix CMatrix::GetRow(long Row);
	CMatrix CMatrix::EigenVectors(); 
	CMatrix CMatrix::EigenValues(); 
	CMatrix CMatrix::operator*(double multi);//Scalar multiplication operator
	double** v; //This could be private. However, if one wants to speed up the code, referencing this one directly as 
				//CMatrix.v[i][j] in stead of CMatrix[i][j] roughly decrease CPU time to  1/3. The only way to improve
private:		//that time would be to use fixed arrays (decleared as double array[r][c]). 
	long ur;//The actual dimensions. These are private and cannot be changed after construction
	long uc; 
	double** EVal;
	double** EVec;
	bool initialized;
};
CMatrix::CMatrix()
{
	initialized=false;
}
CMatrix::CMatrix(long Rows,long Cols)//Constructor
{
	if(Rows<0||Cols<0){
		v = new double*[1];v[0]=new double[1];v[0][0]=0;
		r=0;c=0;ur=0;uc=0;return;
	};
	v=new double*[Rows+1];
	
	for(long i=0;i<=Rows;i++){
		v[i]=new double[Cols+1];
		for(long j=0;j<=Cols;j++){
			v[i][j]=0;
		}
	}
	r=max(Rows,0);
	c=max(Cols,0);
	ur=r;
	uc=c;
	initialized=true;


};
CMatrix::CMatrix(long Rows,long Cols,double** newMatr)//Constructor when a matrix all ready exists
{
	if(Rows<0||Cols<0){
		v = new double*[1];v[0]=new double[1];v[0][0]=0;return;
		r=0;c=0;ur=0;uc=0;return;
	};
	r=max(Rows,0);
	c=max(Cols,0);
	ur=r;
	uc=c;

	double tmp=0;
	__try{tmp=newMatr[0][0];newMatr[0][0]=tmp;}// is it initialized?
	__except(EXCEPTION_EXECUTE_HANDLER){//If exception it is not initialized and thus is empty
		v = new double*[1];v[0]=new double[1];v[0][0]=0;
		r=0;c=0;ur=0;uc=0;return;
	}

	v=new double*[Rows+1];
	for(long i=0;i<=Rows;i++){
		v[i]=new double[Cols+1];
		for(long j=0;j<=Cols;j++){
			__try{v[i][j]=newMatr[i][j];}// exit if it does not exist;
			__except(EXCEPTION_EXECUTE_HANDLER){ur=i;r=i;c=j;uc=j;return;}
		}
	}
	initialized=true;
};
CMatrix::CMatrix(long Cols,double* newVec)//Constructor when a vector all ready exists
{
	if(Cols<0){
		v = new double*[1];v[0]=new double[1];v[0][0]=0;return;
		r=0;c=0;ur=0;uc=0;return;
	};
	r=0;
	ur=0;
	c=max(Cols,0);
	uc=c;


	double tmp=0;
	__try{tmp=newVec[0];newVec[0]=tmp;}// is it initialized?
	__except(EXCEPTION_EXECUTE_HANDLER){//If exception it is not initialized and thus is empty
		v = new double*[1];v[0]=new double[1];v[0][0]=0;
		c=0;uc=0;return;
	}
	v=new double*[1];
	v[0]=new double[Cols+1];
	for(long j=0;j<=Cols;j++){
		__try{v[0][j]=newVec[j];}// exit if it does not exist;
		__except(EXCEPTION_EXECUTE_HANDLER){uc=j;c=j;return;}
	}

	initialized=true;

};

CMatrix::CMatrix(const CMatrix& old)//Copy constructor
{
	__try{
		double test=old.v[0][0];
	}// If this does not exist then just return old CMatrix
	__except(EXCEPTION_EXECUTE_HANDLER){
		return;
	};
	if(old.r>old.ur){r=old.ur;}else{r=old.ur;}
	if(old.c>old.uc){c=old.uc;}else{c=old.uc;}
	ur=r;
	uc=c;
	
	if(r<0||c<0){double** v = new double*[1];v[0]=new double[1];v[0][0]=0;};
	v=new double*[r+1];
	for(long i=0;i<=r;i++){
		v[i]=new double[c+1];
		for(long j=0;j<=c;j++){
			v[i][j]=old.v[i][j];
		}
	}
	initialized=true;
};
CMatrix::~CMatrix()//Destructor
{
	__try{v[0][0]=0;}// exit if it does not exist;
	__except(EXCEPTION_EXECUTE_HANDLER){
		return;
	}
	if((*this).initialized==false){//If this is not initialized then just return
		return;
	};
	for(long i=0;i<=ur;i++){
		delete [] v[i];
	}
	delete [] v;
	return;
};
CMatrix& CMatrix::operator&(CMatrix& old){
	//address-of operator definition
	return old;
};

CMatrix& CMatrix::operator=(const CMatrix& old){
	//Assignment operator definition
	__try{
		double test=(*this).v[0][0];
	}// If this does not exist then just return old CMatrix
	__except(EXCEPTION_EXECUTE_HANDLER){
		(*this).CMatrix::CMatrix(old);
		return *this;
	};
	if((*this).initialized==false){//If this is not initialized then just return old
		(*this).CMatrix::CMatrix(old);
		return *this;
	};
	if(&(*this)==&old){return (*this);}; //If the left hand CMatrix is the same, then just keep it. 
	//If this does exist then delete it and copy
	(*this).CMatrix::~CMatrix();
	(*this).CMatrix::CMatrix(old);
	return *this;
};
CMatrix CMatrix::operator*(const CMatrix& B){
	//Matrix multiplication operator
	
	CMatrix &A=(*this);

	long ubRows=A.r;
	long ubRwCol=A.c;
	long ubCols=B.c;

	CMatrix Target(ubRows,ubCols);
	if(ubRwCol!=B.r){return Target;};
	if(ubRows<1||ubRwCol<1||ubCols<1){return Target;};

	for(long i=1;i<=ubRows;i++)
	{
		for(long j=1;j<=ubCols;j++)
		{
			Target[i][j]=0;
			for(long k=1;k<=ubRwCol;k++)
			{
				Target[i][j]+=(A.v[i][k])*(B.v[k][j]);
			};
		};
	};
	return Target;
};
CMatrix CMatrix::operator^(char *power){
	//Matrix transpose operator
	

	CMatrix Target(1,1);
	if(strcmp(power,"T")){return Target;}
	else{return Target;};
};
CMatrix CMatrix::operator^(double power){
	//Matrix multiplication operator
	

	CMatrix Target(1,1);

	if(power==1){return Target;}
	else{return Target;};
};
double*& CMatrix::operator[](long subscr){
	//subscript operator definition
	return v[subscr];
};

CMatrix CMatrix::GetRow(long Row)
{	CMatrix tmp(c,(*this).v[Row]);
	return tmp;
};

CMatrix CMatrix::operator*(double multi){
	//double (scalar) multiplication operator
	long i,j;
	CMatrix retmatr((*this).r,(*this).c);
	for(i=1;i<=(*this).r;i++){
		for(j=1;j<=(*this).c;j++){
			retmatr.v[i][j]=(*this).v[i][j]*multi;
		};
	};
	return retmatr;
};
//CMatrix CMatrix::EigenValues()
//{	
//	return (*this).EVal;
//};
//CMatrix CMatrix::EigenVectors()
//{	
//	return (*this).EVec;
//};


//For testing:
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <time.h>


