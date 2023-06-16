/* File : cfit.cpp */

/*Use "cl /LD /O2 /fp:fast ctypes.cpp" to compile for windows */
/*Linux suggestion (check): gcc -O3 and march=native */
/*Linux: g++ -shared -o ctypes.so -fPIC ctypes.cpp*/
/*#include <cstdio>
FILE *fp = fopen("coutput.txt","w"); */

#include "MatrixClass.h";
#include "MatrixFunctions.h";
#include <cmath>;


#if defined(_MSC_VER)
	//  Microsoft 
	#define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
	//  GCC
	#define EXPORT extern "C" 
#else
	#define EXPORT extern "C" 
#endif

EXPORT int  fit(double *matrix, 
				double *inverted_marix, 
				long n
				) {
				
	CMatrix c(matrix, n);
	CMatrix inv = MInverse(c);
    return 0;
}
	