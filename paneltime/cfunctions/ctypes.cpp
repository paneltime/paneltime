/* File : ctypes.cpp */

/*Use "cl /LD /Ox /Ot /Oi /GL /LTCG /fp:fast ctypes.cpp" to compile for windows */
/*Linux suggestion (check): gcc -O3 and march=native */
/*Linux: g++ -shared -o ctypes.so -fPIC ctypes.cpp*/
/*Mac: clang++ -O3 -shared -o ctypes.dylib -fPIC ctypes.cpp*/
/*include <cstdio>
FILE *fp = fopen("coutput.txt","w"); */

#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <cctype>

#include <iostream>


#if defined(_MSC_VER)
	//  Microsoft 
	#define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
	//  GCC
	#define EXPORT extern "C" 
#else
	#define EXPORT extern "C" 
#endif

#include "mathexp.cpp"

void inverse(long n, double *x_args, long nx, double *b_args, long nb, 
				double *a, double *ab) {
	
	long j,i;
	
	double sum_ax;
	double sum_ab;
	
	for(i=0;i<n;i++){a[i]=0.0;};
	a[0]=1.0;
	ab[0] = b_args[0];

	for(i=1;i<n;i++){
		sum_ax=0;
		sum_ab=0;
		for(j=0;j<i && j<nx;j++){
			sum_ax+=x_args[j]*a[i-j-1];
			//fprintf(fp, "%f, %f, %d, %d,%d\n", x_args[j], a[i-j-1], j, i, i-j-1);
		}
		a[i]=-sum_ax;
		for(j=0;j<i+1 && j<nb;j++){
			sum_ab+=b_args[j]*a[i-j];
		}
		ab[i]=sum_ab;
	}
	//fclose(fp);
}
	
EXPORT int  armas(double *parameters,
				double *lambda, double *rho, double *gamma, double *psi,
				double *AMA_1, double *AMA_1AR, 
				double *GAR_1, double *GAR_1MA, 
				double *u, double *e, double *var, double *h, double *W, double *T_array, 
				char* h_expr
				) {


	double sum, esq;
	long k,j,i;

	long N = (int) parameters[0];
	long T = (int) parameters[1];
	long nlm = (int) parameters[2];
	long nrh = (int) parameters[3];
	long ngm = (int) parameters[4];
	long npsi = (int) parameters[5];
	long egarch = (int) parameters[6];
	double z = parameters[7];
	long rw;


	
	inverse(T, lambda, nlm, rho, nrh, AMA_1, AMA_1AR);
	inverse(T, gamma, ngm, psi, npsi, GAR_1, GAR_1MA);

	if(h_expr != NULL && *h_expr != '\0'){
	//if(false){
		auto* h_func = exprtk_create_from_string(h_expr);
		for(k=0;k<N;k++){//individual dimension
			for(i=0;i<(int) T_array[k];i++){//time dimension
				//ARMA:
				sum = 0;
				for(j=0;j<=i;j++){//time dimesion, back tracking
					sum += AMA_1AR[j]*u[(i-j) + k*T];
					}
				e[i + k*T] = sum;
				
				esq = sum*sum + 1e-8;
				//GARCH:
				esq = exprtk_eval(h_func, sum, esq, z);
				
				h[i + k*T] = esq;

				sum =0;
				for(j=0;j<=i;j++){//time dimension, back tracking
					sum += GAR_1[j] * W[(i-j) + k*T] + GAR_1MA[j]*h[(i-j) + k*T];
				}
				var[i + k*T] = sum;
			}
		}
	//**   EGARCH ESTIMATION:   */
	}else if(egarch){
		for(k=0;k<N;k++){//individual dimension
			for(i=0;i<(int) T_array[k];i++){//time dimension
				//ARMA:
				sum = 0;
				for(j=0;j<=i;j++){//time dimension, back tracking
					sum += AMA_1AR[j]*u[(i-j) + k*T];
					}
				e[i + k*T] = sum;
				
				//GARCH:
				esq = sum*sum + 1e-8;
				
				//EGARCH:
				h[i + k*T] = log(esq);
				
				sum =0;
				for(j=0;j<=i;j++){//time dimension, back tracking
					sum += GAR_1[j] * W[(i-j) + k*T] + GAR_1MA[j]*h[(i-j) + k*T];
				}
				var[i + k*T] = sum;
			}
		}
	}else{
		//**  NOT EGARCH ESTIMATION:   */
		for(k=0;k<N;k++){//individual dimension
			for(i=0;i<(int) T_array[k];i++){//time dimension
				//ARMA:
				sum = 0;
				for(j=0;j<=i;j++){//time dimesion, back tracking
					sum += AMA_1AR[j]*u[(i-j) + k*T];
					}
				e[i + k*T] = sum;

				//GARCH:
				esq = sum*sum + 1e-8;
				
				h[i + k*T] = esq;

				sum =0;
				for(j=0;j<=i;j++){//time dimension, back tracking
					sum += GAR_1[j] * W[(i-j) + k*T] + GAR_1MA[j]*h[(i-j) + k*T];
				}
				var[i + k*T] = sum;
			}
		}
	}
	return 0;
}
	

void print(double *r){
		int i;
		for (i = 0; i < 10; i++) {
				printf("%.2f ", r[i]);
		}
		printf("\n"); // Print a newline character at the end
		fflush(stdout);
}


EXPORT int  fast_dot(double *r, double *a, double *b, long n, long m) {
	int i, j, k;
	for(i=1;i<n;i++){//individual dimension
		for(j=0;j<m;j++){
			for(k=0;k<n-i;k++){
				r[i+k + j*n] += a[i]*b[k + j*n];
			}
		}
		
	}
	return 0;
}





