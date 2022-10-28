/* File : cfunctions.c */

/*Use "cl /LD /O2 /fp:fast ctypes.cpp" to compile for windows */
/*Linux: gcc -O3 and –march=native */

/*#include <cstdio>
FILE *fp = fopen("coutput.txt","w"); */


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
	
extern "C" __declspec(dllexport) int  armas(long *lengths, 
				double *lambda, double *rho, double *gamma, double *psi,
				double *AMA_1, double *AMA_1AR, 
				double *GAR_1, double *GAR_1MA, 
				double *u, double *e, double *lnv
				) {
				
	double sum;
	long k,j,i;

    long N = lengths[0];
	long T = lengths[1];
    long nlm = lengths[2];
    long nrh = lengths[3];
    long ngm = lengths[4];
    long npsi = lengths[5];
	long rw;


	inverse(T, lambda, nlm, rho, nrh, AMA_1, AMA_1AR);

	inverse(T, gamma, ngm, psi, npsi, GAR_1, GAR_1MA);
	

	for(k=0;k<N;k++){//individual dimension

		for(i=0;i<T;i++){//time dimension
			//ARMA:
			sum = 0;
			for(j=0;j<=i;j++){//time dimesion, back tracking
				sum += AMA_1AR[j]*u[k + (i-j)*N];
				}
			e[k + i*N] = sum;
			//GARCH:
			sum =0;
			for(j=0;j<=i;j++){//time dimension, back tracking
					sum += GAR_1MA[j]*e[k + (i-j)*N]*e[k + (i-j)*N];
				}
			lnv[k + i*N] = sum;
			}
		}

    return 0;
}
	