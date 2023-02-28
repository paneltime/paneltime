/* File : ctypes_c.c */

/*Use "cl /LD ctypes_c.c" to compile for windows */

//#include <stdio.h>
#ifdef linux
#include <cmath>
#elif _WIN32
#include <math.h>
#endif

double min(double a, double b){
    if(a>b){
	return b;
	}
	else
	{
	return a;
		}
	}

void inverse(long n, double *x_args, long nx, double *b_args, long nb, 
             double *a, double *ab) {

                    long j,i;

    double sum_ax;
    double sum_ab;

    a[0]=1.0;
    ab[0] = b_args[0];

    for(i=1;i<n;i++){
        sum_ax=0;
        sum_ab=0;
        for(j=0;j<min(nx,i);j++){
            sum_ax+=x_args[j]*a[i-j-1];
        }
        a[i]=-sum_ax;
        for(j=0;j<min(nb,i+1);j++){
            sum_ab+=b_args[j]*a[i-j];
        }
        ab[i]=sum_ab;
    }

}



__declspec(dllexport) int  armas(double *parameters, 
                                 double *lambda, double *rho, double *gamma, double *psi,
                double *AMA_1, double *AMA_1AR, 
                double *GAR_1, double *GAR_1MA, 
                double *u, double *e, double *var, double *h
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
	long lost_obs = (int) parameters[7];
	double egarch_add = parameters[8];
	long rw;



    inverse(T, lambda, nlm, rho, nrh, AMA_1, AMA_1AR);

    inverse(T, gamma, ngm, psi, npsi, GAR_1, GAR_1MA);

    //FILE *fp;




    for(k=0;k<N;k++){//individual dimension

		for(i=0;i<T;i++){//time dimension
			//ARMA:
            sum = 0;
            for(j=0;j<=i;j++){//time dimesion, back tracking
				sum += AMA_1AR[j]*u[k + (i-j)*N];
                }
            e[k + i*N] = sum;
            //GARCH:
			if(i>=lost_obs){
				h[k + i*N] = sum*sum;
				if(egarch){
					h[k + i*N] += egarch_add;
					h[k + i*N] = log(h[k + i*N] + (h[k + i*N]==0)*1e-18);
				}
			}
			sum =0;
			for(j=0;j<=i;j++){//time dimension, back tracking
				sum += GAR_1MA[j]*h[k + (i-j)*N];
			}
			var[k + i*N] = sum;
            }
        }

	return 0;
}