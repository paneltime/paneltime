/* File : cfunctions.c */

/*Use "cl /LD /O2 /fp:fast ctypes.cpp" to compile for windows */
/*Linux: gcc -O3 and –march=native */

	
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

    AMA_1[0]=1.0;
	AMA_1AR[0] = rho[0];
    GAR_1[0]=1.0;
	GAR_1MA[0] = psi[0];
    for(i=1;i<T;i++){//time dimension
            //inversion ARMA:
            sum = 0;
            for(j=0;j<i && j<nlm;j++){
                sum+=lambda[j]*AMA_1[i-j-1];
            }
            AMA_1[i]=-sum;
            sum = 0;
            for(j=0;j<i+1 && j<nrh;j++){
                sum+=rho[j]*AMA_1[i-j];
            }
            AMA_1AR[i]=sum;

            //inversion GARCH:
            sum = 0;
            for(j=0;j<i && j<ngm;j++){
                sum+=gamma[j]*GAR_1[i-j-1];
            }
            GAR_1[i]=-sum;
            sum = 0;
            for(j=0;j<i+1 && j<npsi;j++){
                sum+=psi[j]*GAR_1[i-j];
            }
            GAR_1MA[i]=sum;
    }


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
	