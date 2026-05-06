/* File : ctypes.cpp */

/*Use "cl /bigobj /LD /O2 /Oi /fp:fast /GL /arch:AVX2 /DNDEBUG /EHsc ctypes.cpp /link /LTCG /OPT:REF /OPT:ICF" 
/*to compile for windows */
/*Linux suggestion (check): gcc -O3 and march=native */
/*Linux: g++ -shared -o ctypes.so -fPIC ctypes.cpp*/
/*Mac: clang++ -O3 -shared -o ctypes.dylib -fPIC ctypes.cpp*/
/*include <cstdio>
FILE *fp = fopen("coutput.txt","w"); */

#include <cmath>
#include <cstdio>
#include <cctype>
#include <iostream>

#if defined(_MSC_VER)
    // Microsoft
    #define RESTRICT __restrict
    #define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    // GCC / Clang
    #define RESTRICT __restrict
    #define EXPORT extern "C"
#else
    #define RESTRICT __restrict
    #define EXPORT extern "C"
#endif

// Bring in exprtk wrapper (evaluator_handle, exprtk_create_from_string, exprtk_eval, exprtk_destroy)
#include "mathexp.cpp"


inline void inverse(long n,
                         const double* RESTRICT x, long nx,
                         const double* RESTRICT b, long nb,
                         double* RESTRICT a,
                         double* RESTRICT ab)
{
    if (n <= 0) return;

    // a[0] = 1, ab[0] = b[0] (or 0 if nb==0)
    a[0]  = 1.0;
    ab[0] = (nb > 0) ? b[0] : 0.0;

    // Main recursion
    for (long i = 1; i < n; ++i) {
        // a[i] = - sum_{j=0..min(i-1,nx-1)} x[j] * a[i-j-1]
        const long mx = (i < nx) ? i : (nx - 1);   // note: when i>=1, mx>=0 if nx>0
        double sum_ax = 0.0;

        if (nx > 0) {
            const double* px = x;          // x[0]
            const double* pa = a + (i - 1); // a[i-1]
            for (long j = 0; j <= mx; ++j) {
                sum_ax += (*px++) * (*pa--);
            }
        }
        a[i] = -sum_ax;

        // ab[i] = sum_{j=0..min(i,nb-1)} b[j] * a[i-j]
        const long mb = (i < nb) ? i : (nb - 1);
        double sum_ab = 0.0;

        if (nb > 0) {
            const double* pb = b;      // b[0]
            const double* pa2 = a + i; // a[i]
            for (long j = 0; j <= mb; ++j) {
                sum_ab += (*pb++) * (*pa2--);
            }
        }
        ab[i] = sum_ab;
    }
}

//---------------------------------------------------------------------
// armas: FAST recursive version (no O(T^2) convolutions)
//---------------------------------------------------------------------
EXPORT int armas(double* parameters,
                 double* lambda, double* rho,
                 double* gamma,  double* psi,
                 double* AMA_1,  double* AMA_1AR,   // kept for API compatibility
                 double* GAR_1,  double* GAR_1MA,   // kept for API compatibility
                 double* u,      double* e,
                 double* var,    double* h,
                 double* W,      double* T_array,
                 char*  h_expr)
{
    const long N      = static_cast<long>(parameters[0]);
    const long T      = static_cast<long>(parameters[1]);
    const long nlm    = static_cast<long>(parameters[2]); // lambda length (lags 1..nlm)
    const long nrh    = static_cast<long>(parameters[3]); // rho length (lags 0..nrh-1)
    const long ngm    = static_cast<long>(parameters[4]); // gamma length (lags 1..ngm)
    const long npsi   = static_cast<long>(parameters[5]); // psi length (lags 0..npsi-1)
    const long egarch = static_cast<long>(parameters[6]);
    const double z    = parameters[7];

    // Optional: If callers rely on these arrays being filled, keep this.
    // (But this is NOT used for the fast computation below.)
    inverse(T, lambda, nlm, rho,  nrh,  AMA_1,  AMA_1AR);
    inverse(T, gamma,  ngm, psi,  npsi, GAR_1,  GAR_1MA);

    // Decide how to compute h
    int mode = 0; // 0 plain, 1 exprtk, 2 egarch
    evaluator_handle* h_func = nullptr;

    if (h_expr != nullptr && *h_expr != '\0') {
        mode   = 1;
        h_func = exprtk_create_from_string(h_expr);
    } else if (egarch) {
        mode = 2;
    } else {
        mode = 0;
    }

    for (long k = 0; k < N; ++k) {
        const long Tk   = static_cast<long>(T_array[k]);
        const long base = k * T;

        double*       e_k   = e   + base;
        double*       h_k   = h   + base;
        double*       var_k = var + base;
        const double* u_k   = u   + base;
        const double* W_k   = W   + base;

        for (long i = 0; i < Tk; ++i) {

            // -----------------------------
            // ARMA recursion:
            // (1 + lambda(L)) e_t = rho(L) u_t
            // where lambda is lag-1 indexed: lambda[0] is L^1 coefficient
            // and rho is lag indexed: rho[0] is L^0 coefficient (typically 1)
            // -----------------------------
            double rhs_e = 0.0;

            if (nrh > 0) {
                rhs_e += rho[0] * u_k[i];
                const long max_r = (i < (nrh - 1)) ? i : (nrh - 1);
                for (long lag = 1; lag <= max_r; ++lag) {
                    rhs_e += rho[lag] * u_k[i - lag];
                }
            } else {
                // degenerate: no rho terms -> treat as u_t
                rhs_e = u_k[i];
            }

            double et = rhs_e;
            const long max_l = (i < nlm) ? i : nlm;
            for (long lag = 1; lag <= max_l; ++lag) {
                et -= lambda[lag - 1] * e_k[i - lag];
            }
            e_k[i] = et;

            // -----------------------------
            // base GARCH term
            // -----------------------------
            double esq = et * et + 1e-8;

            // -----------------------------
            // h[i]
            // -----------------------------
            if (mode == 1) {
                if (h_func) {
                    esq    = exprtk_eval(h_func, et, esq, z);
                    h_k[i] = esq;
                } else {
                    h_k[i] = esq;
                }
            } else if (mode == 2) {
                h_k[i] = std::log(esq);
            } else {
                h_k[i] = esq;
            }

            // -----------------------------
            // VAR/GARCH recursion:
            // (1 + gamma(L)) var_t = W_t + psi(L) h_t
            // gamma is lag-1 indexed: gamma[0] is L^1 coefficient
            // psi is lag indexed: psi[0] is L^0 coefficient (often 1)
            // -----------------------------
            double rhs_v = W_k[i];

            if (npsi > 0) {
                rhs_v += psi[0] * h_k[i];
                const long max_p = (i < (npsi - 1)) ? i : (npsi - 1);
                for (long lag = 1; lag <= max_p; ++lag) {
                    rhs_v += psi[lag] * h_k[i - lag];
                }
            }

            double vt = rhs_v;
            const long max_g = (i < ngm) ? i : ngm;
            for (long lag = 1; lag <= max_g; ++lag) {
                vt -= gamma[lag - 1] * var_k[i - lag];
            }
            var_k[i] = vt;
        }
    }

    if (h_func) {
        exprtk_destroy(h_func);
    }

    return 0;
}

//---------------------------------------------------------------------
// armas: main exported routine
//---------------------------------------------------------------------
EXPORT int armas_debug(double* parameters,
                 double* lambda, double* rho,
                 double* gamma,  double* psi,
                 double* AMA_1,  double* AMA_1AR,
                 double* GAR_1,  double* GAR_1MA,
                 double* u,      double* e,
                 double* var,    double* h,
                 double* W,      double* T_array,
                 char*  h_expr)
{
    const long N      = static_cast<long>(parameters[0]);
    const long T      = static_cast<long>(parameters[1]);
    const long nlm    = static_cast<long>(parameters[2]);
    const long nrh    = static_cast<long>(parameters[3]);
    const long ngm    = static_cast<long>(parameters[4]);
    const long npsi   = static_cast<long>(parameters[5]);
    const long egarch = static_cast<long>(parameters[6]);
    const double z    = parameters[7];

    // Precompute inverse filters
    inverse(T, lambda, nlm, rho,  nrh,  AMA_1,  AMA_1AR);
    inverse(T, gamma,  ngm, psi,  npsi, GAR_1,  GAR_1MA);

    // Decide how to compute h:
    //  mode = 0 -> plain GARCH (h = esq)
    //  mode = 1 -> custom expression via exprtk
    //  mode = 2 -> EGARCH (h = log(esq))
    int mode = 0;
    evaluator_handle* h_func = nullptr;

    if (h_expr != nullptr && *h_expr != '\0') {
        mode   = 1; // exprtk
        h_func = exprtk_create_from_string(h_expr);
        // If compilation failed, h_func may still be non-null but with error_message inside.
        // armas() keeps behaviour identical to your original: just calls exprtk_eval().
    } else if (egarch) {
        mode = 2;   // EGARCH
    } else {
        mode = 0;   // plain GARCH
    }

    // Main loops
    for (long k = 0; k < N; ++k) {          // individual dimension
        const long Tk   = static_cast<long>(T_array[k]);
        const long base = k * T;

        double*       e_k   = e   + base;
        double*       h_k   = h   + base;
        double*       var_k = var + base;
        const double* u_k   = u   + base;
        const double* W_k   = W   + base;

        for (long i = 0; i < Tk; ++i) {     // time dimension
            // ---------- ARMA part ----------
            double sum = 0.0;
            for (long j = 0; j <= i; ++j) {
                sum += AMA_1AR[j] * u_k[i - j];
            }
            e_k[i] = sum;

            // ---------- base GARCH term (esq) ----------
            double esq = sum * sum + 1e-8;

            // ---------- h[i] ----------
            switch (mode) {
                case 1: // custom exprtk
                    if (h_func) {
                        esq    = exprtk_eval(h_func, sum, esq, z);
                        h_k[i] = esq;
                    } else {
                        // If exprtk failed to create a handle, fall back to plain GARCH
                        h_k[i] = esq;
                    }
                    break;

                case 2: // EGARCH
                    h_k[i] = std::log(esq);
                    break;

                default: // plain GARCH
                    h_k[i] = esq;
                    break;
            }

            // ---------- GARCH/VAR part ----------
            double gsum = 0.0;
            for (long j = 0; j <= i; ++j) {
                gsum += GAR_1[j]   * W_k[i - j]
                      + GAR_1MA[j] * h_k[i - j];
            }
            var_k[i] = gsum;
        }
    }

    // Clean up exprtk handle if we created one
    if (h_func) {
        exprtk_destroy(h_func);
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




EXPORT int fast_dot(double* __restrict r,
                    const double* __restrict a,
                    const double* __restrict b,
                    long n, long m)
{
    // Find first zero (assumes tail is all zeros)
    long n_a = n;
    for (long i = 1; i < n; ++i) {
        if (a[i] == 0.0) { n_a = i; break; }
    }

    for (long j = 0; j < m; ++j) {
        double* __restrict rcol       = r + j * n;
        const double* __restrict bcol = b + j * n;

        for (long i = 1; i < n_a; ++i) {
            const double ai = a[i];

            double* __restrict       rptr = rcol + i;
            const double* __restrict bptr = bcol;
            const long len = n - i;

            // Hint compiler for vectorization
            #if defined(__GNUC__)
            #pragma GCC ivdep
            #elif defined(_MSC_VER)
            #pragma loop(ivdep)
            #endif
            for (long k = 0; k < len; ++k) {
                rptr[k] += ai * bptr[k];
            }
        }
    }
    return 0;
}