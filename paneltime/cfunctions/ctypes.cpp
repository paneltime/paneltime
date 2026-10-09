// Build from this directory. ctypes.cpp includes mathexp.cpp.
// Windows, from an x64 MSVC Developer Command Prompt:
// cl /nologo /std:c++14 /O2 /DNDEBUG /EHsc /LD /bigobj ctypes.cpp /link /OUT:ctypes.dll
// Linux: build on the oldest supported distribution to set the glibc baseline.
// g++ -std=c++11 -O2 -DNDEBUG -fPIC -shared ctypes.cpp -o ctypes.so
// macOS: set the minimum OS target and architectures for the release you support.
// clang++ -std=c++11 -O2 -DNDEBUG -dynamiclib -fPIC ctypes.cpp -o ctypes.dylib
#define LOGGING_ENABLED 0 

#include <cmath>
#include <cstdio>
#include <cctype>
#include <iostream>
#include <cstdint>
#include <cstring>   // strerror
#include <cerrno>

// ── Logging ───────────────────────────────────────────────────────────────────
// All functions write to this file before and after every operation.
// If the process crashes, the last completed line tells you exactly where.
// Flush after every write so the log is intact even on a hard crash.
static FILE* fp = nullptr;

static void log_open() {
    if (!fp) {
        fp = fopen("coutput.txt", "w");
        if (!fp) fp = stderr;   // fallback: at least write somewhere
    }
}

#if LOGGING_ENABLED
  #define LOG(...) do { \
      log_open(); \
      fprintf(fp, __VA_ARGS__); \
      fflush(fp); \
  } while(0)
#else
  #define LOG(...) do {} while(0)   // compiles to nothing
#endif

// Log a pointer + first/last values so you can spot null or garbage arrays
static void log_array(const char* name, const double* p, long len) {
    if (!p) {
        LOG("  %s = NULL  <-- LIKELY CRASH CAUSE\n", name);
        return;
    }
    if (len <= 0) {
        LOG("  %s ptr=%p  len=%ld  (empty)\n", name, (void*)p, len);
        return;
    }
    LOG("  %s ptr=%p  len=%ld  first=%.6g  last=%.6g\n",
        name, (void*)p, len, p[0], p[len-1]);
}
// ─────────────────────────────────────────────────────────────────────────────

#if defined(_MSC_VER)
    // Microsoft
    #define RESTRICT 
    #define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    // GCC / Clang
    #define RESTRICT 
    #define EXPORT extern "C"
#else
    #define RESTRICT 
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
    LOG("  inverse() n=%ld nx=%ld nb=%ld  x=%p b=%p a=%p ab=%p\n",
        n, nx, nb, (void*)x, (void*)b, (void*)a, (void*)ab);

    if (!a || !ab)       { LOG("  inverse ERROR: null output pointer\n"); return; }
    if (!x && nx > 0)    { LOG("  inverse ERROR: x is NULL but nx=%ld\n", nx); return; }
    if (!b && nb > 0)    { LOG("  inverse ERROR: b is NULL but nb=%ld\n", nb); return; }
    if (n <= 0)          { LOG("  inverse ERROR: n=%ld\n", n); return; }

    a[0]  = 1.0;
    ab[0] = (nb > 0) ? b[0] : 0.0;
    LOG("  inverse() initial writes OK\n");

    for (long i = 1; i < n; ++i) {
        if (i % 100 == 0) LOG("  inverse() i=%ld\n", i);

        // a[i] = -sum_{j=1..min(i,nx)} x[j-1] * a[i-j]
        // j starts at 1 so a[i-j] is always a[i-1] down to a[0]: never out of bounds
        double sum_ax = 0.0;
        const long lim_a = (i < nx) ? i : nx;
        for (long j = 1; j <= lim_a; ++j) {
            sum_ax += x[j - 1] * a[i - j];
        }
        a[i] = -sum_ax;

        // ab[i] = sum_{j=0..min(i,nb-1)} b[j] * a[i-j]
        // a[i-j] is always >= a[0] since j <= i: never out of bounds
        double sum_ab = 0.0;
        const long lim_b = (i < nb) ? i : (nb - 1);
        for (long j = 0; j <= lim_b; ++j) {
            sum_ab += b[j] * a[i - j];
        }
        ab[i] = sum_ab;
    }

    LOG("  inverse() done\n");
}

//---------------------------------------------------------------------
// armas: FAST recursive version (no O(T^2) convolutions)
//---------------------------------------------------------------------
EXPORT int armas(double* parameters,
                 double* lambda, double* rho,
                 double* gamma,  double* psi,
                 double* AMA_1,  double* AMA_1AR,
                 double* GAR_1,  double* GAR_1MA,
                 double* u,      double* e,
                 double* var,    double* h,
                 double* W,      const int64_t* T_array,
                 char*  h_expr)
{
    LOG("\n=== armas ENTER ===\n");

    if (!parameters) { LOG("  ERROR: parameters is NULL\n"); return -1; }

    const long N      = static_cast<long>(parameters[0]);
    const long T      = static_cast<long>(parameters[1]);
    const long nlm    = static_cast<long>(parameters[2]);
    const long nrh    = static_cast<long>(parameters[3]);
    const long ngm    = static_cast<long>(parameters[4]);
    const long npsi   = static_cast<long>(parameters[5]);
    const long egarch = static_cast<long>(parameters[6]);
    const double z    = parameters[7];

    LOG("  N=%ld T=%ld nlm=%ld nrh=%ld ngm=%ld npsi=%ld egarch=%ld z=%.6g\n",
        N, T, nlm, nrh, ngm, npsi, egarch, z);
    LOG("  h_expr=%s\n", (h_expr && *h_expr) ? h_expr : "(none)");

    // Validate all pointers up front
    if (!lambda)  { LOG("  ERROR: lambda  is NULL\n"); return -1; }
    if (!rho)     { LOG("  ERROR: rho     is NULL\n"); return -1; }
    if (!gamma)   { LOG("  ERROR: gamma   is NULL\n"); return -1; }
    if (!psi)     { LOG("  ERROR: psi     is NULL\n"); return -1; }
    if (!AMA_1)   { LOG("  ERROR: AMA_1   is NULL\n"); return -1; }
    if (!AMA_1AR) { LOG("  ERROR: AMA_1AR is NULL\n"); return -1; }
    if (!GAR_1)   { LOG("  ERROR: GAR_1   is NULL\n"); return -1; }
    if (!GAR_1MA) { LOG("  ERROR: GAR_1MA is NULL\n"); return -1; }
    if (!u)       { LOG("  ERROR: u       is NULL\n"); return -1; }
    if (!e)       { LOG("  ERROR: e       is NULL\n"); return -1; }
    if (!var)     { LOG("  ERROR: var     is NULL\n"); return -1; }
    if (!h)       { LOG("  ERROR: h       is NULL\n"); return -1; }
    if (!W)       { LOG("  ERROR: W       is NULL\n"); return -1; }
    if (!T_array) { LOG("  ERROR: T_array is NULL\n"); return -1; }

    LOG("  All pointers OK\n");
    LOG("  Calling inverse() for AMA...\n");
    inverse(T, lambda, nlm, rho,  nrh,  AMA_1,  AMA_1AR);
    LOG("  inverse AMA done\n");
    LOG("  Calling inverse() for GAR...\n");
    inverse(T, gamma,  ngm, psi,  npsi, GAR_1,  GAR_1MA);
    LOG("  inverse GAR done\n");

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

        LOG("  series k=%ld  Tk=%ld  base=%ld\n", k, Tk, base);

        if (Tk <= 0 || Tk > T) {
            LOG("  WARNING: Tk=%ld out of range [1, T=%ld] for series k=%ld\n", Tk, T, k);
        }

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

    LOG("=== armas EXIT  OK ===\n");
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




EXPORT int fast_dot(double* r,
                    const double* a,
                    const double* b,
                    long n, long m)
{
    LOG("\n=== fast_dot ENTER  n=%ld  m=%ld ===\n", n, m);
    log_array("r", r, n * m);
    log_array("a", a, n);
    log_array("b", b, n * m);

    // ── Input validation ──────────────────────────────────────────────────────
    if (!r || !a || !b) {
        LOG("  ERROR: null pointer  r=%p a=%p b=%p\n", (void*)r, (void*)a, (void*)b);
        return -1;
    }
    if (n <= 0 || m <= 0) {
        LOG("  ERROR: bad dimensions  n=%ld m=%ld\n", n, m);
        return -2;
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Find last non-zero in a[1..n-1] to skip trailing zeros
    long n_a = 1;   // default: inner loop won't execute
    for (long i = n - 1; i >= 1; --i) {
        if (a[i] != 0.0) { n_a = i + 1; break; }
    }
    LOG("  n_a (effective lag length) = %ld\n", n_a);

    if (n_a <= 1) {
        LOG("  a[1..n-1] all zero – nothing to do, returning early\n");
        return 0;
    }

    for (long j = 0; j < m; ++j) {
        LOG("  column j=%ld\n", j);
        double*       rcol = r + j * n;
        const double* bcol = b + j * n;

        for (long i = 1; i < n_a; ++i) {
            const double  ai  = a[i];
            double*       rptr = rcol + i;
            const double* bptr = bcol;
            const long    len  = n - i;

            LOG("    i=%ld  ai=%.6g  rptr offset=%ld  len=%ld\n",
                i, ai, (long)(rptr - r), len);

#if defined(__GNUC__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
//#pragma loop(ivdep)
#endif
            for (long k = 0; k < len; ++k) {
                rptr[k] += ai * bptr[k];
            }

            LOG("    i=%ld  done\n", i);
        }
    }

    LOG("=== fast_dot EXIT  OK ===\n");
    return 0;
}