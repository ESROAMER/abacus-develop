#ifndef ABACUS_ILP64_SCALAPACK_CONNECTOR_H
#define ABACUS_ILP64_SCALAPACK_CONNECTOR_H

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include <mkl_types.h>

static_assert(sizeof(MKL_INT) == 8, "MKL_ILP64 requires 64-bit MKL_INT");

namespace abacus_ilp64_scalapack_detail
{
inline std::vector<MKL_INT> copy_in(const int* src, std::size_t count)
{
    std::vector<MKL_INT> dst(count);
    for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<MKL_INT>(src[i]);
    return dst;
}
inline void copy_out(int* dst, const std::vector<MKL_INT>& src, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<int>(src[i]);
}
inline std::size_t pivot_size(const int* desc)
{
    const int lld = desc ? std::max(0, desc[8]) : 0;
    const int mb = desc ? std::max(0, desc[4]) : 0;
    return static_cast<std::size_t>(std::max(1, lld + mb));
}
inline std::size_t workspace_size(const int* length)
{
    return static_cast<std::size_t>(std::max(1, length ? *length : 1));
}
inline std::size_t cluster_size(const int* desc)
{
    MKL_INT context = static_cast<MKL_INT>(desc ? desc[1] : 0);
    MKL_INT nprow = 0, npcol = 0, myprow = 0, mypcol = 0;
    extern void abacus_mkl_scalapack_Cblacs_gridinfo(MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("Cblacs_gridinfo");
    abacus_mkl_scalapack_Cblacs_gridinfo(context, &nprow, &npcol, &myprow, &mypcol);
    return static_cast<std::size_t>(2 * std::max<MKL_INT>(1, nprow * npcol));
}
}

extern "C"
{
MKL_INT abacus_mkl_scalapack_numroc_(const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("numroc");
void abacus_mkl_scalapack_descinit_(MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*) __asm__("descinit_");
void abacus_mkl_scalapack_pddot_(const MKL_INT*, double*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pddot_");
void abacus_mkl_scalapack_pzdotc_(const MKL_INT*, std::complex<double>*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pzdotc_");
void abacus_mkl_scalapack_pdpotrf_(char*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*) __asm__("pdpotrf_");
void abacus_mkl_scalapack_pzpotrf_(char*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*) __asm__("pzpotrf_");
void abacus_mkl_scalapack_pdtran_(const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pdtran_");
void abacus_mkl_scalapack_pztranu_(const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pztranu_");
void abacus_mkl_scalapack_pztranc_(const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pztranc_");
double abacus_mkl_scalapack_pdlange_(const char*, const MKL_INT*, const MKL_INT*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*) __asm__("pdlange_");
double abacus_mkl_scalapack_pzlange_(const char*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*) __asm__("pzlange_");
void abacus_mkl_scalapack_pzgemv_(const char*, const MKL_INT*, const MKL_INT*, const double*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pzgemv_");
void abacus_mkl_scalapack_pdgemv_(const char*, const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pdgemv_");
void abacus_mkl_scalapack_pdgemm_(const char*, const char*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pdgemm_");
void abacus_mkl_scalapack_pzgemm_(const char*, const char*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pzgemm_");
void abacus_mkl_scalapack_pdsymm_(char*, char*, MKL_INT*, MKL_INT*, double*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, double*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("pdsymm_");
void abacus_mkl_scalapack_pdtrmm_(char*, char*, char*, char*, MKL_INT*, MKL_INT*, double*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("pdtrmm_");
void abacus_mkl_scalapack_pztrmm_(char*, char*, char*, char*, MKL_INT*, MKL_INT*, std::complex<double>*, std::complex<double>*, MKL_INT*, MKL_INT*, MKL_INT*, std::complex<double>*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("pztrmm_");
void abacus_mkl_scalapack_pzhemm_(char*, char*, MKL_INT*, MKL_INT*, std::complex<double>*, std::complex<double>*, MKL_INT*, MKL_INT*, MKL_INT*, std::complex<double>*, MKL_INT*, MKL_INT*, MKL_INT*, std::complex<double>*, std::complex<double>*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("pzhemm_");
void abacus_mkl_scalapack_pzgetrf_(const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*) __asm__("pzgetrf_");
void abacus_mkl_scalapack_pzgesv_(const MKL_INT*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*) __asm__("pzgesv_");
void abacus_mkl_scalapack_pzgetri_(const MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*) __asm__("pzgetri_");
}

extern "C"
{
void abacus_mkl_scalapack_pdsygvx_(const MKL_INT*, const char*, const char*, const char*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*, MKL_INT*, MKL_INT*, double*, const double*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*, const MKL_INT*, MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*) __asm__("pdsygvx_");
void abacus_mkl_scalapack_pzhegvx_(const MKL_INT*, const char*, const char*, const char*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*, MKL_INT*, MKL_INT*, double*, const double*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, double*, const MKL_INT*, MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*) __asm__("pzhegvx_");
void abacus_mkl_scalapack_pssygvx_(const MKL_INT*, const char*, const char*, const char*, const MKL_INT*, float*, const MKL_INT*, const MKL_INT*, const MKL_INT*, float*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*, MKL_INT*, MKL_INT*, float*, const float*, float*, const MKL_INT*, const MKL_INT*, const MKL_INT*, float*, const MKL_INT*, MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*, float*, MKL_INT*) __asm__("pssygvx_");
void abacus_mkl_scalapack_pchegvx_(const MKL_INT*, const char*, const char*, const char*, const MKL_INT*, std::complex<float>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<float>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*, MKL_INT*, MKL_INT*, float*, const float*, std::complex<float>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<float>*, const MKL_INT*, float*, const MKL_INT*, MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*, float*, MKL_INT*) __asm__("pchegvx_");
void abacus_mkl_scalapack_pzgeadd_(const char*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pzgeadd_");
void abacus_mkl_scalapack_pdgemr2d_(const MKL_INT*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pdgemr2d_");
void abacus_mkl_scalapack_pzgemr2d_(const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*) __asm__("pzgemr2d_");
void abacus_mkl_scalapack_Cpigemr2d(MKL_INT, MKL_INT, int*, MKL_INT, MKL_INT, MKL_INT*, int*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cpigemr2d");
void abacus_mkl_scalapack_Cpdgemr2d(MKL_INT, MKL_INT, double*, MKL_INT, MKL_INT, MKL_INT*, double*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cpdgemr2d");
void abacus_mkl_scalapack_Cpzgemr2d(MKL_INT, MKL_INT, std::complex<double>*, MKL_INT, MKL_INT, MKL_INT*, std::complex<double>*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cpzgemr2d");
void abacus_mkl_scalapack_Cpsgemr2d(MKL_INT, MKL_INT, float*, MKL_INT, MKL_INT, MKL_INT*, float*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cpsgemr2d");
void abacus_mkl_scalapack_Cpcgemr2d(MKL_INT, MKL_INT, std::complex<float>*, MKL_INT, MKL_INT, MKL_INT*, std::complex<float>*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cpcgemr2d");
}

namespace abacus_ilp64_scalapack_detail
{
inline MKL_INT gcd(MKL_INT a, MKL_INT b)
{
    while (b != 0)
    {
        const MKL_INT r = a % b;
        a = b;
        b = r;
    }
    return a > 0 ? a : 1;
}

inline MKL_INT ceil_div(MKL_INT a, MKL_INT b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

inline MKL_INT pzgetri_iwork_size(const MKL_INT ja, const int* desc)
{
    if (desc == nullptr)
    {
        return 1;
    }

    const MKL_INT ctxt = static_cast<MKL_INT>(desc[1]);
    const MKL_INT ma = static_cast<MKL_INT>(desc[2]);
    const MKL_INT na = static_cast<MKL_INT>(desc[3]);
    const MKL_INT mb = static_cast<MKL_INT>(desc[4]);
    const MKL_INT nb = static_cast<MKL_INT>(desc[5]);
    const MKL_INT rsrc = static_cast<MKL_INT>(desc[6]);
    const MKL_INT csrc = static_cast<MKL_INT>(desc[7]);
    if (mb <= 0 || nb <= 0)
    {
        return 1;
    }

    MKL_INT nprow = 0, npcol = 0, myprow = 0, mypcol = 0;
    extern void abacus_mkl_scalapack_Cblacs_gridinfo(MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("Cblacs_gridinfo");
    abacus_mkl_scalapack_Cblacs_gridinfo(ctxt, &nprow, &npcol, &myprow, &mypcol);
    if (nprow <= 0 || npcol <= 0)
    {
        return std::max<MKL_INT>(1, na + nb);
    }

    const MKL_INT local_n = na + (ja - 1) % nb;
    const MKL_INT local_cols = abacus_mkl_scalapack_numroc_(&local_n, &nb, &mypcol, &csrc, &npcol);
    if (nprow == npcol)
    {
        return std::max<MKL_INT>(1, local_cols + nb);
    }

    const MKL_INT local_rows = abacus_mkl_scalapack_numroc_(&ma, &mb, &myprow, &rsrc, &nprow);
    const MKL_INT lcm = nprow / gcd(nprow, npcol) * npcol;
    const MKL_INT row_term = ceil_div(ceil_div(local_rows, mb), std::max<MKL_INT>(1, lcm / nprow));
    return std::max<MKL_INT>(1, local_cols + std::max(row_term, nb));
}
}

inline int abacus_ilp64_numroc_(const int* n, const int* nb, const int* iproc, const int* srcproc, const int* nprocs)
{
    const MKL_INT n64 = *n, nb64 = *nb, iproc64 = *iproc, srcproc64 = *srcproc, nprocs64 = *nprocs;
    return static_cast<int>(abacus_mkl_scalapack_numroc_(&n64, &nb64, &iproc64, &srcproc64, &nprocs64));
}

inline void abacus_ilp64_descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info)
{
    std::vector<MKL_INT> desc64(9);
    const MKL_INT m64 = *m, n64 = *n, mb64 = *mb, nb64 = *nb, irsrc64 = *irsrc, icsrc64 = *icsrc, ictxt64 = *ictxt, lld64 = *lld;
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_descinit_(desc64.data(), &m64, &n64, &mb64, &nb64, &irsrc64, &icsrc64, &ictxt64, &lld64, &info64);
    abacus_ilp64_scalapack_detail::copy_out(desc, desc64, 9);
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pddot_(int* n, double* dot, double* x, int* ix, int* jx, int* descx, int* incx, double* y, int* iy, int* jy, int* descy, int* incy)
{
    const MKL_INT n64 = *n, ix64 = *ix, jx64 = *jx, incx64 = *incx, iy64 = *iy, jy64 = *jy, incy64 = *incy;
    auto descx64 = abacus_ilp64_scalapack_detail::copy_in(descx, 9);
    auto descy64 = abacus_ilp64_scalapack_detail::copy_in(descy, 9);
    abacus_mkl_scalapack_pddot_(&n64, dot, x, &ix64, &jx64, descx64.data(), &incx64, y, &iy64, &jy64, descy64.data(), &incy64);
}

inline void abacus_ilp64_pzdotc_(int* n, std::complex<double>* dot, std::complex<double>* x, int* ix, int* jx, int* descx, int* incx, std::complex<double>* y, int* iy, int* jy, int* descy, int* incy)
{
    const MKL_INT n64 = *n, ix64 = *ix, jx64 = *jx, incx64 = *incx, iy64 = *iy, jy64 = *jy, incy64 = *incy;
    auto descx64 = abacus_ilp64_scalapack_detail::copy_in(descx, 9);
    auto descy64 = abacus_ilp64_scalapack_detail::copy_in(descy, 9);
    abacus_mkl_scalapack_pzdotc_(&n64, dot, x, &ix64, &jx64, descx64.data(), &incx64, y, &iy64, &jy64, descy64.data(), &incy64);
}

inline void abacus_ilp64_pdpotrf_(char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    const MKL_INT n64 = *n, ia64 = *ia, ja64 = *ja;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_pdpotrf_(uplo, &n64, a, &ia64, &ja64, desca64.data(), &info64);
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pzpotrf_(char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    const MKL_INT n64 = *n, ia64 = *ia, ja64 = *ja;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_pzpotrf_(uplo, &n64, a, &ia64, &ja64, desca64.data(), &info64);
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pdtran_(const int* m, const int* n, const double* alpha, const double* a, const int* ia, const int* ja, const int* desca, const double* beta, double* c, const int* ic, const int* jc, const int* descc)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pdtran_(&m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline void abacus_ilp64_pztranu_(const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* a, const int* ia, const int* ja, const int* desca, const std::complex<double>* beta, std::complex<double>* c, const int* ic, const int* jc, const int* descc)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pztranu_(&m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline void abacus_ilp64_pztranc_(const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* a, const int* ia, const int* ja, const int* desca, const std::complex<double>* beta, std::complex<double>* c, const int* ic, const int* jc, const int* descc)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pztranc_(&m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline double abacus_ilp64_pdlange_(const char* norm, const int* m, const int* n, const double* a, const int* ia, const int* ja, const int* desca, double* work)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    return abacus_mkl_scalapack_pdlange_(norm, &m64, &n64, a, &ia64, &ja64, desca64.data(), work);
}

inline double abacus_ilp64_pzlange_(const char* norm, const int* m, const int* n, const std::complex<double>* a, const int* ia, const int* ja, const int* desca, double* work)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    return abacus_mkl_scalapack_pzlange_(norm, &m64, &n64, a, &ia64, &ja64, desca64.data(), work);
}

inline void abacus_ilp64_pzgemv_(const char* transa, const int* M, const int* N, const double* alpha, const std::complex<double>* A, const int* IA, const int* JA, const int* DESCA, const std::complex<double>* B, const int* IB, const int* JB, const int* DESCB, const int* K, const double* beta, std::complex<double>* C, const int* IC, const int* JC, const int* DESCC, const int* L)
{
    const MKL_INT M64 = *M, N64 = *N, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, K64 = *K, IC64 = *IC, JC64 = *JC, L64 = *L;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    auto DESCC64 = abacus_ilp64_scalapack_detail::copy_in(DESCC, 9);
    abacus_mkl_scalapack_pzgemv_(transa, &M64, &N64, alpha, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), &K64, beta, C, &IC64, &JC64, DESCC64.data(), &L64);
}

inline void abacus_ilp64_pdgemv_(const char* transa, const int* M, const int* N, const double* alpha, const double* A, const int* IA, const int* JA, const int* DESCA, const double* B, const int* IB, const int* JB, const int* DESCB, const int* K, const double* beta, double* C, const int* IC, const int* JC, const int* DESCC, const int* L)
{
    const MKL_INT M64 = *M, N64 = *N, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, K64 = *K, IC64 = *IC, JC64 = *JC, L64 = *L;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    auto DESCC64 = abacus_ilp64_scalapack_detail::copy_in(DESCC, 9);
    abacus_mkl_scalapack_pdgemv_(transa, &M64, &N64, alpha, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), &K64, beta, C, &IC64, &JC64, DESCC64.data(), &L64);
}

inline void abacus_ilp64_pdgemm_(const char* transa, const char* transb, const int* M, const int* N, const int* K, const double* alpha, const double* A, const int* IA, const int* JA, const int* DESCA, const double* B, const int* IB, const int* JB, const int* DESCB, const double* beta, double* C, const int* IC, const int* JC, const int* DESCC)
{
    const MKL_INT M64 = *M, N64 = *N, K64 = *K, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, IC64 = *IC, JC64 = *JC;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    auto DESCC64 = abacus_ilp64_scalapack_detail::copy_in(DESCC, 9);
    abacus_mkl_scalapack_pdgemm_(transa, transb, &M64, &N64, &K64, alpha, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), beta, C, &IC64, &JC64, DESCC64.data());
}

inline void abacus_ilp64_pzgemm_(const char* transa, const char* transb, const int* M, const int* N, const int* K, const std::complex<double>* alpha, const std::complex<double>* A, const int* IA, const int* JA, const int* DESCA, const std::complex<double>* B, const int* IB, const int* JB, const int* DESCB, const std::complex<double>* beta, std::complex<double>* C, const int* IC, const int* JC, const int* DESCC)
{
    const MKL_INT M64 = *M, N64 = *N, K64 = *K, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, IC64 = *IC, JC64 = *JC;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    auto DESCC64 = abacus_ilp64_scalapack_detail::copy_in(DESCC, 9);
    abacus_mkl_scalapack_pzgemm_(transa, transb, &M64, &N64, &K64, alpha, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), beta, C, &IC64, &JC64, DESCC64.data());
}

inline void abacus_ilp64_pdsymm_(char* side, char* uplo, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pdsymm_(side, uplo, &m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), b, &ib64, &jb64, descb64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline void abacus_ilp64_pdtrmm_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_pdtrmm_(side, uplo, transa, diag, &m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), b, &ib64, &jb64, descb64.data());
}

inline void abacus_ilp64_pztrmm_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_pztrmm_(side, uplo, transa, diag, &m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), b, &ib64, &jb64, descb64.data());
}

inline void abacus_ilp64_pzhemm_(char* side, char* uplo, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pzhemm_(side, uplo, &m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), b, &ib64, &jb64, descb64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline void abacus_ilp64_pzgetrf_(const int* M, const int* N, std::complex<double>* A, const int* IA, const int* JA, const int* DESCA, int* ipiv, int* info)
{
    const MKL_INT M64 = *M, N64 = *N, IA64 = *IA, JA64 = *JA;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto ipiv64 = abacus_ilp64_scalapack_detail::copy_in(ipiv, abacus_ilp64_scalapack_detail::pivot_size(DESCA));
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_pzgetrf_(&M64, &N64, A, &IA64, &JA64, DESCA64.data(), ipiv64.data(), &info64);
    abacus_ilp64_scalapack_detail::copy_out(ipiv, ipiv64, ipiv64.size());
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pzgesv_(const int* n, const int* nrhs, const std::complex<double>* A, const int* ia, const int* ja, const int* desca, int* ipiv, std::complex<double>* B, const int* ib, const int* jb, const int* descb, int* info)
{
    const MKL_INT n64 = *n, nrhs64 = *nrhs, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto ipiv64 = abacus_ilp64_scalapack_detail::copy_in(ipiv, abacus_ilp64_scalapack_detail::pivot_size(desca));
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_pzgesv_(&n64, &nrhs64, A, &ia64, &ja64, desca64.data(), ipiv64.data(), B, &ib64, &jb64, descb64.data(), &info64);
    abacus_ilp64_scalapack_detail::copy_out(ipiv, ipiv64, ipiv64.size());
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pzgetri_(const int* n, const std::complex<double>* A, const int* ia, const int* ja, const int* desca, int* ipiv, const std::complex<double>* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    const MKL_INT n64 = *n, ia64 = *ia, ja64 = *ja, lwork64 = *lwork;
    const bool query_iwork = *liwork < 0;
    MKL_INT liwork64 = query_iwork ? abacus_ilp64_scalapack_detail::pzgetri_iwork_size(ja64, desca) : *liwork;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto ipiv64 = abacus_ilp64_scalapack_detail::copy_in(ipiv, abacus_ilp64_scalapack_detail::pivot_size(desca));
    auto iwork64 = query_iwork
        ? std::vector<MKL_INT>(static_cast<std::size_t>(std::max<MKL_INT>(1, liwork64)), 0)
        : abacus_ilp64_scalapack_detail::copy_in(iwork, abacus_ilp64_scalapack_detail::workspace_size(liwork));
    MKL_INT info64 = 0;
    abacus_mkl_scalapack_pzgetri_(&n64, A, &ia64, &ja64, desca64.data(), ipiv64.data(), work, &lwork64, iwork64.data(), &liwork64, &info64);
    if (query_iwork)
    {
        *iwork = static_cast<int>(iwork64[0]);
    }
    *info = static_cast<int>(info64);
}

inline void abacus_ilp64_pzgeadd_(const char* transa, const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* a, const int* ia, const int* ja, const int* desca, const std::complex<double>* beta, const std::complex<double>* c, const int* ic, const int* jc, const int* descc)
{
    const MKL_INT m64 = *m, n64 = *n, ia64 = *ia, ja64 = *ja, ic64 = *ic, jc64 = *jc;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descc64 = abacus_ilp64_scalapack_detail::copy_in(descc, 9);
    abacus_mkl_scalapack_pzgeadd_(transa, &m64, &n64, alpha, a, &ia64, &ja64, desca64.data(), beta, c, &ic64, &jc64, descc64.data());
}

inline void abacus_ilp64_pdgemr2d_(const int* M, const int* N, double* A, const int* IA, const int* JA, const int* DESCA, double* B, const int* IB, const int* JB, const int* DESCB, const int* ICTXT)
{
    const MKL_INT M64 = *M, N64 = *N, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, ICTXT64 = *ICTXT;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    abacus_mkl_scalapack_pdgemr2d_(&M64, &N64, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), &ICTXT64);
}

inline void abacus_ilp64_pzgemr2d_(const int* M, const int* N, std::complex<double>* A, const int* IA, const int* JA, const int* DESCA, std::complex<double>* B, const int* IB, const int* JB, const int* DESCB, const int* ICTXT)
{
    const MKL_INT M64 = *M, N64 = *N, IA64 = *IA, JA64 = *JA, IB64 = *IB, JB64 = *JB, ICTXT64 = *ICTXT;
    auto DESCA64 = abacus_ilp64_scalapack_detail::copy_in(DESCA, 9);
    auto DESCB64 = abacus_ilp64_scalapack_detail::copy_in(DESCB, 9);
    abacus_mkl_scalapack_pzgemr2d_(&M64, &N64, A, &IA64, &JA64, DESCA64.data(), B, &IB64, &JB64, DESCB64.data(), &ICTXT64);
}

inline void abacus_ilp64_Cpigemr2d(int m, int n, int* A, int ia, int ja, int* desca, int* B, int ib, int jb, int* descb, int ictxt)
{
    const MKL_INT m64 = m, n64 = n, ia64 = ia, ja64 = ja, ib64 = ib, jb64 = jb, ictxt64 = ictxt;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_Cpigemr2d(m64, n64, A, ia64, ja64, desca64.data(), B, ib64, jb64, descb64.data(), ictxt64);
}

inline void abacus_ilp64_Cpdgemr2d(int m, int n, double* A, int ia, int ja, int* desca, double* B, int ib, int jb, int* descb, int ictxt)
{
    const MKL_INT m64 = m, n64 = n, ia64 = ia, ja64 = ja, ib64 = ib, jb64 = jb, ictxt64 = ictxt;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_Cpdgemr2d(m64, n64, A, ia64, ja64, desca64.data(), B, ib64, jb64, descb64.data(), ictxt64);
}

inline void abacus_ilp64_Cpzgemr2d(int m, int n, std::complex<double>* A, int ia, int ja, int* desca, std::complex<double>* B, int ib, int jb, int* descb, int ictxt)
{
    const MKL_INT m64 = m, n64 = n, ia64 = ia, ja64 = ja, ib64 = ib, jb64 = jb, ictxt64 = ictxt;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_Cpzgemr2d(m64, n64, A, ia64, ja64, desca64.data(), B, ib64, jb64, descb64.data(), ictxt64);
}

inline void abacus_ilp64_Cpsgemr2d(int m, int n, float* A, int ia, int ja, int* desca, float* B, int ib, int jb, int* descb, int ictxt)
{
    const MKL_INT m64 = m, n64 = n, ia64 = ia, ja64 = ja, ib64 = ib, jb64 = jb, ictxt64 = ictxt;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_Cpsgemr2d(m64, n64, A, ia64, ja64, desca64.data(), B, ib64, jb64, descb64.data(), ictxt64);
}

inline void abacus_ilp64_Cpcgemr2d(int m, int n, std::complex<float>* A, int ia, int ja, int* desca, std::complex<float>* B, int ib, int jb, int* descb, int ictxt)
{
    const MKL_INT m64 = m, n64 = n, ia64 = ia, ja64 = ja, ib64 = ib, jb64 = jb, ictxt64 = ictxt;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    abacus_mkl_scalapack_Cpcgemr2d(m64, n64, A, ia64, ja64, desca64.data(), B, ib64, jb64, descb64.data(), ictxt64);
}

#define numroc_ abacus_ilp64_numroc_
#define descinit_ abacus_ilp64_descinit_
#define pddot_ abacus_ilp64_pddot_
#define pzdotc_ abacus_ilp64_pzdotc_
#define pdpotrf_ abacus_ilp64_pdpotrf_
#define pzpotrf_ abacus_ilp64_pzpotrf_
#define pdtran_ abacus_ilp64_pdtran_
#define pztranu_ abacus_ilp64_pztranu_
#define pztranc_ abacus_ilp64_pztranc_
#define pdlange_ abacus_ilp64_pdlange_
#define pzlange_ abacus_ilp64_pzlange_
#define pzgemv_ abacus_ilp64_pzgemv_
#define pdgemv_ abacus_ilp64_pdgemv_
#define pdgemm_ abacus_ilp64_pdgemm_
#define pzgemm_ abacus_ilp64_pzgemm_
#define pdsymm_ abacus_ilp64_pdsymm_
#define pdtrmm_ abacus_ilp64_pdtrmm_
#define pztrmm_ abacus_ilp64_pztrmm_
#define pzhemm_ abacus_ilp64_pzhemm_
#define pzgetrf_ abacus_ilp64_pzgetrf_
#define pzgesv_ abacus_ilp64_pzgesv_
#define pzgetri_ abacus_ilp64_pzgetri_
#define pdsygvx_ abacus_ilp64_pdsygvx_
#define pzhegvx_ abacus_ilp64_pzhegvx_
#define pssygvx_ abacus_ilp64_pssygvx_
#define pchegvx_ abacus_ilp64_pchegvx_
#define pzgeadd_ abacus_ilp64_pzgeadd_
#define pdgemr2d_ abacus_ilp64_pdgemr2d_
#define pzgemr2d_ abacus_ilp64_pzgemr2d_
#define Cpigemr2d abacus_ilp64_Cpigemr2d
#define Cpdgemr2d abacus_ilp64_Cpdgemr2d
#define Cpzgemr2d abacus_ilp64_Cpzgemr2d
#define Cpsgemr2d abacus_ilp64_Cpsgemr2d
#define Cpcgemr2d abacus_ilp64_Cpcgemr2d

#endif

inline void abacus_ilp64_pdsygvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, double* A, const int* ia, const int* ja, const int* desca, double* B, const int* ib, const int* jb, const int* descb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, int* nz, double* w, const double* orfac, double* Z, const int* iz, const int* jz, const int* descz, double* work, int* lwork, int* iwork, int* liwork, int* ifail, int* iclustr, double* gap, int* info)
{
    const MKL_INT itype64 = *itype, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, il64 = *il, iu64 = *iu, iz64 = *iz, jz64 = *jz, lwork64 = *lwork;
    MKL_INT m64 = 0, nz64 = 0, liwork64 = *liwork, info64 = 0;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descz64 = abacus_ilp64_scalapack_detail::copy_in(descz, 9);
    auto iwork64 = abacus_ilp64_scalapack_detail::copy_in(iwork, abacus_ilp64_scalapack_detail::workspace_size(liwork));
    auto ifail64 = abacus_ilp64_scalapack_detail::copy_in(ifail, static_cast<std::size_t>(std::max(1, *n)));
    auto iclustr64 = abacus_ilp64_scalapack_detail::copy_in(iclustr, abacus_ilp64_scalapack_detail::cluster_size(desca));
    abacus_mkl_scalapack_pdsygvx_(&itype64, jobz, range, uplo, &n64, A, &ia64, &ja64, desca64.data(), B, &ib64, &jb64, descb64.data(), vl, vu, &il64, &iu64, abstol, &m64, &nz64, w, orfac, Z, &iz64, &jz64, descz64.data(), work, &lwork64, iwork64.data(), &liwork64, ifail64.data(), iclustr64.data(), gap, &info64);
    *m = static_cast<int>(m64); *nz = static_cast<int>(nz64); *liwork = static_cast<int>(liwork64); *info = static_cast<int>(info64);
    abacus_ilp64_scalapack_detail::copy_out(iwork, iwork64, iwork64.size());
    abacus_ilp64_scalapack_detail::copy_out(ifail, ifail64, ifail64.size());
    abacus_ilp64_scalapack_detail::copy_out(iclustr, iclustr64, iclustr64.size());
}

inline void abacus_ilp64_pzhegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<double>* A, const int* ia, const int* ja, const int* desca, std::complex<double>* B, const int* ib, const int* jb, const int* descb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, int* nz, double* w, const double* orfac, std::complex<double>* Z, const int* iz, const int* jz, const int* descz, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* iwork, int* liwork, int* ifail, int* iclustr, double* gap, int* info)
{
    const MKL_INT itype64 = *itype, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, il64 = *il, iu64 = *iu, iz64 = *iz, jz64 = *jz, lwork64 = *lwork, lrwork64 = *lrwork;
    MKL_INT m64 = 0, nz64 = 0, liwork64 = *liwork, info64 = 0;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descz64 = abacus_ilp64_scalapack_detail::copy_in(descz, 9);
    auto iwork64 = abacus_ilp64_scalapack_detail::copy_in(iwork, abacus_ilp64_scalapack_detail::workspace_size(liwork));
    auto ifail64 = abacus_ilp64_scalapack_detail::copy_in(ifail, static_cast<std::size_t>(std::max(1, *n)));
    auto iclustr64 = abacus_ilp64_scalapack_detail::copy_in(iclustr, abacus_ilp64_scalapack_detail::cluster_size(desca));
    abacus_mkl_scalapack_pzhegvx_(&itype64, jobz, range, uplo, &n64, A, &ia64, &ja64, desca64.data(), B, &ib64, &jb64, descb64.data(), vl, vu, &il64, &iu64, abstol, &m64, &nz64, w, orfac, Z, &iz64, &jz64, descz64.data(), work, &lwork64, rwork, &lrwork64, iwork64.data(), &liwork64, ifail64.data(), iclustr64.data(), gap, &info64);
    *m = static_cast<int>(m64); *nz = static_cast<int>(nz64); *liwork = static_cast<int>(liwork64); *info = static_cast<int>(info64);
    abacus_ilp64_scalapack_detail::copy_out(iwork, iwork64, iwork64.size());
    abacus_ilp64_scalapack_detail::copy_out(ifail, ifail64, ifail64.size());
    abacus_ilp64_scalapack_detail::copy_out(iclustr, iclustr64, iclustr64.size());
}

inline void abacus_ilp64_pssygvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, float* A, const int* ia, const int* ja, const int* desca, float* B, const int* ib, const int* jb, const int* descb, const float* vl, const float* vu, const int* il, const int* iu, const float* abstol, int* m, int* nz, float* w, const float* orfac, float* Z, const int* iz, const int* jz, const int* descz, float* work, int* lwork, int* iwork, int* liwork, int* ifail, int* iclustr, float* gap, int* info)
{
    const MKL_INT itype64 = *itype, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, il64 = *il, iu64 = *iu, iz64 = *iz, jz64 = *jz, lwork64 = *lwork;
    MKL_INT m64 = 0, nz64 = 0, liwork64 = *liwork, info64 = 0;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descz64 = abacus_ilp64_scalapack_detail::copy_in(descz, 9);
    auto iwork64 = abacus_ilp64_scalapack_detail::copy_in(iwork, abacus_ilp64_scalapack_detail::workspace_size(liwork));
    auto ifail64 = abacus_ilp64_scalapack_detail::copy_in(ifail, static_cast<std::size_t>(std::max(1, *n)));
    auto iclustr64 = abacus_ilp64_scalapack_detail::copy_in(iclustr, abacus_ilp64_scalapack_detail::cluster_size(desca));
    abacus_mkl_scalapack_pssygvx_(&itype64, jobz, range, uplo, &n64, A, &ia64, &ja64, desca64.data(), B, &ib64, &jb64, descb64.data(), vl, vu, &il64, &iu64, abstol, &m64, &nz64, w, orfac, Z, &iz64, &jz64, descz64.data(), work, &lwork64, iwork64.data(), &liwork64, ifail64.data(), iclustr64.data(), gap, &info64);
    *m = static_cast<int>(m64); *nz = static_cast<int>(nz64); *liwork = static_cast<int>(liwork64); *info = static_cast<int>(info64);
    abacus_ilp64_scalapack_detail::copy_out(iwork, iwork64, iwork64.size());
    abacus_ilp64_scalapack_detail::copy_out(ifail, ifail64, ifail64.size());
    abacus_ilp64_scalapack_detail::copy_out(iclustr, iclustr64, iclustr64.size());
}

inline void abacus_ilp64_pchegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<float>* A, const int* ia, const int* ja, const int* desca, std::complex<float>* B, const int* ib, const int* jb, const int* descb, const float* vl, const float* vu, const int* il, const int* iu, const float* abstol, int* m, int* nz, float* w, const float* orfac, std::complex<float>* Z, const int* iz, const int* jz, const int* descz, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* iwork, int* liwork, int* ifail, int* iclustr, float* gap, int* info)
{
    const MKL_INT itype64 = *itype, n64 = *n, ia64 = *ia, ja64 = *ja, ib64 = *ib, jb64 = *jb, il64 = *il, iu64 = *iu, iz64 = *iz, jz64 = *jz, lwork64 = *lwork, lrwork64 = *lrwork;
    MKL_INT m64 = 0, nz64 = 0, liwork64 = *liwork, info64 = 0;
    auto desca64 = abacus_ilp64_scalapack_detail::copy_in(desca, 9);
    auto descb64 = abacus_ilp64_scalapack_detail::copy_in(descb, 9);
    auto descz64 = abacus_ilp64_scalapack_detail::copy_in(descz, 9);
    auto iwork64 = abacus_ilp64_scalapack_detail::copy_in(iwork, abacus_ilp64_scalapack_detail::workspace_size(liwork));
    auto ifail64 = abacus_ilp64_scalapack_detail::copy_in(ifail, static_cast<std::size_t>(std::max(1, *n)));
    auto iclustr64 = abacus_ilp64_scalapack_detail::copy_in(iclustr, abacus_ilp64_scalapack_detail::cluster_size(desca));
    abacus_mkl_scalapack_pchegvx_(&itype64, jobz, range, uplo, &n64, A, &ia64, &ja64, desca64.data(), B, &ib64, &jb64, descb64.data(), vl, vu, &il64, &iu64, abstol, &m64, &nz64, w, orfac, Z, &iz64, &jz64, descz64.data(), work, &lwork64, rwork, &lrwork64, iwork64.data(), &liwork64, ifail64.data(), iclustr64.data(), gap, &info64);
    *m = static_cast<int>(m64); *nz = static_cast<int>(nz64); *liwork = static_cast<int>(liwork64); *info = static_cast<int>(info64);
    abacus_ilp64_scalapack_detail::copy_out(iwork, iwork64, iwork64.size());
    abacus_ilp64_scalapack_detail::copy_out(ifail, ifail64, ifail64.size());
    abacus_ilp64_scalapack_detail::copy_out(iclustr, iclustr64, iclustr64.size());
}
