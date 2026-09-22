#ifndef ABACUS_ILP64_BLAS_CONNECTOR_H
#define ABACUS_ILP64_BLAS_CONNECTOR_H

#include <complex>
#include <mkl_types.h>

static_assert(sizeof(MKL_INT) == 8, "ABACUS MKL ILP64 requires 8-byte MKL_INT");

extern "C"
{
void abacus_mkl_sscal_(const MKL_INT * N, const float * alpha, float * X, const MKL_INT * incX) __asm__("sscal_");
void abacus_mkl_dscal_(const MKL_INT * N, const double * alpha, double * X, const MKL_INT * incX) __asm__("dscal_");
void abacus_mkl_cscal_(const MKL_INT * N, const std::complex<float> * alpha, std::complex<float> * X, const MKL_INT * incX) __asm__("cscal_");
void abacus_mkl_zscal_(const MKL_INT * N, const std::complex<double> * alpha, std::complex<double> * X, const MKL_INT * incX) __asm__("zscal_");
void abacus_mkl_saxpy_(const MKL_INT * N, const float * alpha, const float * X, const MKL_INT * incX, float * Y, const MKL_INT * incY) __asm__("saxpy_");
void abacus_mkl_daxpy_(const MKL_INT * N, const double * alpha, const double * X, const MKL_INT * incX, double * Y, const MKL_INT * incY) __asm__("daxpy_");
void abacus_mkl_caxpy_(const MKL_INT * N, const std::complex<float> * alpha, const std::complex<float> * X, const MKL_INT * incX, std::complex<float> * Y, const MKL_INT * incY) __asm__("caxpy_");
void abacus_mkl_zaxpy_(const MKL_INT * N, const std::complex<double> * alpha, const std::complex<double> * X, const MKL_INT * incX, std::complex<double> * Y, const MKL_INT * incY) __asm__("zaxpy_");
void abacus_mkl_scopy_(const MKL_INT * n, const float * a, const MKL_INT * incx, float * b, const MKL_INT * incy) __asm__("scopy_");
void abacus_mkl_dcopy_(const MKL_INT * n, const double * a, const MKL_INT * incx, double * b, const MKL_INT * incy) __asm__("dcopy_");
void abacus_mkl_ccopy_(const MKL_INT * n, const std::complex<float> * a, const MKL_INT * incx, std::complex<float> * b, const MKL_INT * incy) __asm__("ccopy_");
void abacus_mkl_zcopy_(const MKL_INT * n, const std::complex<double> * a, const MKL_INT * incx, std::complex<double> * b, const MKL_INT * incy) __asm__("zcopy_");
float abacus_mkl_sdot_(const MKL_INT * N, const float * X, const MKL_INT * incX, const float * Y, const MKL_INT * incY) __asm__("sdot_");
double abacus_mkl_ddot_(const MKL_INT * N, const double * X, const MKL_INT * incX, const double * Y, const MKL_INT * incY) __asm__("ddot_");
float abacus_mkl_snrm2_(const MKL_INT * n, const float * X, const MKL_INT * incX) __asm__("snrm2_");
double abacus_mkl_dnrm2_(const MKL_INT * n, const double * X, const MKL_INT * incX) __asm__("dnrm2_");
float abacus_mkl_scnrm2_(const MKL_INT * n, const std::complex<float> * X, const MKL_INT * incX) __asm__("scnrm2_");
double abacus_mkl_dznrm2_(const MKL_INT * n, const std::complex<double> * X, const MKL_INT * incX) __asm__("dznrm2_");
void abacus_mkl_sgemv_(const char * transa, const MKL_INT * m, const MKL_INT * n, const float * alpha, const float * a, const MKL_INT * lda, const float * x, const MKL_INT * incx, const float * beta, float * y, const MKL_INT * incy) __asm__("sgemv_");
void abacus_mkl_dgemv_(const char * transa, const MKL_INT * m, const MKL_INT * n, const double * alpha, const double * a, const MKL_INT * lda, const double * x, const MKL_INT * incx, const double * beta, double * y, const MKL_INT * incy) __asm__("dgemv_");
void abacus_mkl_cgemv_(const char * trans, const MKL_INT * m, const MKL_INT * n, const std::complex<float> * alpha, const std::complex<float> * a, const MKL_INT * lda, const std::complex<float> * x, const MKL_INT * incx, const std::complex<float> * beta, std::complex<float> * y, const MKL_INT * incy) __asm__("cgemv_");
void abacus_mkl_zgemv_(const char * trans, const MKL_INT * m, const MKL_INT * n, const std::complex<double> * alpha, const std::complex<double> * a, const MKL_INT * lda, const std::complex<double> * x, const MKL_INT * incx, const std::complex<double> * beta, std::complex<double> * y, const MKL_INT * incy) __asm__("zgemv_");
void abacus_mkl_dsymv_(const char * uplo, const MKL_INT * n, const double * alpha, const double * a, const MKL_INT * lda, const double * x, const MKL_INT * incx, const double * beta, double * y, const MKL_INT * incy) __asm__("dsymv_");
void abacus_mkl_dger_(const MKL_INT * m, const MKL_INT * n, const double * alpha, const double * x, const MKL_INT * incx, const double * y, const MKL_INT * incy, double * a, const MKL_INT * lda) __asm__("dger_");
void abacus_mkl_zgerc_(const MKL_INT * m, const MKL_INT * n, const std::complex<double> * alpha, const std::complex<double> * x, const MKL_INT * incx, const std::complex<double> * y, const MKL_INT * incy, std::complex<double> * a, const MKL_INT * lda) __asm__("zgerc_");
void abacus_mkl_sgemm_(const char * transa, const char * transb, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const float * alpha, const float * a, const MKL_INT * lda, const float * b, const MKL_INT * ldb, const float * beta, float * c, const MKL_INT * ldc) __asm__("sgemm_");
void abacus_mkl_dgemm_(const char * transa, const char * transb, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const double * alpha, const double * a, const MKL_INT * lda, const double * b, const MKL_INT * ldb, const double * beta, double * c, const MKL_INT * ldc) __asm__("dgemm_");
void abacus_mkl_cgemm_(const char * transa, const char * transb, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const std::complex<float> * alpha, const std::complex<float> * a, const MKL_INT * lda, const std::complex<float> * b, const MKL_INT * ldb, const std::complex<float> * beta, std::complex<float> * c, const MKL_INT * ldc) __asm__("cgemm_");
void abacus_mkl_zgemm_(const char * transa, const char * transb, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const std::complex<double> * alpha, const std::complex<double> * a, const MKL_INT * lda, const std::complex<double> * b, const MKL_INT * ldb, const std::complex<double> * beta, std::complex<double> * c, const MKL_INT * ldc) __asm__("zgemm_");
void abacus_mkl_ssymm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const float * alpha, const float * a, const MKL_INT * lda, const float * b, const MKL_INT * ldb, const float * beta, float * c, const MKL_INT * ldc) __asm__("ssymm_");
void abacus_mkl_dsymm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const double * alpha, const double * a, const MKL_INT * lda, const double * b, const MKL_INT * ldb, const double * beta, double * c, const MKL_INT * ldc) __asm__("dsymm_");
void abacus_mkl_csymm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const std::complex<float> * alpha, const std::complex<float> * a, const MKL_INT * lda, const std::complex<float> * b, const MKL_INT * ldb, const std::complex<float> * beta, std::complex<float> * c, const MKL_INT * ldc) __asm__("csymm_");
void abacus_mkl_zsymm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const std::complex<double> * alpha, const std::complex<double> * a, const MKL_INT * lda, const std::complex<double> * b, const MKL_INT * ldb, const std::complex<double> * beta, std::complex<double> * c, const MKL_INT * ldc) __asm__("zsymm_");
void abacus_mkl_chemm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const std::complex<float> * alpha, const std::complex<float> * a, const MKL_INT * lda, const std::complex<float> * b, const MKL_INT * ldb, const std::complex<float> * beta, std::complex<float> * c, const MKL_INT * ldc) __asm__("chemm_");
void abacus_mkl_zhemm_(const char * side, const char * uplo, const MKL_INT * m, const MKL_INT * n, const std::complex<double> * alpha, const std::complex<double> * a, const MKL_INT * lda, const std::complex<double> * b, const MKL_INT * ldb, const std::complex<double> * beta, std::complex<double> * c, const MKL_INT * ldc) __asm__("zhemm_");
void abacus_mkl_dtrsm_(const char * side, const char * uplo, const char * transa, const char * diag, const MKL_INT * m, const MKL_INT * n, const double * alpha, const double * a, const MKL_INT * lda, double * b, const MKL_INT * ldb) __asm__("dtrsm_");
void abacus_mkl_ztrsm_(const char * side, const char * uplo, const char * transa, const char * diag, const MKL_INT * m, const MKL_INT * n, const std::complex<double> * alpha, const std::complex<double> * a, const MKL_INT * lda, std::complex<double> * b, const MKL_INT * ldb) __asm__("ztrsm_");
void abacus_mkl_cherk_(const char* uplo, const char* trans, const MKL_INT* n, const MKL_INT* k, const float* alpha, const std::complex<float>* a, const MKL_INT* lda, const float* beta, std::complex<float>* c, const MKL_INT* ldc) __asm__("cherk_");
void abacus_mkl_zherk_(const char* uplo, const char* trans, const MKL_INT* n, const MKL_INT* k, const double* alpha, const std::complex<double>* a, const MKL_INT* lda, const double* beta, std::complex<double>* c, const MKL_INT* ldc) __asm__("zherk_");
void abacus_mkl_dsyrk_(const char* uplo, const char* trans, const MKL_INT* n, const MKL_INT* k, const double* alpha, const double* a, const MKL_INT* lda, const double* beta, double* c, const MKL_INT* ldc) __asm__("dsyrk_");
}

extern "C"
{
inline void abacus_ilp64_sscal_(const int * N, const float * alpha, float * X, const int * incX)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    abacus_mkl_sscal_(&N_ilp64, alpha, X, &incX_ilp64);
}

inline void abacus_ilp64_dscal_(const int * N, const double * alpha, double * X, const int * incX)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    abacus_mkl_dscal_(&N_ilp64, alpha, X, &incX_ilp64);
}

inline void abacus_ilp64_cscal_(const int * N, const std::complex<float> * alpha, std::complex<float> * X, const int * incX)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    abacus_mkl_cscal_(&N_ilp64, alpha, X, &incX_ilp64);
}

inline void abacus_ilp64_zscal_(const int * N, const std::complex<double> * alpha, std::complex<double> * X, const int * incX)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    abacus_mkl_zscal_(&N_ilp64, alpha, X, &incX_ilp64);
}

inline void abacus_ilp64_saxpy_(const int * N, const float * alpha, const float * X, const int * incX, float * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    abacus_mkl_saxpy_(&N_ilp64, alpha, X, &incX_ilp64, Y, &incY_ilp64);
}

inline void abacus_ilp64_daxpy_(const int * N, const double * alpha, const double * X, const int * incX, double * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    abacus_mkl_daxpy_(&N_ilp64, alpha, X, &incX_ilp64, Y, &incY_ilp64);
}

inline void abacus_ilp64_caxpy_(const int * N, const std::complex<float> * alpha, const std::complex<float> * X, const int * incX, std::complex<float> * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    abacus_mkl_caxpy_(&N_ilp64, alpha, X, &incX_ilp64, Y, &incY_ilp64);
}

inline void abacus_ilp64_zaxpy_(const int * N, const std::complex<double> * alpha, const std::complex<double> * X, const int * incX, std::complex<double> * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    abacus_mkl_zaxpy_(&N_ilp64, alpha, X, &incX_ilp64, Y, &incY_ilp64);
}

inline void abacus_ilp64_scopy_(const int * n, const float * a, const int * incx, float * b, const int * incy)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_scopy_(&n_ilp64, a, &incx_ilp64, b, &incy_ilp64);
}

inline void abacus_ilp64_dcopy_(const int * n, const double * a, const int * incx, double * b, const int * incy)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_dcopy_(&n_ilp64, a, &incx_ilp64, b, &incy_ilp64);
}

inline void abacus_ilp64_ccopy_(const int * n, const std::complex<float> * a, const int * incx, std::complex<float> * b, const int * incy)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_ccopy_(&n_ilp64, a, &incx_ilp64, b, &incy_ilp64);
}

inline void abacus_ilp64_zcopy_(const int * n, const std::complex<double> * a, const int * incx, std::complex<double> * b, const int * incy)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_zcopy_(&n_ilp64, a, &incx_ilp64, b, &incy_ilp64);
}

inline float abacus_ilp64_sdot_(const int * N, const float * X, const int * incX, const float * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    return abacus_mkl_sdot_(&N_ilp64, X, &incX_ilp64, Y, &incY_ilp64);
}

inline double abacus_ilp64_ddot_(const int * N, const double * X, const int * incX, const double * Y, const int * incY)
{
    const MKL_INT N_ilp64 = static_cast<MKL_INT>(*N);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    const MKL_INT incY_ilp64 = static_cast<MKL_INT>(*incY);
    return abacus_mkl_ddot_(&N_ilp64, X, &incX_ilp64, Y, &incY_ilp64);
}

inline float abacus_ilp64_snrm2_(const int * n, const float * X, const int * incX)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    return abacus_mkl_snrm2_(&n_ilp64, X, &incX_ilp64);
}

inline double abacus_ilp64_dnrm2_(const int * n, const double * X, const int * incX)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    return abacus_mkl_dnrm2_(&n_ilp64, X, &incX_ilp64);
}

inline float abacus_ilp64_scnrm2_(const int * n, const std::complex<float> * X, const int * incX)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    return abacus_mkl_scnrm2_(&n_ilp64, X, &incX_ilp64);
}

inline double abacus_ilp64_dznrm2_(const int * n, const std::complex<double> * X, const int * incX)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incX_ilp64 = static_cast<MKL_INT>(*incX);
    return abacus_mkl_dznrm2_(&n_ilp64, X, &incX_ilp64);
}

inline void abacus_ilp64_sgemv_(const char * transa, const int * m, const int * n, const float * alpha, const float * a, const int * lda, const float * x, const int * incx, const float * beta, float * y, const int * incy)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_sgemv_(transa, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, x, &incx_ilp64, beta, y, &incy_ilp64);
}

inline void abacus_ilp64_dgemv_(const char * transa, const int * m, const int * n, const double * alpha, const double * a, const int * lda, const double * x, const int * incx, const double * beta, double * y, const int * incy)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_dgemv_(transa, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, x, &incx_ilp64, beta, y, &incy_ilp64);
}

inline void abacus_ilp64_cgemv_(const char * trans, const int * m, const int * n, const std::complex<float> * alpha, const std::complex<float> * a, const int * lda, const std::complex<float> * x, const int * incx, const std::complex<float> * beta, std::complex<float> * y, const int * incy)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_cgemv_(trans, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, x, &incx_ilp64, beta, y, &incy_ilp64);
}

inline void abacus_ilp64_zgemv_(const char * trans, const int * m, const int * n, const std::complex<double> * alpha, const std::complex<double> * a, const int * lda, const std::complex<double> * x, const int * incx, const std::complex<double> * beta, std::complex<double> * y, const int * incy)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_zgemv_(trans, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, x, &incx_ilp64, beta, y, &incy_ilp64);
}

inline void abacus_ilp64_dsymv_(const char * uplo, const int * n, const double * alpha, const double * a, const int * lda, const double * x, const int * incx, const double * beta, double * y, const int * incy)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    abacus_mkl_dsymv_(uplo, &n_ilp64, alpha, a, &lda_ilp64, x, &incx_ilp64, beta, y, &incy_ilp64);
}

inline void abacus_ilp64_dger_(const int * m, const int * n, const double * alpha, const double * x, const int * incx, const double * y, const int * incy, double * a, const int * lda)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    abacus_mkl_dger_(&m_ilp64, &n_ilp64, alpha, x, &incx_ilp64, y, &incy_ilp64, a, &lda_ilp64);
}

inline void abacus_ilp64_zgerc_(const int * m, const int * n, const std::complex<double> * alpha, const std::complex<double> * x, const int * incx, const std::complex<double> * y, const int * incy, std::complex<double> * a, const int * lda)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT incx_ilp64 = static_cast<MKL_INT>(*incx);
    const MKL_INT incy_ilp64 = static_cast<MKL_INT>(*incy);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    abacus_mkl_zgerc_(&m_ilp64, &n_ilp64, alpha, x, &incx_ilp64, y, &incy_ilp64, a, &lda_ilp64);
}

inline void abacus_ilp64_sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k, const float * alpha, const float * a, const int * lda, const float * b, const int * ldb, const float * beta, float * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_sgemm_(transa, transb, &m_ilp64, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_dgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k, const double * alpha, const double * a, const int * lda, const double * b, const int * ldb, const double * beta, double * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_dgemm_(transa, transb, &m_ilp64, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_cgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k, const std::complex<float> * alpha, const std::complex<float> * a, const int * lda, const std::complex<float> * b, const int * ldb, const std::complex<float> * beta, std::complex<float> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_cgemm_(transa, transb, &m_ilp64, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_zgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k, const std::complex<double> * alpha, const std::complex<double> * a, const int * lda, const std::complex<double> * b, const int * ldb, const std::complex<double> * beta, std::complex<double> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_zgemm_(transa, transb, &m_ilp64, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_ssymm_(const char * side, const char * uplo, const int * m, const int * n, const float * alpha, const float * a, const int * lda, const float * b, const int * ldb, const float * beta, float * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_ssymm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_dsymm_(const char * side, const char * uplo, const int * m, const int * n, const double * alpha, const double * a, const int * lda, const double * b, const int * ldb, const double * beta, double * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_dsymm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_csymm_(const char * side, const char * uplo, const int * m, const int * n, const std::complex<float> * alpha, const std::complex<float> * a, const int * lda, const std::complex<float> * b, const int * ldb, const std::complex<float> * beta, std::complex<float> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_csymm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_zsymm_(const char * side, const char * uplo, const int * m, const int * n, const std::complex<double> * alpha, const std::complex<double> * a, const int * lda, const std::complex<double> * b, const int * ldb, const std::complex<double> * beta, std::complex<double> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_zsymm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_chemm_(const char * side, const char * uplo, const int * m, const int * n, const std::complex<float> * alpha, const std::complex<float> * a, const int * lda, const std::complex<float> * b, const int * ldb, const std::complex<float> * beta, std::complex<float> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_chemm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_zhemm_(const char * side, const char * uplo, const int * m, const int * n, const std::complex<double> * alpha, const std::complex<double> * a, const int * lda, const std::complex<double> * b, const int * ldb, const std::complex<double> * beta, std::complex<double> * c, const int * ldc)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_zhemm_(side, uplo, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_dtrsm_(const char * side, const char * uplo, const char * transa, const char * diag, const int * m, const int * n, const double * alpha, const double * a, const int * lda, double * b, const int * ldb)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    abacus_mkl_dtrsm_(side, uplo, transa, diag, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64);
}

inline void abacus_ilp64_ztrsm_(const char * side, const char * uplo, const char * transa, const char * diag, const int * m, const int * n, const std::complex<double> * alpha, const std::complex<double> * a, const int * lda, std::complex<double> * b, const int * ldb)
{
    const MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    abacus_mkl_ztrsm_(side, uplo, transa, diag, &m_ilp64, &n_ilp64, alpha, a, &lda_ilp64, b, &ldb_ilp64);
}

inline void abacus_ilp64_cherk_(const char* uplo, const char* trans, const int* n, const int* k, const float* alpha, const std::complex<float>* a, const int* lda, const float* beta, std::complex<float>* c, const int* ldc)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_cherk_(uplo, trans, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_zherk_(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha, const std::complex<double>* a, const int* lda, const double* beta, std::complex<double>* c, const int* ldc)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_zherk_(uplo, trans, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, beta, c, &ldc_ilp64);
}

inline void abacus_ilp64_dsyrk_(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha, const double* a, const int* lda, const double* beta, double* c, const int* ldc)
{
    const MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    const MKL_INT k_ilp64 = static_cast<MKL_INT>(*k);
    const MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    const MKL_INT ldc_ilp64 = static_cast<MKL_INT>(*ldc);
    abacus_mkl_dsyrk_(uplo, trans, &n_ilp64, &k_ilp64, alpha, a, &lda_ilp64, beta, c, &ldc_ilp64);
}

}
#define sscal_ abacus_ilp64_sscal_
#define dscal_ abacus_ilp64_dscal_
#define cscal_ abacus_ilp64_cscal_
#define zscal_ abacus_ilp64_zscal_
#define saxpy_ abacus_ilp64_saxpy_
#define daxpy_ abacus_ilp64_daxpy_
#define caxpy_ abacus_ilp64_caxpy_
#define zaxpy_ abacus_ilp64_zaxpy_
#define scopy_ abacus_ilp64_scopy_
#define dcopy_ abacus_ilp64_dcopy_
#define ccopy_ abacus_ilp64_ccopy_
#define zcopy_ abacus_ilp64_zcopy_
#define sdot_ abacus_ilp64_sdot_
#define ddot_ abacus_ilp64_ddot_
#define snrm2_ abacus_ilp64_snrm2_
#define dnrm2_ abacus_ilp64_dnrm2_
#define scnrm2_ abacus_ilp64_scnrm2_
#define dznrm2_ abacus_ilp64_dznrm2_
#define sgemv_ abacus_ilp64_sgemv_
#define dgemv_ abacus_ilp64_dgemv_
#define cgemv_ abacus_ilp64_cgemv_
#define zgemv_ abacus_ilp64_zgemv_
#define dsymv_ abacus_ilp64_dsymv_
#define dger_ abacus_ilp64_dger_
#define zgerc_ abacus_ilp64_zgerc_
#define sgemm_ abacus_ilp64_sgemm_
#define dgemm_ abacus_ilp64_dgemm_
#define cgemm_ abacus_ilp64_cgemm_
#define zgemm_ abacus_ilp64_zgemm_
#define ssymm_ abacus_ilp64_ssymm_
#define dsymm_ abacus_ilp64_dsymm_
#define csymm_ abacus_ilp64_csymm_
#define zsymm_ abacus_ilp64_zsymm_
#define chemm_ abacus_ilp64_chemm_
#define zhemm_ abacus_ilp64_zhemm_
#define dtrsm_ abacus_ilp64_dtrsm_
#define ztrsm_ abacus_ilp64_ztrsm_
#define cherk_ abacus_ilp64_cherk_
#define zherk_ abacus_ilp64_zherk_
#define dsyrk_ abacus_ilp64_dsyrk_

#endif
