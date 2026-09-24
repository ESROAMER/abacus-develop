#ifndef ABACUS_ILP64_LAPACK_CONNECTOR_H
#define ABACUS_ILP64_LAPACK_CONNECTOR_H

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include <mkl_types.h>

static_assert(sizeof(MKL_INT) == 8, "ABACUS MKL ILP64 requires 8-byte MKL_INT");

extern "C"
{
void abacus_mkl_dsygvd_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, double* w, double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("dsygvd_");
void abacus_mkl_chegvd_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, std::complex<float>* b, const MKL_INT* ldb, float* w, std::complex<float>* work, const MKL_INT* lwork, float* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("chegvd_");
void abacus_mkl_zhegvd_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, std::complex<double>* b, const MKL_INT* ldb, double* w, std::complex<double>* work, const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("zhegvd_");
void abacus_mkl_dsyevx_(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz, double* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("dsyevx_");
void abacus_mkl_cheevx_(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, std::complex<float>* z, const MKL_INT* ldz, std::complex<float>* work, const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("cheevx_");
void abacus_mkl_zheevx_(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, std::complex<double>* z, const MKL_INT* ldz, std::complex<double>* work, const MKL_INT* lwork, double* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("zheevx_");
void abacus_mkl_dsygvx_(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz, double* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("dsygvx_");
void abacus_mkl_chegvx_(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, std::complex<float>* b, const MKL_INT* ldb, const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, std::complex<float>* z, const MKL_INT* ldz, std::complex<float>* work, const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("chegvx_");
void abacus_mkl_zhegvx_(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, std::complex<double>* b, const MKL_INT* ldb, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, std::complex<double>* z, const MKL_INT* ldz, std::complex<double>* work, const MKL_INT* lwork, double* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("zhegvx_");
void abacus_mkl_dsygv_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, double* w, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dsygv_");
void abacus_mkl_chegv_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, std::complex<float>* b, const MKL_INT* ldb, float* w, std::complex<float>* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) __asm__("chegv_");
void abacus_mkl_zhegv_(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, std::complex<double>* b, const MKL_INT* ldb, double* w, std::complex<double>* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) __asm__("zhegv_");
void abacus_mkl_ssyev_(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork, MKL_INT* info) __asm__("ssyev_");
void abacus_mkl_dsyev_(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dsyev_");
void abacus_mkl_cheev_(const char* jobz, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, float* w, std::complex<float>* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) __asm__("cheev_");
void abacus_mkl_zheev_(const char* jobz, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, double* w, std::complex<double>* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) __asm__("zheev_");
void abacus_mkl_ssyevd_(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("ssyevd_");
void abacus_mkl_dsyevd_(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("dsyevd_");
void abacus_mkl_cheevd_(const char* jobz, const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, float* w, std::complex<float>* work, const MKL_INT* lwork, float* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("cheevd_");
void abacus_mkl_zheevd_(const char* jobz, const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, double* w, std::complex<double>* work, const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) __asm__("zheevd_");
void abacus_mkl_dgeev_(const char* jobvl, const char* jobvr, const MKL_INT* n, double* a, const MKL_INT* lda, double* wr, double* wi, double* vl, const MKL_INT* ldvl, double* vr, const MKL_INT* ldvr, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dgeev_");
void abacus_mkl_zgeev_(const char* jobvl, const char* jobvr, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, std::complex<double>* w, std::complex<double>* vl, const MKL_INT* ldvl, std::complex<double>* vr, const MKL_INT* ldvr, std::complex<double>* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) __asm__("zgeev_");
void abacus_mkl_dgetrf_(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) __asm__("dgetrf_");
void abacus_mkl_dgetri_(const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dgetri_");
void abacus_mkl_dsytrf_(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dsytrf_");
void abacus_mkl_dsytri_(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv, double* work, MKL_INT* info) __asm__("dsytri_");
void abacus_mkl_spotrf_(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) __asm__("spotrf_");
void abacus_mkl_dpotrf_(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) __asm__("dpotrf_");
void abacus_mkl_cpotrf_(const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, MKL_INT* info) __asm__("cpotrf_");
void abacus_mkl_zpotrf_(const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, MKL_INT* info) __asm__("zpotrf_");
void abacus_mkl_spotri_(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) __asm__("spotri_");
void abacus_mkl_dpotri_(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) __asm__("dpotri_");
void abacus_mkl_cpotri_(const char* uplo, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, MKL_INT* info) __asm__("cpotri_");
void abacus_mkl_zpotri_(const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, MKL_INT* info) __asm__("zpotri_");
void abacus_mkl_zgetrf_(const MKL_INT* m, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) __asm__("zgetrf_");
void abacus_mkl_zgetri_(const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, const MKL_INT* ipiv, std::complex<double>* work, const MKL_INT* lwork, MKL_INT* info) __asm__("zgetri_");
void abacus_mkl_strtri_(const char* uplo, const char* diag, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) __asm__("strtri_");
void abacus_mkl_dtrtri_(const char* uplo, const char* diag, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) __asm__("dtrtri_");
void abacus_mkl_ctrtri_(const char* uplo, const char* diag, const MKL_INT* n, std::complex<float>* a, const MKL_INT* lda, MKL_INT* info) __asm__("ctrtri_");
void abacus_mkl_ztrtri_(const char* uplo, const char* diag, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, MKL_INT* info) __asm__("ztrtri_");
void abacus_mkl_dsterf_(const MKL_INT* n, double* d, double* e, MKL_INT* info) __asm__("dsterf_");
void abacus_mkl_dstein_(const MKL_INT* n, const double* d, const double* e, const MKL_INT* m, const double* w, const MKL_INT* iblock, const MKL_INT* isplit, double* z, const MKL_INT* ldz, double* work, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("dstein_");
void abacus_mkl_zstein_(const MKL_INT* n, const double* d, const double* e, const MKL_INT* m, const double* w, const MKL_INT* iblock, const MKL_INT* isplit, std::complex<double>* z, const MKL_INT* ldz, double* work, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) __asm__("zstein_");
void abacus_mkl_dpotf2_(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) __asm__("dpotf2_");
void abacus_mkl_zpotf2_(const char* uplo, const MKL_INT* n, std::complex<double>* a, const MKL_INT* lda, MKL_INT* info) __asm__("zpotf2_");
void abacus_mkl_dgtsv_(const MKL_INT* n, const MKL_INT* nrhs, double* dl, double* d, double* du, double* b, const MKL_INT* ldb, MKL_INT* info) __asm__("dgtsv_");
void abacus_mkl_dsysv_(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda, MKL_INT* ipiv, double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info) __asm__("dsysv_");
double abacus_mkl_dlange_(const char* norm, const MKL_INT* m, const MKL_INT* n, const double* A, const MKL_INT* lda, double* work) __asm__("dlange_");
double abacus_mkl_zlange_(const char* norm, const MKL_INT* m, const MKL_INT* n, const std::complex<double>* A, const MKL_INT* lda, double* work) __asm__("zlange_");
}

extern "C"
{
inline void abacus_ilp64_dsygvd_(const int* itype, const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* b, const int* ldb, double* w, double* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsygvd_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_chegvd_(const int* itype, const char* jobz, const char* uplo, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, float* w, std::complex<float>* work, const int* lwork, float* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT lrwork_ilp64 = static_cast<MKL_INT>(*lrwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_chegvd_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, rwork, &lrwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zhegvd_(const int* itype, const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, double* w, std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT lrwork_ilp64 = static_cast<MKL_INT>(*lrwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zhegvd_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, rwork, &lrwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsyevx_(const char* jobz, const char* range, const char* uplo, const int* n, double* a, const int* lda, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, double* z, const int* ldz, double* work, const int* lwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsyevx_(jobz, range, uplo, &n_ilp64, a, &lda_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_cheevx_(const char* jobz, const char* range, const char* uplo, const int* n, std::complex<float>* a, const int* lda, const float* vl, const float* vu, const int* il, const int* iu, const float* abstol, int* m, float* w, std::complex<float>* z, const int* ldz, std::complex<float>* work, const int* lwork, float* rwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_cheevx_(jobz, range, uplo, &n_ilp64, a, &lda_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, rwork, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zheevx_(const char* jobz, const char* range, const char* uplo, const int* n, std::complex<double>* a, const int* lda, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, std::complex<double>* z, const int* ldz, std::complex<double>* work, const int* lwork, double* rwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zheevx_(jobz, range, uplo, &n_ilp64, a, &lda_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, rwork, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsygvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, double* a, const int* lda, double* b, const int* ldb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, double* z, const int* ldz, double* work, const int* lwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsygvx_(&itype_ilp64, jobz, range, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_chegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, const float* vl, const float* vu, const int* il, const int* iu, const float* abstol, int* m, float* w, std::complex<float>* z, const int* ldz, std::complex<float>* work, const int* lwork, float* rwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_chegvx_(&itype_ilp64, jobz, range, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, rwork, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zhegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, std::complex<double>* z, const int* ldz, std::complex<double>* work, const int* lwork, double* rwork, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT il_ilp64 = static_cast<MKL_INT>(*il);
    MKL_INT iu_ilp64 = static_cast<MKL_INT>(*iu);
    MKL_INT m_ilp64 = 0;
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zhegvx_(&itype_ilp64, jobz, range, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, vl, vu, &il_ilp64, &iu_ilp64, abstol, &m_ilp64, w, z, &ldz_ilp64, work, &lwork_ilp64, rwork, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    if (*lwork != -1)
    {
        for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    }
    *m = static_cast<int>(m_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsygv_(const int* itype, const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* b, const int* ldb, double* w, double* work, const int* lwork, int* info)
{
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsygv_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_chegv_(const int* itype, const char* jobz, const char* uplo, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, float* w, std::complex<float>* work, const int* lwork, float* rwork, int* info)
{
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_chegv_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, rwork, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zhegv_(const int* itype, const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info)
{
    MKL_INT itype_ilp64 = static_cast<MKL_INT>(*itype);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zhegv_(&itype_ilp64, jobz, uplo, &n_ilp64, a, &lda_ilp64, b, &ldb_ilp64, w, work, &lwork_ilp64, rwork, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_ssyev_(const char* jobz, const char* uplo, const int* n, float* a, const int* lda, float* w, float* work, const int* lwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_ssyev_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsyev_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_cheev_(const char* jobz, const char* uplo, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_cheev_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, rwork, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zheev_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zheev_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, rwork, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_ssyevd_(const char* jobz, const char* uplo, const int* n, float* a, const int* lda, float* w, float* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, *liwork)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_ssyevd_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < iwork_ilp64.size(); ++i) iwork[i] = static_cast<int>(iwork_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, *liwork)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsyevd_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < iwork_ilp64.size(); ++i) iwork[i] = static_cast<int>(iwork_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_cheevd_(const char* jobz, const char* uplo, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, *liwork)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT lrwork_ilp64 = static_cast<MKL_INT>(*lrwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_cheevd_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, rwork, &lrwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < iwork_ilp64.size(); ++i) iwork[i] = static_cast<int>(iwork_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zheevd_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, *liwork)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT lrwork_ilp64 = static_cast<MKL_INT>(*lrwork);
    MKL_INT liwork_ilp64 = static_cast<MKL_INT>(*liwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zheevd_(jobz, uplo, &n_ilp64, a, &lda_ilp64, w, work, &lwork_ilp64, rwork, &lrwork_ilp64, iwork_ilp64.data(), &liwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < iwork_ilp64.size(); ++i) iwork[i] = static_cast<int>(iwork_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda, double* wr, double* wi, double* vl, const int* ldvl, double* vr, const int* ldvr, double* work, const int* lwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldvl_ilp64 = static_cast<MKL_INT>(*ldvl);
    MKL_INT ldvr_ilp64 = static_cast<MKL_INT>(*ldvr);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dgeev_(jobvl, jobvr, &n_ilp64, a, &lda_ilp64, wr, wi, vl, &ldvl_ilp64, vr, &ldvr_ilp64, work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zgeev_(const char* jobvl, const char* jobvr, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* w, std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr, std::complex<double>* work, const int* lwork, double* rwork, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldvl_ilp64 = static_cast<MKL_INT>(*ldvl);
    MKL_INT ldvr_ilp64 = static_cast<MKL_INT>(*ldvr);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zgeev_(jobvl, jobvr, &n_ilp64, a, &lda_ilp64, w, vl, &ldvl_ilp64, vr, &ldvr_ilp64, work, &lwork_ilp64, rwork, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dgetrf_(&m_ilp64, &n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), &info_ilp64);
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv[i] = static_cast<int>(ipiv_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv_ilp64[i] = static_cast<MKL_INT>(ipiv[i]);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dgetri_(&n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv, double* work, const int* lwork, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsytrf_(uplo, &n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), work, &lwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv[i] = static_cast<int>(ipiv_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsytri_(const char* uplo, const int* n, double* a, const int* lda, const int* ipiv, double* work, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv_ilp64[i] = static_cast<MKL_INT>(ipiv[i]);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsytri_(uplo, &n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), work, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_spotrf_(const char* uplo, const int* n, float* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_spotrf_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dpotrf_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_cpotrf_(const char* uplo, const int* n, std::complex<float>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_cpotrf_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zpotrf_(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zpotrf_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_spotri_(const char* uplo, const int* n, float* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_spotri_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dpotri_(const char* uplo, const int* n, double* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dpotri_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_cpotri_(const char* uplo, const int* n, std::complex<float>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_cpotri_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zpotri_(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zpotri_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zgetrf_(&m_ilp64, &n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), &info_ilp64);
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv[i] = static_cast<int>(ipiv_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zgetri_(const int* n, std::complex<double>* a, const int* lda, const int* ipiv, std::complex<double>* work, const int* lwork, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv_ilp64[i] = static_cast<MKL_INT>(ipiv[i]);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zgetri_(&n_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), work, &lwork_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsterf_(const int* n, double* d, double* e, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsterf_(&n_ilp64, d, e, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dstein_(const int* n, const double* d, const double* e, const int* m, const double* w, const int* iblock, const int* isplit, double* z, const int* ldz, double* work, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iblock_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < iblock_ilp64.size(); ++i) iblock_ilp64[i] = static_cast<MKL_INT>(iblock[i]);
    std::vector<MKL_INT> isplit_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < isplit_ilp64.size(); ++i) isplit_ilp64[i] = static_cast<MKL_INT>(isplit[i]);
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dstein_(&n_ilp64, d, e, &m_ilp64, w, iblock_ilp64.data(), isplit_ilp64.data(), z, &ldz_ilp64, work, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zstein_(const int* n, const double* d, const double* e, const int* m, const double* w, const int* iblock, const int* isplit, std::complex<double>* z, const int* ldz, double* work, int* iwork, int* ifail, int* info)
{
    std::vector<MKL_INT> iblock_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < iblock_ilp64.size(); ++i) iblock_ilp64[i] = static_cast<MKL_INT>(iblock[i]);
    std::vector<MKL_INT> isplit_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    for (std::size_t i = 0; i < isplit_ilp64.size(); ++i) isplit_ilp64[i] = static_cast<MKL_INT>(isplit[i]);
    std::vector<MKL_INT> iwork_ilp64(static_cast<std::size_t>(std::max(1, 5 * (*n) + 3)));
    std::vector<MKL_INT> ifail_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT ldz_ilp64 = static_cast<MKL_INT>(*ldz);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zstein_(&n_ilp64, d, e, &m_ilp64, w, iblock_ilp64.data(), isplit_ilp64.data(), z, &ldz_ilp64, work, iwork_ilp64.data(), ifail_ilp64.data(), &info_ilp64);
    for (std::size_t i = 0; i < ifail_ilp64.size(); ++i) ifail[i] = static_cast<int>(ifail_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dpotf2_(const char* uplo, const int* n, double* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dpotf2_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_zpotf2_(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_zpotf2_(uplo, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dgtsv_(const int* n, const int* nrhs, double* dl, double* d, double* du, double* b, const int* ldb, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT nrhs_ilp64 = static_cast<MKL_INT>(*nrhs);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dgtsv_(&n_ilp64, &nrhs_ilp64, dl, d, du, b, &ldb_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dsysv_(const char* uplo, const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, double* work, const int* lwork, int* info)
{
    std::vector<MKL_INT> ipiv_ilp64(static_cast<std::size_t>(std::max(1, *n)));
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT nrhs_ilp64 = static_cast<MKL_INT>(*nrhs);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT ldb_ilp64 = static_cast<MKL_INT>(*ldb);
    MKL_INT lwork_ilp64 = static_cast<MKL_INT>(*lwork);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dsysv_(uplo, &n_ilp64, &nrhs_ilp64, a, &lda_ilp64, ipiv_ilp64.data(), b, &ldb_ilp64, work, &lwork_ilp64, &info_ilp64);
    for (std::size_t i = 0; i < ipiv_ilp64.size(); ++i) ipiv[i] = static_cast<int>(ipiv_ilp64[i]);
    *info = static_cast<int>(info_ilp64);
}

inline double abacus_ilp64_dlange_(const char* norm, const int* m, const int* n, const double* A, const int* lda, double* work)
{
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    return abacus_mkl_dlange_(norm, &m_ilp64, &n_ilp64, A, &lda_ilp64, work);
}

inline double abacus_ilp64_zlange_(const char* norm, const int* m, const int* n, const std::complex<double>* A, const int* lda, double* work)
{
    MKL_INT m_ilp64 = static_cast<MKL_INT>(*m);
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    return abacus_mkl_zlange_(norm, &m_ilp64, &n_ilp64, A, &lda_ilp64, work);
}

inline void abacus_ilp64_strtri_(const char* uplo, const char* diag, const int* n, float* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_strtri_(uplo, diag, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_dtrtri_(const char* uplo, const char* diag, const int* n, double* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_dtrtri_(uplo, diag, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_ctrtri_(const char* uplo, const char* diag, const int* n, std::complex<float>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_ctrtri_(uplo, diag, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

inline void abacus_ilp64_ztrtri_(const char* uplo, const char* diag, const int* n, std::complex<double>* a, const int* lda, int* info)
{
    MKL_INT n_ilp64 = static_cast<MKL_INT>(*n);
    MKL_INT lda_ilp64 = static_cast<MKL_INT>(*lda);
    MKL_INT info_ilp64 = 0;
    abacus_mkl_ztrtri_(uplo, diag, &n_ilp64, a, &lda_ilp64, &info_ilp64);
    *info = static_cast<int>(info_ilp64);
}

}
#define dsygvd_ abacus_ilp64_dsygvd_
#define chegvd_ abacus_ilp64_chegvd_
#define zhegvd_ abacus_ilp64_zhegvd_
#define dsyevx_ abacus_ilp64_dsyevx_
#define cheevx_ abacus_ilp64_cheevx_
#define zheevx_ abacus_ilp64_zheevx_
#define dsygvx_ abacus_ilp64_dsygvx_
#define chegvx_ abacus_ilp64_chegvx_
#define zhegvx_ abacus_ilp64_zhegvx_
#define dsygv_ abacus_ilp64_dsygv_
#define chegv_ abacus_ilp64_chegv_
#define zhegv_ abacus_ilp64_zhegv_
#define ssyev_ abacus_ilp64_ssyev_
#define dsyev_ abacus_ilp64_dsyev_
#define cheev_ abacus_ilp64_cheev_
#define zheev_ abacus_ilp64_zheev_
#define ssyevd_ abacus_ilp64_ssyevd_
#define dsyevd_ abacus_ilp64_dsyevd_
#define cheevd_ abacus_ilp64_cheevd_
#define zheevd_ abacus_ilp64_zheevd_
#define dgeev_ abacus_ilp64_dgeev_
#define zgeev_ abacus_ilp64_zgeev_
#define dgetrf_ abacus_ilp64_dgetrf_
#define dgetri_ abacus_ilp64_dgetri_
#define dsytrf_ abacus_ilp64_dsytrf_
#define dsytri_ abacus_ilp64_dsytri_
#define spotrf_ abacus_ilp64_spotrf_
#define dpotrf_ abacus_ilp64_dpotrf_
#define cpotrf_ abacus_ilp64_cpotrf_
#define zpotrf_ abacus_ilp64_zpotrf_
#define spotri_ abacus_ilp64_spotri_
#define dpotri_ abacus_ilp64_dpotri_
#define cpotri_ abacus_ilp64_cpotri_
#define zpotri_ abacus_ilp64_zpotri_
#define zgetrf_ abacus_ilp64_zgetrf_
#define zgetri_ abacus_ilp64_zgetri_
#define strtri_ abacus_ilp64_strtri_
#define dtrtri_ abacus_ilp64_dtrtri_
#define ctrtri_ abacus_ilp64_ctrtri_
#define ztrtri_ abacus_ilp64_ztrtri_
#define dsterf_ abacus_ilp64_dsterf_
#define dstein_ abacus_ilp64_dstein_
#define zstein_ abacus_ilp64_zstein_
#define dpotf2_ abacus_ilp64_dpotf2_
#define zpotf2_ abacus_ilp64_zpotf2_
#define dgtsv_ abacus_ilp64_dgtsv_
#define dsysv_ abacus_ilp64_dsysv_
#define dlange_ abacus_ilp64_dlange_
#define zlange_ abacus_ilp64_zlange_

#endif
