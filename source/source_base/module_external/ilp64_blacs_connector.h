#ifndef ABACUS_ILP64_BLACS_CONNECTOR_H
#define ABACUS_ILP64_BLACS_CONNECTOR_H

#include <complex>
#include <mkl_types.h>

#ifdef __MPI
#include <mpi.h>
#endif

static_assert(sizeof(MKL_INT) == 8, "ABACUS MKL ILP64 requires 8-byte MKL_INT");

extern "C"
{
void abacus_mkl_Cblacs_pinfo(MKL_INT*, MKL_INT*) __asm__("Cblacs_pinfo");
void abacus_mkl_Cblacs_get(MKL_INT, MKL_INT, MKL_INT*) __asm__("Cblacs_get");
void abacus_mkl_Cblacs_gridinfo(MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*) __asm__("Cblacs_gridinfo");
void abacus_mkl_Cblacs_gridinit(MKL_INT*, char*, MKL_INT, MKL_INT) __asm__("Cblacs_gridinit");
void abacus_mkl_Cblacs_gridexit(MKL_INT) __asm__("Cblacs_gridexit");
void abacus_mkl_Cblacs_exit(MKL_INT) __asm__("Cblacs_exit");
MKL_INT abacus_mkl_Cblacs_pnum(MKL_INT, MKL_INT, MKL_INT) __asm__("Cblacs_pnum");
void abacus_mkl_Cblacs_pcoord(MKL_INT, MKL_INT, MKL_INT*, MKL_INT*) __asm__("Cblacs_pcoord");
void abacus_mkl_Cblacs_barrier(MKL_INT, char*) __asm__("Cblacs_barrier");

void abacus_mkl_Cigebs2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT) __asm__("Cigebs2d");
void abacus_mkl_Cigebr2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT, MKL_INT, MKL_INT) __asm__("Cigebr2d");
void abacus_mkl_Csgebs2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, float*, MKL_INT) __asm__("Csgebs2d");
void abacus_mkl_Csgebr2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, float*, MKL_INT, MKL_INT, MKL_INT) __asm__("Csgebr2d");
void abacus_mkl_Cdgebs2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, double*, MKL_INT) __asm__("Cdgebs2d");
void abacus_mkl_Cdgebr2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, double*, MKL_INT, MKL_INT, MKL_INT) __asm__("Cdgebr2d");
void abacus_mkl_Ccgebs2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, std::complex<float>*, MKL_INT) __asm__("Ccgebs2d");
void abacus_mkl_Ccgebr2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, std::complex<float>*, MKL_INT, MKL_INT, MKL_INT) __asm__("Ccgebr2d");
void abacus_mkl_Czgebs2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, std::complex<double>*, MKL_INT) __asm__("Czgebs2d");
void abacus_mkl_Czgebr2d(MKL_INT, char*, char*, MKL_INT, MKL_INT, std::complex<double>*, MKL_INT, MKL_INT, MKL_INT) __asm__("Czgebr2d");

#ifdef __MPI
MKL_INT abacus_mkl_Csys2blacs_handle(MPI_Comm) __asm__("Csys2blacs_handle");
MPI_Comm abacus_mkl_Cblacs2sys_handle(MKL_INT) __asm__("Cblacs2sys_handle");
#endif
}

inline void Cblacs_pinfo(int* myid, int* nprocs)
{
    MKL_INT myid64 = 0, nprocs64 = 0;
    abacus_mkl_Cblacs_pinfo(&myid64, &nprocs64);
    *myid = static_cast<int>(myid64);
    *nprocs = static_cast<int>(nprocs64);
}

inline void Cblacs_get(int icontxt, int what, int* val)
{
    MKL_INT val64 = 0;
    abacus_mkl_Cblacs_get(static_cast<MKL_INT>(icontxt), static_cast<MKL_INT>(what), &val64);
    *val = static_cast<int>(val64);
}

inline void Cblacs_gridinfo(int icontxt, int* nprow, int* npcol, int* myprow, int* mypcol)
{
    MKL_INT nprow64 = 0, npcol64 = 0, myprow64 = 0, mypcol64 = 0;
    abacus_mkl_Cblacs_gridinfo(static_cast<MKL_INT>(icontxt), &nprow64, &npcol64, &myprow64, &mypcol64);
    *nprow = static_cast<int>(nprow64);
    *npcol = static_cast<int>(npcol64);
    *myprow = static_cast<int>(myprow64);
    *mypcol = static_cast<int>(mypcol64);
}

inline void Cblacs_gridinit(int* icontxt, char* layout, int nprow, int npcol)
{
    MKL_INT context64 = static_cast<MKL_INT>(*icontxt);
    abacus_mkl_Cblacs_gridinit(&context64, layout, static_cast<MKL_INT>(nprow), static_cast<MKL_INT>(npcol));
    *icontxt = static_cast<int>(context64);
}

inline void Cblacs_gridexit(int icontxt)
{
    abacus_mkl_Cblacs_gridexit(static_cast<MKL_INT>(icontxt));
}

inline void Cblacs_exit(int icontxt)
{
    abacus_mkl_Cblacs_exit(static_cast<MKL_INT>(icontxt));
}

inline int Cblacs_pnum(int icontxt, int prow, int pcol)
{
    return static_cast<int>(abacus_mkl_Cblacs_pnum(static_cast<MKL_INT>(icontxt), static_cast<MKL_INT>(prow), static_cast<MKL_INT>(pcol)));
}

inline void Cblacs_pcoord(int icontxt, int pnum, int* prow, int* pcol)
{
    MKL_INT prow64 = 0, pcol64 = 0;
    abacus_mkl_Cblacs_pcoord(static_cast<MKL_INT>(icontxt), static_cast<MKL_INT>(pnum), &prow64, &pcol64);
    *prow = static_cast<int>(prow64);
    *pcol = static_cast<int>(pcol64);
}

inline void Cblacs_barrier(int icontxt, char* scope)
{
    abacus_mkl_Cblacs_barrier(static_cast<MKL_INT>(icontxt), scope);
}

#ifdef __MPI
inline int Csys2blacs_handle(MPI_Comm sys_ctxt)
{
    return static_cast<int>(abacus_mkl_Csys2blacs_handle(sys_ctxt));
}

inline MPI_Comm Cblacs2sys_handle(int blacs_ctxt)
{
    return abacus_mkl_Cblacs2sys_handle(static_cast<MKL_INT>(blacs_ctxt));
}
#endif

inline void Cigebs2d(int ctxt, char* scope, char* top, int m, int n, int* a, int lda)
{
    abacus_mkl_Cigebs2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), reinterpret_cast<MKL_INT*>(a), static_cast<MKL_INT>(lda));
}

inline void Cigebr2d(int ctxt, char* scope, char* top, int m, int n, int* a, int lda, int rsrc, int csrc)
{
    abacus_mkl_Cigebr2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), reinterpret_cast<MKL_INT*>(a), static_cast<MKL_INT>(lda), static_cast<MKL_INT>(rsrc), static_cast<MKL_INT>(csrc));
}

inline void Csgebs2d(int ctxt, char* scope, char* top, int m, int n, float* a, int lda)
{
    abacus_mkl_Csgebs2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda));
}

inline void Csgebr2d(int ctxt, char* scope, char* top, int m, int n, float* a, int lda, int rsrc, int csrc)
{
    abacus_mkl_Csgebr2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda), static_cast<MKL_INT>(rsrc), static_cast<MKL_INT>(csrc));
}

inline void Cdgebs2d(int ctxt, char* scope, char* top, int m, int n, double* a, int lda)
{
    abacus_mkl_Cdgebs2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda));
}

inline void Cdgebr2d(int ctxt, char* scope, char* top, int m, int n, double* a, int lda, int rsrc, int csrc)
{
    abacus_mkl_Cdgebr2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda), static_cast<MKL_INT>(rsrc), static_cast<MKL_INT>(csrc));
}

inline void Ccgebs2d(int ctxt, char* scope, char* top, int m, int n, std::complex<float>* a, int lda)
{
    abacus_mkl_Ccgebs2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda));
}

inline void Ccgebr2d(int ctxt, char* scope, char* top, int m, int n, std::complex<float>* a, int lda, int rsrc, int csrc)
{
    abacus_mkl_Ccgebr2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda), static_cast<MKL_INT>(rsrc), static_cast<MKL_INT>(csrc));
}

inline void Czgebs2d(int ctxt, char* scope, char* top, int m, int n, std::complex<double>* a, int lda)
{
    abacus_mkl_Czgebs2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda));
}

inline void Czgebr2d(int ctxt, char* scope, char* top, int m, int n, std::complex<double>* a, int lda, int rsrc, int csrc)
{
    abacus_mkl_Czgebr2d(static_cast<MKL_INT>(ctxt), scope, top, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(lda), static_cast<MKL_INT>(rsrc), static_cast<MKL_INT>(csrc));
}

#endif
