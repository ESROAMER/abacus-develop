//=======================
// AUTHOR : jiyy
// DATE :   2024-01-01
//=======================

#ifndef EWALD_VQ_CPP
#define EWALD_VQ_CPP

#include "auxiliary_func.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "module_base/global_variable.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

std::vector<double> Auxiliary_Func::cal_erfc_kernel(const std::vector<ModuleBase::Vector3<double>>& gk,
                                                    const double& omega)
{
    const int npw = gk.size();
    std::vector<double> vg(npw);
    const double prefactor = ModuleBase::FOUR_PI;

    for (size_t ig = 0; ig != npw; ++ig)
        vg[ig] = gk[ig].norm2() ? (prefactor / (gk[ig].norm2() * GlobalC::ucell.tpiba2))
                                      * (1 - std::exp(-gk[ig].norm2() * GlobalC::ucell.tpiba2 / (4 * omega * omega)))
                                : prefactor / (4 * omega * omega);

    return vg;
}

std::vector<double> Auxiliary_Func::cal_hf_kernel(const std::vector<ModuleBase::Vector3<double>>& gk)
{
    const int npw = gk.size();
    std::vector<double> vg(npw);

    // set 0 for add Auxiliary functions to eliminate singularities later
    for (size_t ig = 0; ig != npw; ++ig)
        vg[ig] = gk[ig].norm2() ? (ModuleBase::FOUR_PI / (gk[ig].norm2() * GlobalC::ucell.tpiba2)) : 0;

    return vg;
}

// for analytic integral of fq
double Auxiliary_Func::sum_for_solve_chi(const std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                         const T_cal_fq_type& func_cal_fq,
                                         const double& fq_int)
{
    const int nks = kvec_c.size();

    // cal fq sum except q=0
    double fq_sum = 0;
    for (size_t ik = 0; ik != nks; ++ik)
        fq_sum += kvec_c[ik].norm2() ? func_cal_fq(kvec_c[ik]): 0;

    double chi = ModuleBase::FOUR_PI * (fq_int - fq_sum / nks );

    return chi;
}

// for numerical integral of fq
double Auxiliary_Func::solve_chi(const std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                 const T_cal_fq_type& func_cal_fq,
                                 const std::array<int, 3>& nq_arr,
                                 const int& niter,
                                 const double& eps,
                                 const int& a_rate)
{
    // cal fq integral
    double fq_int = Iter_Integral(func_cal_fq, nq_arr, niter, eps, a_rate);

    return sum_for_solve_chi(kvec_c, func_cal_fq, fq_int);
}

// for analytic integral of fq
double Auxiliary_Func::solve_chi(const std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                 const T_cal_fq_type& func_cal_fq,
                                 const double& fq_int)
{
    return sum_for_solve_chi(kvec_c, func_cal_fq, fq_int);
}

double Auxiliary_Func::fq_type_0(const ModuleBase::Vector3<double>& qvec,
                                 const int& qdiv,
                                 std::vector<ModuleBase::Vector3<double>>& avec,
                                 std::vector<ModuleBase::Vector3<double>>& bvec)
{
    assert(qvec.norm2());

    std::vector<ModuleBase::Vector3<double>> baq(3);
    std::vector<ModuleBase::Vector3<double>> baq_2(3);
    const int qexpo = -abs(qdiv);
    const double prefactor = std::pow(ModuleBase::TWO_PI, -qexpo);

    for (size_t i = 0; i != 3; ++i)
    {
        baq[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec * ModuleBase::TWO_PI);
        baq_2[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec * ModuleBase::PI);
    }

    double sum_baq = 0;
    double sum_baq_2 = 0;
    for (size_t i = 1; i != 4; ++i)
    {
        size_t j = i % 3 + 1;
        size_t new_i = i - 1;
        size_t new_j = j - 1;
        sum_baq_2 += baq_2[new_i] * baq_2[new_i];
        sum_baq += baq[new_i] * baq[new_j];
    }
    double fq = prefactor * std::pow(4 * sum_baq_2 + 2 * sum_baq, 0.5 * qexpo);

    return fq;
}

double Auxiliary_Func::cal_type_0(const std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                  const int& qdiv,
                                  const double& qdense,
                                  const int& niter,
                                  const double& eps,
                                  const int& a_rate)
{
    ModuleBase::TITLE("Ewald_Vq", "cal_type_0");
    ModuleBase::timer::tick("Ewald_Vq", "cal_type_0");

    std::vector<ModuleBase::Vector3<double>> avec = {GlobalC::ucell.a1, GlobalC::ucell.a2, GlobalC::ucell.a3};
    std::vector<ModuleBase::Vector3<double>> bvec;
    bvec.resize(3);
    bvec[0].x = GlobalC::ucell.G.e11;
    bvec[0].y = GlobalC::ucell.G.e12;
    bvec[0].z = GlobalC::ucell.G.e13;

    bvec[1].x = GlobalC::ucell.G.e21;
    bvec[1].y = GlobalC::ucell.G.e22;
    bvec[1].z = GlobalC::ucell.G.e23;

    bvec[2].x = GlobalC::ucell.G.e31;
    bvec[2].y = GlobalC::ucell.G.e32;
    bvec[2].z = GlobalC::ucell.G.e33;

    std::array<int, 3> nq_arr;
    std::transform(bvec.begin(),
                   bvec.end(),
                   nq_arr.begin(),
                   [&qdense, &a_rate](ModuleBase::Vector3<double>& vec) -> int {
                       int index = static_cast<int>(vec.norm() * qdense * GlobalC::ucell.tpiba);
                       return index ? index - index % a_rate : a_rate;
                   });
    const T_cal_fq_type func_cal_fq_type_0 = std::bind(&fq_type_0, std::placeholders::_1, qdiv, avec, bvec);

    double val = solve_chi(kvec_c, func_cal_fq_type_0, nq_arr, niter, eps, a_rate);
    ModuleBase::timer::tick("Ewald_Vq", "cal_type_0");
    return val;
}

double Auxiliary_Func::fq_type_1(const ModuleBase::Vector3<double>& qvec,
                                 const int& qdiv,
                                 const ModulePW::PW_Basis_K* wfc_basis,
                                 const double& lambda)
{
    double fq = 0.0;
    const int qexpo = -abs(qdiv);

    for (size_t ig = 0; ig != wfc_basis->npw; ++ig)
    {
        int isz = wfc_basis->ig2isz[ig];
        int iz = isz % wfc_basis->nz;
        int is = isz / wfc_basis->nz;
        int ix = wfc_basis->is2fftixy[is] / wfc_basis->fftny;
        int iy = wfc_basis->is2fftixy[is] % wfc_basis->fftny;
        if (ix >= int(wfc_basis->nx / 2) + 1)
            ix -= wfc_basis->nx;
        if (iy >= int(wfc_basis->ny / 2) + 1)
            iy -= wfc_basis->ny;
        if (iz >= int(wfc_basis->nz / 2) + 1)
            iz -= wfc_basis->nz;
        ModuleBase::Vector3<double> f;
        f.x = ix;
        f.y = iy;
        f.z = iz;
        ModuleBase::Vector3<double> qg = qvec + f * wfc_basis->G;

        fq += std::exp(-lambda * qg.norm2()) * std::pow(qg.norm(), qexpo);
    }

    return fq;
}

double Auxiliary_Func::cal_type_1(const std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                  const int& qdiv,
                                  const ModulePW::PW_Basis_K* wfc_basis,
                                  const double& lambda)
{
    ModuleBase::TITLE("Ewald_Vq", "cal_type_1");
    ModuleBase::timer::tick("Ewald_Vq", "cal_type_1");

    const T_cal_fq_type func_cal_fq_type_1 = std::bind(&fq_type_1, std::placeholders::_1, qdiv, wfc_basis, lambda);
    double prefactor = ModuleBase::TWO_PI * std::pow(lambda, -1 / qdiv);
    double fq_int;
    if (qdiv == 2)
        fq_int = prefactor * std::sqrt(ModuleBase::PI);
    else if (qdiv == 1)
        fq_int = prefactor;
    else
        ModuleBase::WARNING_QUIT("Ewald_Vq::cal_type_1", "Type 1 fq only supports qdiv=1 or qdiv=2!");
    double val = solve_chi(kvec_c, func_cal_fq_type_1, fq_int);

    ModuleBase::timer::tick("Ewald_Vq", "cal_type_1");
    return val;
}

double Auxiliary_Func::Iter_Integral(const T_cal_fq_type& func_cal_fq,
                                     const std::array<int, 3>& nq_arr,
                                     const int& niter,
                                     const double& eps,
                                     const int& a_rate)
{
    bool any_negative = std::any_of(nq_arr.begin(), nq_arr.end(), [](int i) { return i < 0; });
    bool any_nthree = std::any_of(nq_arr.begin(), nq_arr.end(), [&a_rate](int i) { return i % a_rate != 0; });
    if (any_negative || any_nthree)
        ModuleBase::WARNING_QUIT("Ewald_Vq::Iter_Integral",
                                 "The elements of `nq_arr` should be non-negative and multiples of a_rate!");
    bool all_zero = std::all_of(nq_arr.begin(), nq_arr.end(), [](int i) { return i == 0; });
    if (all_zero)
        ModuleBase::WARNING_QUIT("Ewald_Vq::Iter_Integral", "At least one element of `nq_arr` should be non-zero!");

    const int nqs
        = std::accumulate(nq_arr.begin(), nq_arr.end(), 1, [](int a, int b) { return a * (2 * b + 1); });
    std::array<double, 3> qstep{};
    std::array<int, 3> nq_arr_in{};
    int ndim = 0;
    for (size_t i = 0; i != 3; ++i)
    {
        if (nq_arr[i] != 0)
        {
            qstep[i] = 1.0 / (2 * nq_arr[i] + 1);
            ndim += 1;
        }
        nq_arr_in[i] = static_cast<int>(nq_arr[i] / a_rate);
    }

    double integ = 0.0;
    int tot_iter = 0;
    for (size_t iter = 0; iter != niter; ++iter)
    {
        double integ_iter = 0.0;
        for (int ig1 = -nq_arr[0]; ig1 != nq_arr[0]+1; ++ig1)
            for (int ig2 = -nq_arr[1]; ig2 != nq_arr[1]+1; ++ig2)
                for (int ig3 = -nq_arr[2]; ig3 != nq_arr[2]+1; ++ig3)
                {
                    if (std::abs(ig1) <= nq_arr_in[0] && std::abs(ig2) <= nq_arr_in[1] && std::abs(ig3) <= nq_arr_in[2])
                        continue;
                    ModuleBase::Vector3<double> qvec; //direct
                    qvec.x = qstep[0] * ig1;
                    qvec.y = qstep[1] * ig2;
                    qvec.z = qstep[2] * ig3;
                    integ_iter += func_cal_fq(qvec * GlobalC::ucell.G);
                }
        integ_iter /= nqs * pow(a_rate, ndim * iter); // Each iteration reduces dq by a multiple of a_rate
        integ += integ_iter;
        if (iter != 0 && integ_iter < eps)
            break;
        std::for_each(qstep.begin(), qstep.end(), [&a_rate](double& qs) { qs /= a_rate; });
    }

    if (tot_iter == niter)
        ModuleBase::WARNING_QUIT("Ewald_Vq::Iter_Integral", "Integral not converged!");

    return integ;
}

#endif