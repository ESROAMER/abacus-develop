//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_CPP
#define GAUSSIAN_ABFS_CPP

#include "gaussian_abfs.h"

#include <algorithm>

#include "module_base/global_variable.h"
#include "module_base/math_ylmreal.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

RI::Tensor<std::complex<double>> Gaussian_Abfs::get_Vq(
    const int& lp_max,
    const int& lq_max, // Maximum L for which to calculate interaction.
    const ModuleBase::Vector3<double>& qvec,
    const ModulePW::PW_Basis_K* wfc_basis,
    const double& chi, // Singularity corrected value at q=0.
    const double& lambda,
    const ModuleBase::Vector3<double>& tau,
    const ORB_gaunt_table& MGT)
{
    ModuleBase::TITLE("Gaussian_Abfs", "get_Vq");
    ModuleBase::timer::tick("Gaussian_Abfs", "get_Vq");

    const int Lmax = lp_max + lq_max;
    const size_t vq_ndim0 = (lp_max + 1) * (lp_max + 1);
    const size_t vq_ndim1 = (lq_max + 1) * (lq_max + 1);
    RI::Tensor<std::complex<double>> Vq({vq_ndim0, vq_ndim1});

    /*
     n_add_ksq * 2 = lp_max + lq_max - abs(lp_max - lq_max)
        if lp_max < lq_max
            n_add_ksq * 2 = lp_max + lq_max - (lq_max - lp_max)
                          = lp_max * 2
        if lp_max > lq_max
            n_add_ksq * 2 = lp_max + lq_max - (lp_max - lq_max)
                          = lq_max * 2
        thus,
            n_add_ksq = min(lp_max, lq_max)
    */
    const int n_add_ksq = std::min(lp_max, lq_max);
    const int n_LM = (Lmax + 1) * (Lmax + 1);
    std::vector<std::vector<std::complex<double>>> lattice_sum(n_add_ksq + 1,
                                                               std::vector<std::complex<double>>(n_LM, {0.0, 0.0}));

    const double exponent = 1.0 / lambda;
    for (int i_add_ksq = 0; i_add_ksq != n_add_ksq + 1;
         ++i_add_ksq) // integrate lp, lq, L to one index i_add_ksq, i.e. (lp+lq-L)/2
    {
        const double power = -2.0 + 2 * i_add_ksq;
        const int this_Lmax = Lmax - 2 * i_add_ksq; // calculate Lmax at current lp+lq
        const bool exclude_Gamma
            = (qvec.norm2() < 1e-10 && i_add_ksq == 0); // only Gamma point and lq+lp-2>0 need to be corrected
        lattice_sum[i_add_ksq] = get_lattice_sum(-qvec, wfc_basis, power, exponent, exclude_Gamma, this_Lmax, tau);
    }

    /* The exponent term comes in from Taylor expanding the
        Gaussian at zero to first order in k^2, which cancels the k^-2 from the
        Coulomb interaction.  While terms of this order are in principle
        neglected, we make one exception here.  Without this, the final result
        would (slightly) depend on the Ewald lambda.*/
    if (qvec.norm2() < 1e-10)
        lattice_sum[0][0] += chi - exponent;

    for (int lp = 0; lp != lp_max + 1; ++lp)
    {
        double norm_1 = double_factorial(2 * lp - 1) * sqrt(ModuleBase::PI * 0.5);
        for (int lq = 0; lq != lq_max + 1; ++lq)
        {
            double norm_2 = double_factorial(2 * lq - 1) * sqrt(ModuleBase::PI * 0.5);
            std::complex<double> phase = std::pow(ModuleBase::IMAG_UNIT, lq - lp);
            std::complex<double> cfac = ModuleBase::FOUR_PI * phase / (norm_1 * norm_2);
            for (int L = std::abs(lp - lq); L <= lp + lq; L += 2) // if lp+lq-L == odd, then Gaunt_Coefficients = 0
            {
                const int i_add_ksq = (lp + lq - L) / 2;
                for (int mp = 0; mp != 2 * lp + 1; ++mp)
                {
                    const int lmp = MGT.get_lm_index(lp, mp);
                    for (int mq = 0; mq != 2 * lq + 1; ++mq)
                    {
                        const int lmq = MGT.get_lm_index(lq, mq);
                        for (int m = 0; m != 2 * L + 1; ++m)
                        {
                            const int lm = MGT.get_lm_index(L, m);
                            double triple_Y = MGT.Gaunt_Coefficients(lmp, lmq, lm);
                            Vq(lmp, lmq) += triple_Y * cfac * lattice_sum[i_add_ksq][lm];
                        }
                    }
                }
            }
        }
    }

    ModuleBase::timer::tick("Gaussian_Abfs", "get_Vq");
    return Vq;
}

Numerical_Orbital_Lm Gaussian_Abfs::Gauss(const Numerical_Orbital_Lm& orb, const double& lambda)
{
    Numerical_Orbital_Lm gaussian;
    const int angular_momentum_l = orb.getL();
    const int Nr = orb.getNr();
    const double frac = std::pow(lambda, angular_momentum_l + 1.5) / double_factorial(2 * angular_momentum_l - 1)
                        / sqrt(ModuleBase::PI * 0.5);

    std::vector<double> psi(Nr);

    for (size_t ir = 0; ir != Nr; ++ir)
    {
        psi[ir] = frac * std::pow(orb.getRadial(ir), angular_momentum_l)
                  * std::exp(-lambda * orb.getRadial(ir) * orb.getRadial(ir) * 0.5);
    }

    gaussian.set_orbital_info(orb.getLabel(),
                              orb.getType(),
                              angular_momentum_l,
                              orb.getChi(),
                              orb.getNr(),
                              orb.getRab(),
                              orb.getRadial(),
                              Numerical_Orbital_Lm::Psi_Type::Psi,
                              ModuleBase::GlobalFunc::VECTOR_TO_PTR(psi),
                              orb.getNk(),
                              orb.getDk(),
                              orb.getDruniform(),
                              false,
                              true,
                              GlobalV::CAL_FORCE);

    return gaussian;
}

double Gaussian_Abfs::double_factorial(const int& n)
{
    double result = 1.0;
    for (int i = n; i > 0; i -= 2)
    {
        if (i == 1)
            result *= 1.0;
        else
            result *= static_cast<double>(i);
    }
    return result;
}

std::vector<std::complex<double>> Gaussian_Abfs::get_lattice_sum(
    const ModuleBase::Vector3<double>& qvec,
    const ModulePW::PW_Basis_K* wfc_basis,
    const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
    const double& lambda,
    const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
    const int& lmax,           // Maximum angular momentum the sum is needed for.
    const ModuleBase::Vector3<double>& tau)
{
    if (power < 0.0 && !exclude_Gamma && qvec.norm2() < 1e-10)
        ModuleBase::WARNING_QUIT("Gaussian_Abfs::lattice_sum", "Gamma point for power<0.0 cannot be evaluated!");

    std::vector<ModuleBase::Vector3<double>> gk;
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
        gk[ig] = f * wfc_basis->G + qvec;
    }

    const int npw = gk.size();
    const int total_lm = (lmax + 1) * (lmax + 1);
    ModuleBase::matrix ylm(total_lm, npw);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, gk.data(), ylm);

    std::vector<std::complex<double>> result;
    result.resize(total_lm);
    for (int L = 0; L != lmax + 1; ++L)
    {
        for (int m = 0; m != 2 * L + 1; ++m)
        {
            const int lm = L * L + m;
            std::complex<double> val_s(0.0, 0.0);
            for (size_t ig = 0; ig != npw; ++ig)
            {
                ModuleBase::Vector3<double> qvec = gk[ig] * GlobalC::ucell.tpiba;
                if (exclude_Gamma && qvec.norm2() < 1e-10)
                    continue;
                std::complex<double> phase = std::exp(ModuleBase::IMAG_UNIT * (qvec * tau));
                val_s += std::exp(-lambda * qvec.norm2()) * std::pow(qvec.norm(), power + L) * phase * ylm(lm, ig);
            }
            result[lm] = val_s;
        }
    }

    return result;
}

#endif