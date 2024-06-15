//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_H
#define GAUSSIAN_ABFS_H

#include <RI/global/Tensor.h>

#include <array>
#include <map>
#include <vector>

#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_ao/ORB_gen_tables.h"
#include "module_basis/module_pw/pw_basis_k.h"

class Gaussian_Abfs
{
  public:
    void init(const int& Lmax, const K_Vectors* kv_in, const ModuleBase::Matrix3& G, const double& lambda);

    inline RI::Tensor<std::complex<double>> get_Vq(const int& lp_max,
                                                   const int& lq_max, // Maximum L for which to calculate interaction.
                                                   const size_t& ik,
                                                   const double& chi, // Singularity corrected value at q=0.
                                                   const ModuleBase::Vector3<double>& tau,
                                                   const ORB_gaunt_table& MGT);

    inline std::array<RI::Tensor<std::complex<double>>, 3> get_dVq(
        const int& lp_max,
        const int& lq_max, // Maximum L for which to calculate interaction.
        const size_t& ik,
        const ModuleBase::Vector3<double>& tau,
        const ORB_gaunt_table& MGT);

    /*
Calculate the lattice sum over a Gaussian:
  S(k) := \sum_G |k+G|^{power+L} \exp(-lambda*|k+G|^2) Y_{LM}(k+G) * \exp(i(k+G)\tau)
  d_S(k) := S(k) * i * (k+G)
*/
    inline std::vector<std::complex<double>> get_lattice_sum(
        const size_t& ik,
        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& exponent,
        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
        const int& lmax,           // Maximum angular momentum the sum is needed for.
        const ModuleBase::Vector3<double>& tau);

    inline std::vector<std::array<std::complex<double>, 3>> get_d_lattice_sum(
        const size_t& ik,
        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& exponent,
        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
        const int& lmax,           // Maximum angular momentum the sum is needed for.
        const ModuleBase::Vector3<double>& tau);

  private:
    double lambda;
    std::vector<ModuleBase::Vector3<double>> qGvecs;
    std::vector<int> n_cells;
    ModuleBase::matrix ylm;
    template <typename Tresult>
    using T_func_DPcal_phase = std::function<Tresult(const ModuleBase::Vector3<double>& vec)>;
    template <typename Tresult>
    using T_func_DPcal_lattice_sum
        = std::function<std::vector<Tresult>(const size_t& ik,
                                const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
                                const double& exponent,
                                const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
                                const int& lmax,           // Maximum angular momentum the sum is needed for.
                                const ModuleBase::Vector3<double>& tau)>;

    template <typename Tin, typename Tout>
    Tout DPcal_Vq_dVq(const int& lp_max,
                         const int& lq_max, // Maximum L for which to calculate interaction.
                         const size_t& ik,
                         const double& chi, // Singularity corrected value at q=0.
                         const ModuleBase::Vector3<double>& tau,
                         const ORB_gaunt_table& MGT,
                         const T_func_DPcal_lattice_sum<std::vector<Tin>>& func_DPcal_lattice_sum);

    template <typename Tresult>
    std::vector<Tresult> DPcal_lattice_sum(
        const size_t& ik,
        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& exponent,
        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
        const int& lmax,           // Maximum angular momentum the sum is needed for.
        const T_func_DPcal_phase<Tresult>& func_DPcal_phase);

    // construct gaussian basis based on original NAO
    Numerical_Orbital_Lm Gauss(const Numerical_Orbital_Lm& orb, const double& lambda);

    // calculates the double factorial n!! of n
    static double double_factorial(const int& n);
    static std::vector<int> get_n_supercells(const ModuleBase::Matrix3& G, const double& Gmax);
};

#endif