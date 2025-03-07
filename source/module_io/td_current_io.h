#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H

#include "cal_r_overlap_R.h"
#include "module_basis/module_nao/two_center_bundle.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_hamilt_lcao/module_tddft/td_current.h"
#include "module_psi/psi.h"

namespace ModuleIO
{
#ifdef __LCAO
/// @brief func to output current, only used in tddft
void write_current_eachk(const int istep,
                         const psi::Psi<std::complex<double>>* psi,
                         const elecstate::ElecState* pelec,
                         const K_Vectors& kv,
                         const TwoCenterIntegrator* intor,
                         const Parallel_Orbitals* pv,
                         const LCAO_Orbitals& orb,
                         const TD_current* cal_current,
                         Record_adj& ra,
#ifdef __EXX
                         cal_r_overlap_R& r_calculator,
                         const hamilt::HContainer<double>& sR,
                         const Exx_LRI<std::complex<double>>& exx
#endif
);

void write_current(const int istep,
                   const psi::Psi<std::complex<double>>* psi,
                   const elecstate::ElecState* pelec,
                   const K_Vectors& kv,
                   const TwoCenterIntegrator* intor,
                   const Parallel_Orbitals* pv,
                   const LCAO_Orbitals& orb,
                   const TD_current* cal_current,
                   Record_adj& ra,
#ifdef __EXX
                   cal_r_overlap_R& r_calculator,
                   const hamilt::HContainer<double>& sR,
                   const Exx_LRI<std::complex<double>>& exx
#endif
);

/// @brief calculate sum_n[𝜌_(𝑛𝑘,𝜇𝜈)] for current calculation
void cal_tmp_DM_k(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                  elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                  const int ik,
                  const int nspin,
                  const int is,
                  const bool reset = true);

void cal_tmp_DM(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                const int nspin);

template <typename Tdata>
void set_hexxR(const K_Vectors& kv,
               const Parallel_Orbitals* pv,
               const std::vector<std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>>& Hexxs,
               hamilt::HContainer<Tdata>* hexxR);

void set_rR_from_sR(const Parallel_Orbitals* pv,
                    cal_r_overlap_R& r_calculator,
                    const hamilt::HContainer<double>& sR,
                    ModuleBase::Vector3<hamilt::HContainer<double>*>& rR);

template <typename Tdata>
void cal_velocity_basis_k(const LCAO_Orbitals& orb,
                          const Parallel_Orbitals* pv,
                          const K_Vectors& kv,
                          const ModuleBase::Vector3<hamilt::HContainer<double>*>& rR,
                          const hamilt::HContainer<double>& sR,
                          const hamilt::HContainer<Tdata>& hR,
                          std::vector<ModuleBase::Vector3<std::complex<double>*>>& velocity_basis_k);

void cal_velocity_matrix(const psi::Psi<std::complex<double>>* psi,
                         const Parallel_Orbitals* pv,
                         const K_Vectors& kv,
                         const std::vector<ModuleBase::Vector3<std::complex<double>*>>& velocity_basis_k,
                         std::vector<std::array<ModuleBase::ComplexMatrix, 3>>& velocity_k);

template <typename Tdata>
void cal_current_exx_k(const LCAO_Orbitals& orb,
                       const Parallel_Orbitals* pv,
                       const K_Vectors& kv,
                       cal_r_overlap_R& r_calculator,
                       const hamilt::HContainer<double>& sR,
                       const hamilt::HContainer<Tdata>& hR,
                       const psi::Psi<std::complex<double>>* psi,
                       const elecstate::ElecState* pelec,
                       std::vector<ModuleBase::Vector3<double>>& current_k);

#endif // __LCAO
} // namespace ModuleIO
#endif // W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H
