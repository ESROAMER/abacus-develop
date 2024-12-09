#ifndef TD_POT_MIXING_H
#define TD_POT_MIXING_H
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_basis/module_nao/two_center_integrator.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/operator_lcao.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include <vector>
#include "module_io/cal_r_overlap_R.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"
#include "module_elecstate/potentials/H_TDDFT_pw.h"

namespace hamilt
{

#ifndef __TD_POT_MIXINGTEMPLATE
#define __TD_POT_MIXINGTEMPLATE

/// The EkineticNew class template inherits from class T
/// it is used to calculate the electronic kinetic
/// Template parameters:
/// - T: base class, it would be OperatorLCAO<TK> or OperatorPW<TK>
/// - TR: data type of real space Hamiltonian, it would be double or std::complex<double>
template <class T>
class TD_mixing_pot : public T
{
};

#endif

/// EkineticNew class template specialization for OperatorLCAO<TK> base class
/// It is used to calculate the electronic kinetic matrix in real space and fold it to k-space
/// HR = <psi_{mu, 0}|-\Nabla^2|psi_{nu, R}>
/// HK = <psi_{mu, k}|-\Nabla^2|psi_{nu, k}> = \sum_{R} e^{ikR} HR
/// Template parameters:
/// - TK: data type of k-space Hamiltonian
/// - TR: data type of real space Hamiltonian
template <typename TK, typename TR>
class TD_mixing_pot<OperatorLCAO<TK, TR>> : public OperatorLCAO<TK, TR>
{
  public:
    /**
     * @brief Construct a new EkineticNew object
     */
    TD_mixing_pot<OperatorLCAO<TK, TR>>(HS_Matrix_K<TK>* hsk_in,
                                      const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                      HContainer<TR>* hR_in,
                                      HContainer<TR>* SR_in,
                                      const LCAO_Orbitals& orb,
                                      const UnitCell* ucell_in,
                                      const std::vector<double>& orb_cutoff,
                                      Grid_Driver* GridD_in,
                                      const TwoCenterIntegrator* intor);

    /**
     * @brief Destroy the EkineticNew object
     */
    ~TD_mixing_pot<OperatorLCAO<TK, TR>>();

    /**
     * @brief contributeHR() is used to calculate the HR matrix
     * <phi_{\mu, 0}|-\Nabla^2|phi_{\nu, R}>
     */
    virtual void contributeHR() override;
    //ETD
    virtual void contributeHk(int ik) override;
    //ETD

    virtual void set_HR_fixed(void*) override;


  private:
    const UnitCell* ucell = nullptr;
    std::vector<double> orb_cutoff_;
    const LCAO_Orbitals& orb_;

    hamilt::HContainer<TR>* HR_fixed = nullptr;

    hamilt::HContainer<TR>* SR = nullptr;

    const TwoCenterIntegrator* intor_ = nullptr;

    bool allocated = false;

    bool HR_fixed_done = false;
    //tddft part
    TD_Velocity* td_velocity;
    static cal_r_overlap_R r_calculator;
    //ETD
    std::vector<std::complex<double>> hk_hybrid;
    //ETD
    /// @brief Store the vector potential for td_ekinetic term
    ModuleBase::Vector3<double> cart_At;
    ModuleBase::Vector3<double> Et;


    /**
     * @brief initialize HR, search the nearest neighbor atoms
     * HContainer is used to store the electronic kinetic matrix with specific <I,J,R> atom-pairs
     * the size of HR will be fixed after initialization
     */
    void initialize_HR(Grid_Driver* GridD_in);

    void init_td();
    void update_td();

    /**
     * @brief calculate the electronic kinetic matrix with specific <I,J,R> atom-pairs
     * use the adjs_all to calculate the HR matrix
     */
    void calculate_HR();

    /**
     * @brief calculate the HR local matrix of <I,J,R> atom pair
     */
    void cal_HR_IJR(const int& iat1,
                    const int& iat2,
                    const Parallel_Orbitals* paraV,
                    const ModuleBase::Vector3<double>& dtau,
                    TR* hr_mat_p,
                    TR* sr_p);

    /// @brief exact the nearest neighbor atoms from all adjacent atoms
    std::vector<AdjacentAtomInfo> adjs_all;
};

} // namespace hamilt
#endif
