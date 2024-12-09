#include "td_pot_mixing.h"

#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/operator_lcao.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

// Constructor
template <typename TK, typename TR>
cal_r_overlap_R hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::r_calculator;

template <typename TK, typename TR>
hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::TD_mixing_pot(
    HS_Matrix_K<TK>* hsk_in,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
    hamilt::HContainer<TR>* hR_in,
    hamilt::HContainer<TR>* SR_in,
    const LCAO_Orbitals& orb,
    const UnitCell* ucell_in,
    const std::vector<double>& orb_cutoff,
    Grid_Driver* GridD_in,
    const TwoCenterIntegrator* intor)
    : hamilt::OperatorLCAO<TK, TR>(hsk_in, kvec_d_in, hR_in), SR(SR_in), orb_(orb), orb_cutoff_(orb_cutoff), intor_(intor)
{
    this->cal_type = calculation_type::lcao_tddft_velocity;
    this->ucell = ucell_in;
#ifdef __DEBUG
    assert(this->ucell != nullptr);
    assert(this->hsk != nullptr);
#endif
    this->init_td();
    // initialize HR to allocate sparse Ekinetic matrix memory
    this->initialize_HR(GridD_in);
}

// destructor
template <typename TK, typename TR>
hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::~TD_mixing_pot()
{
    if (this->allocated)
    {
        delete this->HR_fixed;
    }
    if (this->td_velocity != nullptr)
    {
        delete this->td_velocity;
    }
    TD_Velocity::td_vel_op = nullptr;
}

// initialize_HR()
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::initialize_HR(Grid_Driver* GridD)
{
    ModuleBase::TITLE("TD_mixing_pot", "initialize_HR");
    ModuleBase::timer::tick("TD_mixing_pot", "initialize_HR");

    auto* paraV = this->hR->get_paraV();// get parallel orbitals from HR
    // TODO: if paraV is nullptr, AtomPair can not use paraV for constructor, I will repair it in the future.

    for (int iat1 = 0; iat1 < ucell->nat; iat1++)
    {
        auto tau1 = ucell->get_tau(iat1);
        int T1, I1;
        ucell->iat2iait(iat1, &I1, &T1);
        AdjacentAtomInfo adjs;
        GridD->Find_atom(*ucell, tau1, T1, I1, &adjs);
        std::vector<bool> is_adj(adjs.adj_num + 1, false);
        for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
        {
            const int T2 = adjs.ntype[ad1];
            const int I2 = adjs.natom[ad1];
            const int iat2 = ucell->itia2iat(T2, I2);
            if (paraV->get_row_size(iat1) <= 0 || paraV->get_col_size(iat2) <= 0)
            {
                continue;
            }
            const ModuleBase::Vector3<int>& R_index2 = adjs.box[ad1];
            // choose the real adjacent atoms
            // Note: the distance of atoms should less than the cutoff radius,
            // When equal, the theoretical value of matrix element is zero,
            // but the calculated value is not zero due to the numerical error, which would lead to result changes.
            if (this->ucell->cal_dtau(iat1, iat2, R_index2).norm() * this->ucell->lat0
                < orb_cutoff_[T1] + orb_cutoff_[T2])
            {
                is_adj[ad1] = true;
            }
        }
        filter_adjs(is_adj, adjs);
        this->adjs_all.push_back(adjs);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T2 = adjs.ntype[ad];
            const int I2 = adjs.natom[ad];
            int iat2 = ucell->itia2iat(T2, I2);
            ModuleBase::Vector3<int>& R_index = adjs.box[ad];
            hamilt::AtomPair<TR> tmp(iat1, iat2, R_index, paraV);
            this->hR->insert_pair(tmp);
        }
    }
    // allocate the memory of BaseMatrix in HR, and set the new values to zero
    this->hR->allocate(nullptr, true);

    ModuleBase::timer::tick("TD_mixing_pot", "initialize_HR");
}

template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::calculate_HR()
{
    ModuleBase::TITLE("TD_mixing_pot", "calculate_HR");
    if (this->HR_fixed == nullptr || this->HR_fixed->size_atom_pairs() <= 0)
    {
        ModuleBase::WARNING_QUIT("hamilt::TD_mixing_pot::calculate_HR", "HR_fixed is nullptr or empty");
    }
    ModuleBase::timer::tick("TD_mixing_pot", "calculate_HR");

    const Parallel_Orbitals* paraV = this->HR_fixed->get_atom_pair(0).get_paraV();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int iat1 = 0; iat1 < this->ucell->nat; iat1++)
    {
        auto tau1 = ucell->get_tau(iat1);
        int T1, I1;
        ucell->iat2iait(iat1, &I1, &T1);
        AdjacentAtomInfo& adjs = this->adjs_all[iat1];
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T2 = adjs.ntype[ad];
            const int I2 = adjs.natom[ad];
            const int iat2 = ucell->itia2iat(T2, I2);
            const ModuleBase::Vector3<int>& R_index2 = adjs.box[ad];
            ModuleBase::Vector3<double> dtau = this->ucell->cal_dtau(iat1, iat2, R_index2);

            hamilt::BaseMatrix<TR>* tmp = this->HR_fixed->find_matrix(iat1, iat2, R_index2);
            hamilt::BaseMatrix<TR>* tmp_overlap = this->SR->find_matrix(iat1, iat2, R_index2);
            if (tmp != nullptr)
            {
                this->cal_HR_IJR(iat1, iat2, paraV, dtau, tmp->get_pointer(), tmp_overlap->get_pointer());
            }
            else
            {
                ModuleBase::WARNING_QUIT("hamilt::TD_mixing_pot::calculate_HR", "R_index not found in HR");
            }
        }
    }

    ModuleBase::timer::tick("TD_mixing_pot", "calculate_HR");
}

// cal_HR_IJR()
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::cal_HR_IJR(const int& iat1,
                                                                   const int& iat2,
                                                                   const Parallel_Orbitals* paraV,
                                                                   const ModuleBase::Vector3<double>& dtau,
                                                                   TR* hr_mat_p,
                                                                   TR* sr_p)
{
    // ---------------------------------------------
    // get info of orbitals of atom1 and atom2 from ucell
    // ---------------------------------------------
    int T1, I1;
    this->ucell->iat2iait(iat1, &I1, &T1);
    int T2, I2;
    this->ucell->iat2iait(iat2, &I2, &T2);
    Atom& atom1 = this->ucell->atoms[T1];
    Atom& atom2 = this->ucell->atoms[T2];

    // npol is the number of polarizations,
    // 1 for non-magnetic (one Hamiltonian matrix only has spin-up or spin-down),
    // 2 for magnetic (one Hamiltonian matrix has both spin-up and spin-down)
    const int npol = this->ucell->get_npol();

    const int* iw2l1 = atom1.iw2l;
    const int* iw2n1 = atom1.iw2n;
    const int* iw2m1 = atom1.iw2m;
    const int* iw2l2 = atom2.iw2l;
    const int* iw2n2 = atom2.iw2n;
    const int* iw2m2 = atom2.iw2m;

    // ---------------------------------------------
    // calculate the Ekinetic matrix for each pair of orbitals
    // ---------------------------------------------
    double olm[3] = {0, 0, 0};
    auto row_indexes = paraV->get_indexes_row(iat1);
    auto col_indexes = paraV->get_indexes_col(iat2);
    const int step_trace = col_indexes.size() + 1;

    const ModuleBase::Vector3<double>& tau1 = this->ucell->get_tau(iat1);
    const ModuleBase::Vector3<double> tau2 = tau1 + dtau;
    for (int iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
    {
        const int iw1 = row_indexes[iw1l] / npol;
        const int L1 = iw2l1[iw1];
        const int N1 = iw2n1[iw1];
        const int m1 = iw2m1[iw1];

        // convert m (0,1,...2l) to M (-l, -l+1, ..., l-1, l)
        int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;

        for (int iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
        {
            const int iw2 = col_indexes[iw2l] / npol;
            const int L2 = iw2l2[iw2];
            const int N2 = iw2n2[iw2];
            const int m2 = iw2m2[iw2];

            ModuleBase::Vector3<double> tmp_r = r_calculator.get_psi_r_psi(tau1 * this->ucell->lat0, T1, L1, m1, N1, tau2 * this->ucell->lat0, T2, L2, m2, N2);
            // convert m (0,1,...2l) to M (-l, -l+1, ..., l-1, l)
            int M2 = (m2 % 2 == 0) ? -m2 / 2 : (m2 + 1) / 2;

            for (int ipol = 0; ipol < npol; ipol++)
            {
                hr_mat_p[ipol * step_trace] += tmp_r * Et;
                hr_mat_p[ipol * step_trace] -= ((dtau + tau1) * Et) * sr_p[ipol * step_trace] * this->ucell->lat0;
                /*std::cout << "dtau:" << dtau[0] << " " << dtau[1] << " " << dtau[2] << std::endl;
                std::cout << "tau1:" << tau1[0] << " " << tau1[1] << " " << tau1[2] << std::endl;
                std::cout << "tau2:" << tau2[0] << " " << tau2[1] << " " << tau2[2] << std::endl;
                std::cout << "Et: " << Et[0] << " " << Et[1] << " " << Et[2] << std::endl;
                std::cout << "lat0:" << this->ucell->lat0 << std::endl;*/
            }
            hr_mat_p += npol;
            sr_p += npol;
        }
        hr_mat_p += (npol - 1) * col_indexes.size();
        sr_p += (npol - 1) * col_indexes.size();
    }
}
// init two center integrals and vector potential for td_ekintic term
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::init_td()
{
    td_velocity = new TD_Velocity(this->ucell);
    TD_Velocity::td_vel_op = td_velocity;
    // initialize the r_calculator
    if(td_velocity->get_istep()==-1)
    {
        std::cout << "init_r_overlap" <<std::endl;
        r_calculator.init(*this->hR->get_paraV(), orb_);
    }
    hk_hybrid.resize(this->hR->get_paraV()->nloc);
    td_velocity->hk_hybrid=this->hk_hybrid.data();
}
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::update_td()
{
    std::cout<<"hybrid gague" <<std::endl;
    elecstate::H_TDDFT_pw::update_At();
    // calculate At in cartesian coorinates.
    td_velocity->cal_cart_At(elecstate::H_TDDFT_pw::At);
    this->cart_At = td_velocity->cart_At;
    std::cout << "cart_At: " << cart_At[0] << " " << cart_At[1] << " " << cart_At[2] << std::endl;
    Et = elecstate::H_TDDFT_pw::Et;
    std::cout<<"Et: "<<Et[0]<<" "<<Et[1]<<" "<<Et[2]<<std::endl;
}
// set_HR_fixed()
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::set_HR_fixed(void* HR_fixed_in)
{
    this->HR_fixed = static_cast<hamilt::HContainer<TR>*>(HR_fixed_in);
    this->allocated = false;
}

// contributeHR()
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::contributeHR()
{
    ModuleBase::TITLE("TD_mixing_pot", "contributeHR");
    ModuleBase::timer::tick("TD_mixing_pot", "contributeHR");

    if (!this->HR_fixed_done || TD_Velocity::evolve_once)
    {
        // if this Operator is the first node of the sub_chain, then HR_fixed is nullptr
        if (this->HR_fixed == nullptr)
        {
            this->HR_fixed = new hamilt::HContainer<TR>(*this->hR);
            this->HR_fixed->set_zero();
            this->allocated = true;
        }
        if (this->next_sub_op != nullptr)
        {
            // pass pointer of HR_fixed to the next node
            static_cast<OperatorLCAO<TK, TR>*>(this->next_sub_op)->set_HR_fixed(this->HR_fixed);
        }
        // calculate the values in HR_fixed
        this->update_td();
        this->HR_fixed->set_zero();
        this->calculate_HR();
        this->HR_fixed_done = true;
        TD_Velocity::evolve_once = false;
    }
    // last node of sub-chain, add HR_fixed into HR
    if (this->next_sub_op == nullptr)
    {
        this->hR->add(*(this->HR_fixed));
    }

    ModuleBase::timer::tick("TD_mixing_pot", "contributeHR");
    return;
}

//ETD
// contributeHk()
template <typename TK, typename TR>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<TK, TR>>::contributeHk(int ik) {
    return;
}

// contributeHk()
template <>
void hamilt::TD_mixing_pot<hamilt::OperatorLCAO<std::complex<double>, double>>::contributeHk(int ik) {
    ModuleBase::TITLE("TD_mixing_pot", "contributeHk");
    ModuleBase::timer::tick("TD_mixing_pot", "contributeHk");
    const Parallel_Orbitals* paraV = this->HR_fixed->get_atom_pair(0).get_paraV();
    this->hk_hybrid.assign(this->hk_hybrid.size(), 0);
    if(ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver))
    {
        const int nrow = paraV->get_row_size();
        TD_Velocity::td_vel_op->folding_HR_td(*this->HR_fixed, this->hk_hybrid.data(), this->kvec_d[ik], nrow, 1);
    }
    else
    {
        const int ncol = paraV->get_col_size();
        TD_Velocity::td_vel_op->folding_HR_td(*this->HR_fixed, this->hk_hybrid.data(), this->kvec_d[ik], ncol, 0);
    }
    ModuleBase::timer::tick("OperatorLCAO", "contributeHk");
}
//ETD
template class hamilt::TD_mixing_pot<hamilt::OperatorLCAO<double, double>>;
template class hamilt::TD_mixing_pot<hamilt::OperatorLCAO<std::complex<double>, double>>;
template class hamilt::TD_mixing_pot<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>;
