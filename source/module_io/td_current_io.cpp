#include "td_current_io.h"

#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/libm/libm.h"
#include "module_base/parallel_reduce.h"
#include "module_base/scalapack_connector.h"
#include "module_base/timer.h"
#include "module_base/tool_threading.h"
#include "module_base/vector3.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_elecstate/potentials/H_TDDFT_pw.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_domain.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#ifdef __EXX
#include "module_ri/Exx_LRI.h"
#endif

#ifdef __LCAO
void ModuleIO::set_rR_from_sR(const Parallel_Orbitals* pv,
                              cal_r_overlap_R& r_calculator,
                              const hamilt::HContainer<double>& sR,
                              ModuleBase::Vector3<hamilt::HContainer<double>*>& rR)
{
    ModuleBase::TITLE("ModuleIO", "set_rR_from_sR");
    ModuleBase::timer::tick("ModuleIO", "set_rR_from_sR");
    
    // init
    for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
    {
        for (int i = 0; i < sR.size_atom_pairs(); i++)
        {
            hamilt::AtomPair<double> atom_ij = sR.get_atom_pair(i);
            rR[i_alpha]->insert_pair(atom_ij);
        }
        rR[i_alpha]->allocate(nullptr, true);
    }

    for (int i = 0; i < sR.size_atom_pairs(); i++)
    {
        hamilt::AtomPair<double> atom_ij = sR.get_atom_pair(i);
        // loop R-index
        for (int iR = 0; iR < atom_ij.get_R_size(); iR++)
        {
            // get reference of target atom-pair
            const int iat1 = atom_ij.get_atom_i();
            const int iat2 = atom_ij.get_atom_j();
            const ModuleBase::Vector3<int> r_index = atom_ij.get_R_index(iR);
            // ---------------------------------------------
            // get info of orbitals of atom1 and atom2 from ucell
            // ---------------------------------------------
            int T1, I1;
            GlobalC::ucell.iat2iait(iat1, &I1, &T1);
            int T2, I2;
            GlobalC::ucell.iat2iait(iat2, &I2, &T2);
            Atom& atom1 = GlobalC::ucell.atoms[T1];
            Atom& atom2 = GlobalC::ucell.atoms[T2];

            // npol is the number of polarizations,
            // 1 for non-magnetic (one Hamiltonian matrix only has spin-up or spin-down),
            // 2 for magnetic (one Hamiltonian matrix has both spin-up and spin-down)
            const int npol = GlobalC::ucell.get_npol();

            const int* iw2l1 = atom1.iw2l;
            const int* iw2n1 = atom1.iw2n;
            const int* iw2m1 = atom1.iw2m;
            const int* iw2l2 = atom2.iw2l;
            const int* iw2n2 = atom2.iw2n;
            const int* iw2m2 = atom2.iw2m;

            auto row_indexes = pv->get_indexes_row(iat1);
            auto col_indexes = pv->get_indexes_col(iat2);
            const int step_trace = col_indexes.size() + 1;

            const ModuleBase::Vector3<double>& tau1 = GlobalC::ucell.get_tau(iat1);
            const ModuleBase::Vector3<double> tau2 = tau1 + GlobalC::ucell.cal_dtau(iat1, iat2, r_index);
            for (int iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
            {
                const int iw1 = row_indexes[iw1l] / npol;
                const int L1 = iw2l1[iw1];
                const int N1 = iw2n1[iw1];
                const int m1 = iw2m1[iw1];

                for (int iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
                {
                    const int iw2 = col_indexes[iw2l] / npol;
                    const int L2 = iw2l2[iw2];
                    const int N2 = iw2n2[iw2];
                    const int m2 = iw2m2[iw2];

                    ModuleBase::Vector3<double> tmp_r = r_calculator.get_psi_r_psi(tau1 * GlobalC::ucell.lat0,
                                                                                   T1,
                                                                                   L1,
                                                                                   m1,
                                                                                   N1,
                                                                                   tau2 * GlobalC::ucell.lat0,
                                                                                   T2,
                                                                                   L2,
                                                                                   m2,
                                                                                   N2);
                    for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
                    {
                        hamilt::BaseMatrix<double>* HlocR
                            = rR[i_alpha]->find_matrix(iat1, iat2, r_index.x, r_index.y, r_index.z);
                        HlocR->add_element(iw1, iw2, tmp_r[i_alpha]);
                    }
                }
            }
        }
    }
    ModuleBase::TITLE("ModuleIO", "set_rR_from_sR");
}

// for molecule, if vacuum size is small, the number of R of Hs is smaller than SR
// which may lead to some errors
void ModuleIO::cal_velocity_basis_k(
    const LCAO_Orbitals& orb,
    const Parallel_Orbitals* pv,
    const K_Vectors& kv,
    const ModuleBase::Vector3<hamilt::HContainer<double>*>& rR,
    const hamilt::HContainer<double>& sR,
    const hamilt::HContainer<double>& hR,
    std::vector<ModuleBase::Vector3<std::complex<double>*>>& velocity_basis_k)
{
    ModuleBase::TITLE("ModuleIO", "cal_velocity_basis_k");
    ModuleBase::timer::tick("ModuleIO", "cal_velocity_basis_k");

    const double coeff = (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Cam
                          || GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Ccp)
                             ? 1.0
                             : GlobalC::exx_info.info_global.hybrid_alpha;
    const int nlocal = PARAM.globalv.nlocal;
    const char N_char = 'N';
    const std::complex<double> one_imag = ModuleBase::IMAG_UNIT;
    const std::complex<double> neg_one_imag = ModuleBase::NEG_IMAG_UNIT;
    const std::complex<double> one_real = ModuleBase::ONE;
    const std::complex<double> neg_one_real = ModuleBase::NEG_ONE;
    const std::complex<double> zero_complex = ModuleBase::ZERO;

    std::complex<double>* hk = new std::complex<double>[pv->nloc];
    std::complex<double>* sk = new std::complex<double>[pv->nloc];
    std::complex<double>* partial_hk = new std::complex<double>[pv->nloc];
    std::complex<double>* partial_sk = new std::complex<double>[pv->nloc];
    std::complex<double>* rk = new std::complex<double>[pv->nloc];
    std::complex<double>* h_is = new std::complex<double>[pv->nloc];
    std::complex<double>* h_is_r = new std::complex<double>[pv->nloc];
    std::complex<double>* r_is = new std::complex<double>[pv->nloc];
    std::complex<double>* r_is_h = new std::complex<double>[pv->nloc];
    std::complex<double>* h_is_ps = new std::complex<double>[pv->nloc];

    for (size_t ik = 0; ik != kv.get_nks(); ++ik)
    {
        // set H(k), S(k)
        // 1.1 set H(k)
        ModuleBase::GlobalFunc::ZEROS(hk, pv->nloc);
        const int nrow = pv->get_row_size();
        hamilt::folding_HR(hR, hk, kv.kvec_d[ik], nrow, 1);
        // 1.2 set S(k)
        ModuleBase::GlobalFunc::ZEROS(sk, pv->nloc);
        hamilt::folding_HR(sR, sk, kv.kvec_d[ik], nrow, 1);
        // 2. set inverse S(k) -> sk will be changed to sk_inv
        int* ipiv = new int[pv->nloc];
        int info = 0;
        // 2.1 compute ipiv
        ScalapackConnector::getrf(nlocal, nlocal, sk, 1, 1, pv->desc, ipiv, &info);
        int lwork = -1;
        int liwotk = -1;
        std::vector<std::complex<double>> work(1, 0);
        std::vector<int> iwork(1, 0);
        // 2.2 compute work
        ScalapackConnector::getri(nlocal, sk, 1, 1, pv->desc, ipiv, work.data(), &lwork, iwork.data(), &liwotk, &info);
        lwork = work[0].real();
        work.resize(lwork, 0);
        liwotk = iwork[0];
        iwork.resize(liwotk, 0);
        // 2.3 compute inverse matrix of Sk
        ScalapackConnector::getri(nlocal,
                                  sk, // return sk^-1
                                  1,
                                  1,
                                  pv->desc,
                                  ipiv,
                                  work.data(),
                                  &lwork,
                                  iwork.data(),
                                  &liwotk,
                                  &info);
        assert(0 == info);
        for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
        {
            // 3. set partial_H(k), partial_S(k) and r(k)
            // 3.1 set partial_H(k)
            ModuleBase::GlobalFunc::ZEROS(partial_hk, pv->nloc);
            hamilt::folding_partial_HR(hR, partial_hk, kv.kvec_d[ik], i_alpha, nrow, 1);
            // 3.2 set partial S(k)
            ModuleBase::GlobalFunc::ZEROS(partial_sk, pv->nloc);
            hamilt::folding_partial_HR(sR, partial_sk, kv.kvec_d[ik], i_alpha, nrow, 1);
            // 3.3 set r(k)
            ModuleBase::GlobalFunc::ZEROS(rk, pv->nloc);
            hamilt::folding_HR(*rR[i_alpha], rk, kv.kvec_d[ik], nrow, 1); // set r(k)
            // 4. calculate <\vu,k|v_a|\mu,k> = partial_Hk + IMAG_UNIT * (Hk * Sk_inv * rk) - IMAG_UNIT * (rk * Sk_inv * Hk) - Hk * Sk_inv * partial_Sk 
            //4.1.1 Hk * Sk_inv (note 2.)  
            ModuleBase::GlobalFunc::ZEROS(h_is, pv->nloc);
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nlocal,
                                     nlocal,
                                     one_real,
                                     hk,
                                     1,
                                     1,
                                     pv->desc,
                                     sk,
                                     1,
                                     1,
                                     pv->desc,
                                     zero_complex,
                                     h_is,
                                     1,
                                     1,
                                     pv->desc);
            // 4.1.2 (Hk * Sk_inv) * rk
            ModuleBase::GlobalFunc::ZEROS(h_is_r, pv->nloc);
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nlocal,
                                     nlocal,
                                     one_real,
                                     h_is,
                                     1,
                                     1,
                                     pv->desc,
                                     rk,
                                     1,
                                     1,
                                     pv->desc,
                                     zero_complex,
                                     h_is_r,
                                     1,
                                     1,
                                     pv->desc);
            // 4.2.1 rk * Sk_inv (note 2.)
            ModuleBase::GlobalFunc::ZEROS(r_is, pv->nloc);
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nlocal,
                                     nlocal,
                                     one_real,
                                     rk,
                                     1,
                                     1,
                                     pv->desc,
                                     sk,
                                     1,
                                     1,
                                     pv->desc,
                                     zero_complex,
                                     r_is,
                                     1,
                                     1,
                                     pv->desc);
            // 4.2.2 (rk * Sk_inv) * Hk
            ModuleBase::GlobalFunc::ZEROS(r_is_h, pv->nloc);
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nlocal,
                                     nlocal,
                                     one_real,
                                     r_is,
                                     1,
                                     1,
                                     pv->desc,
                                     hk,
                                     1,
                                     1,
                                     pv->desc,
                                     zero_complex,
                                     r_is_h,
                                     1,
                                     1,
                                     pv->desc);
            // 4.3.1 (Hk * Sk_inv) * partial_Sk
            ModuleBase::GlobalFunc::ZEROS(h_is_ps, pv->nloc);
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nlocal,
                                     nlocal,
                                     one_real,
                                     h_is,
                                     1,
                                     1,
                                     pv->desc,
                                     partial_sk,
                                     1,
                                     1,
                                     pv->desc,
                                     zero_complex,
                                     h_is_ps,
                                     1,
                                     1,
                                     pv->desc);
            // 4.4 h_is_r will be changed to partial_Hk + IMAG_UNIT * (Hk * Sk_inv * rk)
            ScalapackConnector::geadd('N',
                                      nlocal,
                                      nlocal,
                                      one_real,
                                      partial_hk,
                                      1,
                                      1,
                                      pv->desc,
                                      one_imag,
                                      h_is_r,
                                      1,
                                      1,
                                      pv->desc);
            // 4.5 r_is_h will be changed to h_is_r - IMAG_UNIT * (rk * Sk_inv * Hk)
            ScalapackConnector::geadd('N',
                                      nlocal,
                                      nlocal,
                                      one_real,
                                      h_is_r,
                                      1,
                                      1,
                                      pv->desc,
                                      neg_one_imag,
                                      r_is_h,
                                      1,
                                      1,
                                      pv->desc);
            // 4.6 h_is_ps will be changed to r_is_h - Hk * Sk_inv * partial_Sk
            ScalapackConnector::geadd('N',
                                      nlocal,
                                      nlocal,
                                      one_real,
                                      r_is_h,
                                      1,
                                      1,
                                      pv->desc,
                                      neg_one_real,
                                      h_is_ps,
                                      1,
                                      1,
                                      pv->desc);
            // 5. copy h_is_ps to velocity_basis_k[ik][i_alpha]
            BlasConnector::copy(pv->nloc, h_is_ps, 1, velocity_basis_k[ik][i_alpha], 1);
        }
    }

    delete[] hk;
    delete[] sk;
    delete[] partial_hk;
    delete[] partial_sk;
    delete[] rk;
    delete[] h_is;
    delete[] h_is_r;
    delete[] r_is;
    delete[] r_is_h;
    delete[] h_is_ps;
    ModuleBase::timer::tick("ModuleIO", "cal_velocity_basis_k");
}

void ModuleIO::cal_velocity_matrix(const psi::Psi<std::complex<double>>* psi,
                                   const Parallel_Orbitals* pv,
                                   const K_Vectors& kv,
                                   const std::vector<ModuleBase::Vector3<std::complex<double>*>>& velocity_basis_k,
                                   std::vector<std::array<ModuleBase::ComplexMatrix, 3>>& velocity_k)
{
    ModuleBase::TITLE("ModuleIO", "cal_velocity_matrix");
    ModuleBase::timer::tick("ModuleIO", "cal_velocity_matrix");

    const char N_char = 'N';
    const char C_char = 'C';
    const std::complex<double> one_real = ModuleBase::ONE;
    const std::complex<double> zero_complex = ModuleBase::ZERO;
    const double zero_double = 0.0;
    const int nlocal = PARAM.globalv.nlocal;
    const int nbands = PARAM.inp.nbands;
    std::complex<double>* vk_c = new std::complex<double>[pv->ncol_bands*pv->nrow_bands]; // local one
    std::complex<double>* v_c = new std::complex<double>[pv->nloc_wfc];
    // Parallel_2D pv_bands; 
    // pv_bands.set(nbands, nbands, pv->nb, pv->blacs_ctxt);

    for (int ik = 0; ik < kv.get_nks(); ik++)
    {
        // 1. set C
        psi->fix_k(ik);
        // 2. set <\Psi_{n,\mu}|v_{\mu,\nu}|\Psi_{m,\nu}> = C^\dagger_{n,\mu} * v_{\mu,\nu} * C_{\nu,m}
        for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
        {
            ModuleBase::GlobalFunc::ZEROS(vk_c, pv->ncol_bands*pv->nrow_bands);
            ModuleBase::GlobalFunc::ZEROS(v_c, pv->nloc_wfc);
            // v_c_{\mu,m} = v_{\mu,\nu} * C_{\nu,m}
            ScalapackConnector::gemm(N_char,
                                     N_char,
                                     nlocal,
                                     nbands,
                                     nlocal,
                                     one_real,
                                     velocity_basis_k[ik][i_alpha],
                                     1,
                                     1,
                                     pv->desc,
                                     psi[0].get_pointer(),
                                     1,
                                     1,
                                     pv->desc_wfc,
                                     zero_complex,
                                     v_c,
                                     1,
                                     1,
                                     pv->desc_wfc);
            // velocity_k_{n,m} = C^\dagger_{n,\mu} * v_c_{\mu,m}
            ScalapackConnector::gemm(C_char,
                                     N_char,
                                     nbands,
                                     nbands,
                                     nlocal,
                                     one_real,
                                     psi[0].get_pointer(),
                                     1,
                                     1,
                                     pv->desc_wfc,
                                     v_c,
                                     1,
                                     1,
                                     pv->desc_wfc,
                                     zero_complex,
                                     vk_c,
                                     1,
                                     1,
                                     pv->desc_Eij);

            for (int ir = 0; ir < PARAM.inp.nbands; ++ir)
            {
                for (int ic = 0; ic < PARAM.inp.nbands; ++ic)
                {
                    const int irc = ic * pv_bands.nrow + ir;
                    if (pv_bands.in_this_processor(ir, ic))
                        velocity_k[ik][i_alpha](ir, ic) = vk_c[irc];
                }
            }
        }
    }

    delete[] vk_c;
    delete[] v_c;

    ModuleBase::timer::tick("ModuleIO", "cal_velocity_matrix");
}

void ModuleIO::cal_current_exx_k(const LCAO_Orbitals& orb,
                                 const Parallel_Orbitals* pv,
                                 const K_Vectors& kv,
                                 cal_r_overlap_R& r_calculator,
                                 const hamilt::HContainer<double>& sR,
                                 const hamilt::HContainer<double>& hR,
                                 const psi::Psi<std::complex<double>>* psi,
                                 const elecstate::ElecState* pelec,
                                 std::vector<ModuleBase::Vector3<double>>& current_k)
{
    ModuleBase::TITLE("ModuleIO", "cal_current_exx");
    ModuleBase::timer::tick("ModuleIO", "cal_current_exx");

    const int nlocal = PARAM.globalv.nlocal;
    const int nbands = PARAM.inp.nbands;
    // init
    ModuleBase::Vector3<hamilt::HContainer<double>*> rR;
    std::vector<ModuleBase::Vector3<std::complex<double>*>> velocity_basis_k;
    std::vector<std::array<ModuleBase::ComplexMatrix, 3>> velocity_k;
    velocity_basis_k.resize(kv.get_nks());
    velocity_k.resize(kv.get_nks());
    for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
    {
        rR[i_alpha] = new hamilt::HContainer<double>(pv);
        for (int ik = 0; ik < kv.get_nks(); ik++)
        {
            velocity_basis_k[ik][i_alpha] = new std::complex<double>[pv->nloc];
            ModuleBase::GlobalFunc::ZEROS(velocity_basis_k[ik][i_alpha], pv->nloc);
            velocity_k[ik][i_alpha].create(nbands, nbands);
        }
    }
    // set rR
    set_rR_from_sR(pv, r_calculator, sR, rR);
    // set velocity_basis_k
    cal_velocity_basis_k(orb, pv, kv, rR, sR, hR, velocity_basis_k);
    // set velocity_k
    cal_velocity_matrix(psi, pv, kv, velocity_basis_k, velocity_k);

    // sum n and m for current_k
    for (size_t ik = 0; ik != kv.get_nks(); ++ik)
        for (size_t i_alpha = 0; i_alpha != 3; ++i_alpha)
            for (size_t ib = 0; ib != 3; ++ib)
                current_k[ik][i_alpha] += pelec->wg(ik, ib) * velocity_k[ik][i_alpha](ib, ib).real() / 2.0; // for unit
                
    for (size_t i_alpha = 0; i_alpha < 3; ++i_alpha)
    {
        delete rR[i_alpha];
        for (int ik = 0; ik < kv.get_nks(); ik++)
            delete[] velocity_basis_k[ik][i_alpha];
    }

    ModuleBase::TITLE("ModuleIO", "cal_current_exx");
}

void ModuleIO::cal_tmp_DM(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                          elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                          int nspin)
{
    ModuleBase::TITLE("ModuleIO", "cal_tmp_DM");
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM");
    int ld_hk = DM_real.get_paraV_pointer()->nrow;
    int ld_hk2 = 2 * ld_hk;
    for (int is = 1; is <= nspin; ++is)
    {
        for (int ik = 0; ik < DM_real.get_DMK_nks() / nspin; ++ik)
        {
            cal_tmp_DM_k(DM_real, DM_imag, ik, nspin, is, false);
        }
    }
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM");
}
void ModuleIO::write_current(const int istep,
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
                             const hamilt::HContainer<double>& hR
#endif
)
{

    ModuleBase::TITLE("ModuleIO", "write_current");
    ModuleBase::timer::tick("ModuleIO", "write_current");
    std::vector<hamilt::HContainer<std::complex<double>>*> current_term = {nullptr, nullptr, nullptr};
    if (!TD_Velocity::tddft_velocity)
    {
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = cal_current->get_current_term_pointer(dir);
        }
    }
    else
    {
        if (TD_Velocity::td_vel_op == nullptr)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_current", "velocity gague infos is null!");
        }
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = TD_Velocity::td_vel_op->get_current_term_pointer(dir);
        }
    }
    double omega = GlobalC::ucell.omega;
    // construct a DensityMatrix object
    // Since the function cal_dm_psi do not suport DMR in complex type, I replace it with two DMR in double type. Should
    // be refactored in the future.
    const int nspin_dm = std::map<int, int>({{1, 1}, {2, 2}, {4, 1}})[PARAM.inp.nspin];
    elecstate::DensityMatrix<std::complex<double>, double> DM_real(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    elecstate::DensityMatrix<std::complex<double>, double> DM_imag(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    // calculate DMK
    elecstate::cal_dm_psi(DM_real.get_paraV_pointer(), pelec->wg, psi[0], DM_real);

    // init DMR
    DM_real.init_DMR(ra, &GlobalC::ucell);
    DM_imag.init_DMR(ra, &GlobalC::ucell);
    cal_tmp_DM(DM_real, DM_imag, PARAM.inp.nspin);
    DM_real.sum_DMR_spin();
    DM_imag.sum_DMR_spin();

    double current_total[3] = {0.0, 0.0, 0.0};
#ifdef _OPENMP
#pragma omp parallel
    {
        double local_current[3] = {0.0, 0.0, 0.0};
#else
    // ModuleBase::matrix& local_soverlap = soverlap;
    double* local_current = current_total;
#endif
        ModuleBase::Vector3<double> tau1, dtau, tau2;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int iat = 0; iat < GlobalC::ucell.nat; iat++)
        {
            const int T1 = GlobalC::ucell.iat2it[iat];
            Atom* atom1 = &GlobalC::ucell.atoms[T1];
            const int I1 = GlobalC::ucell.iat2ia[iat];
            // get iat1
            int iat1 = GlobalC::ucell.itia2iat(T1, I1);
            const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
            for (int cb = 0; cb < ra.na_each[iat]; ++cb)
            {
                const int T2 = ra.info[iat][cb][3];
                const int I2 = ra.info[iat][cb][4];

                const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);

                Atom* atom2 = &GlobalC::ucell.atoms[T2];

                // get iat2
                int iat2 = GlobalC::ucell.itia2iat(T2, I2);
                double Rx = ra.info[iat][cb][0];
                double Ry = ra.info[iat][cb][1];
                double Rz = ra.info[iat][cb][2];
                // std::cout<< "iat1: " << iat1 << " iat2: " << iat2 << " Rx: " << Rx << " Ry: " << Ry << " Rz:" << Rz
                // << std::endl;
                //   get BaseMatrix
                hamilt::BaseMatrix<double>* tmp_matrix_real
                    = DM_real.get_DMR_pointer(1)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<double>* tmp_matrix_imag
                    = DM_imag.get_DMR_pointer(1)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                // refactor
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvx
                    = current_term[0]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvy
                    = current_term[1]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvz
                    = current_term[2]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                if (tmp_matrix_real == nullptr)
                {
                    continue;
                }
                int row_ap = pv->atom_begin_row[iat1];
                int col_ap = pv->atom_begin_col[iat2];
                // get DMR
                for (int mu = 0; mu < pv->get_row_size(iat1); ++mu)
                {
                    for (int nu = 0; nu < pv->get_col_size(iat2); ++nu)
                    {
                        double dm2d1_real = tmp_matrix_real->get_value(mu, nu);
                        double dm2d1_imag = tmp_matrix_imag->get_value(mu, nu);

                        std::complex<double> rvx = {0, 0};
                        std::complex<double> rvy = {0, 0};
                        std::complex<double> rvz = {0, 0};

                        if (tmp_m_rvx != nullptr)
                        {
                            rvx = tmp_m_rvx->get_value(mu, nu);
                            rvy = tmp_m_rvy->get_value(mu, nu);
                            rvz = tmp_m_rvz->get_value(mu, nu);
                        }
                        // std::cout<<"mu: "<< mu <<" nu: "<< nu << std::endl;
                        // std::cout<<"dm2d1_real: "<< dm2d1_real << " dm2d1_imag: "<< dm2d1_imag << std::endl;
                        // std::cout<<"rvz: "<< rvz.real() << " " << rvz.imag() << std::endl;
                        local_current[0] -= dm2d1_real * rvx.real() - dm2d1_imag * rvx.imag();
                        local_current[1] -= dm2d1_real * rvy.real() - dm2d1_imag * rvy.imag();
                        local_current[2] -= dm2d1_real * rvz.real() - dm2d1_imag * rvz.imag();
                    } // end kk
                } // end jj
            } // end cb
        } // end iat
#ifdef _OPENMP
#pragma omp critical(cal_current_k_reduce)
        {
            for (int i = 0; i < 3; ++i)
            {
                current_total[i] += local_current[i];
            }
        }
    }
#endif
    Parallel_Reduce::reduce_all(current_total, 3);
#ifdef __EXX
    //if (GlobalC::exx_info.info_global.cal_exx)
    //{
        std::vector<ModuleBase::Vector3<double>> current_k_exx;
        current_k_exx.resize(kv.get_nks());
        //TODO: set HexxR to hContainer
        ModuleBase::Vector3<double> current_new;
        cal_current_exx_k(orb, pv, kv, r_calculator, sR, hR, psi, pelec, current_k_exx);
        for (int dir = 0; dir < 3; dir++)
        {
            for (int ik = 0; ik < kv.get_nks(); ik++)
            {
                current_new[dir] -= current_k_exx[ik][dir];
            }
        }
    //}
#endif
    // write end
    if (GlobalV::MY_RANK == 0)
    {
        std::string filename = PARAM.globalv.global_out_dir + "current_total.dat";
        std::ofstream fout;
        fout.open(filename, std::ios::app);
        fout << std::setprecision(16);
        fout << std::scientific;
        fout << istep << " " << current_total[0] / omega << " " << current_total[1] / omega << " "
             << current_total[2] / omega << std::endl;
        fout.close();

        std::string filename_new = PARAM.globalv.global_out_dir + "current_total_new.dat";
        std::ofstream fout_new;
        fout_new.open(filename_new, std::ios::app);
        fout_new << std::setprecision(16);
        fout_new << std::scientific;
        fout_new << istep << " " << current_new[0] / omega << " " << current_new[1] / omega << " "
             << current_new[2] / omega << std::endl;
        fout_new.close();
    }

    ModuleBase::timer::tick("ModuleIO", "write_current");
    return;
}
void ModuleIO::cal_tmp_DM_k(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                            elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                            const int ik,
                            const int nspin,
                            const int is,
                            const bool reset)
{
    ModuleBase::TITLE("ModuleIO", "cal_tmp_DM_k");
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM_k");
    int ld_hk = DM_real.get_paraV_pointer()->nrow;
    int ld_hk2 = 2 * ld_hk;
    // tmp for is
    int ik_begin = DM_real.get_DMK_nks() / nspin * (is - 1); // jump nk for spin_down if nspin==2

    hamilt::HContainer<double>* tmp_DMR_real = DM_real.get_DMR_vector()[is - 1];
    hamilt::HContainer<double>* tmp_DMR_imag = DM_imag.get_DMR_vector()[is - 1];
    if (reset)
    {
        tmp_DMR_real->set_zero();
        tmp_DMR_imag->set_zero();
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < tmp_DMR_real->size_atom_pairs(); ++i)
    {
        hamilt::AtomPair<double>& tmp_ap_real = tmp_DMR_real->get_atom_pair(i);
        hamilt::AtomPair<double>& tmp_ap_imag = tmp_DMR_imag->get_atom_pair(i);
        int iat1 = tmp_ap_real.get_atom_i();
        int iat2 = tmp_ap_real.get_atom_j();
        // get global indexes of whole matrix for each atom in this process
        int row_ap = DM_real.get_paraV_pointer()->atom_begin_row[iat1];
        int col_ap = DM_real.get_paraV_pointer()->atom_begin_col[iat2];
        for (int ir = 0; ir < tmp_ap_real.get_R_size(); ++ir)
        {
            const ModuleBase::Vector3<int> r_index = tmp_ap_real.get_R_index(ir);
            hamilt::BaseMatrix<double>* tmp_matrix_real = tmp_ap_real.find_matrix(r_index);
            hamilt::BaseMatrix<double>* tmp_matrix_imag = tmp_ap_imag.find_matrix(r_index);
#ifdef __DEBUG
            if (tmp_matrix_real == nullptr)
            {
                std::cout << "tmp_matrix is nullptr" << std::endl;
                continue;
            }
#endif
            // only ik
            if (PARAM.inp.nspin != 4)
            {
                double arg_td = 0.0;
                if (elecstate::H_TDDFT_pw::stype == 2)
                {
                    // new
                    // cal tddft phase for mixing gague
                    const int iat1 = tmp_ap_real.get_atom_i();
                    const int iat2 = tmp_ap_real.get_atom_j();
                    ModuleBase::Vector3<double> dtau
                        = TD_Velocity::td_vel_op->get_ucell()->cal_dtau(iat1, iat2, r_index);
                    double& tmp_lat0 = TD_Velocity::td_vel_op->get_ucell()->lat0;
                    arg_td = TD_Velocity::td_vel_op->cart_At * dtau * tmp_lat0;

                    /*std::cout << "arg_td " << arg_td << std::endl;
                    std::cout << "cart_At " << TD_Velocity::td_vel_op->cart_At[0] << " "<<
                    TD_Velocity::td_vel_op->cart_At[1] << " " << TD_Velocity::td_vel_op->cart_At[2] << std::endl;
                    std::cout << "dtau " << dtau[0] << " "<< dtau[1] << " " << dtau[2] << std::endl;
                    std::cout << "ucell->lat0 " << tmp_lat0 << std::endl;
                    std::cout << "iat1 " << iat1 << " " << "iat2 " << iat2 << std::endl;*/
                    // new
                }
                // cal k_phase
                // if TK==std::complex<double>, kphase is e^{ikR}
                const ModuleBase::Vector3<double> dR(r_index.x, r_index.y, r_index.z);
                const double arg = (DM_real.get_kvec_d()[ik] * dR) * ModuleBase::TWO_PI + arg_td;
                double sinp, cosp;
                ModuleBase::libm::sincos(arg, &sinp, &cosp);
                std::complex<double> kphase = std::complex<double>(cosp, sinp);
                // set DMR element
                double* tmp_DMR_real_pointer = tmp_matrix_real->get_pointer();
                double* tmp_DMR_imag_pointer = tmp_matrix_imag->get_pointer();
                std::complex<double>* tmp_DMK_pointer = DM_real.get_DMK_pointer(ik + ik_begin);
                double* DMK_real_pointer = nullptr;
                double* DMK_imag_pointer = nullptr;
                // jump DMK to fill DMR
                // DMR is row-major, DMK is column-major
                tmp_DMK_pointer += col_ap * DM_real.get_paraV_pointer()->nrow + row_ap;
                for (int mu = 0; mu < DM_real.get_paraV_pointer()->get_row_size(iat1); ++mu)
                {
                    DMK_real_pointer = (double*)tmp_DMK_pointer;
                    DMK_imag_pointer = DMK_real_pointer + 1;
                    // calculate real part
                    BlasConnector::axpy(DM_real.get_paraV_pointer()->get_col_size(iat2),
                                        -kphase.imag(),
                                        DMK_imag_pointer,
                                        ld_hk2,
                                        tmp_DMR_real_pointer,
                                        1);
                    BlasConnector::axpy(DM_real.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.real(),
                                        DMK_real_pointer,
                                        ld_hk2,
                                        tmp_DMR_real_pointer,
                                        1);
                    // calculate imag part
                    BlasConnector::axpy(DM_imag.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.imag(),
                                        DMK_real_pointer,
                                        ld_hk2,
                                        tmp_DMR_imag_pointer,
                                        1);
                    BlasConnector::axpy(DM_imag.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.real(),
                                        DMK_imag_pointer,
                                        ld_hk2,
                                        tmp_DMR_imag_pointer,
                                        1);
                    tmp_DMK_pointer += 1;
                    tmp_DMR_real_pointer += DM_real.get_paraV_pointer()->get_col_size(iat2);
                    tmp_DMR_imag_pointer += DM_imag.get_paraV_pointer()->get_col_size(iat2);
                }
            }
        }
    }
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM_k");
}

void ModuleIO::write_current_eachk(const int istep,
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
                                   const hamilt::HContainer<double>& hR
#endif
)
{

    ModuleBase::TITLE("ModuleIO", "write_current");
    ModuleBase::timer::tick("ModuleIO", "write_current");
    std::vector<hamilt::HContainer<std::complex<double>>*> current_term = {nullptr, nullptr, nullptr};
    if (!TD_Velocity::tddft_velocity)
    {
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = cal_current->get_current_term_pointer(dir);
        }
    }
    else
    {
        if (TD_Velocity::td_vel_op == nullptr)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_current", "velocity gague infos is null!");
        }
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = TD_Velocity::td_vel_op->get_current_term_pointer(dir);
        }
    }
    double omega = GlobalC::ucell.omega;
    // construct a DensityMatrix object
    // Since the function cal_dm_psi do not suport DMR in complex type, I replace it with two DMR in double type. Should
    // be refactored in the future.
    const int nspin_dm = std::map<int, int>({{1, 1}, {2, 2}, {4, 1}})[PARAM.inp.nspin];
    elecstate::DensityMatrix<std::complex<double>, double> DM_real(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    elecstate::DensityMatrix<std::complex<double>, double> DM_imag(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    // calculate DMK
    elecstate::cal_dm_psi(DM_real.get_paraV_pointer(), pelec->wg, psi[0], DM_real);

    // init DMR
    DM_real.init_DMR(ra, &GlobalC::ucell);
    DM_imag.init_DMR(ra, &GlobalC::ucell);

    std::vector<ModuleBase::Vector3<double>> current_k_exx;
#ifdef __EXX
    //if (GlobalC::exx_info.info_global.cal_exx)
    //{
        current_k_exx.resize(kv.get_nks());
        cal_current_exx_k(orb, pv, kv, r_calculator, sR, hR, psi, pelec, current_k_exx);
    //}
#endif

    int nks = DM_real.get_DMK_nks();
    if (PARAM.inp.nspin == 2)
    {
        nks /= 2;
    }
    double current_total[3] = {0.0, 0.0, 0.0};
    for (int is = 1; is <= PARAM.inp.nspin; ++is)
    {
        for (int ik = 0; ik < nks; ++ik)
        {
            cal_tmp_DM_k(DM_real, DM_imag, ik, PARAM.inp.nspin, is);
            // check later
            double current_ik[3] = {0.0, 0.0, 0.0};
#ifdef _OPENMP
#pragma omp parallel
            {
                int num_threads = omp_get_num_threads();
                double local_current_ik[3] = {0.0, 0.0, 0.0};
#else
            // ModuleBase::matrix& local_soverlap = soverlap;
            double* local_current_ik = current_ik;
#endif

                ModuleBase::Vector3<double> tau1, dtau, tau2;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (int iat = 0; iat < GlobalC::ucell.nat; iat++)
                {
                    const int T1 = GlobalC::ucell.iat2it[iat];
                    Atom* atom1 = &GlobalC::ucell.atoms[T1];
                    const int I1 = GlobalC::ucell.iat2ia[iat];
                    // get iat1
                    int iat1 = GlobalC::ucell.itia2iat(T1, I1);
                    const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
                    for (int cb = 0; cb < ra.na_each[iat]; ++cb)
                    {
                        const int T2 = ra.info[iat][cb][3];
                        const int I2 = ra.info[iat][cb][4];

                        const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);

                        Atom* atom2 = &GlobalC::ucell.atoms[T2];

                        // get iat2
                        int iat2 = GlobalC::ucell.itia2iat(T2, I2);
                        double Rx = ra.info[iat][cb][0];
                        double Ry = ra.info[iat][cb][1];
                        double Rz = ra.info[iat][cb][2];
                        // std::cout<< "iat1: " << iat1 << " iat2: " << iat2 << " Rx: " << Rx << " Ry: " << Ry << " Rz:"
                        // << Rz << std::endl;
                        //   get BaseMatrix
                        hamilt::BaseMatrix<double>* tmp_matrix_real
                            = DM_real.get_DMR_pointer(is)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<double>* tmp_matrix_imag
                            = DM_imag.get_DMR_pointer(is)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        // refactor
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvx
                            = current_term[0]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvy
                            = current_term[1]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvz
                            = current_term[2]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        if (tmp_matrix_real == nullptr)
                        {
                            continue;
                        }
                        int row_ap = pv->atom_begin_row[iat1];
                        int col_ap = pv->atom_begin_col[iat2];
                        // get DMR
                        for (int mu = 0; mu < pv->get_row_size(iat1); ++mu)
                        {
                            for (int nu = 0; nu < pv->get_col_size(iat2); ++nu)
                            {
                                double dm2d1_real = tmp_matrix_real->get_value(mu, nu);
                                double dm2d1_imag = tmp_matrix_imag->get_value(mu, nu);

                                std::complex<double> rvx = {0, 0};
                                std::complex<double> rvy = {0, 0};
                                std::complex<double> rvz = {0, 0};

                                if (tmp_m_rvx != nullptr)
                                {
                                    rvx = tmp_m_rvx->get_value(mu, nu);
                                    rvy = tmp_m_rvy->get_value(mu, nu);
                                    rvz = tmp_m_rvz->get_value(mu, nu);
                                }
                                // std::cout<<"mu: "<< mu <<" nu: "<< nu << std::endl;
                                // std::cout<<"dm2d1_real: "<< dm2d1_real << " dm2d1_imag: "<< dm2d1_imag << std::endl;
                                // std::cout<<"rvz: "<< rvz.real() << " " << rvz.imag() << std::endl;
                                local_current_ik[0] -= dm2d1_real * rvx.real() - dm2d1_imag * rvx.imag();
                                local_current_ik[1] -= dm2d1_real * rvy.real() - dm2d1_imag * rvy.imag();
                                local_current_ik[2] -= dm2d1_real * rvz.real() - dm2d1_imag * rvz.imag();
                            } // end kk
                        } // end jj
                    } // end cb
                } // end iat
#ifdef _OPENMP
#pragma omp critical(cal_current_k_reduce)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        current_ik[i] += local_current_ik[i];
                    }
                }
            }
#endif
            Parallel_Reduce::reduce_all(current_ik, 3);
            for (int i = 0; i < 3; ++i)
            {
#ifdef __EXX
                if (GlobalC::exx_info.info_global.cal_exx)
                {
                    current_ik[i] += current_k_exx[ik][i];
                }
#endif
                current_total[i] += current_ik[i];
            }
            // MPI_Reduce(local_current_ik, current_ik, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (GlobalV::MY_RANK == 0 && TD_Velocity::out_current_k)
            {
                std::string filename = PARAM.globalv.global_out_dir + "current_spin" + std::to_string(is) + "_ik"
                                       + std::to_string(ik) + ".dat";
                std::ofstream fout;
                fout.open(filename, std::ios::app);
                fout << std::setprecision(16);
                fout << std::scientific;
                fout << istep << " " << current_ik[0] / omega << " " << current_ik[1] / omega << " "
                     << current_ik[2] / omega << std::endl;
                fout.close();
            }
            // write end
        } // end nks
    } // end is
    if (GlobalV::MY_RANK == 0)
    {
        std::string filename = PARAM.globalv.global_out_dir + "current_total.dat";
        std::ofstream fout;
        fout.open(filename, std::ios::app);
        fout << std::setprecision(16);
        fout << std::scientific;
        fout << istep << " " << current_total[0] / omega << " " << current_total[1] / omega << " "
             << current_total[2] / omega << std::endl;
        fout.close();
    }

    ModuleBase::timer::tick("ModuleIO", "write_current");
    return;
}
#endif //__LCAO
