#include "esolver_ks_lcao_tddft.h"

#include "module_io/cal_r_overlap_R.h"
#include "module_io/dipole_io.h"
#include "module_io/dm_io.h"
#include "module_io/rho_io.h"
#include "module_io/write_HS.h"
#include "module_io/write_HS_R.h"

//--------------temporary----------------------------
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_base/scalapack_connector.h"
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_lcao/module_tddft/evolve_elec.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/print_info.h"

//-----HSolver ElecState Hamilt--------
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/elecstate_lcao_tddft.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/global_fp.h"
#include "module_hsolver/hsolver_lcao.h"
#include "module_psi/psi.h"

//-----force& stress-------------------
#include "module_hamilt_lcao/hamilt_lcaodft/FORCE_STRESS.h"

//---------------------------------------------------

namespace ModuleESolver
{

ESolver_KS_LCAO_TDDFT::ESolver_KS_LCAO_TDDFT()
{
    classname = "ESolver_KS_LCAO_TDDFT";
    basisname = "LCAO";
}
ESolver_KS_LCAO_TDDFT::~ESolver_KS_LCAO_TDDFT()
{
    // this->orb_con.clear_after_ions(GlobalC::UOT, GlobalC::ORB, GlobalV::deepks_setorb, GlobalC::ucell.infoNL.nproj);
    delete psi_laststep;
    if (Hk_laststep != nullptr)
    {
        for (int ik = 0; ik < kv.nks; ++ik)
        {
            delete Hk_laststep[ik];
        }
        delete Hk_laststep;
    }
}

void ESolver_KS_LCAO_TDDFT::Init(Input& inp, UnitCell& ucell)
{
    ESolver_KS::Init(inp, ucell);

    // Initialize the FFT.
    // this function belongs to cell LOOP

    // output is GlobalC::ppcell.vloc 3D local pseudopotentials
    // without structure factors
    // this function belongs to cell LOOP
    GlobalC::ppcell.init_vloc(GlobalC::ppcell.vloc, pw_rho);

    if (this->pelec == nullptr)
    {
        this->pelec = new elecstate::ElecStateLCAO_TDDFT(&(this->chr),
                                                         &(kv),
                                                         kv.nks,
                                                         &(this->LOC),
                                                         &(this->UHM),
                                                         &(this->LOWF),
                                                         this->pw_rho,
                                                         pw_big);
    }

    //------------------init Basis_lcao----------------------
    // Init Basis should be put outside of Ensolver.
    // * reading the localized orbitals/projectors
    // * construct the interpolation tables.
    this->Init_Basis_lcao(this->orb_con, inp, ucell);
    //------------------init Basis_lcao----------------------

    //------------------init Hamilt_lcao----------------------
    // * allocate H and S matrices according to computational resources
    // * set the 'trace' between local H/S and global H/S
    this->LM.divide_HS_in_frag(GlobalV::GAMMA_ONLY_LOCAL, orb_con.ParaV, kv.nks);
    //------------------init Hamilt_lcao----------------------

#ifdef __EXX
    // PLEASE simplify the Exx_Global interface
    // mohan add 2021-03-25
    // Peize Lin 2016-12-03
    if (GlobalV::CALCULATION == "scf" || GlobalV::CALCULATION == "relax" || GlobalV::CALCULATION == "cell-relax"
        || GlobalV::CALCULATION == "md")
    {
        if (GlobalC::exx_info.info_global.cal_exx)
        {
            /* In the special "two-level" calculation case,
            first scf iteration only calculate the functional without exact exchange.
            but in "nscf" calculation, there is no need of "two-level" method. */
            if (ucell.atoms[0].ncpp.xc_func == "HSE" || ucell.atoms[0].ncpp.xc_func == "PBE0" ||
                ucell.atoms[0].ncpp.xc_func == "LC_PBE" || ucell.atoms[0].ncpp.xc_func == "LC_WPBE" ||
                ucell.atoms[0].ncpp.xc_func == "LRC_WPBEH" || ucell.atoms[0].ncpp.xc_func == "CAM_PBEH")
            {
                XC_Functional::set_xc_type("pbe");
            }
            else if (ucell.atoms[0].ncpp.xc_func == "SCAN0")
            {
                XC_Functional::set_xc_type("scan");
            }

            // GlobalC::exx_lcao.init();
            if (GlobalC::exx_info.info_ri.real_number)
                GlobalC::exx_lri_double.init(MPI_COMM_WORLD, GlobalC::kv);
            else
                GlobalC::exx_lri_complex.init(MPI_COMM_WORLD, GlobalC::kv);
        }
    }
#endif

    // pass Hamilt-pointer to Operator
    this->UHM.genH.LM = this->UHM.LM = &this->LM;
    // pass basis-pointer to EState and Psi
    this->LOC.ParaV = this->LOWF.ParaV = this->LM.ParaV;

    // init Psi, HSolver, ElecState, Hamilt
    if (this->phsol == nullptr)
    {
        this->phsol = new hsolver::HSolverLCAO(this->LOWF.ParaV);
        this->phsol->method = GlobalV::KS_SOLVER;
    }

    // Inititlize the charge density.
    this->pelec->charge->allocate(GlobalV::NSPIN);
    this->pelec->omega = GlobalC::ucell.omega;

    // Initializee the potential.
    this->pelec->pot = new elecstate::Potential(pw_rho,
                                                &GlobalC::ucell,
                                                &(GlobalC::ppcell.vloc),
                                                &(sf),
                                                &(pelec->f_en.etxc),
                                                &(pelec->f_en.vtxc));
    this->pelec_td = dynamic_cast<elecstate::ElecStateLCAO_TDDFT*>(this->pelec);
}

void ESolver_KS_LCAO_TDDFT::hamilt2density(int istep, int iter, double ethr)
{

    pelec->charge->save_rho_before_sum_band();

    if (wf.init_wfc == "file")
    {
        if (istep >= 1)
        {
            module_tddft::Evolve_elec::solve_psi(istep,
                                                 GlobalV::NBANDS,
                                                 GlobalV::NLOCAL,
                                                 this->p_hamilt,
                                                 this->LOWF,
                                                 this->psi,
                                                 this->psi_laststep,
                                                 this->Hk_laststep,
                                                 this->pelec_td->ekb,
                                                 td_htype,
                                                 INPUT.propagator,
                                                 kv.nks);
            this->pelec_td->psiToRho_td(this->psi[0]);
        }
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    else if (istep >= 2)
    {
        module_tddft::Evolve_elec::solve_psi(istep,
                                             GlobalV::NBANDS,
                                             GlobalV::NLOCAL,
                                             this->p_hamilt,
                                             this->LOWF,
                                             this->psi,
                                             this->psi_laststep,
                                             this->Hk_laststep,
                                             this->pelec_td->ekb,
                                             td_htype,
                                             INPUT.propagator,
                                             kv.nks);
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    // using HSolverLCAO::solve()
    else if (this->phsol != nullptr)
    {
        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        if (this->psi != nullptr)
        {
            this->phsol->solve(this->p_hamilt, this->psi[0], this->pelec_td, GlobalV::KS_SOLVER);
        }
        else if (this->psid != nullptr)
        {
            this->phsol->solve(this->p_hamilt, this->psid[0], this->pelec_td, GlobalV::KS_SOLVER);
        }
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO", "HSolver has not been initialed!");
    }

    // print occupation of each band
    if (iter == 1 && istep <= 2)
    {
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << endl;
        GlobalV::ofs_running << "occupation : " << endl;
        GlobalV::ofs_running << "ik  iband     occ " << endl;
        GlobalV::ofs_running << std::setprecision(6);
        GlobalV::ofs_running << std::setiosflags(ios::showpoint);
        for (int ik = 0; ik < kv.nks; ik++)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ib++)
            {
                std::setprecision(6);
                GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      " << this->pelec_td->wg(ik, ib) << endl;
            }
        }
        GlobalV::ofs_running << endl;
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << endl;
    }

    for (int ik = 0; ik < kv.nks; ++ik)
    {
        this->pelec_td->print_band(ik, INPUT.printe, iter);
    }

#ifdef __EXX
    // Peize Lin add 2020.04.04
    if (XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5)
    {
        // add exx
        // Peize Lin add 2016-12-03
        this->pelec->set_exx();

        if (GlobalC::restart.info_load.load_H && GlobalC::restart.info_load.load_H_finish
            && !GlobalC::restart.info_load.restart_exx)
        {
            XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);
            // GlobalC::exx_lcao.cal_exx_elec(this->LOC, this->LOWF.wfc_k_grid);
            if (GlobalC::exx_info.info_ri.real_number)
                GlobalC::exx_lri_double.cal_exx_elec(this->mix_DMk_2D, *this->LOWF.ParaV);
            else
                GlobalC::exx_lri_complex.cal_exx_elec(this->mix_DMk_2D, *this->LOWF.ParaV);
            GlobalC::restart.info_load.restart_exx = true;
        }
    }
    else
    {
        this->pelec->f_en.exx = 0.;
    }
#endif

    // using new charge density.
    this->pelec->cal_energies(1);

    // symmetrize the charge density only for ground state
    if (istep <= 1)
    {
        Symmetry_rho srho;
        for (int is = 0; is < GlobalV::NSPIN; is++)
        {
            srho.begin(is, *(pelec->charge), pw_rho, GlobalC::Pgrid, this->symm);
        }
    }

    // (6) compute magnetization, only for spin==2
    GlobalC::ucell.magnet.compute_magnetization(this->pelec->charge->nrxx,
                                                this->pelec->charge->nxyz,
                                                this->pelec->charge->rho,
                                                pelec->nelec_spin.data());

    // (7) calculate delta energy
    this->pelec->f_en.deband = this->pelec->cal_delta_eband();
}

bool ESolver_KS_LCAO_TDDFT::do_after_converge(const int istep, int &iter)
{
    bool stop = ESolver_KS_LCAO::do_after_converge(istep, iter);

    if (stop)
    {
        if (istep >= 1 && this->conv_elec)
        {
            if (this->psi != nullptr)
            {
                this->psi[0].fix_k(ik);
                this->pelec->print_psi(this->psi[0]);
            }
            else
            {
                this->psid[0].fix_k(ik);
                this->pelec->print_psi(this->psid[0]);
            }
        }
        elecstate::ElecStateLCAO::out_wfc_flag = 0;
    }

    // Calculate new potential according to new Charge Density
    if (!this->conv_elec)
    {
        if (GlobalV::NSPIN == 4)
            GlobalC::ucell.cal_ux();
        this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);
        this->pelec->f_en.descf = this->pelec->cal_delta_escf();
    }
    else
    {
        this->pelec->cal_converged();
    }

    // store wfc and Hk laststep
    if (istep >= 1 && this->conv_elec)
    {
        if (this->psi_laststep == nullptr)
#ifdef __MPI
                this->psi_laststep = new psi::Psi<std::complex<double>>(kv.nks,
                                                                        this->LOWF.ParaV->ncol_bands,
                                                                        this->LOWF.ParaV->nrow,
                                                                        nullptr);
#else
            this->psi_laststep = new psi::Psi<std::complex<double>>(kv.nks, GlobalV::NBANDS, GlobalV::NLOCAL, nullptr);
#endif

            if (td_htype == 1)
            {
                if (this->Hk_laststep == nullptr)
                {
                    this->Hk_laststep = new std::complex<double>*[kv.nks];
                    for (int ik = 0; ik < kv.nks; ++ik)
                    {
                        this->Hk_laststep[ik] = new std::complex<double>[this->LOC.ParaV->nloc];
                        ModuleBase::GlobalFunc::ZEROS(Hk_laststep[ik], this->LOC.ParaV->nloc);
                    }
                }
            }

            for (int ik = 0; ik < kv.nks; ++ik)
            {
                this->psi->fix_k(ik);
                this->psi_laststep->fix_k(ik);
                int size0 = psi->get_nbands() * psi->get_nbasis();
                for (int index = 0; index < size0; ++index)
                    psi_laststep[0].get_pointer()[index] = psi[0].get_pointer()[index];

                // store Hamiltonian
                if (td_htype == 1)
                {
                    this->p_hamilt->updateHk(ik);
                    hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
                    this->p_hamilt->matrix(h_mat, s_mat);
                    BlasConnector::copy(this->LOC.ParaV->nloc, h_mat.p, 1, Hk_laststep[ik], 1);
                }
            }

            // calculate energy density matrix for tddft
            if (istep > 1 && module_tddft::Evolve_elec::td_edm == 0)
                this->cal_edm_tddft();
        }
        if (this->conv_elec)
        {
            GlobalV::ofs_running
                << "------------------------------------------------------------------------------------------------"
                << endl;
            GlobalV::ofs_running << "Eii : " << endl;
            GlobalV::ofs_running << "ik  iband    Eii (eV)" << endl;
            GlobalV::ofs_running << std::setprecision(6);
            GlobalV::ofs_running << std::setiosflags(ios::showpoint);
            for (int ik = 0; ik < kv.nks; ik++)
            {
                for (int ib = 0; ib < GlobalV::NBANDS; ib++)
                {
                    GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      "
                                         << this->pelec_td->ekb(ik, ib) * ModuleBase::Ry_to_eV << endl;
                }
            }
            GlobalV::ofs_running << endl;
            GlobalV::ofs_running
                << "------------------------------------------------------------------------------------------------"
                << endl;
        }
    }
    
    return stop;
}


void ESolver_KS_LCAO_TDDFT::afterscf(const int istep)
{
    for (int is = 0; is < GlobalV::NSPIN; is++)
    {
        if (module_tddft::Evolve_elec::out_dipole == 1)
        {
            std::stringstream ss_dipole;
            ss_dipole << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DIPOLE";
            ModuleIO::write_dipole(pelec->charge->rho_save[is], pelec->charge->rhopw, is, istep, ss_dipole.str());
        }
    }

    ESolver_KS_LCAO::afterscf(istep);
}

// use the original formula (Hamiltonian matrix) to calculate energy density matrix
void ESolver_KS_LCAO_TDDFT::cal_edm_tddft()
{
    this->LOC.edm_k_tddft.resize(kv.nks);
    for (int ik = 0; ik < kv.nks; ++ik)
    {
#ifdef __MPI
        this->LOC.edm_k_tddft[ik].create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        complex<double>* Htmp = new complex<double>[this->LOC.ParaV->nloc];
        complex<double>* Sinv = new complex<double>[this->LOC.ParaV->nloc];
        complex<double>* tmp1 = new complex<double>[this->LOC.ParaV->nloc];
        complex<double>* tmp2 = new complex<double>[this->LOC.ParaV->nloc];
        complex<double>* tmp3 = new complex<double>[this->LOC.ParaV->nloc];
        complex<double>* tmp4 = new complex<double>[this->LOC.ParaV->nloc];
        ModuleBase::GlobalFunc::ZEROS(Htmp, this->LOC.ParaV->nloc);
        ModuleBase::GlobalFunc::ZEROS(Sinv, this->LOC.ParaV->nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp1, this->LOC.ParaV->nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp2, this->LOC.ParaV->nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp3, this->LOC.ParaV->nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp4, this->LOC.ParaV->nloc);
        const int inc = 1;
        int nrow = this->LOC.ParaV->nrow;
        int ncol = this->LOC.ParaV->ncol;
        hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
        p_hamilt->matrix(h_mat, s_mat);
        zcopy_(&this->LOC.ParaV->nloc, h_mat.p, &inc, Htmp, &inc);
        zcopy_(&this->LOC.ParaV->nloc, s_mat.p, &inc, Sinv, &inc);

        int* ipiv = new int[this->LOC.ParaV->nloc];
        int info;
        const int one_int = 1;
        pzgetrf_(&GlobalV::NLOCAL, &GlobalV::NLOCAL, Sinv, &one_int, &one_int, this->LOC.ParaV->desc, ipiv, &info);

        int LWORK = -1, liWORK = -1;
        std::vector<std::complex<double>> WORK(1, 0);
        std::vector<int> iWORK(1, 0);

        pzgetri_(&GlobalV::NLOCAL,
                 Sinv,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 ipiv,
                 WORK.data(),
                 &LWORK,
                 iWORK.data(),
                 &liWORK,
                 &info);

        LWORK = WORK[0].real();
        WORK.resize(LWORK, 0);
        liWORK = iWORK[0];
        iWORK.resize(liWORK, 0);

        pzgetri_(&GlobalV::NLOCAL,
                 Sinv,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 ipiv,
                 WORK.data(),
                 &LWORK,
                 iWORK.data(),
                 &liWORK,
                 &info);

        const char N_char = 'N', T_char = 'T';
        const complex<double> one_float = {1.0, 0.0}, zero_float = {0.0, 0.0};
        const complex<double> half_float = {0.5, 0.0};
        pzgemm_(&N_char,
                &N_char,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &one_float,
                this->LOC.dm_k[ik].c,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Htmp,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp1,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &one_float,
                tmp1,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Sinv,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp2,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &one_float,
                Sinv,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Htmp,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp3,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &GlobalV::NLOCAL,
                &one_float,
                tmp3,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                this->LOC.dm_k[ik].c,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp4,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgeadd_(&N_char,
                 &GlobalV::NLOCAL,
                 &GlobalV::NLOCAL,
                 &half_float,
                 tmp2,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 &half_float,
                 tmp4,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc);

        zcopy_(&this->LOC.ParaV->nloc, tmp4, &inc, this->LOC.edm_k_tddft[ik].c, &inc);

        delete[] Htmp;
        delete[] Sinv;
        delete[] tmp1;
        delete[] tmp2;
        delete[] tmp3;
        delete[] tmp4;
        delete[] ipiv;
#else
        this->LOC.edm_k_tddft[ik].create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        ModuleBase::ComplexMatrix Sinv(GlobalV::NLOCAL, GlobalV::NLOCAL);
        ModuleBase::ComplexMatrix Htmp(GlobalV::NLOCAL, GlobalV::NLOCAL);
        hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
        p_hamilt->matrix(h_mat, s_mat);
        // cout<<"hmat "<<h_mat.p[0]<<endl;
        for (int i = 0; i < GlobalV::NLOCAL; i++)
        {
            for (int j = 0; j < GlobalV::NLOCAL; j++)
            {
                Htmp(i, j) = h_mat.p[i * GlobalV::NLOCAL + j];
                Sinv(i, j) = s_mat.p[i * GlobalV::NLOCAL + j];
            }
        }
        int INFO;

        int LWORK = 3 * GlobalV::NLOCAL - 1; // tmp
        std::complex<double>* WORK = new std::complex<double>[LWORK];
        ModuleBase::GlobalFunc::ZEROS(WORK, LWORK);
        int IPIV[GlobalV::NLOCAL];

        LapackConnector::zgetrf(GlobalV::NLOCAL, GlobalV::NLOCAL, Sinv, GlobalV::NLOCAL, IPIV, &INFO);
        LapackConnector::zgetri(GlobalV::NLOCAL, Sinv, GlobalV::NLOCAL, IPIV, WORK, LWORK, &INFO);

        this->LOC.edm_k_tddft[ik] = 0.5 * (Sinv * Htmp * this->LOC.dm_k[ik] + this->LOC.dm_k[ik] * Htmp * Sinv);
        delete[] WORK;
#endif
    }
    return;
}
} // namespace ModuleESolver