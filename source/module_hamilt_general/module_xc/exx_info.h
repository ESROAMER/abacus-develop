#ifndef EXX_INFO_H
#define EXX_INFO_H

#include "module_ri/conv_coulomb_pot_k.h"
#include "module_ri/ewald_Vq.h"
#include "xc_functional.h"

struct Exx_Info
{
    struct Exx_Info_Global
    {
        bool cal_exx = false;
        bool use_ewald = false;

        Conv_Coulomb_Pot_K::Ccp_Type ccp_type;
        double hybrid_alpha = 0.25;
        double cam_alpha = 0.0;
        double cam_beta = 0.0;
        double hse_omega = 0.11;
        double mixing_beta_for_loop1 = 1.0;

        bool separate_loop = true;
        size_t hybrid_step = 1;
    };
    Exx_Info_Global info_global;

    struct Exx_Info_Lip
    {
        const Conv_Coulomb_Pot_K::Ccp_Type& ccp_type;
        const double& hse_omega;
        double lambda;

        Exx_Info_Lip(const Exx_Info::Exx_Info_Global& info_global)
            : ccp_type(info_global.ccp_type), hse_omega(info_global.hse_omega)
        {
        }
    };
    Exx_Info_Lip info_lip;

    struct Exx_Info_Ewald
    {
        Auxiliary_Func::Kernal_Type ker_type;
        Auxiliary_Func::Fq_type fq_type;
        const bool& use_ewald;

        double ewald_ecut = 150;
        double ewald_qdiv = 2;
        double ewald_qdense = 40;
        double ewald_lambda = 1;
        int ewald_niter = 100;
        double ewald_eps = 1e-6;
        int ewald_arate = 3;

        Exx_Info_Ewald(const Exx_Info::Exx_Info_Global& info_global) : use_ewald(info_global.use_ewald)
        {
        }
    };
    Exx_Info_Ewald info_ewald;

    struct Exx_Info_RI
    {
        const Conv_Coulomb_Pot_K::Ccp_Type& ccp_type;
        const double& hse_omega;
        const double& cam_alpha;
        const double& cam_beta;

        bool real_number = false;

        double pca_threshold = 0;
        std::vector<std::string> files_abfs;
        double C_threshold = 0;
        double V_threshold = 0;
        double dm_threshold = 0;
        double cauchy_threshold = 0;
        double C_grad_threshold = 0;
        double V_grad_threshold = 0;
        double cauchy_force_threshold = 0;
        double cauchy_stress_threshold = 0;
        double ccp_threshold = 0;
        double ccp_rmesh_times = 10;
        double kmesh_times = 4;

        int abfs_Lmax = 0; // tmp

        Exx_Info_RI(const Exx_Info::Exx_Info_Global& info_global)
            : ccp_type(info_global.ccp_type),
              hse_omega(info_global.hse_omega),
              cam_alpha(info_global.cam_alpha),
              cam_beta(info_global.cam_beta)
        {
        }
    };
    Exx_Info_RI info_ri;

    Exx_Info() : info_lip(this->info_global), info_ewald(this->info_global), info_ri(this->info_global)
    {
    }
};

#endif
