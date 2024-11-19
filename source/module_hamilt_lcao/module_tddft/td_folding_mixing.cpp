#include "td_velocity.h"
#include "module_base/libm/libm.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"

void TD_Velocity::folding_HR_td(const hamilt::HContainer<double>& hR,
                std::complex<double>* hk,
                const ModuleBase::Vector3<double>& kvec_d_in,
                const int ncol,
                const int hk_type)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < hR.size_atom_pairs(); ++i)
    {
        hamilt::AtomPair<double>& tmp = hR.get_atom_pair(i);
        for(int ir = 0;ir < tmp.get_R_size(); ++ir )
        {
            const ModuleBase::Vector3<int> r_index = tmp.get_R_index(ir);

            //new
            //cal tddft phase for mixing gague
            const int iat1 = tmp.get_atom_i();
            const int iat2 = tmp.get_atom_j();
            ModuleBase::Vector3<double> dtau = ucell->cal_dtau(iat1, iat2, r_index);
            const double arg_td = cart_At * dtau * ucell->lat0;
            /*std::cout << "arg_td" << arg_td << std::endl;
            std::cout << "cart_At" << cart_At[0] << " "<< cart_At[1] << " " << cart_At[2] << std::endl;
            std::cout << "dtau" << dtau[0] << " "<< dtau[1] << " " << dtau[2] << std::endl;
            std::cout << "ucell->lat0" << ucell->lat0 << std::endl;
            std::cout << "iat1" << iat1 << " " << "iat2" << iat2 << std::endl;*/

            //new
            // cal k_phase
            // if TK==std::complex<double>, kphase is e^{ikR}
            const ModuleBase::Vector3<double> dR(r_index.x, r_index.y, r_index.z);
            const double arg = (kvec_d_in * dR) * ModuleBase::TWO_PI + arg_td;
            double sinp, cosp;
            ModuleBase::libm::sincos(arg, &sinp, &cosp);
            std::complex<double> kphase = std::complex<double>(cosp, sinp);

            tmp.find_R(r_index);
            tmp.add_to_matrix(hk, ncol, kphase, hk_type);
        }
    }
}
