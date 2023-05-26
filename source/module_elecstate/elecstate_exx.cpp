#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_hamilt_lcao/hamilt_lcaodft/global_fp.h"

namespace elecstate
{

#ifdef __EXX
#ifdef __LCAO
/// @brief calculation if converged
/// @date Peize Lin add 2016-12-03
void ElecState::set_exx()
{
    ModuleBase::TITLE("energy", "set_exx");

    auto exx_energy = []() -> double {
        if ("lcao_in_pw" == GlobalV::BASIS_TYPE)
        {
            return GlobalC::exx_lip.get_exx_energy();
        }
        else if ("lcao" == GlobalV::BASIS_TYPE)
        {
            if (GlobalC::exx_info.info_ri.real_number)
                return GlobalC::exx_lri_double.Eexx;
            else
                return std::real(GlobalC::exx_lri_complex.Eexx);
        }
        else
        {
            throw std::invalid_argument(ModuleBase::GlobalFunc::TO_STRING(__FILE__)
                                        + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
        }
    };
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        const double coeff = (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Cam) ? 1.0 : GlobalC::exx_info.info_global.hybrid_alpha;
        this->f_en.exx = coeff * exx_energy();
    }

    return;
}
#endif //__LCAO
#endif //__EXX

}