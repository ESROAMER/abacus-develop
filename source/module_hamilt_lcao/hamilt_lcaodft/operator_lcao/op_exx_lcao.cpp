#ifdef __EXX
#include "op_exx_lcao.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_ri/RI_2D_Comm.h"

namespace hamilt
{

template class OperatorEXX<OperatorLCAO<double>>;

template class OperatorEXX<OperatorLCAO<std::complex<double>>>;

template<typename T>
void OperatorEXX<OperatorLCAO<T>>::contributeHR()
{

}

// double and complex<double> are the same temperarily
template<>
void OperatorEXX<OperatorLCAO<double>>::contributeHk(int ik)
{
    // Peize Lin add 2016-12-03
    if(XC_Functional::get_func_type()==4 || XC_Functional::get_func_type()==5)
    {
		const double coeff = (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Cam) ? 1.0 : GlobalC::exx_info.info_global.hybrid_alpha;
		if(GlobalC::exx_info.info_ri.real_number)
        RI_2D_Comm::add_Hexx(
            kv,
            ik,
            coeff,
				*this->LM->Hexxd,
				*this->LM);
		else
            RI_2D_Comm::add_Hexx(
                kv,
                ik,
				coeff,
				*this->LM->Hexxc,
				*this->LM);
    }
}

template<>
void OperatorEXX<OperatorLCAO<std::complex<double>>>::contributeHk(int ik)
{
    // Peize Lin add 2016-12-03
    if(XC_Functional::get_func_type()==4 || XC_Functional::get_func_type()==5)
    {
		const double coeff = (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Cam) ? 1.0 : GlobalC::exx_info.info_global.hybrid_alpha;
		if(GlobalC::exx_info.info_ri.real_number)
            RI_2D_Comm::add_Hexx(
                kv,
                ik,
				coeff,
				*this->LM->Hexxd,
				*this->LM);
		else
            RI_2D_Comm::add_Hexx(
                kv,
                ik,
				coeff,
				*this->LM->Hexxc,
				*this->LM);
    }
}
} // namespace hamilt
#endif