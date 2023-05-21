#ifndef CONV_COULOMB_POT_K_H
#define CONV_COULOMB_POT_K_H

#include <vector>
#include <map>
#include <string>

class Conv_Coulomb_Pot_K
{
public:

	enum class Ccp_Type{		//  parameter:
		Ccp,                 // 
		Hf,					//
		Hse,					//  	"hse_omega"
		Cam};				//  	"hse_omega", "cam_alpha", "cam_beta"

	template<typename T> static T cal_orbs_ccp(
		const T &orbs,
		const Ccp_Type &ccp_type,
		const std::map<std::string,double> &parameter,
        const double rmesh_times,
        const int& nks);
	
private:
		
	template< typename T > static double get_rmesh_proportion(
		const T &orbs,
		const double psi_threshold);
		
private:

	static std::vector<double> cal_psi_ccp( const std::vector<double> & psif );
	
	static std::vector<double> cal_psi_hf(const int& nks, const std::vector<double> &psif,
                                          const std::vector<double> &k_radial,
                                          const double omega);

	static std::vector<double> cal_psi_hse( 
		const std::vector<double> & psif,
		const std::vector<double> & k_radial,
		const double omega);

	// cam_alpha: Fraction of Hartree-Fock exchange
    // cam_beta: Fraction of short-range exact exchange
	static std::vector<double> cal_psi_cam( 
		const std::vector<double> & psif,
		const std::vector<double> & k_radial,
		const double omega,
		const double cam_alpha,
		const double cam_beta);
};

#endif