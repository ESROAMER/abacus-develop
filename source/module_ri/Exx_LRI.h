//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef EXX_LRI_H
#define EXX_LRI_H

#include "LRI_CV.h"
#include "ewald_Vq.h"
#include "module_hamilt_general/module_xc/exx_info.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_base/matrix.h"
#include "module_ri/Mix_DMk_2D.h"
#include <RI/physics/Exx.h>

#include <vector>
#include <array>
#include <map>
#include <deque>
#include <mpi.h>

	class Local_Orbital_Charge;
	class Parallel_Orbitals;
	
    template<typename T, typename Tdata>
	class RPA_LRI;

    template<typename T, typename Tdata>
    class Exx_LRI_Interface;

template<typename Tdata>
class Exx_LRI
{
private:
	using TA = int;
	using Tcell = int;
	static constexpr std::size_t Ndim = 3;
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using TatomR = std::array<double,Ndim>;		// tmp

public:
	Exx_LRI( const Exx_Info::Exx_Info_RI &info_in, const Exx_Info::Exx_Info_Ewald &info_ewald_in) 
			:info(info_in), info_ewald(info_ewald_in){}

	void init(const MPI_Comm &mpi_comm_in, const K_Vectors &kv_in);
	void cal_exx_force();
	void cal_exx_stress();

	std::vector< std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>> Hexxs;
	Tdata Eexx;
	ModuleBase::matrix force_exx;
	ModuleBase::matrix stress_exx;

private:
	const Exx_Info::Exx_Info_RI &info;
	const Exx_Info::Exx_Info_Ewald &info_ewald;
    MPI_Comm mpi_comm;
    const K_Vectors *p_kv;

	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> lcaos;
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_ccp;
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_ccp_sr;

	LRI_CV<Tdata> cv;
	LRI_CV<Tdata> cv_sr;
	RI::Exx<TA,Tcell,Ndim,Tdata> exx_lri;
	Ewald_Vq<Tdata> evq;

	void cal_exx_ions(const ModulePW::PW_Basis_K* wfc_basis);
	void cal_exx_elec(const Parallel_Orbitals &pv);
	void post_process_Hexx( std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> &Hexxs_io ) const;
	Tdata post_process_Eexx( const Tdata &Eexx_in ) const;

    int two_level_step = 0;
    Mix_DMk_2D mix_DMk_2D;
    
    friend class RPA_LRI<double, Tdata>;
    friend class RPA_LRI<std::complex<double>, Tdata>;
    friend class Exx_LRI_Interface<double, Tdata>;
    friend class Exx_LRI_Interface<std::complex<double>, Tdata>;
};

#include "Exx_LRI.hpp"

#endif