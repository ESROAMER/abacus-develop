//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef EXX_LRI_H
#define EXX_LRI_H

#include <RI/physics/Exx.h>


#include "LRI_CV.h"
#include "ewald_Vq.h"
#include "module_hamilt_general/module_xc/exx_info.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_base/matrix.h"

#include <vector>
#include <array>
#include <map>
#include <deque>
#include <mpi.h>

	class Parallel_Orbitals;
	
	template<typename T, typename Tdata>
	class RPA_LRI;

	template<typename T, typename Tdata>
	class Exx_LRI_Interface;

template <typename Tdata>
class Exx_LRI
{
  private:
    using TA = int;
    using Tcell = int;
    static constexpr std::size_t Ndim = 3;
    using TC = std::array<Tcell, Ndim>;
    using TAC = std::pair<TA, TC>;
    using TatomR = std::array<double, Ndim>; // tmp

  public:
    Exx_LRI(const Exx_Info::Exx_Info_RI& info_in, const Exx_Info::Exx_Info_Ewald& info_ewald_in)
        : info(info_in), info_ewald(info_ewald_in), evq(info, info_ewald){};
    
    void init(const MPI_Comm& mpi_comm_in, const K_Vectors& kv_in);
    void cal_exx_force();
    void cal_exx_stress();

	std::vector< std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>> Hexxs;
    double Eexx;
	ModuleBase::matrix force_exx;
	ModuleBase::matrix stress_exx;

  private:
    const Exx_Info::Exx_Info_RI& info;
    const Exx_Info::Exx_Info_Ewald& info_ewald;
	MPI_Comm mpi_comm;
	const K_Vectors* p_kv = nullptr;

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> lcaos;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_ccp;
    LRI_CV<Tdata> cv;
    RI::Exx<TA, Tcell, Ndim, Tdata> exx_lri;
    Ewald_Vq<Tdata> evq;

	void cal_exx_ions();
	void cal_exx_elec(const std::vector<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>> &Ds, const Parallel_Orbitals &pv);
	void post_process_Hexx( std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> &Hexxs_io ) const;
    double post_process_Eexx(const double& Eexx_in) const;

	friend class RPA_LRI<double, Tdata>;
	friend class RPA_LRI<std::complex<double>, Tdata>;
	friend class Exx_LRI_Interface<double, Tdata>;
	friend class Exx_LRI_Interface<std::complex<double>, Tdata>;
};

#include "Exx_LRI.hpp"

#endif