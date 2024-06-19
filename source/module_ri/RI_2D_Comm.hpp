//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef RI_2D_COMM_HPP
#define RI_2D_COMM_HPP

#include <Comm/Comm_Assemble/Comm_Assemble.h>
#include <Comm/example/Communicate_Map-1.h>
#include <Comm/example/Communicate_Map-2.h>
#include <RI/comm/example/Communicate_Map_Period.h>
#include <RI/comm/mix/Communicate_Tensors_Map.h>
#include <RI/global/Global_Func-2.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "RI_2D_Comm.h"
#include "RI_Util.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

// inline RI::Tensor<double> tensor_conj(const RI::Tensor<double>& t) { return t; }
// inline RI::Tensor<std::complex<double>> tensor_conj(const RI::Tensor<std::complex<double>>& t)
// {
//     RI::Tensor<std::complex<double>> r(t.shape);
//     for (int i = 0;i < t.data->size();++i)(*r.data)[i] = std::conj((*t.data)[i]);
//     return r;
// }
// inline RI::Tensor<double> tensor_real(const RI::Tensor<double>& t) { return t; }
// inline RI::Tensor<std::complex<double>> tensor_real(const RI::Tensor<std::complex<double>>& t)
// {
// 	RI::Tensor<std::complex<double>> r(t.shape);
// 	for (int i = 0;i < t.data->size();++i)(*r.data)[i] = ((*t.data)[i]).real();
// 	return r;
// }

template <typename Tdata, typename Tmatrix>
auto RI_2D_Comm::split_m2D_ktoR(const K_Vectors& kv,
                                const std::vector<const Tmatrix*>& mks_2D,
                                const Parallel_Orbitals& pv)
    -> std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>
{
    ModuleBase::TITLE("RI_2D_Comm", "split_m2D_ktoR");
    ModuleBase::timer::tick("RI_2D_Comm", "split_m2D_ktoR");

    const TC period = RI_Util::get_Born_vonKarmen_period(kv);
    const std::map<int, int> nspin_k = {
        {1, 1},
        {2, 2},
        {4, 1}
    };
    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN); // why?

    std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>> mRs_a2D(GlobalV::NSPIN);
    for (int is_k = 0; is_k < nspin_k.at(GlobalV::NSPIN); ++is_k)
    {
        const std::vector<int> ik_list = RI_2D_Comm::get_ik_list(kv, is_k);
        for (const TC& cell: RI_Util::get_Born_von_Karmen_cells(period))
        {
            RI::Tensor<Tdata> mR_2D;
            for (const int ik: ik_list)
            {
                using Tdata_m = typename Tmatrix::value_type;
                RI::Tensor<Tdata_m> mk_2D
                    = RI_Util::Vector_to_Tensor<Tdata_m>(*mks_2D[ik], pv.get_col_size(), pv.get_row_size());
                const Tdata_m frac
                    = SPIN_multiple
                      * RI::Global_Func::convert<Tdata_m>(
                          std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                                   * (kv.kvec_c[ik] * (RI_Util::array3_to_Vector3(cell) * GlobalC::ucell.latvec))));
                auto set_mR_2D = [&mR_2D](auto&& mk_frac) {
                    if (mR_2D.empty())
                        mR_2D = RI::Global_Func::convert<Tdata>(mk_frac);
                    else
                        mR_2D = mR_2D + RI::Global_Func::convert<Tdata>(mk_frac);
                };
                // if (static_cast<int>(std::round(SPIN_multiple * kv.wk[ik] * kv.nkstot_full)) == 2)
                //     set_mR_2D(mk_2D * (frac * 0.5) + tensor_conj(mk_2D * (frac * 0.5)));
                if (static_cast<int>(std::round(SPIN_multiple * kv.wk[ik] * kv.nkstot_full)) == 2)
                    set_mR_2D(tensor_real(mk_2D * frac));
                else
                    set_mR_2D(mk_2D * frac);
            }

            for (int iwt0_2D = 0; iwt0_2D != mR_2D.shape[0]; ++iwt0_2D)
            {
                const int iwt0 = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? pv.local2global_col(iwt0_2D)
                                                                                     : pv.local2global_row(iwt0_2D);
                int iat0, iw0_b, is0_b;
                std::tie(iat0, iw0_b, is0_b) = RI_2D_Comm::get_iat_iw_is_block(iwt0);
                const int it0 = GlobalC::ucell.iat2it[iat0];
                for (int iwt1_2D = 0; iwt1_2D != mR_2D.shape[1]; ++iwt1_2D)
                {
                    const int iwt1 = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? pv.local2global_row(iwt1_2D)
                                                                                         : pv.local2global_col(iwt1_2D);
                    int iat1, iw1_b, is1_b;
                    std::tie(iat1, iw1_b, is1_b) = RI_2D_Comm::get_iat_iw_is_block(iwt1);
                    const int it1 = GlobalC::ucell.iat2it[iat1];

                    const int is_b = RI_2D_Comm::get_is_block(is_k, is0_b, is1_b);
                    RI::Tensor<Tdata>& mR_a2D = mRs_a2D[is_b][iat0][{iat1, cell}];
                    if (mR_a2D.empty())
                        mR_a2D = RI::Tensor<Tdata>({static_cast<size_t>(GlobalC::ucell.atoms[it0].nw),
                                                    static_cast<size_t>(GlobalC::ucell.atoms[it1].nw)});
                    mR_a2D(iw0_b, iw1_b) = mR_2D(iwt0_2D, iwt1_2D);
                }
            }
        }
    }
    ModuleBase::timer::tick("RI_2D_Comm", "split_m2D_ktoR");
    return mRs_a2D;
}

template <typename Tdata>
void RI_2D_Comm::add_Hexx(const K_Vectors& kv,
                          const int ik,
                          const double alpha,
                          const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& Hs,
                          LCAO_Matrix& lm)
{
    ModuleBase::TITLE("RI_2D_Comm", "add_Hexx");
    ModuleBase::timer::tick("RI_2D_Comm", "add_Hexx");

    const Parallel_Orbitals& pv = *lm.ParaV;

    const std::map<int, std::vector<int>> is_list = {
        {1, {0}         },
        {2, {kv.isk[ik]}},
        {4, {0, 1, 2, 3}}
    };
    for (const int is_b: is_list.at(GlobalV::NSPIN))
    {
        int is0_b, is1_b;
        std::tie(is0_b, is1_b) = RI_2D_Comm::split_is_block(is_b);
        for (const auto& Hs_tmpA: Hs[is_b])
        {
            const TA& iat0 = Hs_tmpA.first;
            for (const auto& Hs_tmpB: Hs_tmpA.second)
            {
                const TA& iat1 = Hs_tmpB.first.first;
                const TC& cell1 = Hs_tmpB.first.second;
                const std::complex<double> frac
                    = alpha
                      * std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                                 * (kv.kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)));
                const RI::Tensor<Tdata>& H = Hs_tmpB.second;
                for (size_t iw0_b = 0; iw0_b < H.shape[0]; ++iw0_b)
                {
                    const int iwt0 = RI_2D_Comm::get_iwt(iat0, iw0_b, is0_b);
                    if (pv.global2local_row(iwt0) < 0)
                        continue;
                    for (size_t iw1_b = 0; iw1_b < H.shape[1]; ++iw1_b)
                    {
                        const int iwt1 = RI_2D_Comm::get_iwt(iat1, iw1_b, is1_b);
                        if (pv.global2local_col(iwt1) < 0)
                            continue;

                        if (GlobalV::GAMMA_ONLY_LOCAL)
                            lm.set_HSgamma(iwt0,
                                           iwt1,
                                           RI::Global_Func::convert<double>(H(iw0_b, iw1_b))
                                               * RI::Global_Func::convert<double>(frac),
                                           lm.Hloc.data());
                        else
                            lm.set_HSk(iwt0,
                                       iwt1,
                                       RI::Global_Func::convert<std::complex<double>>(H(iw0_b, iw1_b)) * frac,
                                       'L',
                                       -1);
                    }
                }
            }
        }
    }
    ModuleBase::timer::tick("RI_2D_Comm", "add_Hexx");
}

std::tuple<int, int, int> RI_2D_Comm::get_iat_iw_is_block(const int iwt)
{
    const int iat = GlobalC::ucell.iwt2iat[iwt];
    const int iw = GlobalC::ucell.iwt2iw[iwt];
    switch (GlobalV::NSPIN)
    {
    case 1:
    case 2:
        return std::make_tuple(iat, iw, 0);
    case 4:
        return std::make_tuple(iat, iw / 2, iw % 2);
    default:
        throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
}

int RI_2D_Comm::get_is_block(const int is_k, const int is_row_b, const int is_col_b)
{
    switch (GlobalV::NSPIN)
    {
    case 1:
        return 0;
    case 2:
        return is_k;
    case 4:
        return is_row_b * 2 + is_col_b;
    default:
        throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
}

std::tuple<int, int> RI_2D_Comm::split_is_block(const int is_b)
{
    switch (GlobalV::NSPIN)
    {
    case 1:
    case 2:
        return std::make_tuple(0, 0);
    case 4:
        return std::make_tuple(is_b / 2, is_b % 2);
    default:
        throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
}

int RI_2D_Comm::get_iwt(const int iat, const int iw_b, const int is_b)
{
    const int it = GlobalC::ucell.iat2it[iat];
    const int ia = GlobalC::ucell.iat2ia[iat];
    int iw = -1;
    switch (GlobalV::NSPIN)
    {
    case 1:
    case 2:
        iw = iw_b;
        break;
    case 4:
        iw = iw_b * 2 + is_b;
        break;
    default:
        throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
    const int iwt = GlobalC::ucell.itiaiw2iwt(it, ia, iw);
    return iwt;
}

template <typename TA, typename TAC, typename T>
std::map<TA, std::map<TAC, T>> RI_2D_Comm::comm_map2_first(const MPI_Comm& mpi_comm,
                                                           const std::map<TA, std::map<TAC, T>>& Ds_in,
                                                           const std::set<TA>& s0,
                                                           const std::set<TA>& s1)
{
    RI::Communicate_Map_Period::Judge_Map2_First<TA> judge;
    judge.s0 = s0;
    judge.s1 = s1;
    return comm_map2(mpi_comm, Ds_in, judge);
}

template <typename TA, typename TAC, typename T, typename Tjudge>
std::map<TA, std::map<TAC, T>> RI_2D_Comm::comm_map2(const MPI_Comm& mpi_comm,
                                                     const std::map<TA, std::map<TAC, T>>& Ds_in,
                                                     const Tjudge& judge)
{
    Comm::Comm_Assemble<std::tuple<TA, TAC>, T, std::map<TA, std::map<TAC, T>>, Tjudge, std::map<TA, std::map<TAC, T>>>
        com(mpi_comm);

    com.traverse_keys_provide = Comm::Communicate_Map::traverse_keys<TA, TAC, T>;
    com.get_value_provide = Comm::Communicate_Map::get_value<TA, TAC, T>;
    com.set_value_require = set_value_add<TA, TAC, T>;
    com.flag_lock_set_value = Comm::Comm_Tools::Lock_Type::Copy_merge;
    com.init_datas_local = Comm::Communicate_Map::init_datas_local<TA, TAC, T>;
    com.add_datas = add_datas<TA, TAC, T>;

    std::map<TA, std::map<TAC, T>> Ds_out;
    com.communicate(Ds_in, judge, Ds_out);
    return Ds_out;
}

template <typename Tkey, typename Tvalue>
void RI_2D_Comm::set_value_add(Tkey&& key, Tvalue&& value, std::map<Tkey, Tvalue>& data)
{
    using namespace RI::Array_Operator;
    auto ptr = data.find(key);
    if (ptr == data.end())
        data[key] = std::move(value);
    else
        ptr->second = ptr->second + std::move(value);
}

template <typename Tkey0, typename Tkey1, typename Tvalue>
void RI_2D_Comm::set_value_add(std::tuple<Tkey0, Tkey1>&& key,
                               Tvalue&& value,
                               std::map<Tkey0, std::map<Tkey1, Tvalue>>& data)
{
    set_value_add(std::move(std::get<1>(key)), std::move(value), data[std::get<0>(key)]);
}

template <typename Tkey, typename Tvalue>
void RI_2D_Comm::add_datas(std::map<Tkey, Tvalue>&& data_local, std::map<Tkey, Tvalue>& data_recv)
{
    using namespace RI::Array_Operator;
    auto ptr_local = data_local.begin();
    auto ptr_recv = data_recv.begin();
    for (; ptr_local != data_local.end() && ptr_recv != data_recv.end();)
    {
        const Tkey& key_local = ptr_local->first;
        const Tkey& key_recv = ptr_recv->first;
        if (key_local == key_recv)
        {
            ptr_recv->second = ptr_recv->second + std::move(ptr_local->second);
            ++ptr_local;
            ++ptr_recv;
        }
        else if (key_local < key_recv)
        {
            ptr_recv = data_recv.emplace_hint(ptr_recv, key_local, std::move(ptr_local->second));
            ++ptr_local;
        }
        else
        {
            ++ptr_recv;
        }
    }
    for (; ptr_local != data_local.end(); ++ptr_local)
    {
        ptr_recv = data_recv.emplace_hint(ptr_recv, ptr_local->first, std::move(ptr_local->second));
    }
}

template <typename Tkey0, typename Tkey1, typename Tvalue>
void RI_2D_Comm::add_datas(std::map<Tkey0, std::map<Tkey1, Tvalue>>&& data_local,
                           std::map<Tkey0, std::map<Tkey1, Tvalue>>& data_recv)
{
    auto ptr_local = data_local.begin();
    auto ptr_recv = data_recv.begin();
    for (; ptr_local != data_local.end() && ptr_recv != data_recv.end();)
    {
        const Tkey0& key_local = ptr_local->first;
        const Tkey0& key_recv = ptr_recv->first;
        if (key_local == key_recv)
        {
            add_datas(std::move(ptr_local->second), ptr_recv->second);
            ++ptr_local;
            ++ptr_recv;
        }
        else if (key_local < key_recv)
        {
            ptr_recv = data_recv.emplace_hint(ptr_recv, key_local, std::move(ptr_local->second));
            ++ptr_local;
        }
        else
        {
            ++ptr_recv;
        }
    }
    for (; ptr_local != data_local.end(); ++ptr_local)
    {
        ptr_recv = data_recv.emplace_hint(ptr_recv, ptr_local->first, std::move(ptr_local->second));
    }
}

#endif