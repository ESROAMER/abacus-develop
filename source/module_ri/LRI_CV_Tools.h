//=======================
// AUTHOR : Peize Lin
// DATE :   2022-10-24
//=======================

#ifndef LRI_CV_TOOLS_H
#define LRI_CV_TOOLS_H

#include <RI/global/Tensor.h>

#include <array>
#include <cstddef>
#include <vector>

#include "abfs.h"

namespace LRI_CV_Tools
{
template <typename Tdata>
extern RI::Tensor<Tdata> cal_I(const RI::Tensor<Tdata>& m);
template <typename Tdata>
extern std::vector<std::vector<RI::Tensor<Tdata>>> cal_I(const std::vector<std::vector<RI::Tensor<Tdata>>>& ms);

template <typename Tdata>
inline RI::Tensor<Tdata> transform_Rm(const RI::Tensor<Tdata>& V);
template <typename Tdata>
inline std::array<RI::Tensor<Tdata>, 3> transform_Rm(const std::array<RI::Tensor<Tdata>, 3>& dV);

// template<typename T> inline bool exist(const T &V);

// template<typename T1, typename T2, typename Treturn>
// extern Treturn mul1(const T1 &t1, const T2 &t2);
// template<typename T1, typename T2, typename Treturn>
// extern Treturn mul2(const T1 &mat, const T2 &vec);

template <typename Tdata>
inline bool exist(const RI::Tensor<Tdata>& V);
template <typename T, std::size_t N>
inline bool exist(const std::array<T, N>& dV);

template <typename Tdata>
extern RI::Tensor<Tdata> mul1(const RI::Tensor<Tdata>& t1, const RI::Tensor<Tdata>& t2);
template <typename T>
extern std::array<T, 3> mul1(const std::array<T, 3>& t1, const T& t2);

template <typename Tdata>
extern std::vector<RI::Tensor<Tdata>> mul2(const std::vector<std::vector<RI::Tensor<Tdata>>>& mat,
                                           const std::vector<RI::Tensor<Tdata>>& vec);
template <typename T1, typename T2>
extern std::array<T2, 3> mul2(const T1& t1, const std::array<T2, 3>& t2);

// template<typename T, std::size_t N>
// std::array<T,N> operator-(const std::array<T,N> &v1, const std::array<T,N> &v2);
// template<typename T>
// std::vector<T> operator-(const std::vector<T> &v1, const std::vector<T> &v2);
template <typename T, std::size_t N>
extern std::vector<std::array<T, N>> minus(const std::vector<std::array<T, N>>& v1,
                                           const std::vector<std::array<T, N>>& v2);

template <typename T, std::size_t N>
extern std::array<T, N> negative(const std::array<T, N>& v_in);

// template<typename T> T transpose12(const T &c_in);
template <typename Tdata>
RI::Tensor<Tdata> transpose12(const RI::Tensor<Tdata>& c_in);
template <typename T, std::size_t N>
std::array<T, N> transpose12(const std::array<T, N>& c_in);

template <typename T, std::size_t N>
extern std::array<std::vector<T>, N> change_order(std::vector<std::array<T, N>>&& ds_in);
template <typename T, std::size_t N>
std::vector<std::array<T, N>> change_order(std::array<std::vector<T>, N>&& ds_in);
template <typename T, std::size_t N>
extern std::array<std::vector<std::vector<T>>, N> change_order(std::vector<std::vector<std::array<T, N>>>&& ds_in);
template <typename TkeyA, typename TkeyB, typename Tvalue, std::size_t N>
extern std::array<std::map<TkeyA, std::map<TkeyB, Tvalue>>, N> change_order(
    std::map<TkeyA, std::map<TkeyB, std::array<Tvalue, N>>>&& ds_in);

template <typename Tcell>
extern std::array<Tcell, 3> cal_latvec_range(const double& rcut_times);

template <typename TA, typename Tcell, typename Tdata>
extern std::map<int, std::map<int, std::map<Abfs::Vector3_Order<double>, RI::Tensor<Tdata>>>> get_CVws(
    const std::map<TA, std::map<std::pair<TA, std::array<Tcell, 3>>, RI::Tensor<Tdata>>>& CVs);
template <typename TA, typename Tcell, typename Tdata>
extern std::map<int, std::map<int, std::map<Abfs::Vector3_Order<double>, std::array<RI::Tensor<Tdata>, 3>>>> get_dCVws(
    const std::map<TA, std::map<std::pair<TA, std::array<Tcell, 3>>, std::array<RI::Tensor<Tdata>, 3>>>& dCVs);

template <typename Tdata>
inline void init_elem(RI::Tensor<Tdata>& data, const size_t ndim0, const size_t ndim1);
template <typename T, std::size_t N>
inline void init_elem(std::array<RI::Tensor<T>, N>& data, const size_t ndim0, const size_t ndim1);

template <typename Tdata>
inline void add_elem(Tdata& data, Tdata& val, Tdata& frac);
template <typename T, std::size_t N>
extern void add_elem(std::array<T, N>& data, T& val, T& frac);
template <typename Tdata>
inline void add_elem(RI::Tensor<Tdata>& data, const int lmp, const int lmq, Tdata& val, Tdata& frac);
template <typename T, std::size_t N>
extern void add_elem(std::array<RI::Tensor<T>, N>& data, const int lmp, const int lmq, std::array<T, N>& val, T& frac);
template <typename T, std::size_t N>
extern void add_elem(std::array<RI::Tensor<T>, N>& data,
                     const int lmp0,
                     const int lmq0,
                     std::array<RI::Tensor<T>, N>& val,
                     const int lmp1,
                     const int lmq1,
                     T& frac);

template <typename Tout, typename Tin>
inline RI::Tensor<Tout> convert(RI::Tensor<Tin>& data);
template <typename Tout, typename Tin, std::size_t N>
extern std::array<RI::Tensor<Tout>, N> convert(std::array<RI::Tensor<Tin>, N>& data);
} // namespace LRI_CV_Tools

#include "LRI_CV_Tools.hpp"

#endif