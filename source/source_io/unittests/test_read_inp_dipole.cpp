#include "source_io/module_parameter/read_input.h"

#include "gtest/gtest.h"
#include <functional>
#include <limits>

namespace
{

Input_para dipole_input()
{
    Input_para input;
    input.esolver_type = "tddft";
    input.basis_type = "lcao";
    input.calculation = "md";
    input.mdp.md_nstep = 1;
    input.symmetry = "-1";
    input.td_space_type = {2};
    input.td_dipole_position = {10.0, 11.0, 12.0};
    input.td_dipole_width = {0.5};
    return input;
}

TEST(PeriodicDipoleInput, ValidAndLegacy)
{
    const Input_para dipole = dipole_input();
    EXPECT_NO_FATAL_FAILURE(ModuleIO::check_td_efield_parameters(dipole));
    const Input_para legacy;
    EXPECT_NO_FATAL_FAILURE(ModuleIO::check_td_efield_parameters(legacy));
}

TEST(PeriodicDipoleInput, RejectUnsupportedAndMalformedInputs)
{
    const std::vector<std::function<void(Input_para&)>> changes = {
        [](Input_para& input) { input.td_space_type = {1, 0}; },
        [](Input_para& input) { input.td_space_type = {1}; },
        [](Input_para& input) { input.td_dipole_position.pop_back(); },
        [](Input_para& input) { input.td_dipole_width.clear(); },
        [](Input_para& input) { input.td_dipole_width[0] = 0.0; },
        [](Input_para& input) { input.td_dipole_width[0] = -1.0; },
        [](Input_para& input) { input.td_dipole_position[0] = std::numeric_limits<double>::infinity(); },
        [](Input_para& input) { input.td_stype = 1; },
        [](Input_para& input) { input.td_stype = 2; },
        [](Input_para& input) { input.init_vecpot = true; },
        [](Input_para& input) { input.basis_type = "pw"; },
        [](Input_para& input) { input.mdp.md_nstep = 2; },
        [](Input_para& input) { input.mdp.md_restart = true; },
        [](Input_para& input) { input.cal_stress = true; },
        [](Input_para& input) { input.symmetry = "0"; },
        [](Input_para& input) { input.symmetry = "1"; }
    };
    for (std::size_t i = 0; i < changes.size(); ++i)
    {
        Input_para input = dipole_input();
        changes[i](input);
        EXPECT_EXIT(ModuleIO::check_td_efield_parameters(input), testing::ExitedWithCode(1), "") << "case " << i;
    }
}

} // namespace
