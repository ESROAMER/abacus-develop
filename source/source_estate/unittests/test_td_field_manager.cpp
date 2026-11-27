#include "source_base/constants.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_io/module_parameter/input_parameter.h"

#include "gtest/gtest.h"

TEST(TDFieldManager, PeriodicMomentsUseAllFiveWaveforms)
{
    Input_para input;
    input.td_vext = true;
    input.td_dt = 0.01;
    input.td_tstart = 0;
    input.td_tend = 4;
    input.td_ttype = {0, 1, 2, 3, 4};
    input.td_vext_dire = {1, 2, 3, 1, 2};
    input.td_gauss_t0 = {2.0};
    input.td_trape_t1 = {1.0};
    input.td_trigo_phase2 = {0.8};
    input.td_heavi_t0 = {3.0};
    input.td_supsine_tstart = {0};
    input.td_supsine_tend = {4};
    const auto uniform = elecstate::create_td_field_manager(input);
    input.td_stype = 3;
    input.td_dipole_position = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    input.td_dipole_width = {0.5, 0.6, 0.7, 0.8, 0.9};
    const auto periodic = elecstate::create_td_field_manager(input);
    for (int step = 0; step <= 2; ++step)
    {
        uniform->prepare_sample(step);
        periodic->prepare_sample(step);
    }
    const double field_unit = ModuleBase::BOHR_TO_A / ModuleBase::Hartree_to_eV;
    for (std::size_t i = 0; i < input.td_ttype.size(); ++i)
    {
        const auto* source = periodic->fields()[i].periodic_dipole();
        ASSERT_NE(source, nullptr);
        EXPECT_DOUBLE_EQ(source->width, input.td_dipole_width[i]);
        EXPECT_DOUBLE_EQ(source->position.z, input.td_dipole_position[3 * i + 2]);
        const double moment = periodic->field_vals_ha()[i];
        EXPECT_NE(moment, 0.0);
        EXPECT_NEAR(moment * field_unit, uniform->field_vals_ha()[i], 1e-14);
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_DOUBLE_EQ(periodic->efield_ha()[axis], 0.0);
    }
}
