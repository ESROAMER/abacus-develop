#include "source_base/constants.h"
#include "source_estate/module_pot/pot_periodic_dipole.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_io/module_parameter/input_parameter.h"

#include "gtest/gtest.h"
#include <cmath>
#include <vector>

namespace
{

// A low cutoff leaves precisely +/- (2pi/20,0,0) and G=0.
// This checks the real FFT normalization, sign, translated source and ionic force.
TEST(PotPeriodicDipoleTest, PeriodicSingleMode)
{
    UnitCell cell;
    cell.lat0 = 2.0;
    cell.latvec = ModuleBase::Matrix3(10, 0, 0, 0, 3, 0, 0, 0, 3);
    cell.tpiba = ModuleBase::TWO_PI / cell.lat0;
    cell.tpiba2 = cell.tpiba * cell.tpiba;
    cell.omega = 720.0;
    cell.ntype = 1;
    cell.nat = 1;
    std::vector<Atom> atoms(1);
    atoms[0].na = 1;
    atoms[0].taud.resize(1);
    atoms[0].taud[0].set(0.31, 0.5, 0.5);
    atoms[0].ncpp.zv = 1.0;
    cell.atoms = atoms.data();
    const double g = ModuleBase::TWO_PI / 20.0;
    const double exponent = -0.125 * g * g;
    const double scale = -16.0 * ModuleBase::PI * std::exp(exponent) / (cell.omega * g);
    for (const bool gamma: {false, true})
    {
        ModulePW::PW_Basis grid("cpu", "double");
        grid.initgrids(cell.lat0, cell.latvec, 16, 8, 8);
        grid.initparameters(gamma, 0.18);
        grid.setuptransform();
        grid.collect_local_pw();
        Input_para input;
        input.td_vext = true;
        input.td_dt = 0.02;
        input.td_tstart = 0;
        input.td_tend = 0;
        input.td_ttype = {3, 3};
        input.td_vext_dire = {1, 1};
        input.td_heavi_t0 = {10.0, 10.0};
        input.td_dipole_width = {0.5, 0.5};
        input.td_stype = 3;
        input.td_dipole_moment = {-0.25, 1.25};
        input.td_dipole_position = {2.3, 1.0, 2.0, 22.3, 1.0, 2.0};
        auto manager = elecstate::create_td_field_manager(input);
        manager->prepare_sample(0);
        const auto unit_moment_grid = std::make_shared<elecstate::PeriodicDipoleGrid>(grid, cell, *manager);
        elecstate::PotPeriodicDipole potential(unit_moment_grid, manager);
        std::vector<double> v(grid.nrxx, 0.0);
        potential.cal_fixed_v(v.data());
        for (int ix = 0; ix < grid.nx; ++ix)
        {
            const double phase = g * (20.0 * ix / grid.nx - 2.3);
            const double expected = scale * std::sin(phase);
            const int ir = ix * grid.ny * grid.nz;
            EXPECT_NEAR(v[ir], expected, 1e-12);
        }
        ModuleBase::matrix force(1, 3);
        elecstate::PotPeriodicDipole::compute_force(cell, grid, *manager, force);
        const double ion_phase = g * (6.2 - 2.3);
        const double expected_force = scale * g * std::cos(ion_phase);
        EXPECT_NEAR(force(0, 0), expected_force, 1e-12);
        EXPECT_NEAR(force(0, 1), 0.0, 1e-12);
        EXPECT_NEAR(force(0, 2), 0.0, 1e-12);

        manager->prepare_sample(1);
        elecstate::PotPeriodicDipole next_step(unit_moment_grid, manager);
        std::vector<double> inactive(grid.nrxx, 0.0);
        next_step.cal_fixed_v(inactive.data());
        for (const double value: inactive)
        {
            EXPECT_DOUBLE_EQ(value, 0.0);
        }
    }
    cell.atoms = nullptr;
}

} // namespace
