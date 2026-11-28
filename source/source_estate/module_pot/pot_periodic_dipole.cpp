#include "pot_periodic_dipole.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "td_field_manager.h"

#include <cmath>
#include <complex>

namespace elecstate
{
namespace
{

// Fourier coefficient of the electronic potential (Ry per e*Bohr).
std::complex<double> dipole_coefficient(const PeriodicDipole& source, int axis, const UnitCell& cell,
                                        const ModulePW::PW_Basis& basis, int ig)
{
    const auto& n = basis.gdirect[ig];
    // Suppress G=0 and unpaired Nyquist planes on even meshes.
    if (basis.gg[ig] == 0.0 || 2 * std::abs(n.x) == basis.nx || 2 * std::abs(n.y) == basis.ny
        || 2 * std::abs(n.z) == basis.nz)
    {
        return {};
    }
    const auto g = basis.gcar[ig] * cell.tpiba;
    const double g2 = basis.gg[ig] * cell.tpiba2;
    const double phase = g * source.position;
    const double exponent = -0.5 * source.width * source.width * g2;
    const double scale = 2.0 * ModuleBase::FOUR_PI * g[axis] * std::exp(exponent) / (cell.omega * g2);
    return {scale * std::sin(phase), scale * std::cos(phase)};
}

void add_dipole_force(const PeriodicDipole& source, int axis, double moment, const UnitCell& cell,
                      const ModulePW::PW_Basis& basis, ModuleBase::matrix& force)
{
    const int count = 3 * cell.nat;
    std::vector<double> gradients(count, 0.0);
    const int half_axis = basis.xprime ? 0 : 1;
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        const auto v = dipole_coefficient(source, axis, cell, basis, ig);
        const auto& n = basis.gdirect[ig];
        const auto g = basis.gcar[ig] * cell.tpiba;
        const double weight = basis.gamma_only && n[half_axis] != 0.0 ? 2.0 : 1.0;
        int iat = 0;
        for (int it = 0; it < cell.ntype; ++it)
        {
            const auto& atoms = cell.atoms[it];
            for (int ia = 0; ia < atoms.na; ++ia, ++iat)
            {
                const double phase = ModuleBase::TWO_PI * (n * atoms.taud[ia]);
                // The positive ion feels +Z grad(V_e), in Ry/Bohr.
                const double derivative = -weight * (v.real() * std::sin(phase) + v.imag() * std::cos(phase));
                for (int d = 0; d < 3; ++d)
                {
                    gradients[3 * iat + d] += atoms.ncpp.zv * moment * derivative * g[d];
                }
            }
        }
    }
    if (basis.poolnproc > 1)
    {
        Parallel_Reduce::reduce_pool(gradients.data(), count);
    }
    for (int ia = 0; ia < cell.nat; ++ia)
    {
        for (int d = 0; d < 3; ++d)
        {
            force(ia, d) += gradients[3 * ia + d];
        }
    }
}

} // namespace

PeriodicDipoleGrid::PeriodicDipoleGrid(const ModulePW::PW_Basis& basis, const UnitCell& cell, const TDFieldManager& fields)
{
    const auto& sources = fields.fields();
    dipole_space_.resize(sources.size());
    for (std::size_t i = 0; i < sources.size(); ++i)
    {
        const auto& source = *sources[i].periodic_dipole();
        const int axis = sources[i].direction();
        std::vector<std::complex<double>> coefficients(basis.npw);
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            coefficients[ig] = dipole_coefficient(source, axis, cell, basis, ig);
        }
        dipole_space_[i].resize(basis.nrxx);
        basis.recip2real(coefficients.data(), dipole_space_[i].data());
    }
}

void PeriodicDipoleGrid::add_to(double* potential, const std::vector<double>& moments) const
{
    for (std::size_t i = 0; i < moments.size(); ++i)
    {
        for (std::size_t ir = 0; ir < dipole_space_[i].size(); ++ir)
        {
            potential[ir] += moments[i] * dipole_space_[i][ir];
        }
    }
}

PotPeriodicDipole::PotPeriodicDipole(const std::shared_ptr<const PeriodicDipoleGrid>& grid,
                                     const std::shared_ptr<TDFieldManager>& fields)
    : fields_(fields), grid_(grid)
{
    fixed_mode = true;
}

void PotPeriodicDipole::cal_fixed_v(double* potential)
{
    if (fields_->active())
    {
        const auto& moments = fields_->field_vals_ha();
        grid_->add_to(potential, moments);
    }
}

void PotPeriodicDipole::compute_force(const UnitCell& cell, const ModulePW::PW_Basis& basis,
                                      const TDFieldManager& fields, ModuleBase::matrix& force)
{
    force.zero_out();
    const auto& sources = fields.fields();
    const auto& moments = fields.field_vals_ha();
    for (std::size_t i = 0; i < sources.size(); ++i)
    {
        if (moments[i] != 0.0)
        {
            const auto& source = *sources[i].periodic_dipole();
            const int axis = sources[i].direction();
            add_dipole_force(source, axis, moments[i], cell, basis, force);
        }
    }
}

} // namespace elecstate
