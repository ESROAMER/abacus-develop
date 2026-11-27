#include "h_tddft_pw.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
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

H_TDDFT_pw::H_TDDFT_pw(const ModulePW::PW_Basis* rho_basis_in,
                       const UnitCell* ucell_in,
                       const std::shared_ptr<TDFieldManager>& field_manager)
    : ucell_(ucell_in), field_manager_(field_manager)
{
    this->dynamic_mode = false;
    this->fixed_mode = true;
    this->rho_basis_ = rho_basis_in;

    if (!field_manager_)
    {
        ModuleBase::WARNING_QUIT("H_TDDFT_pw", "RT-TDDFT field manager is not initialized.");
    }
    const auto& fields = field_manager_->fields();
    dipole_space_.resize(fields.size());
    for (std::size_t i = 0; i < fields.size(); ++i)
    {
        const PeriodicDipole* source = fields[i].periodic_dipole();
        if (source)
        {
            const int axis = fields[i].direction();
            std::vector<std::complex<double>> coefficients(rho_basis_->npw);
            for (int ig = 0; ig < rho_basis_->npw; ++ig)
            {
                coefficients[ig] = dipole_coefficient(*source, axis, *ucell_, *rho_basis_, ig);
            }
            dipole_space_[i].resize(rho_basis_->nrxx);
            rho_basis_->recip2real(coefficients.data(), dipole_space_[i].data());
        }
    }
}

void H_TDDFT_pw::cal_fixed_v(double* vl_pseudo)
{
    ModuleBase::TITLE("H_TDDFT_pw", "cal_fixed_v");
    if (field_manager_->gauge() != 0)
    {
        return;
    }

    if (!field_manager_->active())
    {
        return;
    }

    ModuleBase::timer::start("H_TDDFT_pw", "cal_fixed_v");
    const std::vector<TDField>& fields = field_manager_->fields();
    const std::vector<double>& field_values = field_manager_->field_vals_ha();
    for (std::size_t field_index = 0; field_index < fields.size(); ++field_index)
    {
        if (fields[field_index].periodic_dipole())
        {
            const double moment = field_values[field_index];
            for (int ir = 0; ir < rho_basis_->nrxx; ++ir)
            {
                vl_pseudo[ir] += moment * dipole_space_[field_index][ir];
            }
            continue;
        }
        std::vector<double> vext_space(this->rho_basis_->nrxx, 0.0);
        const double field_value = 2.0 * field_values[field_index]; // Scalar potential is stored in Rydberg.

        cal_v_space_length(vext_space, fields[field_index].direction() + 1);
        for (std::size_t ir = 0; ir < static_cast<std::size_t>(this->rho_basis_->nrxx); ++ir)
        {
            vl_pseudo[ir] += vext_space[ir] * field_value;
        }
    }
    ModuleBase::timer::end("H_TDDFT_pw", "cal_fixed_v");
}

void H_TDDFT_pw::cal_v_space_length(std::vector<double>& vext_space, const int direction)
{
    ModuleBase::TITLE("H_TDDFT_pw", "cal_v_space_length");
    ModuleBase::timer::start("H_TDDFT_pw", "cal_v_space_length");

    for (int ir = 0; ir < this->rho_basis_->nrxx; ++ir)
    {
        const int i = ir / (this->rho_basis_->ny * this->rho_basis_->nplane);
        const int j = ir / this->rho_basis_->nplane - i * this->rho_basis_->ny;
        const int k = ir % this->rho_basis_->nplane + this->rho_basis_->startz_current;
        const double x = static_cast<double>(i) / this->rho_basis_->nx;
        const double y = static_cast<double>(j) / this->rho_basis_->ny;
        const double z = static_cast<double>(k) / this->rho_basis_->nz;

        if (direction == 1)
        {
            vext_space[ir] = cal_v_space_length_potential(x) * this->ucell_->latvec.e11
                             + cal_v_space_length_potential(y) * this->ucell_->latvec.e21
                             + cal_v_space_length_potential(z) * this->ucell_->latvec.e31;
        }
        else if (direction == 2)
        {
            vext_space[ir] = cal_v_space_length_potential(x) * this->ucell_->latvec.e12
                             + cal_v_space_length_potential(y) * this->ucell_->latvec.e22
                             + cal_v_space_length_potential(z) * this->ucell_->latvec.e32;
        }
        else
        {
            vext_space[ir] = cal_v_space_length_potential(x) * this->ucell_->latvec.e13
                             + cal_v_space_length_potential(y) * this->ucell_->latvec.e23
                             + cal_v_space_length_potential(z) * this->ucell_->latvec.e33;
        }
    }

    ModuleBase::timer::end("H_TDDFT_pw", "cal_v_space_length");
}

double H_TDDFT_pw::cal_v_space_length_potential(const double coordinate) const
{
    const double lower_cut = field_manager_->length_cut1();
    const double upper_cut = field_manager_->length_cut2();
    if (coordinate < lower_cut)
    {
        return -((coordinate - lower_cut) * (upper_cut - lower_cut) / (lower_cut + 1.0 - upper_cut) - lower_cut) * this->ucell_->lat0;
    }
    if (coordinate < upper_cut)
    {
        return coordinate * this->ucell_->lat0;
    }
    return -((coordinate - upper_cut) * (upper_cut - lower_cut) / (lower_cut + 1.0 - upper_cut) - upper_cut) * this->ucell_->lat0;
}

void H_TDDFT_pw::compute_force(const UnitCell& cell, const ModulePW::PW_Basis& basis,
                               const TDFieldManager& manager, ModuleBase::matrix& force)
{
    // Velocity gauge has no scalar electric-field force. Forces use Rydberg units.
    const double ry_scale = manager.gauge() == 1 ? 0.0 : 2.0;
    const ModuleBase::Vector3<double> field = manager.efield_ha() * ry_scale;
    int atom_index = 0;
    for (int type = 0; type < cell.ntype; ++type)
    {
        for (int atom = 0; atom < cell.atoms[type].na; ++atom)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                force(atom_index, direction) = field[direction] * cell.atoms[type].ncpp.zv;
            }
            ++atom_index;
        }
    }
    const auto& fields = manager.fields();
    const auto& amplitudes = manager.field_vals_ha();
    for (std::size_t i = 0; i < fields.size(); ++i)
    {
        const PeriodicDipole* source = fields[i].periodic_dipole();
        if (source && amplitudes[i] != 0.0)
        {
            const int axis = fields[i].direction();
            add_dipole_force(*source, axis, amplitudes[i], cell, basis, force);
        }
    }
}

} // namespace elecstate
