#include "td_field.h"

#include <utility>

namespace elecstate
{

double TDFieldSample::step_position() const
{
    return electronic_step + static_cast<double>(simpson_node) / subdivisions;
}

TDField::TDField(const int direction, std::unique_ptr<TDFieldProfile> profile, const int subdivisions)
    : direction_(direction), profile_(std::move(profile)), subdivisions_(subdivisions)
{
}

TDField::TDField(const int direction, std::unique_ptr<TDFieldProfile> profile, const int subdivisions,
                 std::unique_ptr<PeriodicDipole> dipole)
    : TDField(direction, std::move(profile), subdivisions)
{
    dipole_ = std::move(dipole);
}

const PeriodicDipole* TDField::periodic_dipole() const
{
    return dipole_.get();
}

int TDField::direction() const
{
    return direction_;
}

int TDField::subdivisions() const
{
    return subdivisions_;
}

double TDField::electric_field(const TDFieldSample& sample) const
{
    return profile_->electric_field(sample);
}

} // namespace elecstate
