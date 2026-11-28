#ifndef POT_PERIODIC_DIPOLE_H
#define POT_PERIODIC_DIPOLE_H

#include "pot_base.h"
#include <memory>
#include <vector>

namespace elecstate
{
class TDFieldManager;

/// Unit-moment potentials on the rank-local density grid, in source order.
class PeriodicDipoleGrid
{
  public:
    PeriodicDipoleGrid(const ModulePW::PW_Basis& basis, const UnitCell& cell, const TDFieldManager& fields);
    void add_to(double* potential, const std::vector<double>& moments) const;

  private:
    std::vector<std::vector<double>> dipole_space_;
};

/// Three-dimensional periodic scalar potential for analytic dipole sources (td_stype=3).
class PotPeriodicDipole : public PotBase
{
  public:
    PotPeriodicDipole(const std::shared_ptr<const PeriodicDipoleGrid>& grid, const std::shared_ptr<TDFieldManager>& fields);
    void cal_fixed_v(double* potential) override;
    static void compute_force(const UnitCell& cell, const ModulePW::PW_Basis& basis,
                              const TDFieldManager& fields, ModuleBase::matrix& force);

  private:
    std::shared_ptr<TDFieldManager> fields_;
    std::shared_ptr<const PeriodicDipoleGrid> grid_;
};
} // namespace elecstate
#endif
