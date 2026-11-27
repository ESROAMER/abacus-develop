#ifndef POT_PERIODIC_DIPOLE_H
#define POT_PERIODIC_DIPOLE_H

#include "pot_base.h"
#include <memory>
#include <vector>

namespace elecstate
{
class TDFieldManager;

/// Three-dimensional periodic scalar potential for analytic dipole sources (td_stype=3).
class PotPeriodicDipole : public PotBase
{
  public:
    PotPeriodicDipole(const ModulePW::PW_Basis& basis, const UnitCell& cell, const std::shared_ptr<TDFieldManager>& fields);
    void cal_fixed_v(double* potential) override;
    static void compute_force(const UnitCell& cell, const ModulePW::PW_Basis& basis,
                              const TDFieldManager& fields, ModuleBase::matrix& force);

  private:
    std::shared_ptr<TDFieldManager> fields_;
    std::vector<std::vector<double>> dipole_space_; ///< Unit-moment grid potential in input order.
};
} // namespace elecstate
#endif
