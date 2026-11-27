#include "read_input.h"
#include "read_input_tool.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cmath>

namespace ModuleIO
{
namespace
{

void check_dipole_setup(const Input_para& input)
{
    if (input.basis_type != "lcao" || input.esolver_type != "tddft" || input.td_stype != 0 || input.init_vecpot)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "Periodic dipoles require analytic LCAO RT-TDDFT fields in length gauge.");
    }
    if (input.calculation != "md" || input.mdp.md_nstep != 1 || input.mdp.md_restart || input.cal_stress || input.symmetry != "-1")
    {
        ModuleBase::WARNING_QUIT("ReadInput", "Periodic dipoles require a single MD step, no restart or stress, and symmetry=-1.");
    }
}

} // namespace

void check_periodic_dipoles(const Input_para& input)
{
    const auto& types = input.td_space_type;
    if (!types.empty() && types.size() != input.td_ttype.size())
    {
        ModuleBase::WARNING_QUIT("ReadInput", "td_space_type must match the number of td_ttype entries.");
    }
    const std::size_t count = std::count(types.begin(), types.end(), 2);
    const std::size_t uniform = std::count(types.begin(), types.end(), 0);
    if (count + uniform != types.size())
    {
        ModuleBase::WARNING_QUIT("ReadInput", "td_space_type accepts 0 (uniform) or 2 (3D periodic dipole).");
    }
    if (input.td_dipole_position.size() != 3 * count || input.td_dipole_width.size() != count)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "Each dipole needs three td_dipole_position values and one td_dipole_width.");
    }
    for (const double width: input.td_dipole_width)
    {
        if (!std::isfinite(width) || width <= 0.0)
        {
            ModuleBase::WARNING_QUIT("ReadInput", "td_dipole_width values must be finite and positive.");
        }
    }
    for (const double position: input.td_dipole_position)
    {
        if (!std::isfinite(position))
        {
            ModuleBase::WARNING_QUIT("ReadInput", "td_dipole_position values must be finite.");
        }
    }
    if (count > 0)
    {
        check_dipole_setup(input);
    }
}

void ReadInput::item_periodic_dipole()
{
    {
        Input_Item item("td_space_type");
        item.annotation = "spatial model of each external source";
        item.category = "Real-Time TDDFT (Common)";
        item.type = "Vector of Integer";
        item.description = R"(Spatial model per td_ttype entry. Use 0 for a uniform field and 2 for a Gaussian dipole repeated in all three lattice directions with zero macroscopic field. Omission selects uniform fields. Dipoles require LCAO length-gauge RT-TDDFT with fixed ions and cell (calculation=md, md_nstep=1, md_restart=false, cal_stress=false, symmetry=-1). td_vext_dire selects the Cartesian moment axis. All five waveforms accept signed dipole moments in e*Bohr.)";
        item.default_value = "empty (all uniform)";
        item.unit = "";
        item.set_availability("esolver_type==tddft");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            parse_expression(item.str_values, para.input.td_space_type);
        };
        sync_intvec(input.td_space_type, para.input.td_space_type.size(), 0);
        this->add_item(item);
    }
    {
        Input_Item item("td_dipole_position");
        item.annotation = "periodic dipole positions";
        item.category = "Real-Time TDDFT (Common)";
        item.type = "Vector of Real";
        item.description = R"(Cartesian x y z coordinates in Bohr for each periodic dipole, in occurrence order, skipping uniform components. Translating a source by a lattice vector gives the same potential.)";
        item.default_value = "empty";
        item.unit = "Bohr";
        item.set_availability("esolver_type==tddft");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            parse_expression(item.str_values, para.input.td_dipole_position);
        };
        sync_doublevec(input.td_dipole_position, para.input.td_dipole_position.size(), 0.0);
        this->add_item(item);
    }
    {
        Input_Item item("td_dipole_width");
        item.annotation = "periodic dipole widths";
        item.category = "Real-Time TDDFT (Common)";
        item.type = "Vector of Real";
        item.description = R"(One finite, strictly positive Gaussian standard deviation in Bohr per periodic dipole, in occurrence order. The density plane-wave basis must resolve this length.)";
        item.default_value = "empty";
        item.unit = "Bohr";
        item.set_availability("esolver_type==tddft");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            parse_expression(item.str_values, para.input.td_dipole_width);
        };
        sync_doublevec(input.td_dipole_width, para.input.td_dipole_width.size(), 0.0);
        this->add_item(item);
    }
}

} // namespace ModuleIO
