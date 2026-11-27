#include "source_io/module_efield/td_efield_io.h"

#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include "source_estate/module_pot/td_field_manager.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace
{

std::string field_out_path(const std::string& output_dir, const std::size_t field_index, const bool dipole)
{
    std::stringstream path;
    const char* prefix = dipole ? "dipole_" : "efield_";
    path << output_dir << prefix << field_index + 1 << ".txt";
    return path.str();
}

void log_output_path(std::ostream& log, const std::string& path)
{
    char* resolved = ::realpath(path.c_str(), nullptr);
    if (!resolved)
    {
        ModuleBase::WARNING_QUIT("ModuleIO::prep_td_field_out", "Cannot resolve output path " + path);
    }
    log << " Write RT-TDDFT field: " << resolved << '\n';
    std::free(resolved);
}

} // namespace

namespace ModuleIO
{

void prep_td_field_out(const std::string& output_dir, const std::size_t field_count, const bool restart)
{
    // A restart continues an existing time series. A fresh calculation opens
    // every configured file with `out`, which truncates any previous series.
    if (restart)
    {
        return;
    }

    for (std::size_t field_index = 0; field_index < field_count; ++field_index)
    {
        const std::string output_path = field_out_path(output_dir, field_index, false);
        std::ofstream output(output_path.c_str(), std::ofstream::out);
        if (!output)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::prep_td_field_out", "Cannot prepare electric-field file " + output_path + "!");
        }
    }
}

void prep_td_field_out(const std::string& output_dir, const elecstate::TDFieldManager& manager, const bool restart,
                       std::ostream& running_log)
{
    if (restart)
    {
        return;
    }
    const auto& fields = manager.fields();
    const std::size_t count = manager.file_source() ? 0 : fields.size();
    bool has_dipole = false;
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto* source = fields[i].periodic_dipole();
        const bool dipole = source != nullptr;
        const std::string path = field_out_path(output_dir, i, dipole);
        std::ofstream output(path.c_str());
        if (!output)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::prep_td_field_out", "Cannot prepare source file " + path);
        }
        if (source)
        {
            has_dipole = true;
            const int direction = fields[i].direction() + 1;
            output << std::setprecision(17) << "# time(fs) moment(e*Bohr); Cartesian direction=" << direction
                   << "; source(Bohr)=" << source->position.x << " " << source->position.y << " " << source->position.z
                   << "; sigma(Bohr)=" << source->width << "; periodic_3d zero_mean_field\n";
        }
        log_output_path(running_log, path);
    }
    const std::string total_path = output_dir + "efield_total.txt";
    std::ofstream output(total_path.c_str());
    if (!output)
    {
        ModuleBase::WARNING_QUIT("ModuleIO::prep_td_field_out", "Cannot prepare efield_total.txt.");
    }
    log_output_path(running_log, total_path);
    output << std::left << std::setw(10) << "# time(fs)" << std::right;
    for (const char* label: {"Ex(V/Angstrom)", "Ey(V/Angstrom)", "Ez(V/Angstrom)"})
    {
        output << "  " << std::setw(19) << label;
    }
    output << "  # ";
    if (manager.file_source())
    {
        output << "endpoint finite differences\n";
    }
    else if (has_dipole)
    {
        output << "sum of uniform analytic fields (dipole sources excluded)\n";
    }
    else
    {
        output << "sum of analytic fields\n";
    }
}

void write_td_field(const elecstate::TDFieldManager& manager, const std::string& output_dir)
{
    const double time_fs = manager.current_step() * manager.dt_ha() * ModuleBase::AU_to_FS;
    // Keep the complete time series, including zero fields outside the analytic window.
    {
        std::ofstream output((output_dir + "efield_total.txt").c_str(), std::ofstream::app);
        if (!output)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_td_field", "Cannot append efield_total.txt.");
        }
        output << std::right << std::fixed << std::setprecision(6) << std::setw(10) << time_fs;
        output << std::scientific << std::setprecision(12);
        for (int direction = 0; direction < 3; ++direction)
        {
            output << "  " << std::setw(19) << manager.efield_ha()[direction] * ModuleBase::Hartree_to_eV / ModuleBase::BOHR_TO_A;
        }
        output << '\n';
    }
    if (manager.file_source())
    {
        return;
    }

    const std::vector<double>& field_values = manager.field_vals_ha();
    for (std::size_t field_index = 0; field_index < field_values.size(); ++field_index)
    {
        // Keep one file per input occurrence even when directions repeat.
        const bool dipole = manager.fields()[field_index].periodic_dipole() != nullptr;
        const std::string output_path = field_out_path(output_dir, field_index, dipole);
        std::ofstream output(output_path.c_str(), std::ofstream::app);
        if (!output)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_td_field", "Cannot append electric-field file " + output_path + "!");
        }
        const double value = dipole ? field_values[field_index]
                                    : field_values[field_index] * ModuleBase::Hartree_to_eV / ModuleBase::BOHR_TO_A;
        output << std::right << std::fixed << std::setprecision(6) << std::setw(10) << time_fs << "  " << std::scientific
               << std::setprecision(12) << std::setw(19) << value
               << std::endl;
    }
}

} // namespace ModuleIO
