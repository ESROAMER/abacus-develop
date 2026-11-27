#include "source_io/module_efield/td_efield_io.h"

#include "source_base/constants.h"
#include "source_estate/module_pot/td_field_manager.h"
#include "source_io/module_parameter/input_parameter.h"

#include "gtest/gtest.h"
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

class TDEFieldIOTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        clean_output_files();
    }

    void TearDown() override
    {
        clean_output_files();
    }

    std::string output_path(const int field_index) const
    {
        return output_prefix_ + "efield_" + std::to_string(field_index) + ".txt";
    }

    void clean_output_files() const
    {
        for (int field_index = 0; field_index <= 3; ++field_index)
        {
            std::remove(output_path(field_index).c_str());
        }
        std::remove(blocking_path_.c_str());
        std::remove((output_prefix_ + "efield_total.txt").c_str());
    }

    std::shared_ptr<elecstate::TDFieldManager> create_manager(const std::vector<double>& amplitudes) const
    {
        Input_para input;
        input.td_vext = true;
        input.td_stype = 0;
        input.td_tstart = 0;
        input.td_tend = 10;
        input.td_dt = ModuleBase::AU_to_FS;
        input.td_ttype.clear();
        input.td_vext_dire.clear();
        input.td_heavi_t0.clear();
        input.td_heavi_amp.clear();
        for (std::size_t field_index = 0; field_index < amplitudes.size(); ++field_index)
        {
            input.td_ttype.push_back(3);
            input.td_vext_dire.push_back(static_cast<int>(field_index % 3) + 1);
            input.td_heavi_t0.push_back(10.0);
            input.td_heavi_amp.push_back(amplitudes[field_index]);
        }
        return elecstate::create_td_field_manager(input);
    }

    std::vector<std::pair<double, double>> read_samples(const int field_index) const
    {
        std::ifstream input(output_path(field_index).c_str());
        std::vector<std::pair<double, double>> samples;
        double time = 0.0;
        double field = 0.0;
        while (input >> time >> field)
        {
            samples.push_back(std::make_pair(time, field));
        }
        return samples;
    }

    const std::string output_prefix_ = "td_efield_io_test_";
    const std::string blocking_path_ = "td_efield_io_test_blocking_path";
};

TEST_F(TDEFieldIOTest, FreshCalculationTruncatesAndUsesOneBasedFiles)
{
    {
        std::ofstream output_1(output_path(1).c_str());
        std::ofstream output_2(output_path(2).c_str());
        output_1 << "old data\n";
        output_2 << "old data\n";
    }

    std::shared_ptr<elecstate::TDFieldManager> manager = create_manager({2.0, 3.0});
    ModuleIO::prep_td_field_out(output_prefix_, manager->fields().size(), false);
    EXPECT_TRUE(read_samples(1).empty());
    EXPECT_TRUE(read_samples(2).empty());

    manager->prepare_sample(0);
    ModuleIO::write_td_field(*manager, output_prefix_);
    manager->prepare_sample(1);
    ModuleIO::write_td_field(*manager, output_prefix_);

    const std::vector<std::pair<double, double>> samples_1 = read_samples(1);
    const std::vector<std::pair<double, double>> samples_2 = read_samples(2);
    ASSERT_EQ(samples_1.size(), 2U);
    ASSERT_EQ(samples_2.size(), 2U);
    EXPECT_DOUBLE_EQ(samples_1[0].first, 0.0);
    EXPECT_DOUBLE_EQ(samples_2[0].first, 0.0);
    EXPECT_NEAR(samples_1[0].second, 2.0, 1.0e-12);
    EXPECT_NEAR(samples_2[0].second, 3.0, 1.0e-12);
    // One atomic time unit is rounded to six decimal places in fs.
    EXPECT_DOUBLE_EQ(samples_1[1].first, 0.024189);
    EXPECT_DOUBLE_EQ(samples_2[1].first, 0.024189);

    std::ifstream zero_based_file(output_path(0).c_str());
    EXPECT_FALSE(zero_based_file.good());
}

TEST_F(TDEFieldIOTest, RestartPreservesExistingSamples)
{
    {
        std::ofstream output(output_path(1).c_str());
        output << "7 8\n";
    }

    std::shared_ptr<elecstate::TDFieldManager> manager = create_manager({4.0});
    ModuleIO::prep_td_field_out(output_prefix_, manager->fields().size(), true);
    const int next_step = manager->current_step() + 1;
    manager->prepare_sample(next_step);
    ModuleIO::write_td_field(*manager, output_prefix_);

    const std::vector<std::pair<double, double>> samples = read_samples(1);
    ASSERT_EQ(samples.size(), 2U);
    EXPECT_DOUBLE_EQ(samples[0].first, 7.0);
    EXPECT_DOUBLE_EQ(samples[0].second, 8.0);
    EXPECT_DOUBLE_EQ(samples[1].first, 0.0);
    EXPECT_NEAR(samples[1].second, 4.0, 1.0e-12);
}

TEST_F(TDEFieldIOTest, ReportsOutputOpenFailures)
{
    {
        std::ofstream blocking_file(blocking_path_.c_str());
        blocking_file << "not a directory\n";
    }
    const std::string invalid_directory = blocking_path_ + "/";
    std::shared_ptr<elecstate::TDFieldManager> manager = create_manager({4.0});

    EXPECT_EXIT(ModuleIO::prep_td_field_out(invalid_directory, manager->fields().size(), false), testing::ExitedWithCode(1), "");

    const int next_step = manager->current_step() + 1;
    manager->prepare_sample(next_step);
    EXPECT_EXIT(ModuleIO::write_td_field(*manager, invalid_directory), testing::ExitedWithCode(1), "");
}

TEST(TDEFieldIO, TwoPeriodicDipoles)
{
    Input_para input;
    input.td_vext = true;
    input.td_tstart = 0;
    input.td_tend = 2;
    input.td_dt = ModuleBase::AU_to_FS;
    input.td_ttype = {3, 3};
    input.td_vext_dire = {1, 1};
    input.td_stype = 3;
    input.td_dipole_position = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    input.td_dipole_width = {0.5, 0.75};
    input.td_heavi_t0 = {2.0, 2.0};
    input.td_heavi_amp = {-0.25, 3.0};
    const auto manager = elecstate::create_td_field_manager(input);
    const std::string prefix = testing::TempDir() + "mixed_periodic_td_";
    std::ostringstream log;
    ModuleIO::prep_td_field_out(prefix, *manager, false, log);
    manager->prepare_sample(0);
    manager->prepare_sample(1);
    const auto full = manager->field_state();
    const auto half = manager->subinterval(full, 0, 2);
    manager->bind_field(half);
    EXPECT_DOUBLE_EQ(manager->field_vals_ha()[0], -0.25);
    EXPECT_DOUBLE_EQ(manager->field_vals_ha()[1], 3.0);
    EXPECT_DOUBLE_EQ(manager->efield_ha()[0], 0.0);
    manager->bind_field(full);
    ModuleIO::write_td_field(*manager, prefix);

    const std::string dipole_path = prefix + "dipole_1.txt";
    const std::string field_path = prefix + "dipole_2.txt";
    const std::string total_path = prefix + "efield_total.txt";
    std::ifstream dipole(dipole_path.c_str());
    std::ifstream field(field_path.c_str());
    std::ifstream total(total_path.c_str());
    std::string header;
    std::getline(dipole, header);
    EXPECT_NE(header.find("moment(e*Bohr)"), std::string::npos);
    EXPECT_NE(header.find("source(Bohr)=1 2 3"), std::string::npos);
    EXPECT_NE(header.find("sigma(Bohr)=0.5"), std::string::npos);
    std::getline(total, header);
    double time = 0.0;
    double value = 0.0;
    ASSERT_TRUE(dipole >> time >> value);
    EXPECT_DOUBLE_EQ(value, -0.25);
    std::getline(field, header);
    EXPECT_NE(header.find("sigma(Bohr)=0.75"), std::string::npos);
    ASSERT_TRUE(field >> time >> value);
    EXPECT_DOUBLE_EQ(value, 3.0);
    ASSERT_TRUE(total >> time >> value);
    EXPECT_DOUBLE_EQ(value, 0.0);
    EXPECT_NE(log.str().find(dipole_path), std::string::npos);
    std::remove(dipole_path.c_str());
    std::remove(field_path.c_str());
    std::remove(total_path.c_str());
}

} // namespace
