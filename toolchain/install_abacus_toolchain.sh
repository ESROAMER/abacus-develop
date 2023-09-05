#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")" && pwd -P)"

# +---------------------------------------------------------------------------+
# |   ABACUS: (Atomic-orbital Based Ab-initio Computation at UStc)            |
# |   -- an open-source package based on density functional theory(DFT)       |
# |   Copyright 2004-2022 ABACUS developers group                             |
# |   <https://github.com/deepmodeling/abacus-develop>                        |
# |                                                                           |
# |   SPDX-License-Identifier: GPL-2.0-or-later                               |
# +---------------------------------------------------------------------------+
#
#
# *****************************************************************************
#> \brief    This script will compile and install or link existing tools and
#>           libraries ABACUS depends on and generate setup files that
#>           can be used to compile and use ABACUS
#> \history  Created on Friday, 2023/08/18
#            Update for Intel (18.08.2023, MK)
#> \author   Zhaoqing Liu quanmisaka@stu.pku.edu.cn
#> with the reference of Lianheng Tong (ltong) lianheng.tong@kcl.ac.uk
# *****************************************************************************

# ------------------------------------------------------------------------
# Work directories and used files
# ------------------------------------------------------------------------
export ROOTDIR="${PWD}"
export SCRIPTDIR="${ROOTDIR}/scripts"
export BUILDDIR="${ROOTDIR}/build"
export INSTALLDIR="${ROOTDIR}/install"
export SETUPFILE="${INSTALLDIR}/setup"
export SHA256_CHECKSUM="${SCRIPTDIR}/checksums.sha256"
export ARCH_FILE_TEMPLATE="${SCRIPTDIR}/arch_base.tmpl"

# ------------------------------------------------------------------------
# Make a copy of all options for $SETUPFILE
# ------------------------------------------------------------------------
TOOLCHAIN_OPTIONS="$@"

# ------------------------------------------------------------------------
# Load common variables and tools
# ------------------------------------------------------------------------
source "${SCRIPTDIR}"/common_vars.sh
source "${SCRIPTDIR}"/tool_kit.sh

# ------------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------------
show_help() {
  cat << EOF
This script will help you compile and install, 
or link libraries ABACUS depends on, 
and give setup files that you can use to compile ABACUS.

USAGE:

$(basename $SCRIPT_NAME) [options]

OPTIONS:

-h, --help                Show this message.
-j <n>                    Number of processors to use for compilation, if
                          this option is not present, then the script
                          automatically tries to determine the number of
                          processors you have and try to use all of the
                          processors.
--no-check-certificate    If you encounter "certificate verification" errors
                          from wget or ones saying that "common name doesn't
                          match requested host name" while at tarball downloading
                          stage, then the recommended solution is to install
                          the newest wget release. Alternatively, you can use
                          this option to bypass the verification and proceed with
                          the download. Security wise this should still be okay
                          as the installation script will check file checksums
                          after every tarball download. Nevertheless use this
                          option at your own risk.
--install-all             This option will set value of all --with-PKG
                          options to "install". You can selectively set
                          --with-PKG to another value again by having the
                          --with-PKG option placed AFTER this option on the
                          command line.
--mpi-mode                Selects which MPI flavour to use. Available options
                          are: mpich, openmpi, intelmpi, and no. By selecting "no",
                          MPI is not supported and disabled. By default the script
                          will try to determine the flavour based on the MPI library
                          currently available in your system path. For CRAY (CLE)
                          systems, the default flavour is mpich. Note that explicitly
                          setting --with-mpich, --with-openmpi or --with-intelmpi
                          options to values other than no will also switch --mpi-mode
                          to the respective mode.
--math-mode               Selects which core math library to use. Available options
                          are: acml, cray, mkl, and openblas. The option "cray"
                          corresponds to cray libsci, and is the default for CRAY
                          (CLE) systems. For non-CRAY systems, if env variable MKLROOT
                          exists then mkl will be default, otherwise openblas is the
                          default option. Explicitly setting --with-acml, --with-mkl,
                          or --with-openblas options will switch --math-mode to the
                          respective modes.
--gpu-ver                 Selects the GPU architecture for which to compile. Available
                          options are: K20X, K40, K80, P100, V100, Mi50, Mi100, Mi250, 
                          and no.
                          This setting determines the value of nvcc's '-arch' flag.
                          Default = no.
--log-lines               Number of log file lines dumped in case of a non-zero exit code.
                          Default = 200
--target-cpu              Compile for the specified target CPU (e.g. haswell or generic), i.e.
                          do not optimize for the actual host system which is the default (native)
--no-arch-files           Do not generate arch files
--dry-run                 Write only config files, but don't actually build packages.

The --enable-FEATURE options follow the rules:
  --enable-FEATURE=yes    Enable this particular feature
  --enable-FEATURE=no     Disable this particular feature
  --enable-FEATURE        The option keyword alone is equivalent to
                          --enable-FEATURE=yes

  --enable-cuda           Turn on GPU (CUDA) support (can be combined
                          with --enable-opencl).
                          Default = no
  --enable-hip            Turn on GPU (HIP) support.
                          Default = no
  --enable-opencl         Turn on OpenCL (GPU) support. Requires the OpenCL
                          development packages and runtime. If combined with
                          --enable-cuda, OpenCL alongside of CUDA is used.
                          Default = no
  --enable-cray           Turn on or off support for CRAY Linux Environment
                          (CLE) manually. By default the script will automatically
                          detect if your system is CLE, and provide support
                          accordingly.

The --with-PKG options follow the rules:
  --with-PKG=install      Will download the package in \$PWD/build and
                          install the library package in \$PWD/install.
  --with-PKG=system       The script will then try to find the required
                          libraries of the package from the system path
                          variables such as PATH, LD_LIBRARY_PATH and
                          CPATH etc.
  --with-PKG=no           Do not use the package.
  --with-PKG=<path>       The package will be assumed to be installed in
                          the given <path>, and be linked accordingly.
  --with-PKG              The option keyword alone will be equivalent to
                          --with-PKG=install

  --with-gcc              The GCC compiler to use to compile ABACUS.
                          Default = system
  --with-intel            Use the Intel compiler to compile ABACUS.
                          Default = system
  --with-intel-classic    Use the classic Intel compiler to compile ABACUS.
                          Default = no
  --with-cmake            Cmake utilities
                          Default = install
  --with-openmpi          OpenMPI, important if you want a parallel version of ABACUS.
                          Default = system
  --with-mpich            MPICH, MPI library like OpenMPI. one should
                          use only one of OpenMPI, MPICH or Intel MPI.
                          Default = system
  --with-mpich-device     Select the MPICH device, implies the use of MPICH as MPI library
                          Default = ch4
  --with-intelmpi         Intel MPI, MPI library like OpenMPI. one should
                          use only one of OpenMPI, MPICH or Intel MPI.
                          Default = system
  --with-libxc            libxc, exchange-correlation library. Needed for
                          QuickStep DFT and hybrid calculations.
                          Default = install
  --with-fftw             FFTW3, library for fast fourier transform
                          Default = install
  --with-acml             AMD core maths library, which provides LAPACK and BLAS
                          Default = system
  --with-mkl              Intel Math Kernel Library, which provides LAPACK, and BLAS.
                          If MKL's FFTW3 interface is suitable (no FFTW-MPI support),
                          it replaces the FFTW library. If the ScaLAPACK component is
                          found, it replaces the one specified by --with-scalapack.
                          Default = system
  --with-openblas         OpenBLAS is a free high performance LAPACK and BLAS library,
                          the successor to GotoBLAS.
                          Default = install
  --with-scalapack        Parallel linear algebra library, needed for parallel
                          calculations.
                          Default = install
  --with-cereal         Enable cereal for ABACUS LCAO
                          Default = install
  --with-elpa             Eigenvalue SoLvers for Petaflop-Applications library.
                          Fast library for large parallel jobs.
                          Default = install
  --with-libtorch         Enable libtorch the machine learning framework needed for DeePKS
                          Default = no
  --with-libnpy         Enable libnpy the machine learning framework needed for DeePKS
                          Default = no

FURTHER INSTRUCTIONS

All packages to be installed locally will be downloaded and built inside
./build, and then installed into package specific directories inside
./install.

Both ./build and ./install are safe to delete, as they contain
only the files and directories that are generated by this script. However,
once all the packages are installed, and you compile ABACUS using the arch
files provided by this script, then you must keep ./install in exactly
the same location as it was first created, as it contains tools and libraries
your version of ABACUS binary will depend on.

It should be safe to terminate running of this script in the middle of a
build process. The script will know if a package has been successfully
installed, and will just carry on and recompile and install the last
package it is working on. This is true even if you lose the content of
the entire ./build directory.

  +----------------------------------------------------------------+
  |  YOU SHOULD ALWAYS SOURCE ./install/setup BEFORE YOU RUN ABACUS |
  |  COMPILED WITH THIS TOOLCHAIN                                   |
  +----------------------------------------------------------------+

EOF
}

# ------------------------------------------------------------------------
# PACKAGE LIST: register all new dependent tools and libs here. Order
# is important, the first in the list gets installed first
# ------------------------------------------------------------------------
tool_list="gcc intel cmake"
mpi_list="mpich openmpi intelmpi"
math_list="mkl acml openblas"
lib_list="fftw libxc scalapack elpa cereal libtorch libnpy"
package_list="${tool_list} ${mpi_list} ${math_list} ${lib_list}"
# ------------------------------------------------------------------------

# first set everything to __DONTUSE__
for ii in ${package_list}; do
  eval with_${ii}="__DONTUSE__"
done

# ------------------------------------------------------------------------
# Work out default settings
# ------------------------------------------------------------------------

# tools to turn on by default:
with_gcc="__SYSTEM__"

# libs to turn on by default, the math and mpi libraries are chosen by there respective modes:
with_fftw="__INSTALL__"
with_libxc="__INSTALL__"
with_scalapack="__INSTALL__"
# default math library settings, MATH_MODE picks the math library
# to use, and with_* defines the default method of installation if it
# is picked. For non-CRAY systems defaults to mkl if $MKLROOT is
# available, otherwise defaults to openblas
if [ "${MKLROOT}" ]; then
  export MATH_MODE="mkl"
  with_mkl="__SYSTEM__"
else
  export MATH_MODE="openblas"
fi
with_acml="__SYSTEM__"
with_openblas="__INSTALL__"
with_elpa="__INSTALL__"
with_cereal="__INSTALL__"
with_libtorch="__DONTUSE__"
with_libnpy="__DONTUSE__"
# for MPI, we try to detect system MPI variant
if (command -v mpiexec > /dev/null 2>&1); then
  # check if we are dealing with openmpi, mpich or intelmpi
  if (mpiexec --version 2>&1 | grep -s -q "HYDRA"); then
    echo "MPI is detected and it appears to be MPICH"
    export MPI_MODE="mpich"
    with_mpich="__SYSTEM__"
  elif (mpiexec --version 2>&1 | grep -s -q "OpenRTE"); then
    echo "MPI is detected and it appears to be OpenMPI"
    export MPI_MODE="openmpi"
    with_openmpi="__SYSTEM__"
  elif (mpiexec --version 2>&1 | grep -s -q "Intel"); then
    echo "MPI is detected and it appears to be Intel MPI"
    with_gcc="__DONTUSE__"
    with_intel="__SYSTEM__"
    with_intelmpi="__SYSTEM__"
    export MPI_MODE="intelmpi"
  else # default to mpich
    echo "MPI is detected and defaults to MPICH"
    export MPI_MODE="mpich"
    with_mpich="__SYSTEM__"
  fi
else
  report_warning $LINENO "No MPI installation detected (ignore this message in Cray Linux Environment or when MPI installation was requested)."
  export MPI_MODE="no"
fi

# default enable options
dry_run="__FALSE__"
no_arch_files="__FALSE__"
enable_tsan="__FALSE__"
enable_opencl="__FALSE__"
enable_cuda="__FALSE__"
enable_hip="__FALSE__"
export intel_classic="yes" 
# no, then icc->icx, icpc->icpx, 
# which cannot compile elpa-2021 and fftw.3.3.10 in some place
# due to some so-called cross-compile problem
# and will lead to problem in force calculation
# but icx is recommended by intel compiler
# option: --with-intel-classic can change it to yes/no
# zhaoqing by 2023.08.31
export GPUVER="no"
export MPICH_DEVICE="ch4"
export TARGET_CPU="native"

# default for log file dump size
export LOG_LINES="200"

# defaults for CRAY Linux Environment
if [ "${CRAY_LD_LIBRARY_PATH}" ]; then
  enable_cray="__TRUE__"
  export MATH_MODE="cray"
  # Default MPI used by CLE is assumed to be MPICH, in any case
  # do not use the installers for the MPI libraries
  with_mpich="__DONTUSE__"
  with_openmpi="__DONTUSE__"
  with_intelmpi="__DONTUSE__"
  export MPI_MODE="mpich"
  # set default value for some installers appropriate for CLE
  with_gcc="__DONTUSE__"
  with_intel="__DONTUSE__"
  with_fftw="__SYSTEM__"
  with_scalapack="__DONTUSE__"
else
  enable_cray="__FALSE__"
fi

# ------------------------------------------------------------------------
# parse user options
# ------------------------------------------------------------------------
while [ $# -ge 1 ]; do
  case ${1} in
    -j)
      case "${2}" in
        -*)
          export NPROCS_OVERWRITE="$(get_nprocs)"
          ;;
        [0-9]*)
          shift
          export NPROCS_OVERWRITE="${1}"
          ;;
        *)
          report_error ${LINENO} \
            "The -j flag can only be followed by an integer number, found ${2}."
          exit 1
          ;;
      esac
      ;;
    -j[0-9]*)
      export NPROCS_OVERWRITE="${1#-j}"
      ;;
    --no-check-certificate)
      export DOWNLOADER_FLAGS="--no-check-certificate"
      ;;
    --install-all)
      # set all package to the default installation status
      for ii in ${package_list}; do
        if [ "${ii}" != "intel" ] && [ "${ii}" != "intelmpi" ]; then
          eval with_${ii}="__INSTALL__"
        fi
      done
      # Use MPICH as default
      export MPI_MODE="mpich"
      ;;
    --mpi-mode=*)
      user_input="${1#*=}"
      case "$user_input" in
        mpich)
          export MPI_MODE="mpich"
          ;;
        openmpi)
          export MPI_MODE="openmpi"
          ;;
        intelmpi)
          export MPI_MODE="intelmpi"
          ;;
        no)
          export MPI_MODE="no"
          ;;
        *)
          report_error ${LINENO} \
            "--mpi-mode currently only supports openmpi, mpich, intelmpi and no as options"
          exit 1
          ;;
      esac
      ;;
    --math-mode=*)
      user_input="${1#*=}"
      case "$user_input" in
        cray)
          export MATH_MODE="cray"
          ;;
        mkl)
          export MATH_MODE="mkl"
          ;;
        acml)
          export MATH_MODE="acml"
          ;;
        openblas)
          export MATH_MODE="openblas"
          ;;
        *)
          report_error ${LINENO} \
            "--math-mode currently only supports mkl, acml, and openblas as options"
          ;;
      esac
      ;;
    --gpu-ver=*)
      user_input="${1#*=}"
      case "${user_input}" in
        K20X | K40 | K80 | P100 | V100 | A100 | Mi50 | Mi100 | Mi250 | no)
          export GPUVER="${user_input}"
          ;;
        *)
          report_error ${LINENO} \
            "--gpu-ver currently only supports K20X, K40, K80, P100, V100, A100, Mi50, Mi100, Mi250, and no as options"
          exit 1
          ;;
      esac
      ;;
    --target-cpu=*)
      user_input="${1#*=}"
      export TARGET_CPU="${user_input}"
      ;;
    --log-lines=*)
      user_input="${1#*=}"
      export LOG_LINES="${user_input}"
      ;;
    --no-arch-files)
      no_arch_files="__TRUE__"
      ;;
    --dry-run)
      dry_run="__TRUE__"
      ;;
    --enable-tsan*)
      enable_tsan=$(read_enable $1)
      if [ "${enable_tsan}" = "__INVALID__" ]; then
        report_error "invalid value for --enable-tsan, please use yes or no"
        exit 1
      fi
      ;;
    --enable-cuda*)
      enable_cuda=$(read_enable $1)
      if [ $enable_cuda = "__INVALID__" ]; then
        report_error "invalid value for --enable-cuda, please use yes or no"
        exit 1
      fi
      ;;
    --enable-hip*)
      enable_hip=$(read_enable $1)
      if [ "${enable_hip}" = "__INVALID__" ]; then
        report_error "invalid value for --enable-hip, please use yes or no"
        exit 1
      fi
      ;;
    --enable-opencl*)
      enable_opencl=$(read_enable $1)
      if [ $enable_opencl = "__INVALID__" ]; then
        report_error "invalid value for --enable-opencl, please use yes or no"
        exit 1
      fi
      ;;
    --enable-cray*)
      enable_cray=$(read_enable $1)
      if [ "${enable_cray}" = "__INVALID__" ]; then
        report_error "invalid value for --enable-cray, please use yes or no"
        exit 1
      fi
      ;;
    --with-gcc*)
      with_gcc=$(read_with "${1}")
      ;;
    --with-cmake*)
      with_cmake=$(read_with "${1}")
      ;;
    --with-mpich-device=*)
      user_input="${1#*=}"
      export MPICH_DEVICE="${user_input}"
      export MPI_MODE=mpich
      ;;
    --with-mpich*)
      with_mpich=$(read_with "${1}")
      if [ "${with_mpich}" != "__DONTUSE__" ]; then
        export MPI_MODE=mpich
      fi
      ;;
    --with-openmpi*)
      with_openmpi=$(read_with "${1}")
      if [ "${with_openmpi}" != "__DONTUSE__" ]; then
        export MPI_MODE=openmpi
      fi
      ;;
    --with-intelmpi*)
      with_intelmpi=$(read_with "${1}" "__SYSTEM__")
      if [ "${with_intelmpi}" != "__DONTUSE__" ]; then
        export MPI_MODE=intelmpi
      fi
      ;;
    --with-intel-classic*)
      intel_classic=$(read_with "${1}" "yes") # default yes
      ;;
    --with-intel*)
      with_intel=$(read_with "${1}" "__SYSTEM__")
      ;;
    --with-libxc*)
      with_libxc=$(read_with "${1}")
      ;;
    --with-fftw*)
      with_fftw=$(read_with "${1}")
      ;;
    --with-mkl*)
      with_mkl=$(read_with "${1}" "__SYSTEM__")
      if [ "${with_mkl}" != "__DONTUSE__" ]; then
        export MATH_MODE="mkl"
      fi
      ;;
    --with-acml*)
      with_acml=$(read_with "${1}")
      if [ "${with_acml}" != "__DONTUSE__" ]; then
        export MATH_MODE="acml"
      fi
      ;;
    --with-openblas*)
      with_openblas=$(read_with "${1}")
      if [ "${with_openblas}" != "__DONTUSE__" ]; then
        export MATH_MODE="openblas"
      fi
      ;;
    --with-scalapack*)
      with_scalapack=$(read_with "${1}")
      ;;
    --with-elpa*)
      with_elpa=$(read_with "${1}")
      ;;
    --with-libtorch*)
      with_libtorch=$(read_with "${1}")
      ;;
    --with-cereal*)
      with_cereal=$(read_with "${1}")
      ;;
    --with-libnpy*)
      with_libnpy=$(read_with "${1}")
      ;;
    --help*)
      show_help
      exit 0
      ;;
    -h*)
      show_help
      exit 0
      ;;
    *)
      report_error "Unknown flag: $1"
      exit 1
      ;;
  esac
  shift
done

# consolidate settings after user input
export ENABLE_TSAN="${enable_tsan}"
export ENABLE_CUDA="${enable_cuda}"
export ENABLE_HIP="${enable_hip}"
export ENABLE_OPENCL="${enable_opencl}"
export ENABLE_CRAY="${enable_cray}"

# ------------------------------------------------------------------------
# Check and solve known conflicts before installations proceed
# ------------------------------------------------------------------------
# Compiler conflicts
if [ "${with_intel}" != "__DONTUSE__" ] && [ "${with_gcc}" = "__INSTALL__" ]; then
  echo "You have chosen to use the Intel compiler, therefore the installation of the GCC compiler will be skipped."
  with_gcc="__SYSTEM__"
fi
# MPI library conflicts
if [ "${MPI_MODE}" = "no" ]; then
  if [ "${with_scalapack}" != "__DONTUSE__" ]; then
    echo "Not using MPI, so scalapack is disabled."
    with_scalapack="__DONTUSE__"
  fi
  if [ "${with_elpa}" != "__DONTUSE__" ]; then
    echo "Not using MPI, so ELPA is disabled."
    with_elpa="__DONTUSE__"
  fi
else
  # if gcc is installed, then mpi needs to be installed too
  if [ "${with_gcc}" = "__INSTALL__" ]; then
    echo "You have chosen to install the GCC compiler, therefore MPI libraries have to be installed too"
    case ${MPI_MODE} in
      mpich)
        with_mpich="__INSTALL__"
        with_openmpi="__DONTUSE__"
        ;;
      openmpi)
        with_mpich="__DONTUSE__"
        with_openmpi="__INSTALL__"
        ;;
    esac
    echo "and the use of the Intel compiler and Intel MPI will be disabled."
    with_intel="__DONTUSE__"
    with_intelmpi="__DONTUSE__"
  fi
  # Enable only one MPI implementation
  case ${MPI_MODE} in
    mpich)
      with_openmpi="__DONTUSE__"
      with_intelmpi="__DONTUSE__"
      ;;
    openmpi)
      with_mpich="__DONTUSE__"
      with_intelmpi="__DONTUSE__"
      ;;
    intelmpi)
      with_mpich="__DONTUSE__"
      with_openmpi="__DONTUSE__"
      ;;
  esac
fi

# If CUDA or HIP are enabled, make sure the GPU version has been defined.
if [ "${ENABLE_CUDA}" = "__TRUE__" ] || [ "${ENABLE_HIP}" = "__TRUE__" ]; then
  if [ "${GPUVER}" = "no" ]; then
    report_error "Please choose GPU architecture to compile for with --gpu-ver"
    exit 1
  fi
fi

# several packages require cmake.
if [ "${with_scalapack}" = "__INSTALL__" ]; then
  [ "${with_cmake}" = "__DONTUSE__" ] && with_cmake="__INSTALL__"
fi


# ------------------------------------------------------------------------
# Preliminaries
# ------------------------------------------------------------------------

mkdir -p ${INSTALLDIR}

# variables used for generating ABACUS ARCH file
export CP_DFLAGS=""
export CP_LIBS=""
export CP_CFLAGS=""
export CP_LDFLAGS="-Wl,--enable-new-dtags"

# ------------------------------------------------------------------------
# Start writing setup file
# ------------------------------------------------------------------------
cat << EOF > "$SETUPFILE"
#!/bin/bash
source "${SCRIPTDIR}/tool_kit.sh"
export ABACUS_TOOLCHAIN_OPTIONS="${TOOLCHAIN_OPTIONS}"
EOF

# ------------------------------------------------------------------------
# Special settings for CRAY Linux Environment (CLE)
# TODO: CLE should be handle like gcc or Intel using a with_cray flag and
#       this section should be moved to a separate file install_cray.
# ------------------------------------------------------------------------
if [ "${ENABLE_CRAY}" = "__TRUE__" ]; then
  echo "------------------------------------------------------------------------"
  echo "CRAY Linux Environment (CLE) is detected"
  echo "------------------------------------------------------------------------"
  # add cray paths to system search path
  export LIB_PATHS="CRAY_LD_LIBRARY_PATH ${LIB_PATHS}"
  # set compilers to CLE wrappers
  check_command cc
  check_command ftn
  check_command CC
  export CC="cc"
  export CXX="CC"
  export FC="ftn"
  export F90="${FC}"
  export F77="${FC}"
  export MPICC="${CC}"
  export MPICXX="${CXX}"
  export MPIFC="${FC}"
  export MPIFORT="${MPIFC}"
  export MPIF77="${MPIFC}"
  # CRAY libsci should contains core math libraries, scalapack
  # doesn't need LDFLAGS or CFLAGS, nor do the one need to
  # explicitly link the math and scalapack libraries, as all is
  # taken care of by the cray compiler wrappers.
  if [ "$with_scalapack" = "__DONTUSE__" ]; then
    export CP_DFLAGS="${CP_DFLAGS} IF_MPI(-D__SCALAPACK|)"
  fi
  case $MPI_MODE in
    mpich)
      if [ "$MPICH_DIR" ]; then
        cray_mpich_include_path="$MPICH_DIR/include"
        cray_mpich_lib_path="$MPICH_DIR/lib"
        export INCLUDE_PATHS="$INCLUDE_PATHS cray_mpich_include_path"
        export LIB_PATHS="$LIB_PATHS cray_mpich_lib_path"
      fi
      if [ "$with_mpich" = "__DONTUSE__" ]; then
        add_include_from_paths MPI_CFLAGS "mpi.h" $INCLUDE_PATHS
        add_include_from_paths MPI_LDFLAGS "libmpi.*" $LIB_PATHS
        export MPI_CFLAGS
        export MPI_LDFLAGS
        export MPI_LIBS=" "
        export CP_DFLAGS="${CP_DFLAGS} IF_MPI(-D__parallel|)"
      fi
      ;;
    openmpi)
      if [ "$with_openmpi" = "__DONTUSE__" ]; then
        add_include_from_paths MPI_CFLAGS "mpi.h" $INCLUDE_PATHS
        add_include_from_paths MPI_LDFLAGS "libmpi.*" $LIB_PATHS
        export MPI_CFLAGS
        export MPI_LDFLAGS
        export MPI_LIBS="-lmpi -lmpi_cxx"
        export CP_DFLAGS="${CP_DFLAGS} IF_MPI(-D__parallel|)"
      fi
      ;;
    intelmpi)
      if [ "$with_intelmpi" = "__DONTUSE__" ]; then
        with_gcc="__DONTUSE__"
        with_intel="__SYSTEM__"
        add_include_from_paths MPI_CFLAGS "mpi.h" $INCLUDE_PATHS
        add_include_from_paths MPI_LDFLAGS "libmpi.*" $LIB_PATHS
        export MPI_CFLAGS
        export MPI_LDFLAGS
        export MPI_LIBS="-lmpi -lmpi_cxx"
        export CP_DFLAGS="${CP_DFLAGS} IF_MPI(-D__parallel|)"
      fi
      ;;
  esac
  check_lib -lz
  check_lib -ldl
  export CRAY_EXTRA_LIBS="-lz -ldl"
  # the space is intentional, so that the variable is non-empty and
  # can pass require_env checks
  export SCALAPACK_LDFLAGS=" "
  export SCALAPACK_LIBS=" "
fi

# ------------------------------------------------------------------------
# Installing tools required for building ABACUS and associated libraries
# ------------------------------------------------------------------------

echo "Compiling with $(get_nprocs) processes for target ${TARGET_CPU}."

# Select the correct compute number based on the GPU architecture
case ${GPUVER} in
  K20X)
    export ARCH_NUM="35"
    ;;
  K40)
    export ARCH_NUM="35"
    ;;
  K80)
    export ARCH_NUM="37"
    ;;
  P100)
    export ARCH_NUM="60"
    ;;
  V100)
    export ARCH_NUM="70"
    ;;
  A100)
    export ARCH_NUM="80"
    ;;
  Mi50)
    # TODO: export ARCH_NUM=
    ;;
  Mi100)
    # TODO: export ARCH_NUM=
    ;;
  Mi250)
    # TODO: export ARCH_NUM=
    ;;
  no)
    export ARCH_NUM="no"
    ;;
  *)
    report_error ${LINENO} \
      "--gpu-ver currently only supports K20X, K40, K80, P100, V100, A100, Mi50, Mi100, Mi250, and no as options"
    exit 1
    ;;
esac

write_toolchain_env ${INSTALLDIR}

# write toolchain config
echo "tool_list=\"${tool_list}\"" > ${INSTALLDIR}/toolchain.conf
for ii in ${package_list}; do
  install_mode="$(eval echo \${with_${ii}})"
  echo "with_${ii}=\"${install_mode}\"" >> ${INSTALLDIR}/toolchain.conf
done

# ------------------------------------------------------------------------
# Build packages unless dry-run mode is enabled.
# ------------------------------------------------------------------------
if [ "${dry_run}" = "__TRUE__" ]; then
  echo "Wrote only configuration files (--dry-run)."
else
  echo "# Leak suppressions" > ${INSTALLDIR}/lsan.supp
  ./scripts/stage0/install_stage0.sh
  ./scripts/stage1/install_stage1.sh
  ./scripts/stage2/install_stage2.sh
  ./scripts/stage3/install_stage3.sh
  ./scripts/stage4/install_stage4.sh
fi

cat << EOF
========================== usage =========================
Done!
To use the installed tools and libraries and ABACUS version
compiled with it you will first need to execute at the prompt:
  source ${SETUPFILE}
To build ABACUS by gnu-toolchain, just use:
    ./build_abacus_gnu.sh
To build ABACUS by intel-toolchain, just use:
    ./build_abacus_intel.sh
or you can modify the builder scripts to suit your needs.
"""
EOF


#EOF