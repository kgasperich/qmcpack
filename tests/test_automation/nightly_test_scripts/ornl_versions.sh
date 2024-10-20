#
# Versions should be consistent with setup script
#

# GCC
# Dates at https://gcc.gnu.org/releases.html
#gcc_vnew=14.1.0 # Released 2024-05-06
gcc_vnew=13.2.0 # Released 2023-07-27
gcc_vold=11.4.0 # Released 2023-05-29

gcc_vcuda=11.4.0 # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
gcc_vintel=13.2.0 # Compiler for C++ library used by Intel compiler
gcc_vnvhpc=13.2.0 # Use makelocalrc to configure NVHPC with this compiler 
gcc_vllvmoffload=11.4.0 # Version for LLVM offload builds, should be compatible with CUDA version used

# LLVM 
# Dates at https://releases.llvm.org/
llvm_vnew=18.1.6 # Released 2023-05-18
llvm_voffload=18.1.6
cuda_voffload=12.4.0 # CUDA version for offload builds

# HDF5
# Dates at https://portal.hdfgroup.org/display/support/Downloads
hdf5_vnew=1.14.3 # Released 2023-10-3
hdf5_vold=${hdf5_vnew}

# CMake 
# Dates at https://cmake.org/files/
cmake_vnew=3.27.9 # Relased 2023-10-06
cmake_vold=${cmake_vnew}
#cmake_vold=3.22.4 # Release 2022-04-12

# OpenMPI
# Dates at https://www.open-mpi.org/software/ompi/v5.0/
ompi_vnew=5.0.3 # Released 2024-04-08

# Libxml2
libxml2_v=2.10.3 # Released 2022-12? See https://gitlab.gnome.org/GNOME/libxml2/-/releases

# FFTW
# Dates at http://www.fftw.org/release-notes.html
fftw_vnew=3.3.10 # Released 2021-09-15
fftw_vold=3.3.8 # Released 2018-05-28

# BOOST
# Dates at https://www.boost.org/users/history/
boost_vnew=1.85.0 # Released 2024-04-15
boost_vold=1.79.0 # Released 2022-04-13

# Python
# Use a single version to reduce dependencies. Ideally the spack prefered version.
python_version=3.11.9







