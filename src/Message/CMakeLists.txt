#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//                    Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


SET(COMM_SRCS
  Communicate.cpp
  AppAbort.cpp
  MPIObjectBase.cpp
)

ADD_LIBRARY(message ${COMM_SRCS})
TARGET_LINK_LIBRARIES(message PUBLIC platform_host)

IF(HAVE_MPI)
  TARGET_LINK_LIBRARIES(message PUBLIC MPI::MPI_CXX Boost::boost)
  SET_PROPERTY(TARGET message APPEND PROPERTY INCLUDE_DIRECTORIES "${qmcpack_SOURCE_DIR}/external_codes/mpi_wrapper")
  SET_PROPERTY(TARGET message APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${qmcpack_SOURCE_DIR}/external_codes/mpi_wrapper")
ENDIF()

ADD_LIBRARY(catch_main catch_main.cpp)
TARGET_INCLUDE_DIRECTORIES(catch_main PUBLIC "${PROJECT_SOURCE_DIR}/external_codes/catch")
# This plus using the definition of the catch_benchmark_main as INTERFACE target
# would allow just building catch main once for benchmarking and mpi enabled testing
# But it causes a segv on power9 with clang12 and on alcf-gnu-mkl18 
# TARGET_COMPILE_DEFINITIONS(catch_main PRIVATE "CATCH_CONFIG_ENABLE_BENCHMARKING")
TARGET_LINK_LIBRARIES(catch_main message)
IF(HAVE_MPI)
  SET_PROPERTY(TARGET catch_main APPEND PROPERTY COMPILE_DEFINITIONS "CATCH_MAIN_HAVE_MPI")
ENDIF()

add_library(catch_main_no_mpi catch_main.cpp)
TARGET_INCLUDE_DIRECTORIES(catch_main_no_mpi PUBLIC "${PROJECT_SOURCE_DIR}/external_codes/catch")

# ADD_LIBRARY(catch_benchmark_main INTERFACE)
# SET_TARGET_PROPERTIES(catch_benchmark_main PROPERTIES INTERFACE_COMPILE_DEFINITIONS "CATCH_CONFIG_ENABLE_BENCHMARKING")
# TARGET_LINK_LIBRARIES(catch_benchmark_main INTERFACE catch_main)

ADD_LIBRARY(catch_benchmark_main catch_main.cpp)
TARGET_COMPILE_DEFINITIONS(catch_benchmark_main PUBLIC "CATCH_CONFIG_ENABLE_BENCHMARKING")
TARGET_LINK_LIBRARIES(catch_benchmark_main message)
IF(HAVE_MPI)
  SET_PROPERTY(TARGET catch_benchmark_main APPEND PROPERTY COMPILE_DEFINITIONS "CATCH_MAIN_HAVE_MPI")
ENDIF()
TARGET_INCLUDE_DIRECTORIES(catch_benchmark_main PUBLIC "${PROJECT_SOURCE_DIR}/external_codes/catch")
