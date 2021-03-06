#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers
#//
#// File developed by: Cody A. Melton, cmelton@sandia.gov, Sandia National Laboratories
#//
#// File created by: Cody A. Melton, cmelton@sandia.gov, Sandia National Laboratories
#//////////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(SRC_DIR tools)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME deterministic-unit_test_${SRC_DIR})
SET(UTEST_DIR ${qmcpack_BINARY_DIR}/tests/${SRC_DIR})
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E make_directory "${UTEST_DIR}")

SET(SRCS test_qmcfstool.cpp)
         
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E make_directory "${UTEST_DIR}")
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/simple_Sk.dat" ${UTEST_DIR})
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/simple_input.xml" ${UTEST_DIR})

ADD_EXECUTABLE(${UTEST_EXE} ${SRCS})
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main fstool)
IF(USE_OBJECT_TARGET)
  TARGET_LINK_LIBRARIES(${UTEST_EXE} qmcparticle qmcutil containers platform_omptarget)
ENDIF()

ADD_UNIT_TEST(${UTEST_NAME} 1 1 "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
SET_TESTS_PROPERTIES(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${UTEST_DIR})
IF( "${ENABLE_SANITIZER}" STREQUAL "asan" ) 
  SET_PROPERTY(TEST ${UTEST_NAME} APPEND PROPERTY LABELS noasan)
ENDIF()

# Minimal test for qmc-check-affinity. Check for "OpenMP" in output, no errorcodes.
IF ( HAVE_MPI )
  ADD_TEST( NAME check-affinity_runs COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qmc-check-affinity> )
ELSE()
  ADD_TEST(NAME check-affinity_runs COMMAND $<TARGET_FILE:qmc-check-affinity>)
ENDIF()
SET_TESTS_PROPERTIES(check-affinity_runs PROPERTIES
                         TIMEOUT 120 LABELS "deterministic" PASS_REGULAR_EXPRESSION "OpenMP"
                         PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)

# Basic test for qmc-get-supercell. Cubic primitive vectors, request what should optimally be a 2x2x2 tiling
IF ( HAVE_MPI )
  ADD_TEST( NAME get-supercell_verbose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qmc-get-supercell> --ptvs 1 0 0 0 1 0 0 0 1 --target 8 --maxentry 4 --verbose)
ELSE()
  ADD_TEST(NAME get-supercell_verbose COMMAND $<TARGET_FILE:qmc-get-supercell> --ptvs 1 0 0 0 1 0 0 0 1 --target 8 --maxentry 4 --verbose)
ENDIF()
SET_TESTS_PROPERTIES(get-supercell_verbose PROPERTIES TIMEOUT 120 LABELS "deterministic" PASS_REGULAR_EXPRESSION "radius = 1"
                         PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)
IF ( HAVE_MPI )
  ADD_TEST( NAME get-supercell_terse COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qmc-get-supercell> --ptvs 1 0 0 0 1 0 0 0 1 --target 8)
ELSE()
  ADD_TEST(NAME get-supercell_terse COMMAND $<TARGET_FILE:qmc-get-supercell> --ptvs 1 0 0 0 1 0 0 0 1 --target 8)
ENDIF()
SET_TESTS_PROPERTIES(get-supercell_terse PROPERTIES TIMEOUT 120 LABELS "deterministic"
                         PASS_REGULAR_EXPRESSION "1   2   0   0   0   2   0   0   0   2   2   0   0   0   2   0   0   0   2"
                         PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)

# Check qmc-extract-eshdf-kvectors
SET(KTEST_HDF_INPUT1 ${qmcpack_SOURCE_DIR}/tests/solids/diamondC_1x1x1_pp/pwscf.pwscf.h5)
SET(KTEST_HDF_INPUT2 ${qmcpack_SOURCE_DIR}/tests/solids/diamondC_2x1x1_pp/pwscf.pwscf.h5)
IF ( HAVE_MPI )
  ADD_TEST( NAME extract-eshdf-kvectors_1x1x1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qmc-extract-eshdf-kvectors> ${KTEST_HDF_INPUT1})
ELSE()
  ADD_TEST(NAME extract-eshdf-kvectors_1x1x1 COMMAND $<TARGET_FILE:qmc-extract-eshdf-kvectors> ${KTEST_HDF_INPUT1})
ENDIF()
SET_TESTS_PROPERTIES(extract-eshdf-kvectors_1x1x1 PROPERTIES TIMEOUT 120 LABELS "deterministic"
                        PASS_REGULAR_EXPRESSION "0   0   0"
                        PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)
IF ( HAVE_MPI )
  ADD_TEST( NAME extract-eshdf-kvectors_2x1x1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qmc-extract-eshdf-kvectors> ${KTEST_HDF_INPUT2})
ELSE()
  ADD_TEST(NAME extract-eshdf-kvectors_2x1x1 COMMAND $<TARGET_FILE:qmc-extract-eshdf-kvectors> ${KTEST_HDF_INPUT2})
ENDIF()
SET_TESTS_PROPERTIES(extract-eshdf-kvectors_2x1x1 PROPERTIES TIMEOUT 120 LABELS "deterministic"
                        PASS_REGULAR_EXPRESSION "0   0   0\n0.5   0   0"
                        PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)

# Minimal test for convert4qmc. Check for Usage on output if no arguments. Primarily tested through converter tests.
IF ( HAVE_MPI )
  ADD_TEST( NAME convert4qmc_usage COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:convert4qmc> )
ELSE()
  ADD_TEST(NAME convert4qmc_usage COMMAND $<TARGET_FILE:convert4qmc>)
ENDIF()
SET_TESTS_PROPERTIES(convert4qmc_usage PROPERTIES TIMEOUT 120 LABELS "deterministic"
                        PASS_REGULAR_EXPRESSION "Usage:"
                        PROCESSORS 1 ENVIRONMENT OMP_NUM_THREADS=3 PROCESSOR_AFFINITY TRUE)
