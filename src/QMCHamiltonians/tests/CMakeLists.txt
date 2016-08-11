
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(SRC_DIR hamiltonian)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME unit_test_${SRC_DIR})

SET(SRCS test_bare_kinetic.cpp test_coulomb_pbcAB.cpp test_coulomb_pbcAA.cpp)
IF(HAVE_CUDA)
  SET(SRCS ${SRCS}
      test_coulomb_CUDA.cpp 
)
ENDIF()



ADD_EXECUTABLE(${UTEST_EXE} ${SRCS})
TARGET_LINK_LIBRARIES(${UTEST_EXE} qmc qmcwfs qmcham qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#ADD_TEST(NAME ${UTEST_NAME} COMMAND "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
SET_TESTS_PROPERTIES(${UTEST_NAME} PROPERTIES LABELS "unit")
