
INCLUDE("${qmcpack_SOURCE_DIR}/CMake/test_labels.cmake")

# Runs Executable Tests

IF (QMC_NO_SLOW_CUSTOM_TESTING_COMMANDS)
  FUNCTION(ADD_QE_TEST)
  ENDFUNCTION()
  FUNCTION(RUN_QE_TEST)
  ENDFUNCTION()
ELSE (QMC_NO_SLOW_CUSTOM_TESTING_COMMANDS)

FUNCTION( ADD_QE_TEST TESTNAME PROCS TEST_BINARY NPOOL WORKDIR TEST_INPUT)
    IF ( HAVE_MPI )
        ADD_TEST( NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${MPIEXEC_PREFLAGS} ${TEST_BINARY} -npool ${NPOOL} -inp ${TEST_INPUT} )
    ELSE()
        ADD_TEST( NAME ${TESTNAME} COMMAND ${TEST_BINARY} -npool 1 -inp ${TEST_INPUT} )
    ENDIF()
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 PROCESSORS ${PROCS} PROCESSOR_AFFINITY TRUE WORKING_DIRECTORY ${WORKDIR} )
    SET_PROPERTY( TEST ${TESTNAME} APPEND PROPERTY LABELS "converter" )
ENDFUNCTION()

FUNCTION( RUN_QE_TEST BASE_NAME SRC_DIR PROCS1 PROCS2 PROCS3 NPOOL1 NPOOL2 NPOOL3 TEST_INPUT_PREFIX TEST_NAME)
    SET( FULL_NAME ${BASE_NAME}-np-${PROCS1}-${PROCS2}-${PROCS3}-nk-${NPOOL1}-${NPOOL2}-${NPOOL3} )
    SET( ${TEST_NAME} ${FULL_NAME} PARENT_SCOPE)
    SET( MY_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/${FULL_NAME} )
    MESSAGE_VERBOSE("Adding test ${FULL_NAME}")
    COPY_DIRECTORY( "${SRC_DIR}" "${MY_WORKDIR}" )
    ADD_QE_TEST(${FULL_NAME}-scf  ${PROCS1} ${QE_PW_DIR}/pw.x         ${NPOOL1} ${MY_WORKDIR} ${TEST_INPUT_PREFIX}-scf.in )
    IF(PROCS2 EQUAL 0)
        ADD_QE_TEST(${FULL_NAME}-pw2x ${PROCS3} ${QE_PW2Q_DIR}/pw2qmcpack.x ${NPOOL3} ${MY_WORKDIR} ${TEST_INPUT_PREFIX}-pw2x.in )
        SET_TESTS_PROPERTIES(${FULL_NAME}-pw2x PROPERTIES DEPENDS ${FULL_NAME}-scf)
    ELSE(PROCS2 EQUAL 0)
        ADD_QE_TEST(${FULL_NAME}-nscf ${PROCS2} ${QE_PW_DIR}/pw.x         ${NPOOL2} ${MY_WORKDIR} ${TEST_INPUT_PREFIX}-nscf.in )
        SET_TESTS_PROPERTIES(${FULL_NAME}-nscf PROPERTIES DEPENDS ${FULL_NAME}-scf)
        ADD_QE_TEST(${FULL_NAME}-pw2x ${PROCS3} ${QE_PW2Q_DIR}/pw2qmcpack.x ${NPOOL3} ${MY_WORKDIR} ${TEST_INPUT_PREFIX}-pw2x.in )
        SET_TESTS_PROPERTIES(${FULL_NAME}-pw2x PROPERTIES DEPENDS ${FULL_NAME}-nscf)
    ENDIF(PROCS2 EQUAL 0)
ENDFUNCTION()

ENDIF(QMC_NO_SLOW_CUSTOM_TESTING_COMMANDS)


FUNCTION( SOFTLINK_H5 SOURCE TARGET PREFIX FILENAME TEST_NAME)
    SET(${TEST_NAME} "LINK_${SOURCE}_TO_${TARGET}" PARENT_SCOPE)
    ADD_TEST( NAME LINK_${SOURCE}_TO_${TARGET} COMMAND ${qmcpack_SOURCE_DIR}/tests/scripts/clean_and_link_h5.sh ${SOURCE}/out/${PREFIX}.pwscf.h5 ${SOURCE}-${TARGET}/${FILENAME} )
    SET_TESTS_PROPERTIES(LINK_${SOURCE}_TO_${TARGET} PROPERTIES DEPENDS ${SOURCE}-pw2x)
    SET_PROPERTY( TEST LINK_${SOURCE}_TO_${TARGET} APPEND PROPERTY LABELS "converter" )
ENDFUNCTION()


