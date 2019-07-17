
# Boilerplate from Introduction to modern CMake


# Do this to Keep IDE's happy according to introduction to modern cmake
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)


add_library( testutils qdpxx_utils.h qdpxx_latticeinit.h qdpxx_latticeinit.cpp
    	reunit.cpp test_env.h test_env.cpp dslashm_w.cpp )
target_link_libraries( testutils libmg qdp kokkos gtest ) 
    	
target_include_directories(testutils PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(testutils PUBLIC "${CMAKE_BINARY_DIR}/include")
target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/core/src")
target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/containers/src")
target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/algorithms/src")
target_include_directories(testutils PUBLIC "${Kokkos_BINARY_DIR}")
    	
# This macro takes the testname and atts an executable from the argumnets
# (ARGN is every argument after the listed one) which allows you to pass a list
# of sources after the testname
macro(package_add_test TESTNAME)
    # Make the executable
    add_executable(${TESTNAME} ${ARGN})

   target_include_directories(${TESTNAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")
   target_include_directories(${TESTNAME} PUBLIC "${CMAKE_BINARY_DIR}/include")
   target_include_directories(${TESTNAME} PUBLIC "${Kokkos_SOURCE_DIR}/core/src")
   target_include_directories(${TESTNAME} PUBLIC "${Kokkos_SOURCE_DIR}/containers/src")
   target_include_directories(${TESTNAME} PUBLIC "${Kokkos_SOURCE_DIR}/algorithms/src")
   target_include_directories(${TESTNAME} PUBLIC "${Kokkos_BINARY_DIR}")

    # Add on the gtest libs
    target_link_libraries(${TESTNAME} testutils )

    # Add the test to CTest
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})

    # I am not 100% sure what tis is for but it is a property
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

# Here TESTNAME is test1
# and ARGN is test1.cpp
package_add_test(test_kokkos test_kokkos.cpp)
package_add_test(test_kokkos_vnode test_kokkos_vnode.cpp)
package_add_test(test_kokkos_vperf  test_kokkos_vperf.cpp)
package_add_test(test_kokkos_vectype test_kokkos_vectype.cpp)
package_add_test(test_kokkos_perf test_kokkos_perf.cpp)
