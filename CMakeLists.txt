cmake_minimum_required(VERSION 3.13)
project(KokkosDslash VERSION 0.1
                  DESCRIPTION "Kokkos Dslash"
                  LANGUAGES CXX)


# Basic Includes
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)


# Deal with default loglevel
# If code is built in release mode it will be INFO
# Otherwise  it will be build set to DEBUG
# Users can set it of course.
set( MG_DEFAULT_LOGLEVEL "DEBUG" CACHE STRING "Default LogLevel can be INFO, DEBUG, DEBUG2 DEBUG3")
set( MG_VECLEN_SP "4" CACHE STRING "Single Precision Vector Length (default is 4 for AVX2")
option(MG_FORTRANLIKE_COMPLEX "Fortran like complex ordering (r,i,r,i,...r,i) for vectors" OFF)

# Kokkos-Dslash specific options
option(MG_USE_AVX512 "Use AVX512 intrinsics" FALSE)
option(MG_USE_AVX2 "Use AVX512 intrinsics" TRUE)
option(MG_KOKKOS_USE_CUDA "Use CUDA" FALSE)
option(MG_KOKKOS_USE_FLAT_DISPATCH "Flat parallel for..." FALSE)
option(MG_KOKKOS_USE_TEAM_DISPATCH "Team based parallel for" FALSE)
option(MG_KOKKOS_USE_MDRANGE_DISPATCH "MDRange based parallel for" FALSE)

option(MG_KOKKOS_USE_NEIGHBOR_TABLE "Should we compute neighbor index or use a table" TRUE)

if( MG_USE_AVX512 )
	message(STATUS "Using AVX512")
endif()


if( MG_USE_AVX2 )
	message(STATUS "Using AVX2")
endif()

# Find OpenMP for backward compatibiliy
#find_package(OpenMP REQUIRED)
if ( KOKKOS_ENABLE_OPENMP )
  find_package(OpenMP REQUIRED)
endif()

if (Kokkos_ENABLE_OPENMPTARGET)
   message(STATUS "Kokkos is using OpenMPTarget and so am I")
   set(MG_USE_OPENMPTARGET ON)
endif()

# Find QDP++  for tests
list(APPEND CMAKE_MODULE_PATH "${QDPXX_DIR}")
find_package(QDPXX)
if( QDPXX_FOUND ) 
  set( HAVE_QDPXX "1")
  if( QDPXX_PARALLEL_ARCH STREQUAL "parscalar" )
     set(MG_PARSCALAR_ARCH TRUE)
  endif()
endif() 

if( MG_KOKKOS_USE_CUDA ) 


endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/kokkos_dslash_config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/kokkos_dslash_config.h)
  
# Add the Googletest directory and mark its targets etc as advanced
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/kokkos" "extern/kokkos")

add_subdirectory(lib)
#Make tests here
add_subdirectory(test)


# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install the generated headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/sycl_dslash_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# Set some variables to export to the config file if not already set
#set(KokkosDslash_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)

# Make the config file
#configure_package_config_file(SyCLDslashConfig.cmake.in SyCLDslashConfig.cmake INSTALL_DESTINATION lib/cmake/SyCLDslash PATH_VARS SyCLDslash_INCLUDE_DIR)

# Make the version file
#write_basic_package_version_file(
#    SyCLDslashVersion.cmake
#    VERSION ${PACKAGE_VERSION}
#    COMPATIBILITY AnyNewerVersion
#    )

#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashConfig.cmake DESTINATION lib/cmake/SyCLDslash)


#install(EXPORT SyCLDslashTargets NAMESPACE SyCLDslash:: DESTINATION lib/cmake/SyCLDslash )

# Export stuff
# This remains a mystery
#export(PACKAGE SyCLDslash)
