
#include "Kokkos_Core.hpp"
#include "gtest/gtest.h"
#include "utils/initialize.h"


	/* This is a convenience routine to setup the test environment for GTest and its layered test environments */
	int main(int argc, char **argv)
	{

		  ::testing::InitGoogleTest(&argc, argv);
		  ::MG::initialize(&argc, &argv);
		  auto ret_val =  RUN_ALL_TESTS();
		  ::MG::finalize();
		  return ret_val;

	}

