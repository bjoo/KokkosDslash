/*
 * kokkos_defaults.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DEFAULTS_H_
#define TEST_KOKKOS_KOKKOS_DEFAULTS_H_

#include "kokkos_dslash_config.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace MG
{

template<typename T>
using MGComplex = Kokkos::complex<T>;
using HostExec = Kokkos::Serial;

#if defined(MG_USE_CUDA)
  using ExecSpace = Kokkos::Cuda::execution_space;
  using MemorySpace = Kokkos::Cuda::memory_space;
  
  using Layout = Kokkos::LayoutLeft;
  using GaugeLayout = Kokkos::LayoutLeft;
  using NeighLayout = Kokkos::LayoutLeft;
#elif defined(MG_USE_HIP)
  using ExecSpace = Kokkos::Cuda::execution_space;
  using MemorySpace = Kokkos::Cuda::memory_space;
  
  using Layout = Kokkos::LayoutLeft;
  using GaugeLayout = Kokkos::LayoutLeft;
  using NeighLayout = Kokkos::LayoutLeft;
#else
  using ExecSpace = Kokkos::OpenMP::execution_space;
  using MemorySpace = Kokkos::OpenMP::memory_space;
  using Layout = Kokkos::LayoutRight;
  using GaugeLayout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::OpenMP::array_layout;
#endif

#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
using ThreadExecPolicy =  Kokkos::TeamPolicy<ExecSpace,Kokkos::LaunchBounds<128,1>>;
using SimpleRange = Kokkos::RangePolicy<ExecSpace>;

#else
using ThreadExecPolicy = Kokkos::TeamPolicy<ExecSpace>;
using SimpleRange = Kokkos::RangePolicy<ExecSpace>;
#endif


using TeamHandle =  ThreadExecPolicy::member_type;
using VectorPolicy = Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,TeamHandle>;

#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
  // Try an N-dimensional threading policy for cache blocking
 using MDPolicy =  Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,
		 Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>, Kokkos::LaunchBounds<512,1>>;

#else 
  // Try an N-dimensional threading policy for cache blocking
 using MDPolicy =  Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,
		 Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>>;

#endif  
}

#endif /* TEST_KOKKOS_KOKKOS_DEFAULTS_H_ */
