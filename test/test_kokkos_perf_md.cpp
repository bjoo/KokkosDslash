#include "kokkos_dslash_config.h"
#include "gtest/gtest.h"
//#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"


#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

#include <chrono>
#include <ctime>

using namespace MG;
using namespace MGTesting;
using namespace QDP;


TEST(TestKokkos, TestDslashTime)
{
	IndexArray latdims={{32,32,32,32}};
	int iters=200; 

	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());

	using SpinorType = KokkosCBFineSpinor<MGComplex<REAL32>,4>;
	using FullGaugeType = KokkosFineGaugeField<MGComplex<REAL32>>;
       
	SpinorType  kokkos_spinor_even(info,EVEN);
	SpinorType  kokkos_spinor_odd(info,ODD);
	FullGaugeType  kokkos_gauge(info);

	// Import Gauge Field
	QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	int per_team=32; // Arbitrary for now

	KokkosDslash<MGComplex<REAL32>,
		     MGComplex<REAL32>,
		     MGComplex<REAL32>> D(info,per_team);

	IndexArray cb_latdims = kokkos_spinor_even.GetInfo().GetCBLatticeDimensions();
	double num_sites = static_cast<double>(cb_latdims[0]*cb_latdims[1]*cb_latdims[2]*cb_latdims[3]);

	int cb = EVEN; // Just do this on cb=0 for now
	SpinorType& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
	SpinorType& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;

	QDPLatticeFermionToKokkosCBSpinor(psi_in, in_spinor);

	Kokkos::Timer timer;
#if 1
	int titers=50;
	double best_flops = 0;
	IndexArray best_blocks={1,1,1,1};
	for(IndexType t=cb_latdims[3]; t >= 1; t /= 2) {
		for(IndexType z=cb_latdims[2]; z >= 1; z /= 2) {
			for(IndexType y=cb_latdims[1]; y >= 1; y/=2 ) {
				for(IndexType x = cb_latdims[0]; x >= 1; x/= 2 ) {
					int isign=1;
					IndexType num_blocks = 1;
					num_blocks *= cb_latdims[0]/x;
					num_blocks *= cb_latdims[1]/y;
					num_blocks *= cb_latdims[2]/z;
					num_blocks *= cb_latdims[3]/t;
#if defined( MG_USE_CUDA ) || defined(MG_USE_HIP) 
					if( x*y*z*t <= 256) { 
#else
					if ( num_blocks <= 256) {
#endif
					  timer.reset();
					  for(int i=0; i < titers; ++i) {
					    D(in_spinor,kokkos_gauge,out_spinor,isign,{x,y,z,t});
					  }
					  
#if defined(  MG_USE_CUDA ) || defined (MG_USE_HIP )
					  Kokkos::fence();
#endif
					  double time_taken = timer.seconds();
					  double flops = static_cast<double>(1320.0*num_sites*titers);
					  double floprate = flops/(time_taken*1.0e9);
					  MasterLog(INFO,"Tuning: (Bx,By,Bz,Bt)=(%d,%d,%d,%d) GFLOPS=%lf", x,y,z,t,floprate);
					  if (floprate > best_flops){
					    best_flops = floprate;
					    best_blocks[0]=x;
					    best_blocks[1]=y;
					    best_blocks[2]=z;
					    best_blocks[3]=t;
					  }
					}

					
					}
				}
			}
		}
#else

#if defined ( MG_USE_CUDA ) || defined (MG_USE_HIP)
		IndexArray best_blocks={16,16,1,1};
#else
	IndexArray best_blocks={4,2,2,16};
#endif // MG_USE_CUDA


#endif
	MasterLog(INFO, "Main timing: (Bx,By,Bz,Bt)=(%d,%d,%d,%d)",
				best_blocks[0],best_blocks[1],best_blocks[2],best_blocks[3]);

	for(int rep=0; rep < 10; ++rep ) {
	  int isign = 1;
	  //for(int isign=-1; isign < 2; isign+=2) {
	  // Time it.
	  
	  timer.reset();
	  for(int i=0; i < iters; ++i) {
	    D(in_spinor,kokkos_gauge,out_spinor,isign,best_blocks);
	  }
#if defined (MG_USE_CUDA) || defined(MG_USE_HIP) 
	  //	  Kokkos::fence();
#endif
	  
	  double time_taken = timer.seconds();
	  
	  double rfo = 1.0;
	  double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
	  double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
	  double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);
	  double rfo_bytes_out = (1.0 + rfo)*bytes_out;
	  double flops = static_cast<double>(1320.0*num_sites*iters);
	  
	  MasterLog(INFO,"isign=%d Performance: %lf GFLOPS", isign, flops/(time_taken*1.0e9));
	  MasterLog(INFO,"isign=%d Effective BW (RFO=0): %lf GB/sec",isign, (bytes_in+bytes_out)/(time_taken*1.0e9));
	  MasterLog(INFO,"isign=%d Effective BW (RFO=1): %lf GB/sec",  isign, (bytes_in+rfo_bytes_out)/(time_taken*1.0e9));
	  
	  

	  MasterLog(INFO,"");
	} // rep
}
